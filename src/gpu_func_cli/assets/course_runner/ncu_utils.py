import glob
import os
import re
import sys


class NCUError(Exception):
    """Base class for NCU-related errors."""

    pass


class NCUReportNotFoundError(NCUError):
    """Raised when the NCU report file is not found."""

    pass


class NCUMetricNotFoundError(NCUError):
    """Raised when a requested metric is not found in the report."""

    pass


class NCULibraryNotFoundError(NCUError):
    """Raised when ncu_report library is not available."""

    pass


def find_ncu_python_path():
    """Try to find the path to ncu_report.py in common NVIDIA installation locations."""
    search_paths = [
        "/opt/nvidia/nsight-compute/*/extras/python/",
        "/usr/local/cuda/nsight-compute-*/extras/python/",
    ]

    found_paths = []
    for pattern in search_paths:
        found_paths.extend(glob.glob(pattern))

    if not found_paths:
        return None

    # Sort to get the latest version
    found_paths.sort(reverse=True)
    return found_paths[0]


# Add NCU Python path to sys.path (if found at a well-known install location).
# This is best-effort: if it isn't found, ncu_report may still be importable via
# PYTHONPATH (e.g. the Nsight Compute.app bundle on macOS), so we always attempt
# the import below regardless of whether the glob matched.
ncu_path = find_ncu_python_path()
if ncu_path and ncu_path not in sys.path:
    sys.path.append(ncu_path)

try:
    import ncu_report
except ImportError:
    ncu_report = None


def simplify_gpu_name(full_name):
    """Simplify a full GPU name, e.g., 'NVIDIA GeForce RTX 3080' -> 'rtx3080'."""
    if not full_name:
        return None

    full_name = full_name.lower()

    # Mapping for common GPUs
    if "h100" in full_name:
        return "h100"
    if "a100" in full_name:
        return "a100"
    if "v100" in full_name:
        return "v100"
    if "t4" in full_name:
        return "t4"
    if "b200" in full_name:
        return "b200"
    if "b300" in full_name:
        return "b300"

    # RTX Pro blackwell
    if "rtx pro blackwell" in full_name:
        return "rtxproblackwell"
    pro_match = re.search(r"rtx\s*pro\s*(\d+)\s*blackwell", full_name)
    if pro_match:
        return f"b{pro_match.group(1)}"

    # Handle gaming RTX cards (e.g., "nvidia geforce rtx 5060 ti" -> "rtx5060ti")
    rtx_match = re.search(r"rtx\s*(\d+\w*)", full_name)
    if rtx_match:
        return f"rtx{rtx_match.group(1)}"

    # Fallback: remove spaces and non-alphanumeric
    simplified = re.sub(
        r"[^a-z0-9]",
        "",
        full_name.replace("nvidia", "")
        .replace("geforce", "")
        .replace("tesla", "")
        .replace("quadro", ""),
    )
    return simplified if simplified else "gpu"


def get_gpu_name_from_report(report_path):
    return get_ncu_metric(report_path, "", "device__attribute_display_name")


def get_ncu_metric(report_path, kernel_name_pattern, metric_name):
    """
    Extract a metric from an NCU report.
    kernel_name_pattern can be a substring of the kernel name.
    metric_name can be a simple metric name or a metric with an instance name,
    e.g. "sass__inst_executed_per_opcode[FFMA]"
    """
    # Support for instance-specific metrics like metric_name[instance_name]
    instance_name = None
    base_metric_name = metric_name
    if "[" in metric_name and metric_name.endswith("]"):
        base_metric_name, instance_name = metric_name[:-1].split("[", 1)

    try:
        report = load_report(report_path)
        for i in range(report.num_ranges()):
            range_obj = report.range_by_idx(i)
            for j in range(range_obj.num_actions()):
                action = range_obj.action_by_idx(j)
                if kernel_name_pattern in action.name():
                    metric = action.metric_by_name(base_metric_name)
                    if metric:
                        if instance_name:
                            if not metric.has_correlation_ids():
                                raise NCUMetricNotFoundError(
                                    f"Metric {base_metric_name} does not have instances"
                                )

                            cids = metric.correlation_ids()
                            found_idx = -1
                            for k in range(cids.num_instances()):
                                if cids.as_string(k) == instance_name:
                                    found_idx = k
                                    break

                            if found_idx == -1:
                                raise NCUMetricNotFoundError(
                                    f"Instance {instance_name} not found in metric {base_metric_name}"
                                )

                            return _format_metric_value(metric, found_idx)
                        else:
                            return _format_metric_value(metric)

        raise NCUMetricNotFoundError(
            f"Metric {metric_name} not found for kernel {kernel_name_pattern}"
        )
    except NCUError:
        raise
    except Exception as e:
        raise NCUError(str(e)) from e


def _format_metric_value(metric, index=None):
    """Helper to format metric value from IMetric object."""
    # Try different accessors depending on what's available and returns a value
    try:
        val = metric.as_string(index) if index is not None else metric.as_string()
        if val == "None" or val is None:
            raise ValueError("None string")
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Some metrics might be double or uint64
        try:
            # Check if it's an integer value represented as double
            dval = metric.as_double(index) if index is not None else metric.as_double()

            val = str(int(dval)) if dval.is_integer() else f"{dval:.2f}"
        except (AttributeError, RuntimeError, TypeError, ValueError):
            try:
                uint_value = metric.as_uint64(index) if index is not None else metric.as_uint64()
                val = str(uint_value)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                val = "N/A"
    return val


def get_ncu_opcodes(report_path, kernel_name_pattern, metric_name="sass__inst_executed_per_opcode"):
    """
    Extract all instances (opcodes) and their values for a given metric.
    """
    try:
        report = load_report(report_path)
        for i in range(report.num_ranges()):
            range_obj = report.range_by_idx(i)
            for j in range(range_obj.num_actions()):
                action = range_obj.action_by_idx(j)
                if kernel_name_pattern in action.name():
                    metric = action.metric_by_name(metric_name)
                    if metric:
                        if not metric.has_correlation_ids():
                            raise NCUMetricNotFoundError(
                                f"Metric {metric_name} does not have instances"
                            )

                        cids = metric.correlation_ids()
                        results = {}
                        for k in range(cids.num_instances()):
                            opcode = cids.as_string(k)
                            val_str = _format_metric_value(metric, k)
                            try:
                                results[opcode] = float(val_str)
                            except ValueError:
                                results[opcode] = val_str
                        return results

        raise NCUMetricNotFoundError(
            f"Metric {metric_name} not found for kernel {kernel_name_pattern}"
        )
    except NCUError:
        raise
    except Exception as e:
        raise NCUError(str(e)) from e


def get_ncu_stall_reasons(report_path, kernel_name_pattern):
    """
    Extract warp stall reasons from an NCU report.
    """
    prefix = "smsp__pcsamp_warps_issue_stalled_"
    try:
        report = load_report(report_path)
        for i in range(report.num_ranges()):
            range_obj = report.range_by_idx(i)
            for j in range(range_obj.num_actions()):
                action = range_obj.action_by_idx(j)
                if kernel_name_pattern in action.name():
                    results = {}
                    for metric_name in action.metric_names():
                        if metric_name.startswith(prefix) and not metric_name.endswith(
                            "_not_issued"
                        ):
                            reason = metric_name[len(prefix) :].replace("_", " ").title()
                            val = action.metric_by_name(metric_name).as_double()
                            if val > 0:
                                results[reason] = val
                    return results

        raise NCUMetricNotFoundError(f"Stall metrics not found for kernel {kernel_name_pattern}")
    except NCUError:
        raise
    except Exception as e:
        raise NCUError(str(e)) from e


def load_report(report_path):
    if ncu_report is None:
        raise NCULibraryNotFoundError("ncu_report library not found")

    if not os.path.exists(report_path):
        raise NCUReportNotFoundError(f"Report file {report_path} not found")

    return ncu_report.load_report(str(report_path))


if __name__ == "__main__":
    # Test
    if len(sys.argv) > 3:
        print(get_ncu_metric(sys.argv[1], sys.argv[2], sys.argv[3]))
    else:
        print("Usage: ncu_utils.py <report_path> <kernel_name_pattern> <metric_name>")
        # Example test
        report = "examples/reports/saxpy.ncu-rep"
        if os.path.exists(report):
            print(
                f"Test saxpy inst_executed: {get_ncu_metric(report, 'saxpy', 'sm__inst_executed.sum')}"
            )
