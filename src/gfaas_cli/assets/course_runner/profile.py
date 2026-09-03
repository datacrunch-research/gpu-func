# ---------------------------------------------------------------------------
# Curated metric extraction.
#
# A flat dump of every metric (action.metric_names()) is both noisy and
# lossy: instanced metrics (per-opcode, per-stall-reason) collapse to a single
# scalar through as_string(), hiding the per-instance breakdown. And derived
# metrics — arithmetic intensity, achieved/peak ratios — aren't in the report
# at all; they're computed from raws. So we curate instead.
#
# CURATED_METRICS lists the raw metrics to pull. A trailing "[*]" marks an
# instanced metric whose full instance->value breakdown we want (fetched via
# get_ncu_opcodes), stored as a nested dict. Everything else is a scalar.
#
# DERIVED_METRICS maps a derived name to a function over the fetched raw dict;
# each returns a value or None (None when its inputs are missing on this arch).
# Derived keys are namespaced under "derived." in the merged result.
# ---------------------------------------------------------------------------
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .ncu_utils import load_report, simplify_gpu_name

AGG_SUM = "sum"
AGG_MEAN = "mean"
AGG_UNIQUE = "unique"


@dataclass
class MetricSpec:
    name: str
    display: str
    source: list[str]
    aggregate: str


CURATED_METRICS = [
    # Timing / launch
    # MetricSpec("grid_size", "Grid Size", ["launch__grid_size"]),
    # MetricSpec("block_size", "Block Size", ["launch__block_size"]),
    # MetricSpec("registers_per_thread", "Registers per Thread", ["launch__registers_per_thread"]),
    # MetricSpec("shared_mem_per_block_static", "Static Shared Memory per Block", ["launch__shared_mem_per_block_static"]),
    # Throughput (% of peak)
    # MetricSpec("gpu_memory_throughput", "Memory Throughput", ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]),
    # Occupancy
    MetricSpec(
        "occupancy", "Occupancy", ["sm__warps_active.avg.pct_of_peak_sustained_active"], AGG_MEAN
    ),
    # MetricSpec("occupancy_limit_registers", "Occupancy Limit (Registers)", ["launch__occupancy_limit_registers"]),
    # MetricSpec("occupancy_limit_shared_mem", "Occupancy Limit (Shared Memory)", ["launch__occupancy_limit_shared_mem"]),
    # DRAM traffic (bytes)
    MetricSpec(
        "dram_read_bytes",
        "DRAM Read Bytes",
        ["dram__bytes_read.sum", "dram__bytes_op_read.sum"],
        AGG_SUM,
    ),
    MetricSpec(
        "dram_write_bytes",
        "DRAM Write Bytes",
        ["dram__bytes_write.sum", "dram__bytes_op_write.sum"],
        AGG_SUM,
    ),
    MetricSpec(
        "dram_throughput",
        "DRAM Throughput",
        [
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        AGG_MEAN,
    ),
    MetricSpec(
        "sm_throughput",
        "SM Throughput",
        ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        AGG_MEAN,
    ),
    MetricSpec("instructions", "Instructions", ["smsp__inst_executed.sum"], AGG_SUM),
    MetricSpec("cycles", "Cycles", ["gpc__cycles_elapsed.max"], AGG_SUM),
    MetricSpec("duration", "Duration", ["gpu__time_duration.sum"], AGG_SUM),
    MetricSpec("loads", "Loads", ["sass__inst_executed_global_loads"], AGG_SUM),
    MetricSpec("stores", "Stores", ["sass__inst_executed_global_stores"], AGG_SUM),
    MetricSpec("ldgsts", "LDGSTS", ["smsp__inst_executed_op_ldgsts.sum"], AGG_SUM),
    MetricSpec("device_name", "Device", ["device__attribute_display_name"], AGG_UNIQUE),
    MetricSpec("cc_major", "CC Major", ["device__attribute_compute_capability_major"], AGG_UNIQUE),
    MetricSpec("cc_minor", "CC Minor", ["device__attribute_compute_capability_minor"], AGG_UNIQUE),
    MetricSpec(
        "l2_write_sectors", "L2 Write Sectors", ["l1tex__m_l1tex2xbar_write_sectors.sum"], AGG_SUM
    ),
    # L1 Cache ---> Compressor
    # L2 sectors (32B)
    # MetricSpec("l2_tex_read_sectors", "L2 Texture Read Sectors", ["lts__t_sectors_srcunit_tex_op_read.sum"]),
    # MetricSpec("l2_tex_write_sectors", "L2 Texture Write Sectors", ["lts__t_sectors_srcunit_tex_op_write.sum"]),
    # Global memory coalescing: sectors vs requests for global loads/stores
    # MetricSpec("global_load_sectors", "Global Load Sectors", ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]),
    # MetricSpec("global_load_requests", "Global Load Requests", ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"]),
    # MetricSpec("global_store_sectors", "Global Store Sectors", ["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"]),
    # MetricSpec("global_store_requests", "Global Store Requests", ["l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"]),
    # Instructions
    # MetricSpec("instructions_executed", "Instructions Executed", ["smsp__inst_executed.sum"]),
    # MetricSpec("sass_instructions_executed", "SASS Instructions Executed", ["sm__sass_thread_inst_executed.sum"]),
    # Tensor pipe utilization
    # MetricSpec("tensor_pipe_utilization", "Tensor Pipe Utilization", ["sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"]),
    # Instanced: per-opcode instruction mix
    # "sass__inst_executed_per_opcode[*]",
]

if False:
    RAW_METRIC_LIST = [
        "device__attribute_architecture",
        "device__attribute_chip",
        "device__attribute_clock_rate",
        "device__attribute_display_name",
        "device__attribute_ecc_enabled",
        "device__attribute_memory_clock_rate",
    ]


def _num(raw, key):
    """Pull a scalar metric as float, or None if absent/non-numeric."""
    v = raw.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ratio(num, den):
    a, b = num, den
    if a is None or b is None or b == 0:
        return None
    return a / b


if False:
    DERIVED_METRICS = {
        # Sectors per request: 1.0 is perfectly coalesced, higher is worse.
        "global_ld_sectors_per_req": lambda r: _ratio(
            _num(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
            _num(r, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"),
        ),
        "global_st_sectors_per_req": lambda r: _ratio(
            _num(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"),
            _num(r, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"),
        ),
        # Total DRAM bytes moved.
        "dram_bytes_total": lambda r: (
            None
            if _num(r, "dram__bytes_read.sum") is None and _num(r, "dram__bytes_write.sum") is None
            else (_num(r, "dram__bytes_read.sum") or 0.0)
            + (_num(r, "dram__bytes_write.sum") or 0.0)
        ),
        # Achieved DRAM bandwidth (bytes / second), from bytes and duration (ns).
        "dram_achieved_gbps": lambda r: _ratio(
            ((_num(r, "dram__bytes_read.sum") or 0.0) + (_num(r, "dram__bytes_write.sum") or 0.0)),
            _num(r, "gpu__time_duration.sum"),
        ),  # bytes/ns == GB/s
    }


def extract_from_action(action, metric: str):
    value = action.metric_by_name(metric)
    if value is None:
        # print("Missing",  metric)
        return None

    # print(value.kind())
    # print(value.metric_type())
    return value.value()


def extract_curated_metrics(report_path):
    """
    Extract a metric from an NCU report.
    kernel_name_pattern can be a substring of the kernel name.
    metric_name can be a simple metric name or a metric with an instance name,
    e.g. "sass__inst_executed_per_opcode[FFMA]"
    """

    kernel_results: defaultdict[str, dict[str, Any]] = defaultdict(dict)

    report = load_report(report_path)
    for i in range(report.num_ranges()):
        range_obj = report.range_by_idx(i)
        for j in range(range_obj.num_actions()):
            action = range_obj.action_by_idx(j)
            this_kernel = {}
            for metric in CURATED_METRICS:
                for candidate in metric.source:
                    value = extract_from_action(action, candidate)
                    if value is not None:
                        this_kernel[metric.name] = value
                        break
            kernel_results[action.name()] = this_kernel
    first = next(iter(kernel_results.keys()))
    return kernel_results, simplify_gpu_name(kernel_results[first]["device_name"])


def aggregate_profiling_info(profiling: dict):
    num_kernels = len(profiling)
    metric_specs = {metric.name: metric for metric in CURATED_METRICS}
    result = defaultdict(list)
    for metrics in profiling.values():
        for metric, value in metrics.items():
            result[metric].append(value)

    total_duration = sum(result["duration"])
    final: dict[str, Any] = {}
    for metric, values in result.items():
        agg = metric_specs[metric].aggregate
        if agg == AGG_MEAN:
            wvalues = [v * t for v, t in zip(values, result["duration"], strict=True)]
            final[metric] = sum(wvalues) / total_duration
        elif agg == AGG_SUM:
            final[metric] = sum(values)
        elif agg == AGG_UNIQUE:
            all_vals = set(values)
            assert len(all_vals) == 1
            final[metric] = next(iter(all_vals))
        else:
            raise ValueError(f"Unknown aggregation type: {agg}")

    final["num_kernels"] = num_kernels

    return final
