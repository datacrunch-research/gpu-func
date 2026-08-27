import unittest

from gpu_func_cli.parser import build_parser


class ParserTests(unittest.TestCase):
    def test_custom_parser_defaults_and_repeatable_args(self):
        args = build_parser().parse_args(
            ["custom", "profile", "kernel.cu", "--arg", "7", "--arg", "x"]
        )

        self.assertEqual(args.command_name, "custom")
        self.assertEqual(args.custom_command, "profile")
        self.assertEqual(args.source, "kernel.cu")
        self.assertEqual(args.arg, ["7", "x"])
        self.assertIsNone(args.gpu)
        self.assertEqual(args.gpu_count, 1)
        self.assertEqual(args.ncu_args, "--set basic")
        self.assertEqual(args.nvtx_range, "profile_kernel")

    def test_remote_resource_options_parse_human_sizes(self):
        args = build_parser().parse_args(
            [
                "custom",
                "run",
                "kernel.cu",
                "--gpu-type",
                "gb300",
                "--gpu-count",
                "4",
                "--memory",
                "384GiB",
                "--storage",
                "256GiB",
                "--env",
                "MODE=test",
            ]
        )

        self.assertEqual(args.gpu_type, "gb300")
        self.assertEqual(args.gpu_count, 4)
        self.assertEqual(args.memory, 384 * 1024**3)
        self.assertEqual(args.storage, 256 * 1024**3)
        self.assertEqual(args.env, [("MODE", "test")])

    def test_exercise_parser_accepts_specs_and_source_file(self):
        args = build_parser().parse_args(
            [
                "exercise",
                "01-haxpy",
                "benchmark",
                "benchmarks/01_aligned_small.txt",
                "--file",
                "haxpy.cu",
            ]
        )

        self.assertEqual(args.command_name, "exercise")
        self.assertEqual(args.exercise_id, "01-haxpy")
        self.assertEqual(args.exercise_command, "benchmark")
        self.assertEqual(args.specs, ["benchmarks/01_aligned_small.txt"])
        self.assertEqual(args.source_file, "haxpy.cu")

    def test_report_summary_parser_requires_subcommand(self):
        args = build_parser().parse_args(["report", "summary", "profile.ncu-rep", "--per-kernel"])

        self.assertEqual(args.command_name, "report")
        self.assertEqual(args.report_command, "summary")
        self.assertEqual(args.report, "profile.ncu-rep")
        self.assertIs(args.per_kernel, True)


if __name__ == "__main__":
    unittest.main()
