# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DTK gpu-hpl benchmark family."""

import unittest

from superbench.benchmarks.micro_benchmarks.dtk_hpl_mxp_performance import DtkHplMxpBenchmark
from superbench.benchmarks.micro_benchmarks.dtk_hpl_performance import DtkHplBenchmark


class DtkHplFamilyBenchmarkTest(unittest.TestCase):
    """Tests for DTK gpu-hpl and gpu-hpl-mxp benchmarks."""
    def _parse_args(self, benchmark):
        """Parse benchmark arguments without running preprocess."""
        benchmark.add_parser_arguments()
        ret, args, _ = benchmark.parse_args()
        if ret:
            benchmark._args = args
        return ret, args

    def test_dtk_hpl_default_metric_workload(self):
        """Test DTK gpu-hpl default workload formatting."""
        benchmark = DtkHplBenchmark('gpu-hpl')
        ret, args = self._parse_args(benchmark)

        self.assertTrue(ret)
        self.assertEqual(1, args.P)
        self.assertEqual(1, args.Q)
        self.assertEqual(45312, args.N)
        self.assertEqual(384, args.NB)
        self.assertEqual(0, args.BCAST)

        benchmark._tv = benchmark._format_tv()
        self.assertEqual('WC10R2R32_TTN8', benchmark._tv)
        self.assertEqual('WC10R2R32_TTN8_P1_Q1_N45312_NB384', benchmark._format_workload())

    def test_dtk_hpl_mxp_default_metric_workload(self):
        """Test DTK gpu-hpl-mxp default workload formatting."""
        benchmark = DtkHplMxpBenchmark('gpu-hpl-mxp')
        ret, args = self._parse_args(benchmark)

        self.assertTrue(ret)
        self.assertEqual(1, args.P)
        self.assertEqual(1, args.Q)
        self.assertEqual(61440, args.N)
        self.assertEqual(2560, args.NB)
        self.assertEqual(0, args.BCAST)

        benchmark._tv = benchmark._format_tv()
        self.assertEqual('WC0', benchmark._tv)
        self.assertEqual('WC0_P1_Q1_N61440_NB2560', benchmark._format_workload())

    def test_dtk_hpl_sample_metric_workload(self):
        """Test DTK gpu-hpl workload formatting with sample parameters."""
        benchmark = DtkHplBenchmark('gpu-hpl', parameters='--P 4 --Q 1 --N 8192 --NB 512 --BCAST 1')
        ret, _ = self._parse_args(benchmark)

        self.assertTrue(ret)
        benchmark._tv = benchmark._format_tv()
        self.assertEqual('WC11R2R32_TTN8', benchmark._tv)
        self.assertEqual('WC11R2R32_TTN8_P4_Q1_N8192_NB512', benchmark._format_workload())

    def test_dtk_hpl_mxp_sample_metric_workload(self):
        """Test DTK gpu-hpl-mxp workload formatting with sample parameters."""
        benchmark = DtkHplMxpBenchmark('gpu-hpl-mxp', parameters='--P 4 --Q 1 --N 8192 --NB 4096 --BCAST 1')
        ret, _ = self._parse_args(benchmark)

        self.assertTrue(ret)
        benchmark._tv = benchmark._format_tv()
        self.assertEqual('WC1', benchmark._tv)
        self.assertEqual('WC1_P4_Q1_N8192_NB4096', benchmark._format_workload())

    def test_dtk_hpl_only_arguments_are_not_accepted_by_mxp(self):
        """Test rocHPL-only arguments are not accepted by gpu-hpl-mxp."""
        hpl_benchmark = DtkHplBenchmark('gpu-hpl', parameters='--PFACT 2')
        hpl_mxp_benchmark = DtkHplMxpBenchmark('gpu-hpl-mxp', parameters='--PFACT 2')

        hpl_ret, _ = self._parse_args(hpl_benchmark)
        hpl_mxp_ret, _ = self._parse_args(hpl_mxp_benchmark)

        self.assertTrue(hpl_ret)
        self.assertFalse(hpl_mxp_ret)


if __name__ == '__main__':
    unittest.main()
