# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DTK gpu-hpl benchmark family."""

import os
import unittest

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks.micro_benchmarks.dtk_hpl_mxp_performance import DtkHplMxpBenchmark
from superbench.benchmarks.micro_benchmarks.dtk_hpl_performance import DtkHplBenchmark


class DtkHplFamilyBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for DTK gpu-hpl and gpu-hpl-mxp benchmarks."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/run_rochpl', 'bin/run_rochplmxp'])

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

    def test_dtk_hpl_preprocess_generates_dat_file(self):
        """Test DTK gpu-hpl dat file and command generation."""
        benchmark = DtkHplBenchmark('gpu-hpl')

        self.assertTrue(benchmark._preprocess())

        dat_file_name = 'HPL-WC10R2R32_TTN8_P1_Q1_N45312_NB384.dat'
        out_file_name = 'HPL-WC10R2R32_TTN8_P1_Q1_N45312_NB384.out'
        self.assertEqual(os.path.join(self._tmp_dir, 'bin', dat_file_name), benchmark._dat_path)
        self.assertEqual(os.path.join(self._tmp_dir, 'bin', out_file_name), benchmark._out_path)
        self.assertEqual(1, len(benchmark._commands))
        self.assertIn(f'run_rochpl -P 1 -Q 1 --it 1 -i {dat_file_name}', benchmark._commands[0])

        with open(benchmark._dat_path, 'r') as dat_file:
            dat_content = dat_file.read()

        self.assertIn(f'{out_file_name} output file name (if any)', dat_content)
        self.assertIn('45312         Ns', dat_content)
        self.assertIn('384         NBs', dat_content)
        self.assertIn('0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)', dat_content)
        self.assertIn('2            PFACTs (0=left, 1=Crout, 2=Right)', dat_content)
        self.assertIn('8            memory alignment in double (> 0)', dat_content)

    def test_dtk_hpl_mxp_preprocess_generates_dat_file(self):
        """Test DTK gpu-hpl-mxp dat file and command generation."""
        benchmark = DtkHplMxpBenchmark('gpu-hpl-mxp', parameters='--P 4 --Q 1 --N 8192 --NB 4096 --BCAST 1 --it 6')

        self.assertTrue(benchmark._preprocess())

        dat_file_name = 'HPL-MxP-WC1_P4_Q1_N8192_NB4096.dat'
        out_file_name = 'HPL-MxP-WC1_P4_Q1_N8192_NB4096.out'
        self.assertEqual(os.path.join(self._tmp_dir, 'bin', dat_file_name), benchmark._dat_path)
        self.assertEqual(os.path.join(self._tmp_dir, 'bin', out_file_name), benchmark._out_path)
        self.assertEqual(1, len(benchmark._commands))
        self.assertIn(f'run_rochplmxp -P 4 -Q 1 --it 6 -i {dat_file_name}', benchmark._commands[0])

        with open(benchmark._dat_path, 'r') as dat_file:
            dat_content = dat_file.read()

        self.assertIn(f'{out_file_name} output file name (if any)', dat_content)
        self.assertIn('8192         Ns', dat_content)
        self.assertIn('4096         NBs', dat_content)
        self.assertIn('4            P', dat_content)
        self.assertIn('1            Q', dat_content)
        self.assertIn('1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)', dat_content)


if __name__ == '__main__':
    unittest.main()
