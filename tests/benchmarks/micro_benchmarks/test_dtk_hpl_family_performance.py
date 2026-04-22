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

    def _write_output_file(self, benchmark, output):
        """Write generated HPL output for result parsing tests."""
        with open(benchmark._out_path, 'w') as output_file:
            output_file.write(output)

    def _load_data_file(self, file_name):
        """Load test data file content."""
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', file_name)
        with open(data_path, 'r') as data_file:
            return data_file.read()

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
        self.assertEqual(0, args.warmup)
        self.assertEqual(1, args.iterations)
        self.assertEqual('max', args.reduce_op)

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

    def test_dtk_hpl_invalid_sampling_arguments(self):
        """Test invalid HPL sampling arguments are rejected."""
        self.assertFalse(DtkHplBenchmark('gpu-hpl', parameters='--warmup -1')._preprocess())
        self.assertFalse(DtkHplBenchmark('gpu-hpl', parameters='--iterations 0')._preprocess())

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
        benchmark = DtkHplMxpBenchmark(
            'gpu-hpl-mxp', parameters='--P 4 --Q 1 --N 8192 --NB 4096 --BCAST 1 --warmup 1 --iterations 5'
        )

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

    def test_dtk_hpl_result_parsing_with_warmup_and_max_reduce(self):
        """Test DTK gpu-hpl parses generated output and reduces by max FLOPS."""
        benchmark = DtkHplBenchmark(
            'gpu-hpl', parameters='--P 4 --Q 1 --N 8192 --NB 512 --BCAST 1 --warmup 1 --iterations 3 --reduce-op max'
        )

        self.assertTrue(benchmark._preprocess())
        self._write_output_file(
            benchmark, """
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.71              5.167e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.69              5.338e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.67              5.437e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.67              5.450e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
"""
        )

        self.assertTrue(benchmark._process_raw_result(0, 'stdout noise'))

        workload = 'WC11R2R32_TTN8_P4_Q1_N8192_NB512'
        self.assertEqual(545.0, benchmark.result[f'{workload}_flops'][0])
        self.assertEqual(0.67, benchmark.result[f'{workload}_time'][0])
        self.assertEqual(1, benchmark.result[f'{workload}_tests_pass'][0])
        self.assertIn('raw_output_0', benchmark.raw_data)
        self.assertIn('hpl_output_0', benchmark.raw_data)

    def test_dtk_hpl_mxp_result_parsing_does_not_match_output_n(self):
        """Test DTK gpu-hpl-mxp parses output when output N differs from input N."""
        benchmark = DtkHplMxpBenchmark(
            'gpu-hpl-mxp',
            parameters='--P 4 --Q 1 --N 8192 --NB 4096 --BCAST 1 --warmup 1 --iterations 2 --reduce-op min'
        )

        self.assertTrue(benchmark._preprocess())
        self._write_output_file(
            benchmark, """
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC1            16384  4096     4     1               0.78              3.742e+03
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0891789 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC1            16384  4096     4     1               0.80              3.665e+03
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0891789 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC1            16384  4096     4     1               0.78              3.767e+03
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0891789 ...... FAILED
"""
        )

        self.assertTrue(benchmark._process_raw_result(0, 'stdout noise'))

        workload = 'WC1_P4_Q1_N8192_NB4096'
        self.assertEqual(3665.0, benchmark.result[f'{workload}_flops'][0])
        self.assertEqual(0.80, benchmark.result[f'{workload}_time'][0])
        self.assertEqual(0, benchmark.result[f'{workload}_tests_pass'][0])

    def test_dtk_hpl_result_parsing_with_median_reduce(self):
        """Test DTK gpu-hpl median reduce uses reciprocal time."""
        benchmark = DtkHplBenchmark(
            'gpu-hpl', parameters='--P 4 --Q 1 --N 8192 --NB 512 --BCAST 1 --iterations 4 --reduce-op median'
        )

        self.assertTrue(benchmark._preprocess())
        self._write_output_file(
            benchmark, """
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.80              5.000e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.40              6.000e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.20              7.000e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WC11R2R32        8192   512     4     1               0.10              8.000e+02
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0002689 ...... PASSED
"""
        )

        self.assertTrue(benchmark._process_raw_result(0, 'stdout noise'))

        workload = 'WC11R2R32_TTN8_P4_Q1_N8192_NB512'
        self.assertEqual(650.0, benchmark.result[f'{workload}_flops'][0])
        self.assertEqual(0.26666666666666666, benchmark.result[f'{workload}_time'][0])
        self.assertEqual(1, benchmark.result[f'{workload}_tests_pass'][0])

    def test_dtk_hpl_result_parsing_with_sample_output_file(self):
        """Test DTK gpu-hpl parses a full sample output file."""
        benchmark = DtkHplBenchmark(
            'gpu-hpl', parameters='--P 4 --Q 1 --N 8192 --NB 512 --BCAST 1 --warmup 1 --iterations 5'
        )

        self.assertTrue(benchmark._preprocess())
        self._write_output_file(benchmark, self._load_data_file('gpu_hpl_sample.out'))

        self.assertTrue(benchmark._process_raw_result(0, 'stdout noise'))

        workload = 'WC11R2R32_TTN8_P4_Q1_N8192_NB512'
        self.assertEqual(545.0, benchmark.result[f'{workload}_flops'][0])
        self.assertEqual(0.67, benchmark.result[f'{workload}_time'][0])
        self.assertEqual(1, benchmark.result[f'{workload}_tests_pass'][0])

    def test_dtk_hpl_mxp_result_parsing_with_sample_output_file(self):
        """Test DTK gpu-hpl-mxp parses a full sample output file."""
        benchmark = DtkHplMxpBenchmark(
            'gpu-hpl-mxp', parameters='--P 4 --Q 1 --N 8192 --NB 4096 --BCAST 1 --iterations 6'
        )

        self.assertTrue(benchmark._preprocess())
        self._write_output_file(benchmark, self._load_data_file('gpu_hpl_mxp_sample.out'))

        self.assertTrue(benchmark._process_raw_result(0, 'stdout noise'))

        workload = 'WC1_P4_Q1_N8192_NB4096'
        self.assertEqual(3767.0, benchmark.result[f'{workload}_flops'][0])
        self.assertEqual(0.78, benchmark.result[f'{workload}_time'][0])
        self.assertEqual(1, benchmark.result[f'{workload}_tests_pass'][0])

    def test_dtk_hpl_result_parsing_fails_when_output_file_is_missing(self):
        """Test DTK gpu-hpl parsing fails when generated output file is missing."""
        benchmark = DtkHplBenchmark('gpu-hpl')

        self.assertTrue(benchmark._preprocess())
        self.assertFalse(benchmark._process_raw_result(0, 'stdout noise'))


if __name__ == '__main__':
    unittest.main()
