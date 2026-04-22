# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DTK HPL benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import GpuHplBenchmark
from superbench.benchmarks.micro_benchmarks.gpu_hpl_performance_base import format_hpl_tv


class DtkHplBenchmark(GpuHplBenchmark):
    """The DTK HPL benchmark class."""

    _default_bin_name = 'run_rochpl'
    _default_dat_name = 'HPL.dat'
    _default_out_name = 'HPL.out'

    def _add_variant_parser_arguments(self):
        """Add rocHPL-specific arguments."""
        self._parser.add_argument(
            '--PFACT',
            type=int,
            default=2,
            choices=[0, 1, 2],
            required=False,
            help='Panel factorization: 0 for left, 1 for Crout, 2 for right.',
        )
        self._parser.add_argument(
            '--NBMIN',
            type=int,
            default=32,
            required=False,
            help='Recursive stopping criterion.',
        )
        self._parser.add_argument(
            '--NDIV',
            type=int,
            default=2,
            required=False,
            help='Number of panels in recursion.',
        )
        self._parser.add_argument(
            '--RFACT',
            type=int,
            default=2,
            choices=[0, 1, 2],
            required=False,
            help='Recursive panel factorization: 0 for left, 1 for Crout, 2 for right.',
        )
        self._parser.add_argument(
            '--DEPTH',
            type=int,
            default=1,
            required=False,
            help='Lookahead depth.',
        )
        self._parser.add_argument(
            '--SWAP',
            type=int,
            default=1,
            choices=[0, 1, 2],
            required=False,
            help='Swapping algorithm: 0 for binary exchange, 1 for long, 2 for mix.',
        )
        self._parser.add_argument(
            '--swapping-threshold',
            type=int,
            default=64,
            required=False,
            help='Swapping threshold.',
        )
        self._parser.add_argument(
            '--L1',
            type=int,
            default=0,
            choices=[0, 1],
            required=False,
            help='L1 storage form: 0 for transposed, 1 for non-transposed.',
        )
        self._parser.add_argument(
            '--U',
            type=int,
            default=0,
            choices=[0, 1],
            required=False,
            help='U storage form: 0 for transposed, 1 for non-transposed.',
        )
        self._parser.add_argument(
            '--Equilibration',
            type=int,
            default=0,
            choices=[0, 1],
            required=False,
            help='Equilibration: 0 for no, 1 for yes.',
        )
        self._parser.add_argument(
            '--memory-alignment',
            type=int,
            default=8,
            required=False,
            help='Memory alignment in double.',
        )

    def _format_tv(self):
        """Format the expected rocHPL T/V field from input arguments."""
        return format_hpl_tv(
            self._args.PMAP,
            self._args.DEPTH,
            self._args.BCAST,
            self._args.RFACT,
            self._args.NDIV,
            self._args.PFACT,
            self._args.NBMIN,
        )


BenchmarkRegistry.register_benchmark('gpu-hpl', DtkHplBenchmark, platform=Platform.DTK)
