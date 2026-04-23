# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DTK HPL benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import GpuHplBenchmark
from superbench.benchmarks.micro_benchmarks.gpu_hpl_performance_base import format_hpl_extended_tv, format_hpl_tv


class DtkHplBenchmark(GpuHplBenchmark):
    """The DTK HPL benchmark class."""

    _default_bin_name = 'run_rochpl'
    _default_dat_name = 'HPL.dat'
    _default_out_name = 'HPL.out'
    _file_label = 'HPL'

    def _add_variant_parser_arguments(self):
        """Add rocHPL-specific arguments."""
        self._parser.add_argument(
            '--pfact',
            dest='pfact',
            type=int,
            default=2,
            choices=[0, 1, 2],
            required=False,
            help='Panel factorization: 0 for left, 1 for Crout, 2 for right.',
        )
        self._parser.add_argument(
            '--nbmin',
            dest='nbmin',
            type=int,
            default=32,
            required=False,
            help='Recursive stopping criterion.',
        )
        self._parser.add_argument(
            '--ndiv',
            dest='ndiv',
            type=int,
            default=2,
            required=False,
            help='Number of panels in recursion.',
        )
        self._parser.add_argument(
            '--rfact',
            dest='rfact',
            type=int,
            default=2,
            choices=[0, 1, 2],
            required=False,
            help='Recursive panel factorization: 0 for left, 1 for Crout, 2 for right.',
        )
        self._parser.add_argument(
            '--depth',
            dest='depth',
            type=int,
            default=1,
            required=False,
            help='Lookahead depth.',
        )
        self._parser.add_argument(
            '--swap',
            dest='swap',
            type=int,
            default=1,
            choices=[0, 1, 2],
            required=False,
            help='Swapping algorithm: 0 for binary exchange, 1 for long, 2 for mix.',
        )
        self._parser.add_argument(
            '--swapping-threshold',
            dest='swapping_threshold',
            type=int,
            default=64,
            required=False,
            help='Swapping threshold.',
        )
        self._parser.add_argument(
            '--l1',
            dest='l1',
            type=int,
            default=0,
            choices=[0, 1],
            required=False,
            help='L1 storage form: 0 for transposed, 1 for non-transposed.',
        )
        self._parser.add_argument(
            '--u',
            dest='u',
            type=int,
            default=0,
            choices=[0, 1],
            required=False,
            help='U storage form: 0 for transposed, 1 for non-transposed.',
        )
        self._parser.add_argument(
            '--equilibration',
            dest='equilibration',
            type=int,
            default=0,
            choices=[0, 1],
            required=False,
            help='Equilibration: 0 for no, 1 for yes.',
        )
        self._parser.add_argument(
            '--memory-alignment',
            dest='memory_alignment',
            type=int,
            default=8,
            required=False,
            help='Memory alignment in double.',
        )

    def _format_tv(self):
        """Format the expected rocHPL T/V field from input arguments."""
        return format_hpl_extended_tv(
            self._args.pmap,
            self._args.depth,
            self._args.bcast,
            self._args.rfact,
            self._args.ndiv,
            self._args.pfact,
            self._args.nbmin,
            self._args.l1,
            self._args.u,
            self._args.equilibration,
            self._args.memory_alignment,
        )

    def _format_output_tv(self):
        """Format the expected rocHPL T/V field in generated output."""
        return format_hpl_tv(
            self._args.pmap,
            self._args.depth,
            self._args.bcast,
            self._args.rfact,
            self._args.ndiv,
            self._args.pfact,
            self._args.nbmin,
        )

    def _format_dat_content(self):
        """Format generated rocHPL input file content."""
        return (
            'HPLinpack benchmark input file\n'
            'Innovative Computing Laboratory, University of Tennessee\n'
            f'{self._out_file_name} output file name (if any)\n'
            '0            device out (6=stdout,7=stderr,file)\n'
            '1            # of problems sizes (N)\n'
            f'{self._args.n}         Ns\n'
            '1            # of NBs\n'
            f'{self._args.nb}         NBs\n'
            f'{self._args.pmap}            PMAP process mapping (0=Row-,1=Column-major)\n'
            '1            # of process grids (P x Q)\n'
            f'{self._args.p}            Ps\n'
            f'{self._args.q}            Qs\n'
            f'{self._args.threshold}         threshold\n'
            '1            # of panel fact\n'
            f'{self._args.pfact}            PFACTs (0=left, 1=Crout, 2=Right)\n'
            '1            # of recursive stopping criterium\n'
            f'{self._args.nbmin}           NBMINs (>= 1)\n'
            '1            # of panels in recursion\n'
            f'{self._args.ndiv}            NDIVs\n'
            '1            # of recursive panel fact.\n'
            f'{self._args.rfact}            RFACTs (0=left, 1=Crout, 2=Right)\n'
            '1            # of broadcast\n'
            f'{self._args.bcast}            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)\n'
            '1            # of lookahead depth\n'
            f'{self._args.depth}            DEPTHs (>=0)\n'
            f'{self._args.swap}            SWAP (0=bin-exch,1=long,2=mix)\n'
            f'{self._args.swapping_threshold}           swapping threshold\n'
            f'{self._args.l1}            L1 in (0=transposed,1=no-transposed) form\n'
            f'{self._args.u}            U  in (0=transposed,1=no-transposed) form\n'
            f'{self._args.equilibration}            Equilibration (0=no,1=yes)\n'
            f'{self._args.memory_alignment}            memory alignment in double (> 0)\n'
        )


BenchmarkRegistry.register_benchmark('gpu-hpl', DtkHplBenchmark, platform=Platform.DTK)
