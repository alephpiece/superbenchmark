# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DTK HPL-MxP benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import GpuHplBenchmark
from superbench.benchmarks.micro_benchmarks.gpu_hpl_performance_base import format_hpl_mxp_tv


class DtkHplMxpBenchmark(GpuHplBenchmark):
    """The DTK HPL-MxP benchmark class."""

    _default_bin_name = 'run_rochplmxp'
    _default_dat_name = 'HPL-MxP.dat'
    _default_out_name = 'HPL-MxP.out'
    _file_label = 'HPL-MxP'
    _default_n = 61440
    _default_nb = 2560

    def _format_tv(self):
        """Format the expected rocHPL-MxP T/V field from input arguments."""
        return format_hpl_mxp_tv(self._args.PMAP, self._args.BCAST)

    def _format_dat_content(self):
        """Format generated rocHPL-MxP input file content."""
        return (
            'HPLinpack benchmark input file\n'
            'Innovative Computing Laboratory, University of Tennessee\n'
            f'{self._out_file_name} output file name (if any)\n'
            '0            device out (6=stdout,7=stderr,file)\n'
            '1            # of problems sizes (N)\n'
            f'{self._args.N}         Ns\n'
            '1            # of NBs\n'
            f'{self._args.NB}         NBs\n'
            f'{self._args.PMAP}            PMAP process mapping (0=Row-,1=Column-major)\n'
            f'{self._args.P}            P\n'
            f'{self._args.Q}            Q\n'
            f'{self._args.threshold}         threshold\n'
            '1            # of broadcast\n'
            f'{self._args.BCAST}            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)\n'
        )


BenchmarkRegistry.register_benchmark('gpu-hpl-mxp', DtkHplMxpBenchmark, platform=Platform.DTK)
