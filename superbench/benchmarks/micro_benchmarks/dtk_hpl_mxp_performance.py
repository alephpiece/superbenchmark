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
    _default_n = 61440
    _default_nb = 2560

    def _format_tv(self):
        """Format the expected rocHPL-MxP T/V field from input arguments."""
        return format_hpl_mxp_tv(self._args.PMAP, self._args.BCAST)


BenchmarkRegistry.register_benchmark('gpu-hpl-mxp', DtkHplMxpBenchmark, platform=Platform.DTK)
