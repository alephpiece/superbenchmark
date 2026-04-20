# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GPU topology utilities."""

import unittest

from superbench.common.utils.gpu_topology import get_gpu_numa_node_command


class GpuTopologyTest(unittest.TestCase):
    """Test GPU topology utilities."""
    def test_get_gpu_numa_node_command(self):
        """Test get_gpu_numa_node_command returns remote shell command substitution."""
        command = get_gpu_numa_node_command(1)

        self.assertIn('hy-smi --showtoponuma --json', command)
        self.assertIn('card1', command)
        self.assertIn('(Topology) Numa Node', command)
        self.assertNotIn("'", command)
