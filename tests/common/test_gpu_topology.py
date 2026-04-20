# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GPU topology utilities."""

import json
import unittest
from unittest import mock

from superbench.common.utils.gpu_topology import get_gpu_numa_node, get_gpu_numa_node_command


class GpuTopologyTest(unittest.TestCase):
    """Test GPU topology utilities."""
    @mock.patch('superbench.common.utils.gpu_topology.run_command')
    def test_get_gpu_numa_node(self, mock_run_command):
        """Test get_gpu_numa_node parses hy-smi output."""
        mock_run_command.return_value.returncode = 0
        mock_run_command.return_value.stdout = json.dumps(
            {
                'card0': {
                    '(Topology) Numa Node': '3',
                },
                'card1': {
                    '(Topology) Numa Node': '1',
                },
            }
        )

        self.assertEqual(get_gpu_numa_node(1), '1')
        mock_run_command.assert_called_once_with('hy-smi --showtoponuma --json', quiet=True)

    @mock.patch('superbench.common.utils.gpu_topology.run_command')
    def test_get_gpu_numa_node_command_failure(self, mock_run_command):
        """Test get_gpu_numa_node command failure."""
        mock_run_command.return_value.returncode = 1
        mock_run_command.return_value.stdout = 'hy-smi failed'

        with self.assertRaisesRegex(RuntimeError, 'Failed to get GPU NUMA node from hy-smi'):
            get_gpu_numa_node(0)

    @mock.patch('superbench.common.utils.gpu_topology.run_command')
    def test_get_gpu_numa_node_parse_failure(self, mock_run_command):
        """Test get_gpu_numa_node parse failure."""
        mock_run_command.return_value.returncode = 0
        mock_run_command.return_value.stdout = json.dumps({'card0': {}})

        with self.assertRaisesRegex(RuntimeError, 'Failed to parse GPU NUMA node from hy-smi'):
            get_gpu_numa_node(0)

    def test_get_gpu_numa_node_command(self):
        """Test get_gpu_numa_node_command returns remote shell command substitution."""
        self.assertEqual(get_gpu_numa_node_command(1), '$(sb node topo --get gpu-numa-node --proc-rank 1)')
