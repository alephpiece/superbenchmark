# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI node handler test."""

import io
import unittest
import contextlib
from unittest import mock

from knack.util import CLIError

import superbench.cli._node_handler as node_handler


class CLINodeHandlerTestCase(unittest.TestCase):
    """A class for node handler test cases."""
    @mock.patch('superbench.cli._node_handler.get_gpu_numa_node')
    def test_topo_command_handler_gpu_numa_node(self, mock_get_gpu_numa_node):
        """Test topo command handler gets GPU NUMA node."""
        mock_get_gpu_numa_node.return_value = '1'
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            node_handler.topo_command_handler(get='gpu-numa-node', proc_rank=1)

        self.assertEqual(stdout.getvalue(), '1\n')
        mock_get_gpu_numa_node.assert_called_once_with(1)

    def test_topo_command_handler_invalid_get(self):
        """Test topo command handler rejects invalid get value."""
        with self.assertRaises(CLIError):
            node_handler.topo_command_handler(get='invalid', proc_rank=1)

    def test_topo_command_handler_missing_proc_rank(self):
        """Test topo command handler requires proc_rank."""
        with self.assertRaises(CLIError):
            node_handler.topo_command_handler(get='gpu-numa-node')
