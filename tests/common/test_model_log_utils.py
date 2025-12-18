# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for model_log_utils module."""

import json
import tempfile
import pytest
from unittest.mock import Mock
from superbench.common import model_log_utils


class TestRecordStepLoss:
    """Tests for record_step_loss function."""

    def test_record_loss_conversion_failure(self):
        """Test exception handling when loss conversion fails."""
        logger = Mock()
        losses_list = []

        # Create a mock object that raises exception on conversion
        bad_loss = Mock()
        bad_loss.detach.side_effect = RuntimeError("Conversion failed")

        result = model_log_utils.record_step_loss(bad_loss, curr_step=5, losses_list=losses_list, logger=logger)

        assert result is None
        assert losses_list == [None]
        logger.info.assert_called_once_with('Unable to convert loss to float at step 5')


class TestLoadAndValidateReferenceFile:
    """Tests for _load_and_validate_reference_file function."""

    def test_file_not_found(self):
        """Test FileNotFoundError when reference file doesn't exist."""
        with pytest.raises(FileNotFoundError, match='Reference results file not found'):
            model_log_utils._load_and_validate_reference_file('/nonexistent/file.json')

    def test_invalid_json(self):
        """Test ValueError when JSON is malformed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{invalid json')
            f.flush()

            with pytest.raises(ValueError, match='Invalid JSON'):
                model_log_utils._load_and_validate_reference_file(f.name)

    def test_missing_raw_data(self):
        """Test ValueError when raw_data section is missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'some_other_key': {}}, f)
            f.flush()

            with pytest.raises(ValueError, match='does not contain "raw_data" section'):
                model_log_utils._load_and_validate_reference_file(f.name)


class TestFindBenchmarkRawData:
    """Tests for _find_benchmark_raw_data function."""

    def test_benchmark_not_found(self):
        """Test ValueError when benchmark name not found in reference."""
        ref_raw_data = {'pytorch-resnet18': {}, 'pytorch-bert': {}}

        with pytest.raises(ValueError, match='does not contain raw_data for benchmark matching'):
            model_log_utils._find_benchmark_raw_data(ref_raw_data, 'llama')


class TestExtractMetadataFromRawData:
    """Tests for _extract_metadata_from_raw_data function."""

    def test_extract_from_list_of_dicts(self):
        """Test extracting metadata from list of dicts format."""
        metadata_list = [{'batch_size': 32, 'seed': 42}]
        result = model_log_utils._extract_metadata_from_raw_data(metadata_list)
        assert result == {'batch_size': 32, 'seed': 42}

    def test_extract_from_nested_list(self):
        """Test extracting metadata from nested list format."""
        metadata_list = [[{'batch_size': 16, 'seq_len': 128}]]
        result = model_log_utils._extract_metadata_from_raw_data(metadata_list)
        assert result == {'batch_size': 16, 'seq_len': 128}

    def test_extract_from_dict(self):
        """Test extracting metadata from direct dict format."""
        metadata_dict = {'num_steps': 100}
        result = model_log_utils._extract_metadata_from_raw_data(metadata_dict)
        assert result == {'num_steps': 100}

    def test_extract_returns_none_for_invalid(self):
        """Test returns None for invalid metadata format."""
        result = model_log_utils._extract_metadata_from_raw_data([])
        assert result is None


class TestCompareCheckpointValues:
    """Tests for _compare_checkpoint_values function."""

    def test_length_mismatch(self):
        """Test detection of checkpoint count mismatch."""
        logger = Mock()
        curr_run = [1.0, 2.0, 3.0]
        ref_run = [1.0, 2.0]

        mismatches = model_log_utils._compare_checkpoint_values('loss', 0, curr_run, ref_run, logger)

        assert len(mismatches) == 1
        assert 'checkpoint count mismatch (3 vs 2)' in mismatches[0]

    def test_value_mismatch_numeric(self):
        """Test detection of numeric value mismatch with diff calculation."""
        logger = Mock()
        curr_run = [1.0, 2.5, 3.0]
        ref_run = [1.0, 2.0, 3.0]

        mismatches = model_log_utils._compare_checkpoint_values('loss', 0, curr_run, ref_run, logger)

        assert len(mismatches) == 1
        assert 'checkpoint 1' in mismatches[0]
        assert 'diff: 0.5' in mismatches[0]


class TestApplyMetadataOverrides:
    """Tests for apply_metadata_overrides function."""

    def test_no_metadata_provided(self):
        """Test warning when no metadata is provided."""
        logger = Mock()
        args = Mock()

        count = model_log_utils.apply_metadata_overrides(args, None, logger)

        assert count == 0
        logger.warning.assert_called_once_with('No metadata provided for override')

    def test_precision_override_from_string(self):
        """Test precision override converts string to Precision enum list."""
        from superbench.benchmarks.context import Precision

        logger = Mock()
        args = Mock()
        args.batch_size = 32
        args.precision = [Precision.FLOAT16]

        ref_metadata = {'batch_size': 32, 'precision': 'float32'}

        count = model_log_utils.apply_metadata_overrides(args, ref_metadata, logger)

        # Should override precision from string 'float32' to [Precision.FLOAT32]
        assert count == 1
        assert isinstance(args.precision, list)
        assert args.precision[0] == Precision.FLOAT32

    def test_precision_override_from_list(self):
        """Test precision override handles list of strings."""
        from superbench.benchmarks.context import Precision

        logger = Mock()
        args = Mock()
        args.precision = [Precision.FLOAT16]

        ref_metadata = {'precision': ['float32', 'float16']}

        count = model_log_utils.apply_metadata_overrides(args, ref_metadata, logger)

        assert count == 1
        assert isinstance(args.precision, list)
        assert len(args.precision) == 2
        assert args.precision[0] == Precision.FLOAT32
        assert args.precision[1] == Precision.FLOAT16
