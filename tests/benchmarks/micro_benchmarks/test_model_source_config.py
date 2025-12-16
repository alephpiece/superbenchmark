# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for ModelSourceConfig."""

import pytest
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


class TestModelSourceConfig:
    """Test cases for ModelSourceConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ModelSourceConfig(identifier='bert-base')
        assert config.source == 'in-house'
        assert config.identifier == 'bert-base'
        assert config.torch_dtype == 'float32'
        assert config.trust_remote_code is False
        assert config.hf_token is None

    def test_huggingface_config(self):
        """Test HuggingFace configuration."""
        config = ModelSourceConfig(
            source='huggingface',
            identifier='meta-llama/Llama-2-7b-hf',
            torch_dtype='float16'
        )
        assert config.source == 'huggingface'
        assert config.identifier == 'meta-llama/Llama-2-7b-hf'
        assert config.torch_dtype == 'float16'

    def test_invalid_source(self):
        """Test invalid source raises error."""
        with pytest.raises(ValueError, match='Invalid model source'):
            ModelSourceConfig(source='invalid', identifier='test')

    def test_invalid_dtype(self):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError, match='Invalid torch_dtype'):
            ModelSourceConfig(identifier='test', torch_dtype='invalid')

    def test_missing_identifier(self):
        """Test missing identifier raises error."""
        with pytest.raises(ValueError, match='identifier must be provided'):
            ModelSourceConfig(identifier='')

    def test_validate_huggingface_format(self):
        """Test validation of HuggingFace model identifier format."""
        config = ModelSourceConfig(
            source='huggingface',
            identifier='invalid-format'
        )
        is_valid, message = config.validate()
        assert not is_valid
        assert 'organization/model-name' in message

    def test_validate_valid_huggingface(self):
        """Test validation of valid HuggingFace model."""
        config = ModelSourceConfig(
            source='huggingface',
            identifier='meta-llama/Llama-2-7b-hf'
        )
        is_valid, message = config.validate()
        assert is_valid
        assert message == ''

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelSourceConfig(
            source='huggingface',
            identifier='test/model',
            hf_token='secret_token'
        )
        config_dict = config.to_dict()
        assert config_dict['source'] == 'huggingface'
        assert config_dict['identifier'] == 'test/model'
        assert config_dict['hf_token'] == '***'  # Token should be masked

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'source': 'huggingface',
            'identifier': 'test/model',
            'torch_dtype': 'float16',
            'hf_token': 'token',
            'unknown_param': 'value'
        }
        config = ModelSourceConfig.from_dict(config_dict)
        assert config.source == 'huggingface'
        assert config.identifier == 'test/model'
        assert config.torch_dtype == 'float16'
        assert config.hf_token == 'token'
        assert 'unknown_param' in config.additional_kwargs

    def test_is_huggingface(self):
        """Test is_huggingface method."""
        hf_config = ModelSourceConfig(source='huggingface', identifier='test/model')
        inhouse_config = ModelSourceConfig(source='in-house', identifier='bert-base')
        assert hf_config.is_huggingface() is True
        assert inhouse_config.is_huggingface() is False

    def test_is_inhouse(self):
        """Test is_inhouse method."""
        hf_config = ModelSourceConfig(source='huggingface', identifier='test/model')
        inhouse_config = ModelSourceConfig(source='in-house', identifier='bert-base')
        assert hf_config.is_inhouse() is False
        assert inhouse_config.is_inhouse() is True

    def test_get_hf_kwargs(self):
        """Test get_hf_kwargs method."""
        config = ModelSourceConfig(
            source='huggingface',
            identifier='test/model',
            hf_token='token123',
            trust_remote_code=True,
            revision='main'
        )
        kwargs = config.get_hf_kwargs()
        assert kwargs['token'] == 'token123'
        assert kwargs['trust_remote_code'] is True
        assert kwargs['revision'] == 'main'

    def test_deprecated_use_auth_token(self):
        """Test deprecated use_auth_token parameter."""
        config = ModelSourceConfig(
            identifier='test',
            use_auth_token='old_token'
        )
        assert config.hf_token == 'old_token'

    def test_trust_remote_code_warning(self):
        """Test warning for trust_remote_code."""
        with pytest.warns(UserWarning, match='trust_remote_code'):
            config = ModelSourceConfig(
                source='huggingface',
                identifier='test/model',
                trust_remote_code=True
            )
            config.validate()
