# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for HuggingFaceModelLoader."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import (
    HuggingFaceModelLoader,
    ModelLoadError,
    ModelNotFoundError,
    ModelIncompatibleError
)
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


class TestHuggingFaceModelLoader:
    """Test cases for HuggingFaceModelLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing."""
        return HuggingFaceModelLoader(cache_dir='/tmp/test_cache', token=None)

    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader.cache_dir == '/tmp/test_cache'
        assert loader.token is None

    def test_initialization_with_env_token(self, monkeypatch):
        """Test loader picks up token from environment."""
        monkeypatch.setenv('HF_TOKEN', 'env_token')
        loader = HuggingFaceModelLoader()
        assert loader.token == 'env_token'

    def test_supported_architectures(self, loader):
        """Test supported architectures list."""
        architectures = loader.list_supported_architectures()
        assert 'bert' in architectures['supported']
        assert 'gpt2' in architectures['supported']
        assert 'llama' in architectures['supported']
        assert 'qwen' in architectures['supported']

    def test_get_torch_dtype_valid(self, loader):
        """Test torch dtype conversion."""
        assert loader._get_torch_dtype('float32') == torch.float32
        assert loader._get_torch_dtype('float16') == torch.float16
        assert loader._get_torch_dtype('fp16') == torch.float16
        assert loader._get_torch_dtype('bfloat16') == torch.bfloat16

    def test_get_torch_dtype_invalid(self, loader):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError, match='Invalid dtype'):
            loader._get_torch_dtype('invalid_dtype')

    def test_check_architecture_compatibility_supported(self, loader):
        """Test compatibility check for supported architecture."""
        is_compatible, reason = loader._check_architecture_compatibility('bert')
        assert is_compatible is True
        assert 'well-tested' in reason

    def test_check_architecture_compatibility_experimental(self, loader):
        """Test compatibility check for experimental architecture."""
        is_compatible, reason = loader._check_architecture_compatibility('deepseek')
        assert is_compatible is True
        assert 'experimental' in reason.lower()

    def test_check_architecture_compatibility_unsupported(self, loader):
        """Test compatibility check for unsupported architecture."""
        is_compatible, reason = loader._check_architecture_compatibility('unknown_arch')
        assert is_compatible is False
        assert 'not in the tested list' in reason

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoConfig')
    def test_get_model_info_success(self, mock_config, loader):
        """Test getting model info."""
        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.model_type = 'bert'
        mock_cfg.hidden_size = 768
        mock_cfg.num_hidden_layers = 12
        mock_cfg.num_attention_heads = 12
        mock_cfg.vocab_size = 30522
        mock_cfg.max_position_embeddings = 512
        mock_config.from_pretrained.return_value = mock_cfg

        info = loader.get_model_info('test/model')
        
        assert info['model_id'] == 'test/model'
        assert info['architecture'] == 'bert'
        assert info['hidden_size'] == 768
        assert info['num_layers'] == 12

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoConfig')
    def test_get_model_info_not_found(self, mock_config, loader):
        """Test getting info for non-existent model."""
        mock_config.from_pretrained.side_effect = OSError('404 not found')
        
        with pytest.raises(ModelNotFoundError, match='not found'):
            loader.get_model_info('nonexistent/model')

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoModel')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoConfig')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoTokenizer')
    def test_load_model_success(self, mock_tokenizer, mock_config, mock_model, loader):
        """Test successful model loading."""
        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.model_type = 'bert'
        mock_config.from_pretrained.return_value = mock_cfg

        # Mock model
        mock_mdl = MagicMock()
        mock_mdl.parameters.return_value = [torch.randn(100, 100)]
        mock_mdl.eval.return_value = mock_mdl
        mock_mdl.to.return_value = mock_mdl
        mock_model.from_pretrained.return_value = mock_mdl

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        model, config, tokenizer = loader.load_model('test/model', device='cpu')
        
        assert model == mock_mdl
        assert config == mock_cfg
        assert tokenizer == mock_tok

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoModel')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoConfig')
    def test_load_model_not_found(self, mock_config, mock_model, loader):
        """Test loading non-existent model."""
        mock_config.from_pretrained.side_effect = OSError('404 Client Error')
        
        with pytest.raises(ModelNotFoundError, match='not found'):
            loader.load_model('nonexistent/model')

    def test_load_model_from_config_invalid_source(self, loader):
        """Test loading with invalid source in config."""
        config = ModelSourceConfig(source='in-house', identifier='bert-base')
        
        with pytest.raises(ValueError, match='Cannot load model'):
            loader.load_model_from_config(config)

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoModel')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoConfig')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoTokenizer')
    def test_validate_model_compatibility_supported(self, mock_tokenizer, mock_config, mock_model, loader):
        """Test validating a supported model."""
        mock_cfg = MagicMock()
        mock_cfg.model_type = 'bert'
        mock_cfg.use_cache = False
        
        mock_mdl = MagicMock()
        mock_mdl.parameters.return_value = [torch.randn(100, 100)]
        
        is_compatible, reason = loader.validate_model_compatibility(mock_mdl, mock_cfg)
        assert is_compatible is True
        assert 'well-supported' in reason

    def test_get_model_size(self, loader):
        """Test model size calculation."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            torch.randn(1000, 1000),  # 1M params
            torch.randn(500, 500),     # 0.25M params
        ]
        
        size = loader._get_model_size(mock_model)
        assert abs(size - 1.25) < 0.01  # Should be ~1.25M

    def test_repr(self, loader):
        """Test string representation."""
        repr_str = repr(loader)
        assert 'HuggingFaceModelLoader' in repr_str
        assert '/tmp/test_cache' in repr_str
        assert 'no authentication' in repr_str
