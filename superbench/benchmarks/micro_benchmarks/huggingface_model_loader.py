# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugging Face model loader for benchmarking."""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

from superbench.common.utils import logger
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelNotFoundError(ModelLoadError):
    """Exception raised when model is not found."""
    pass


class ModelIncompatibleError(ModelLoadError):
    """Exception raised when model is incompatible with ONNX export."""
    pass


class HuggingFaceModelLoader:
    """Loads models from Hugging Face Hub for benchmarking.

    This class handles downloading, caching, and loading models from
    Hugging Face Hub with support for authentication, device mapping,
    and compatibility validation.

    Attributes:
        cache_dir: Directory to cache downloaded models.
        token: HuggingFace authentication token for private/gated models.
    """

    # Architectures known to work well with ONNX export
    SUPPORTED_ARCHITECTURES = {
        'bert', 'roberta', 'distilbert', 'albert', 'electra',  # BERT family
        'gpt2', 'gpt_neo', 'gptj', 'gpt_neox',  # GPT family
        'llama', 'llama2', 'llama3',  # Llama family
        'qwen', 'qwen2',  # Qwen family
        'mixtral',  # Mixtral MoE
        'bloom',  # BLOOM
        'opt',  # OPT
        'bart', 't5',  # Seq2Seq models
    }

    # Architectures that may have issues with ONNX export
    EXPERIMENTAL_ARCHITECTURES = {
        'deepseek', 'falcon', 'mpt', 'stablelm', 'phi', 'mistral'
    }

    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """Initialize the HuggingFace model loader.

        Args:
            cache_dir: Directory to cache downloaded models. If None, uses HF default.
            token: HuggingFace authentication token for private/gated models.
        """
        self.cache_dir = cache_dir or os.getenv('HF_HOME') or os.path.expanduser('~/.cache/huggingface')
        self.token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f'HuggingFaceModelLoader initialized with cache_dir: {self.cache_dir}')
        if self.token:
            logger.info('Authentication token provided for private/gated models')

    def load_model(
        self,
        model_identifier: str,
        torch_dtype: Optional[str] = None,
        device: str = 'cuda',
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        device_map: Optional[str] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PretrainedConfig, AutoTokenizer]:
        """Load a model from Hugging Face Hub.

        Args:
            model_identifier: HF model ID (e.g., 'meta-llama/Llama-2-7b-hf').
            torch_dtype: Data type for model weights ('float32', 'float16', 'bfloat16').
            device: Device to load model on ('cuda', 'cpu').
            trust_remote_code: Whether to trust remote code execution.
            revision: Specific model version/commit/tag to use.
            device_map: Device mapping strategy for large models.
            **kwargs: Additional arguments passed to from_pretrained().

        Returns:
            Tuple of (model, config, tokenizer).

        Raises:
            ModelNotFoundError: If model doesn't exist on HF Hub.
            ModelLoadError: If model loading fails for any reason.
        """
        logger.info(f'Loading model: {model_identifier}')

        try:
            # Convert torch_dtype string to torch dtype
            dtype = self._get_torch_dtype(torch_dtype) if torch_dtype else None

            # Prepare loading kwargs
            load_kwargs = {
                'cache_dir': self.cache_dir,
                'trust_remote_code': trust_remote_code,
                'revision': revision,
                **kwargs
            }

            # Add token if available
            if self.token:
                load_kwargs['token'] = self.token

            # Add dtype if specified
            if dtype:
                load_kwargs['torch_dtype'] = dtype

            # Load config first to check architecture
            logger.info('Loading model configuration...')
            config = AutoConfig.from_pretrained(model_identifier, **load_kwargs)

            # Validate architecture compatibility
            architecture = config.model_type.lower()
            is_compatible, reason = self._check_architecture_compatibility(architecture)
            if not is_compatible:
                logger.warning(
                    f'Model architecture "{architecture}" may have issues: {reason}'
                )

            # Load tokenizer (may fail for some models, that's ok)
            tokenizer = None
            try:
                logger.info('Loading tokenizer...')
                tokenizer = AutoTokenizer.from_pretrained(model_identifier, **load_kwargs)
            except Exception as e:
                logger.warning(f'Could not load tokenizer: {e}. Continuing without tokenizer.')

            # Load model
            logger.info(f'Loading model weights (dtype={torch_dtype}, device={device})...')
            model_kwargs = load_kwargs.copy()

            # Handle device mapping for large models
            if device_map:
                model_kwargs['device_map'] = device_map
            elif device == 'cuda' and torch.cuda.is_available():
                # Don't set device_map if device is explicitly cuda
                pass
            else:
                model_kwargs['device_map'] = device

            model = AutoModel.from_pretrained(model_identifier, **model_kwargs)

            # Move to device if not using device_map
            if not device_map and device != 'auto':
                model = model.to(device)

            # Set to eval mode
            model.eval()

            logger.info(
                f'Successfully loaded model: {model_identifier} '
                f'({self._get_model_size(model):.2f}M parameters)'
            )

            return model, config, tokenizer

        except OSError as e:
            if 'not found' in str(e).lower() or '404' in str(e):
                raise ModelNotFoundError(
                    f"Model '{model_identifier}' not found on Hugging Face Hub. "
                    f"Please check the model ID at https://huggingface.co/models"
                ) from e
            raise ModelLoadError(f"Failed to load model '{model_identifier}': {e}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading model '{model_identifier}': {e}") from e

    def load_model_from_config(self, config: ModelSourceConfig) -> Tuple[PreTrainedModel, PretrainedConfig, AutoTokenizer]:
        """Load a model using ModelSourceConfig.

        Args:
            config: ModelSourceConfig instance with loading parameters.

        Returns:
            Tuple of (model, config, tokenizer).

        Raises:
            ValueError: If config source is not 'huggingface'.
            ModelLoadError: If model loading fails.
        """
        if not config.is_huggingface():
            raise ValueError(
                f"Cannot load model with source '{config.source}'. "
                "Use 'huggingface' source."
            )

        # Validate config
        is_valid, error = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")

        # Extract loading parameters
        return self.load_model(
            model_identifier=config.identifier,
            torch_dtype=config.torch_dtype,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
            device_map=config.device_map,
            **config.additional_kwargs
        )

    def validate_model_compatibility(
        self,
        model: PreTrainedModel,
        config: PretrainedConfig
    ) -> Tuple[bool, str]:
        """Check if model can be exported to ONNX.

        Args:
            model: The model to validate.
            config: The model configuration.

        Returns:
            Tuple of (is_compatible, reason).
        """
        architecture = config.model_type.lower()

        # Check if architecture is supported
        if architecture in self.SUPPORTED_ARCHITECTURES:
            return (True, f"Architecture '{architecture}' is well-supported for ONNX export")

        if architecture in self.EXPERIMENTAL_ARCHITECTURES:
            return (
                True,
                f"Architecture '{architecture}' is experimental. ONNX export may require adjustments."
            )

        # Check for known incompatible features
        issues = []

        # Check for dynamic operations that may cause issues
        if hasattr(config, 'use_cache') and config.use_cache:
            issues.append("Model uses KV cache which may complicate ONNX export")

        # Check for very large models
        param_count = self._get_model_size(model)
        if param_count > 70_000:  # > 70B parameters
            issues.append(f"Very large model ({param_count/1000:.1f}B params) may cause memory issues")

        if issues:
            return (False, "; ".join(issues))

        return (True, "No obvious compatibility issues detected")

    def get_model_info(self, model_identifier: str) -> Dict[str, Any]:
        """Retrieve model metadata without downloading full model.

        Args:
            model_identifier: HF model ID.

        Returns:
            Dictionary with model information (architecture, size, etc.).

        Raises:
            ModelNotFoundError: If model doesn't exist.
        """
        try:
            # Load just the config to get metadata
            load_kwargs = {'cache_dir': self.cache_dir}
            if self.token:
                load_kwargs['token'] = self.token

            config = AutoConfig.from_pretrained(model_identifier, **load_kwargs)

            # Extract useful information
            info = {
                'model_id': model_identifier,
                'architecture': config.model_type,
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_layers': getattr(config, 'num_hidden_layers', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'vocab_size': getattr(config, 'vocab_size', None),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', None),
            }

            # Check compatibility
            is_compatible, reason = self._check_architecture_compatibility(config.model_type.lower())
            info['onnx_compatible'] = is_compatible
            info['compatibility_notes'] = reason

            return info

        except OSError as e:
            if 'not found' in str(e).lower() or '404' in str(e):
                raise ModelNotFoundError(
                    f"Model '{model_identifier}' not found on Hugging Face Hub"
                ) from e
            raise

    def list_supported_architectures(self) -> Dict[str, List[str]]:
        """Return list of model architectures and their support status.

        Returns:
            Dictionary with 'supported' and 'experimental' architecture lists.
        """
        return {
            'supported': sorted(list(self.SUPPORTED_ARCHITECTURES)),
            'experimental': sorted(list(self.EXPERIMENTAL_ARCHITECTURES)),
        }

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch.dtype.

        Args:
            dtype_str: String representation of dtype ('float32', 'float16', etc.).

        Returns:
            Corresponding torch.dtype.

        Raises:
            ValueError: If dtype string is invalid.
        """
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'int8': torch.int8,
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }

        if dtype_str.lower() not in dtype_map:
            raise ValueError(
                f"Invalid dtype '{dtype_str}'. "
                f"Must be one of {list(dtype_map.keys())}"
            )

        return dtype_map[dtype_str.lower()]

    def _check_architecture_compatibility(self, architecture: str) -> Tuple[bool, str]:
        """Check if architecture is compatible with ONNX export.

        Args:
            architecture: Model architecture name.

        Returns:
            Tuple of (is_compatible, reason/note).
        """
        architecture = architecture.lower()

        if architecture in self.SUPPORTED_ARCHITECTURES:
            return (True, f"Architecture '{architecture}' is well-tested and supported")

        if architecture in self.EXPERIMENTAL_ARCHITECTURES:
            return (
                True,
                f"Architecture '{architecture}' is experimental. "
                "ONNX export may require special handling."
            )

        return (
            False,
            f"Architecture '{architecture}' is not in the tested list. "
            "ONNX export may fail or require modifications."
        )

    def _get_model_size(self, model: PreTrainedModel) -> float:
        """Calculate model size in millions of parameters.

        Args:
            model: The model to measure.

        Returns:
            Number of parameters in millions.
        """
        return sum(p.numel() for p in model.parameters()) / 1_000_000

    def clear_cache(self, model_identifier: Optional[str] = None):
        """Clear cached models.

        Args:
            model_identifier: Specific model to clear. If None, warns but doesn't delete.
        """
        if model_identifier:
            logger.warning(
                f"Selective cache clearing for '{model_identifier}' not implemented. "
                "Please manually delete from {self.cache_dir}"
            )
        else:
            logger.warning(
                f"To clear all cached models, manually delete: {self.cache_dir}"
            )

    def __repr__(self) -> str:
        """String representation of the loader."""
        token_status = 'authenticated' if self.token else 'no authentication'
        return f"HuggingFaceModelLoader(cache_dir='{self.cache_dir}', {token_status})"
