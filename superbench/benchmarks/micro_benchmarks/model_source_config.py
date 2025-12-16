# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Configuration classes for model source and loading."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class ModelSource(Enum):
    """Enumeration of supported model sources."""

    IN_HOUSE = 'in-house'
    HUGGINGFACE = 'huggingface'


@dataclass
class ModelSourceConfig:
    """Configuration for model source and loading parameters.

    This class encapsulates all configuration needed to load a model
    from either in-house definitions or Hugging Face Hub.

    Attributes:
        source: Source of the model ('in-house' or 'huggingface').
        identifier: Model name (in-house) or model ID (HuggingFace).
        hf_token: Optional HuggingFace authentication token for private/gated models.
        torch_dtype: Data type for model weights ('float32', 'float16', 'bfloat16').
        trust_remote_code: Whether to trust remote code execution (HF models).
        revision: Specific model version/commit/tag to use.
        cache_dir: Directory to cache downloaded models.
        device_map: Device mapping strategy for model loading.
        use_auth_token: Deprecated, use hf_token instead.
        additional_kwargs: Additional keyword arguments for model loading.
    """

    source: str = 'in-house'
    identifier: str = ''
    hf_token: Optional[str] = None
    torch_dtype: str = 'float32'
    trust_remote_code: bool = False
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    device_map: str = 'auto'
    use_auth_token: Optional[str] = None  # Deprecated
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and normalization."""
        # Handle deprecated use_auth_token
        if self.use_auth_token is not None and self.hf_token is None:
            self.hf_token = self.use_auth_token

        # Normalize source to enum
        if isinstance(self.source, str):
            self.source = self.source.lower()
            if self.source not in ['in-house', 'huggingface']:
                raise ValueError(
                    f"Invalid model source '{self.source}'. "
                    f"Must be 'in-house' or 'huggingface'."
                )

        # Validate torch_dtype
        valid_dtypes = ['float32', 'float16', 'bfloat16', 'int8']
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid torch_dtype '{self.torch_dtype}'. "
                f"Must be one of {valid_dtypes}."
            )

        # Validate identifier is provided
        if not self.identifier:
            raise ValueError("Model identifier must be provided.")

    def validate(self) -> tuple[bool, str]:
        """Validate configuration parameters.

        Returns:
            Tuple of (is_valid, error_message).
            If is_valid is True, error_message is empty.
        """
        # Check identifier format for HuggingFace models
        if self.source == 'huggingface':
            if '/' not in self.identifier and not self.identifier.startswith('hf://'):
                return (
                    False,
                    f"HuggingFace model identifier '{self.identifier}' should be in "
                    f"format 'organization/model-name' (e.g., 'meta-llama/Llama-2-7b-hf')"
                )

            # Warn about trust_remote_code
            if self.trust_remote_code:
                import warnings
                warnings.warn(
                    "trust_remote_code=True allows execution of remote code. "
                    "Only use with trusted models.",
                    UserWarning
                )

        return (True, '')

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            'source': self.source,
            'identifier': self.identifier,
            'hf_token': '***' if self.hf_token else None,  # Don't expose token
            'torch_dtype': self.torch_dtype,
            'trust_remote_code': self.trust_remote_code,
            'revision': self.revision,
            'cache_dir': self.cache_dir,
            'device_map': self.device_map,
            'additional_kwargs': self.additional_kwargs,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelSourceConfig':
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters.

        Returns:
            ModelSourceConfig instance.
        """
        # Extract known fields
        known_fields = {
            'source', 'identifier', 'hf_token', 'torch_dtype',
            'trust_remote_code', 'revision', 'cache_dir', 'device_map',
            'use_auth_token'
        }
        
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        unknown_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        
        # Add unknown parameters to additional_kwargs
        if unknown_params:
            known_params['additional_kwargs'] = unknown_params
        
        return cls(**known_params)

    def is_huggingface(self) -> bool:
        """Check if this configuration is for a HuggingFace model.

        Returns:
            True if source is 'huggingface', False otherwise.
        """
        return self.source == 'huggingface'

    def is_inhouse(self) -> bool:
        """Check if this configuration is for an in-house model.

        Returns:
            True if source is 'in-house', False otherwise.
        """
        return self.source == 'in-house'

    def get_hf_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for HuggingFace model loading.

        Returns:
            Dictionary of kwargs suitable for HuggingFace's from_pretrained().
        """
        if not self.is_huggingface():
            return {}

        kwargs = {
            'trust_remote_code': self.trust_remote_code,
            'revision': self.revision,
            'cache_dir': self.cache_dir,
        }

        # Add token if provided
        if self.hf_token:
            kwargs['token'] = self.hf_token

        # Add additional kwargs
        kwargs.update(self.additional_kwargs)

        # Remove None values
        return {k: v for k, v in kwargs.items() if v is not None}

    def __repr__(self) -> str:
        """String representation of the configuration."""
        token_status = 'set' if self.hf_token else 'not set'
        return (
            f"ModelSourceConfig(source='{self.source}', "
            f"identifier='{self.identifier}', "
            f"torch_dtype='{self.torch_dtype}', "
            f"hf_token={token_status})"
        )
