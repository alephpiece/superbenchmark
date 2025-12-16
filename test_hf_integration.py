#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Simple integration test for HuggingFace model loading."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import HuggingFaceModelLoader


def test_model_source_config():
    """Test ModelSourceConfig creation and validation."""
    print("Testing ModelSourceConfig...")
    
    # Test in-house config
    config1 = ModelSourceConfig(source='in-house', identifier='bert-base')
    assert config1.is_inhouse()
    print("✓ In-house config created successfully")
    
    # Test HuggingFace config
    config2 = ModelSourceConfig(
        source='huggingface',
        identifier='bert-base-uncased',
        torch_dtype='float16'
    )
    assert config2.is_huggingface()
    print("✓ HuggingFace config created successfully")
    
    # Test to_dict and from_dict
    config_dict = config2.to_dict()
    config3 = ModelSourceConfig.from_dict(config_dict)
    assert config3.identifier == config2.identifier
    print("✓ Dict conversion working")
    
    print("✅ ModelSourceConfig tests passed!\n")


def test_huggingface_loader():
    """Test HuggingFaceModelLoader initialization."""
    print("Testing HuggingFaceModelLoader...")
    
    # Create loader
    loader = HuggingFaceModelLoader(cache_dir='/tmp/test_hf_cache')
    assert loader.cache_dir == '/tmp/test_hf_cache'
    print("✓ Loader initialized successfully")
    
    # Test architecture support
    architectures = loader.list_supported_architectures()
    assert 'bert' in architectures['supported']
    assert 'gpt2' in architectures['supported']
    assert 'llama' in architectures['supported']
    print(f"✓ Supported architectures: {len(architectures['supported'])} models")
    print(f"✓ Experimental architectures: {len(architectures['experimental'])} models")
    
    print("✅ HuggingFaceModelLoader tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Integration Test: HuggingFace Model Support")
    print("=" * 60 + "\n")
    
    try:
        test_model_source_config()
        test_huggingface_loader()
        
        print("=" * 60)
        print("✅ All integration tests PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
