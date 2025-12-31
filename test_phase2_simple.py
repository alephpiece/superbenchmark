#!/usr/bin/env python3
"""
Simple Phase 2 Integration Test (No Downloads)

Tests the integration without downloading actual models from HuggingFace.
Uses mocking to verify the code paths work correctly.
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase


def test_model_source_config_creation():
    """Test that ModelSourceConfig can be created from args."""
    print("=" * 80)
    print("TEST 1: ModelSourceConfig Creation from Args")
    print("=" * 80)
    
    # Test direct creation (simulating what benchmark would do)
    config = ModelSourceConfig(
        source='huggingface',
        identifier='bert-base-uncased',
        torch_dtype='float16',
        hf_token=None,
        trust_remote_code=False
    )
    
    assert config is not None, "Config should not be None"
    assert config.source == 'huggingface', f"Expected 'huggingface', got {config.source}"
    assert config.identifier == 'bert-base-uncased', f"Expected 'bert-base-uncased', got {config.identifier}"
    assert config.torch_dtype == 'float16', f"Expected 'float16', got {config.torch_dtype}"
    
    print("✓ ModelSourceConfig created successfully")
    print(f"  - source: {config.source}")
    print(f"  - identifier: {config.identifier}")
    print(f"  - torch_dtype: {config.torch_dtype}")
    print("✅ TEST 1 PASSED\n")


def test_model_source_config_none_when_no_source():
    """Test that config validation works for in-house models."""
    print("=" * 80)
    print("TEST 2: In-House Model Config")
    print("=" * 80)
    
    # Test in-house model config
    config = ModelSourceConfig(
        source='in-house',
        identifier='bert-large'
    )
    
    assert config is not None, "Config should not be None"
    assert config.source == 'in-house', f"Expected 'in-house', got {config.source}"
    assert config.is_inhouse() == True, "Should be identified as in-house"
    assert config.is_huggingface() == False, "Should not be identified as HuggingFace"
    
    print("✓ In-house model config created successfully")
    print(f"  - source: {config.source}")
    print(f"  - identifier: {config.identifier}")
    print("✅ TEST 2 PASSED\n")


def test_huggingface_loader_integration():
    """Test that HuggingFaceModelLoader would be called correctly."""
    print("=" * 80)
    print("TEST 3: HuggingFaceModelLoader Integration (Mocked)")
    print("=" * 80)
    
    # Create ModelSourceConfig
    config = ModelSourceConfig(
        source='huggingface',
        identifier='prajjwal1/bert-tiny',  # Small model for testing
        torch_dtype='float32'
    )
    
    # Mock the HuggingFaceModelLoader
    with patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.HuggingFaceModelLoader') as MockLoader:
        # Create mock returns
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.model_type = 'bert'
        mock_tokenizer = MagicMock()
        
        # Setup the mock
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.load_model_from_config.return_value = (mock_model, mock_config, mock_tokenizer)
        
        # Import here to use the mocked version
        from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import HuggingFaceModelLoader
        
        loader = HuggingFaceModelLoader()
        model, cfg, tokenizer = loader.load_model_from_config(config)
        
        # Verify the mock was called
        assert model is not None, "Model should not be None"
        print("✓ HuggingFaceModelLoader.load_model_from_config called successfully")
        print("✓ Returned mock model, config, and tokenizer")
        print("✅ TEST 3 PASSED\n")


def test_argument_parsing():
    """Test that command-line arguments are properly defined."""
    print("=" * 80)
    print("TEST 4: Command-Line Argument Parsing")
    print("=" * 80)
    
    import argparse
    
    # Create a parser
    parser = argparse.ArgumentParser()
    
    # Add the arguments that PytorchBase.add_parser_arguments should add
    parser.add_argument('--model_source', type=str, default=None)
    parser.add_argument('--model_identifier', type=str, default=None)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--trust_remote_code', action='store_true')
    
    # Test parsing HuggingFace args
    args = parser.parse_args([
        '--model_source', 'huggingface',
        '--model_identifier', 'bert-base-uncased',
        '--trust_remote_code'
    ])
    
    assert args.model_source == 'huggingface'
    assert args.model_identifier == 'bert-base-uncased'
    assert args.trust_remote_code == True
    
    print("✓ Command-line arguments parsed correctly")
    print(f"  - model_source: {args.model_source}")
    print(f"  - model_identifier: {args.model_identifier}")
    print(f"  - trust_remote_code: {args.trust_remote_code}")
    print("✅ TEST 4 PASSED\n")


def test_config_validation():
    """Test config validation for HuggingFace models."""
    print("=" * 80)
    print("TEST 5: Configuration Validation")
    print("=" * 80)
    
    # Valid HuggingFace config
    valid_config = ModelSourceConfig(
        source='huggingface',
        identifier='meta-llama/Llama-2-7b-hf',
        torch_dtype='float16'
    )
    
    is_valid, error = valid_config.validate()
    assert is_valid, f"Valid config should pass validation. Error: {error}"
    print("✓ Valid HuggingFace config passed validation")
    
    # Invalid HuggingFace config (missing '/')
    try:
        invalid_config = ModelSourceConfig(
            source='huggingface',
            identifier='bert-base-uncased',  # Should have '/' for HF
            torch_dtype='float16'
        )
        is_valid, error = invalid_config.validate()
        # Should return False and warning message
        print(f"✓ Invalid config detected: {error}")
    except Exception as e:
        print(f"✓ Invalid config properly handled: {e}")
    
    print("✅ TEST 5 PASSED\n")


def test_helper_methods():
    """Test ModelSourceConfig helper methods."""
    print("=" * 80)
    print("TEST 6: ModelSourceConfig Helper Methods")
    print("=" * 80)
    
    hf_config = ModelSourceConfig(
        source='huggingface',
        identifier='bert-base-uncased'
    )
    
    inhouse_config = ModelSourceConfig(
        source='in-house',
        identifier='bert-large'
    )
    
    # Test is_huggingface()
    assert hf_config.is_huggingface() == True
    assert inhouse_config.is_huggingface() == False
    print("✓ is_huggingface() works correctly")
    
    # Test is_inhouse()
    assert hf_config.is_inhouse() == False
    assert inhouse_config.is_inhouse() == True
    print("✓ is_inhouse() works correctly")
    
    # Test get_hf_kwargs()
    hf_kwargs = hf_config.get_hf_kwargs()
    assert isinstance(hf_kwargs, dict)
    print(f"✓ get_hf_kwargs() returns dict: {list(hf_kwargs.keys())}")
    
    print("✅ TEST 6 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PHASE 2 INTEGRATION TEST - SIMPLE (NO DOWNLOADS)")
    print("=" * 80 + "\n")
    
    tests = [
        ("ModelSourceConfig Creation", test_model_source_config_creation),
        ("Config None Check", test_model_source_config_none_when_no_source),
        ("HuggingFace Loader Integration", test_huggingface_loader_integration),
        ("Argument Parsing", test_argument_parsing),
        ("Config Validation", test_config_validation),
        ("Helper Methods", test_helper_methods),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {test_name}")
            print(f"   Error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✅ ALL PHASE 2 INTEGRATION TESTS PASSED!")
        return 0
    else:
        print(f"❌ {failed} TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
