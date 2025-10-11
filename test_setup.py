"""
Test script to verify ASAP setup is working correctly
"""

import os
# Fix for macOS mutex lock error with TensorFlow
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

print("Testing ASAP setup...\n")

# Test 1: Import core dependencies
print("1. Testing core dependencies...")
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")

    # Check if Metal (GPU) is available
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ GPU available: {gpus}")
        else:
            print(f"  ✗ No GPU detected (running on CPU)")
    except Exception as gpu_err:
        print(f"  ⚠ GPU detection skipped: {gpu_err}")

except ImportError as e:
    print(f"  ✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    import scipy
    print(f"  ✓ NumPy {np.__version__}")
    print(f"  ✓ Pandas {pd.__version__}")
    print(f"  ✓ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Import ASAP utilities
print("\n2. Testing ASAP utilities...")
try:
    from utils.ASAP_utils import build_model
    from utils.eval_utils import EvalUtils
    print("  ✓ ASAP utilities imported successfully")
except ImportError as e:
    print(f"  ✗ ASAP utilities import failed: {e}")
    sys.exit(1)

# Test 3: Test DTW function
print("\n3. Testing DTW function...")
try:
    test_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_y = np.array([1.1, 2.2, 3.1, 4.0, 5.2])

    result = EvalUtils.dtw_fct(test_x, test_y)
    print(f"  ✓ DTW function works: distance = {result[0]:.4f}")
except Exception as e:
    print(f"  ✗ DTW function failed: {e}")
    sys.exit(1)

# Test 4: Build ASAP model
print("\n4. Testing ASAP model building...")
try:
    params_config = {
        'nb_inputs': 56,
        'in_seq_len': 100,
    }
    params_model = {
        'cell_multiheadatt': 16*4,
        'num_head_multiheadatt': 4,
        'pruning_stat': True,
        'cell_lstm': 20,
        'cell_dense': 20,
    }

    model = build_model(params_config, params_model)
    print(f"  ✓ ASAP model built successfully")
    print(f"  ✓ Model has {model.count_params():,} parameters")

    # Test forward pass with dummy data
    dummy_input = np.random.randn(1, 100, 56).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"  ✓ Forward pass successful (output shape: {len(output)} outputs)")

except Exception as e:
    print(f"  ✗ Model building/testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test preprocessing imports (optional)
print("\n5. Testing preprocessing dependencies (optional)...")
try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError:
    print("  ⚠ OpenCV not installed (needed for preprocessing)")

try:
    import opensmile
    print(f"  ✓ OpenSMILE installed")
except ImportError:
    print("  ⚠ OpenSMILE not installed (needed for preprocessing)")

print("\n" + "="*50)
print("✓ All core tests passed! ASAP is ready to use.")
print("="*50)
print("\nNext steps:")
print("  - Install OpenFace for visual feature extraction")
print("  - Wait for NoXi dataset access")
print("  - Run preprocessing pipeline")
print("  - Train ASAP model on NoXi data")
