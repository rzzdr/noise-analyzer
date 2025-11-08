#!/usr/bin/env python3
"""
Extract Normalization Parameters for ESP32 Deployment
=====================================================

Extracts the feature normalization parameters (mean and std) from the trained model
and saves them in a format suitable for ESP32 deployment.

This script generates C++ header files with the normalization constants.
"""

import numpy as np
import sys
import argparse
from pathlib import Path

# Add app directory to path
sys.path.append('app')

def extract_normalization_params(params_path='models/model_params.npz', output_dir='esp32_deployment'):
    """Extract normalization parameters and generate C++ headers"""
    
    print(f"Loading normalization parameters from {params_path}...")
    
    if not Path(params_path).exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    # Load parameters
    params = np.load(params_path)
    scaler_mean = params['scaler_mean']
    scaler_std = params['scaler_std']
    
    print(f"✓ Parameters loaded")
    print(f"  Mean shape: {scaler_mean.shape}")
    print(f"  Std shape: {scaler_std.shape}")
    print(f"  Mean range: [{scaler_mean.min():.6f}, {scaler_mean.max():.6f}]")
    print(f"  Std range: [{scaler_std.min():.6f}, {scaler_std.max():.6f}]")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate C++ header file
    header_content = f"""#ifndef MODEL_NORMALIZATION_H
#define MODEL_NORMALIZATION_H

/*
 * Model Normalization Parameters
 * Generated automatically from {params_path}
 * 
 * These parameters are used to normalize mel spectrogram features
 * before feeding them to the TensorFlow Lite model.
 * 
 * Usage:
 *   normalized_feature[i] = (feature[i] - FEATURE_MEAN[i]) / FEATURE_STD[i]
 */

#include <stdint.h>

// Feature dimensions
#define N_MEL_BANDS {len(scaler_mean)}
#define N_TIME_FRAMES 100

// Feature normalization parameters
const float FEATURE_MEAN[N_MEL_BANDS] = {{
"""
    
    # Add mean values
    for i, mean_val in enumerate(scaler_mean):
        if i % 8 == 0:
            header_content += "\n    "
        header_content += f"{mean_val:.8f}f"
        if i < len(scaler_mean) - 1:
            header_content += ", "
    
    header_content += f"""
}};

const float FEATURE_STD[N_MEL_BANDS] = {{
"""
    
    # Add std values
    for i, std_val in enumerate(scaler_std):
        if i % 8 == 0:
            header_content += "\n    "
        header_content += f"{std_val:.8f}f"
        if i < len(scaler_std) - 1:
            header_content += ", "
    
    header_content += f"""
}};

// Helper function for feature normalization
inline void normalize_features(float* features, int n_features) {{
    for (int i = 0; i < n_features; i++) {{
        features[i] = (features[i] - FEATURE_MEAN[i % N_MEL_BANDS]) / FEATURE_STD[i % N_MEL_BANDS];
    }}
}}

// Batch normalization for full spectrogram
inline void normalize_spectrogram(float* spectrogram) {{
    for (int frame = 0; frame < N_TIME_FRAMES; frame++) {{
        for (int band = 0; band < N_MEL_BANDS; band++) {{
            int idx = frame * N_MEL_BANDS + band;
            spectrogram[idx] = (spectrogram[idx] - FEATURE_MEAN[band]) / FEATURE_STD[band];
        }}
    }}
}}

#endif // MODEL_NORMALIZATION_H
"""
    
    # Save C++ header
    header_file = output_path / 'model_normalization.h'
    with open(header_file, 'w') as f:
        f.write(header_content)
    
    print(f"✓ C++ header saved: {header_file}")
    
    # Also save as JSON for other uses
    import json
    json_data = {
        'feature_mean': scaler_mean.tolist(),
        'feature_std': scaler_std.tolist(),
        'n_mel_bands': int(len(scaler_mean)),
        'n_time_frames': 100
    }
    
    json_file = output_path / 'normalization_params.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ JSON file saved: {json_file}")
    
    # Generate validation data for testing
    np.savez(output_path / 'validation_params.npz',
             mean=scaler_mean,
             std=scaler_std)
    
    print(f"✓ Validation file saved: {output_path / 'validation_params.npz'}")
    
    return scaler_mean, scaler_std


def main():
    parser = argparse.ArgumentParser(description='Extract normalization parameters for ESP32 deployment')
    parser.add_argument('--params', default='models/model_params.npz', help='Path to model parameters file')
    parser.add_argument('--output', default='esp32_deployment', help='Output directory')
    
    args = parser.parse_args()
    
    print("Normalization Parameter Extraction")
    print("=" * 40)
    
    try:
        scaler_mean, scaler_std = extract_normalization_params(args.params, args.output)
        
        print("\n" + "=" * 40)
        print("✅ EXTRACTION COMPLETED!")
        print(f"📁 Output directory: {args.output}")
        print("📄 Files generated:")
        print("  - model_normalization.h (C++ header)")
        print("  - normalization_params.json (JSON data)")
        print("  - validation_params.npz (validation data)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())