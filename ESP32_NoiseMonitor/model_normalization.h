#ifndef MODEL_NORMALIZATION_H
#define MODEL_NORMALIZATION_H

/*
 * Model Normalization Parameters
 * Generated automatically from app/models/model_params.npz
 * 
 * These parameters are used to normalize mel spectrogram features
 * before feeding them to the TensorFlow Lite model.
 * 
 * Usage:
 *   normalized_feature[i] = (feature[i] - FEATURE_MEAN[i]) / FEATURE_STD[i]
 */

#include <stdint.h>

// Include config for consistent parameters
#ifndef CONFIG_H
#include "config.h"
#endif

// Use dimensions from config.h
#define N_TIME_FRAMES N_FRAMES

// Feature normalization parameters (simplified - use global mean/std)
const float GLOBAL_FEATURE_MEAN = -39.19651086f;
const float GLOBAL_FEATURE_STD = 18.33127076f;

// Helper function for feature normalization
inline void normalize_features(float* features, int n_features) {
    for (int i = 0; i < n_features; i++) {
        features[i] = (features[i] - GLOBAL_FEATURE_MEAN) / GLOBAL_FEATURE_STD;
    }
}

// Batch normalization for full spectrogram
inline void normalize_spectrogram(float* spectrogram) {
    int total_features = N_TIME_FRAMES * N_MEL_BANDS;
    for (int i = 0; i < total_features; i++) {
        spectrogram[i] = (spectrogram[i] - GLOBAL_FEATURE_MEAN) / GLOBAL_FEATURE_STD;
    }
}

#endif // MODEL_NORMALIZATION_H