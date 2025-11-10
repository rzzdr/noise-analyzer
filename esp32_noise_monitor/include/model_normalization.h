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

// Feature dimensions
#define N_MEL_BANDS 1
#define N_TIME_FRAMES 100

// Feature normalization parameters
const float FEATURE_MEAN[N_MEL_BANDS] = {

    -39.19651086f
};

const float FEATURE_STD[N_MEL_BANDS] = {

    18.33127076f
};

// Helper function for feature normalization
inline void normalize_features(float* features, int n_features) {
    for (int i = 0; i < n_features; i++) {
        features[i] = (features[i] - FEATURE_MEAN[i % N_MEL_BANDS]) / FEATURE_STD[i % N_MEL_BANDS];
    }
}

// Batch normalization for full spectrogram
inline void normalize_spectrogram(float* spectrogram) {
    for (int frame = 0; frame < N_TIME_FRAMES; frame++) {
        for (int band = 0; band < N_MEL_BANDS; band++) {
            int idx = frame * N_MEL_BANDS + band;
            spectrogram[idx] = (spectrogram[idx] - FEATURE_MEAN[band]) / FEATURE_STD[band];
        }
    }
}

#endif // MODEL_NORMALIZATION_H
