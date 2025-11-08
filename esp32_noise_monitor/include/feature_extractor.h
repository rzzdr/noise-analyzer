/*
 * Audio Feature Extractor for ESP32-CAM
 * =====================================
 * 
 * Extracts mel spectrogram features from audio for neural network inference.
 * Optimized for ESP32 with PSRAM and based on the training pipeline.
 */

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <Arduino.h>
#include <math.h>
#include "config.h"
#include "arduinoFFT.h"

class FeatureExtractor {
private:
    // FFT processing
    arduinoFFT fft;
    double* fft_input;
    double* fft_output;
    float* power_spectrum;
    
    // Mel filter bank
    float** mel_filters;  // [n_mel_bands][n_fft_bins]
    int* filter_starts;   // Start bin for each mel filter
    int* filter_ends;     // End bin for each mel filter
    
    // Window function
    float* window_function;
    
    // Output buffer
    float* mel_spectrogram;  // [n_frames * n_mel_bands]
    
    // Normalization parameters (loaded from model training)
    float* feature_mean;
    float* feature_std;
    
    // Hop analysis for frame extraction
    int hop_samples;
    int frame_samples;
    
    // Helper functions
    void compute_mel_filters();
    float hz_to_mel(float hz);
    float mel_to_hz(float mel);
    void apply_window(float* audio, int start_idx, int length);
    void compute_power_spectrum();
    void apply_mel_filters(int frame_idx);
    bool load_normalization_params();
    
public:
    FeatureExtractor();
    ~FeatureExtractor();
    
    // Initialization
    bool init();
    
    // Feature extraction
    bool extract_features(float* audio, int audio_length, float* output_features);
    
    // Get output dimensions
    int get_output_size() const { return N_TIME_FRAMES * N_MEL_BANDS; }
    
    // Performance monitoring
    struct ExtractionStats {
        unsigned long total_time_us;
        unsigned long fft_time_us;
        unsigned long mel_time_us;
        unsigned long norm_time_us;
        int n_frames_processed;
    };
    
    ExtractionStats get_last_stats() const { return last_stats; }
    
private:
    ExtractionStats last_stats;
};

#endif // FEATURE_EXTRACTOR_H