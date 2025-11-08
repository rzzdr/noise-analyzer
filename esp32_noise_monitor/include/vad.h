/*
 * Voice Activity Detector (VAD) for ESP32-CAM
 * ===========================================
 * 
 * C++ implementation of the Python VAD class for library environment noise detection.
 * Uses calibration-based thresholds with multiple acoustic features.
 */

#ifndef VAD_H
#define VAD_H

#include <Arduino.h>
#include <math.h>
#include "config.h"
#include "arduinoFFT.h"

class VoiceActivityDetector {
private:
    // Calibration state
    bool is_calibrated;
    int calibration_sample_count;
    float calibration_samples[VAD_CALIBRATION_SAMPLES][4];  // [energy, spectral_centroid, zcr, mfcc_variance]
    
    // Computed thresholds
    float energy_threshold;
    float spectral_centroid_threshold;
    float zero_crossing_threshold;
    float mfcc_variance_threshold;
    
    // Activity buffer for temporal smoothing
    bool activity_buffer[VAD_ACTIVITY_BUFFER_SIZE];
    int activity_buffer_index;
    
    // FFT for spectral analysis
    arduinoFFT fft;
    double* fft_input;
    double* fft_output;
    
    // Working buffers
    float* temp_buffer;
    
    // Statistics computation
    float compute_percentile(float* data, int n, float percentile);
    void insertion_sort(float* arr, int n);
    
public:
    VoiceActivityDetector();
    ~VoiceActivityDetector();
    
    // Initialization
    bool init();
    void reset_calibration();
    
    // Feature extraction
    float extract_energy(float* audio, int length);
    float extract_spectral_centroid(float* audio, int length);
    float extract_zero_crossing_rate(float* audio, int length);
    float extract_mfcc_variance(float* audio, int length);
    
    // Calibration
    bool add_calibration_sample(float* audio, int length);
    bool is_calibration_complete() const { return is_calibrated; }
    float get_calibration_progress() const;
    
    // Activity detection
    struct VADResult {
        bool is_activity;
        float confidence;
        float energy;
        float spectral_centroid;
        float zcr;
        float mfcc_variance;
        unsigned long processing_time_us;
    };
    
    VADResult detect_activity(float* audio, int length);
    
    // Status
    bool is_ready() const { return is_calibrated; }
    void print_thresholds() const;
};

#endif // VAD_H