/*
 * Voice Activity Detector (VAD) Implementation
 * ===========================================
 */

#include "vad.h"

VoiceActivityDetector::VoiceActivityDetector() 
    : is_calibrated(false), calibration_sample_count(0), activity_buffer_index(0) {
    
    // Initialize activity buffer
    for (int i = 0; i < VAD_ACTIVITY_BUFFER_SIZE; i++) {
        activity_buffer[i] = false;
    }
    
    // Initialize thresholds to default values
    energy_threshold = 0.005f;
    spectral_centroid_threshold = 1000.0f;
    zero_crossing_threshold = 0.01f;
    mfcc_variance_threshold = 0.1f;
    
    // Initialize pointers
    fft_input = nullptr;
    fft_output = nullptr;
    temp_buffer = nullptr;
}

VoiceActivityDetector::~VoiceActivityDetector() {
    if (fft_input) {
        free(fft_input);
    }
    if (fft_output) {
        free(fft_output);
    }
    if (temp_buffer) {
        free(temp_buffer);
    }
}

bool VoiceActivityDetector::init() {
    DEBUG_INFO("Initializing VAD...\n");
    
    // Allocate FFT buffers
    int fft_size = N_FFT;
    fft_input = (double*)ps_malloc(fft_size * sizeof(double));  // Use PSRAM
    fft_output = (double*)ps_malloc(fft_size * sizeof(double));
    temp_buffer = (float*)ps_malloc(SAMPLES_PER_WINDOW * sizeof(float));
    
    if (!fft_input || !fft_output || !temp_buffer) {
        DEBUG_ERROR("Failed to allocate VAD buffers\n");
        return false;
    }
    
    // Initialize FFT object
    fft = arduinoFFT(fft_input, fft_output, fft_size, SAMPLE_RATE);
    
    DEBUG_INFO("VAD initialized successfully\n");
    DEBUG_INFO("Calibration required: %.1f seconds (%d samples)\n", 
               VAD_CALIBRATION_DURATION_SEC, VAD_CALIBRATION_SAMPLES);
    
    return true;
}

void VoiceActivityDetector::reset_calibration() {
    is_calibrated = false;
    calibration_sample_count = 0;
    DEBUG_INFO("VAD calibration reset\n");
}

float VoiceActivityDetector::extract_energy(float* audio, int length) {
    float sum_squares = 0.0f;
    
    for (int i = 0; i < length; i++) {
        sum_squares += audio[i] * audio[i];
    }
    
    return sqrtf(sum_squares / length);  // RMS energy
}

float VoiceActivityDetector::extract_spectral_centroid(float* audio, int length) {
    // Window the signal (simple Hamming window approximation)
    for (int i = 0; i < length && i < N_FFT; i++) {
        float window = 0.54f - 0.46f * cosf(2.0f * PI * i / (length - 1));
        fft_input[i] = audio[i] * window;
    }
    
    // Zero pad if necessary
    for (int i = length; i < N_FFT; i++) {
        fft_input[i] = 0.0;
    }
    
    // Compute FFT
    fft.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    fft.Compute(FFT_FORWARD);
    fft.ComplexToMagnitude();
    
    // Compute spectral centroid
    float weighted_sum = 0.0f;
    float magnitude_sum = 0.0f;
    
    int num_bins = N_FFT / 2;  // Only use positive frequencies
    
    for (int i = 1; i < num_bins; i++) {  // Skip DC component
        float magnitude = fft_output[i];
        float frequency = (float)i * SAMPLE_RATE / N_FFT;
        
        weighted_sum += frequency * magnitude;
        magnitude_sum += magnitude;
    }
    
    if (magnitude_sum > 0.0f) {
        return weighted_sum / magnitude_sum;
    } else {
        return 0.0f;
    }
}

float VoiceActivityDetector::extract_zero_crossing_rate(float* audio, int length) {
    int zero_crossings = 0;
    
    for (int i = 1; i < length; i++) {
        if ((audio[i] >= 0 && audio[i-1] < 0) || (audio[i] < 0 && audio[i-1] >= 0)) {
            zero_crossings++;
        }
    }
    
    return (float)zero_crossings / (length - 1);
}

float VoiceActivityDetector::extract_mfcc_variance(float* audio, int length) {
    // Simplified MFCC variance approximation
    // In full implementation, this would compute MFCCs and their variance
    // For now, we'll use spectral variance as a proxy
    
    // Compute power spectrum (already done in spectral centroid if called before)
    for (int i = 0; i < length && i < N_FFT; i++) {
        fft_input[i] = audio[i];
    }
    for (int i = length; i < N_FFT; i++) {
        fft_input[i] = 0.0;
    }
    
    fft.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    fft.Compute(FFT_FORWARD);
    fft.ComplexToMagnitude();
    
    // Compute variance of log power spectrum (MFCC-like feature)
    float mean_log_power = 0.0f;
    int num_bins = N_FFT / 2;
    
    // Convert to log power and compute mean
    for (int i = 1; i < num_bins; i++) {
        fft_output[i] = logf(fft_output[i] + 1e-8f);  // Add small epsilon to avoid log(0)
        mean_log_power += fft_output[i];
    }
    mean_log_power /= (num_bins - 1);
    
    // Compute variance
    float variance = 0.0f;
    for (int i = 1; i < num_bins; i++) {
        float diff = fft_output[i] - mean_log_power;
        variance += diff * diff;
    }
    variance /= (num_bins - 1);
    
    return variance;
}

bool VoiceActivityDetector::add_calibration_sample(float* audio, int length) {
    if (is_calibrated) {
        DEBUG_WARN("VAD already calibrated\n");
        return true;
    }
    
    if (calibration_sample_count >= VAD_CALIBRATION_SAMPLES) {
        DEBUG_ERROR("Calibration buffer full\n");
        return false;
    }
    
    // Extract features
    float energy = extract_energy(audio, length);
    float spectral_centroid = extract_spectral_centroid(audio, length);
    float zcr = extract_zero_crossing_rate(audio, length);
    float mfcc_variance = extract_mfcc_variance(audio, length);
    
    // Store calibration sample
    calibration_samples[calibration_sample_count][0] = energy;
    calibration_samples[calibration_sample_count][1] = spectral_centroid;
    calibration_samples[calibration_sample_count][2] = zcr;
    calibration_samples[calibration_sample_count][3] = mfcc_variance;
    
    calibration_sample_count++;
    
    DEBUG_DEBUG("Calibration sample %d/%d: E=%.6f, SC=%.1f, ZCR=%.4f, MFCC=%.4f\n",
                calibration_sample_count, VAD_CALIBRATION_SAMPLES,
                energy, spectral_centroid, zcr, mfcc_variance);
    
    // Check if calibration is complete
    if (calibration_sample_count >= VAD_CALIBRATION_SAMPLES) {
        // Compute thresholds using 95th percentile
        float energy_values[VAD_CALIBRATION_SAMPLES];
        float spectral_values[VAD_CALIBRATION_SAMPLES];
        float zcr_values[VAD_CALIBRATION_SAMPLES];
        float mfcc_values[VAD_CALIBRATION_SAMPLES];
        
        for (int i = 0; i < VAD_CALIBRATION_SAMPLES; i++) {
            energy_values[i] = calibration_samples[i][0];
            spectral_values[i] = calibration_samples[i][1];
            zcr_values[i] = calibration_samples[i][2];
            mfcc_values[i] = calibration_samples[i][3];
        }
        
        float energy_baseline = compute_percentile(energy_values, VAD_CALIBRATION_SAMPLES, 95.0f);
        float spectral_baseline = compute_percentile(spectral_values, VAD_CALIBRATION_SAMPLES, 95.0f);
        float zcr_baseline = compute_percentile(zcr_values, VAD_CALIBRATION_SAMPLES, 95.0f);
        float mfcc_baseline = compute_percentile(mfcc_values, VAD_CALIBRATION_SAMPLES, 95.0f);
        
        // Set thresholds with safety margins
        energy_threshold = MAX(energy_baseline * VAD_ENERGY_MARGIN, 0.002f);
        spectral_centroid_threshold = MAX(spectral_baseline * VAD_SPECTRAL_MARGIN, 800.0f);
        zero_crossing_threshold = MAX(zcr_baseline * VAD_ZCR_MARGIN, 0.005f);
        mfcc_variance_threshold = MAX(mfcc_baseline * VAD_MFCC_MARGIN, 0.05f);
        
        is_calibrated = true;
        
        DEBUG_INFO("VAD calibration completed!\n");
        print_thresholds();
        
        return true;
    }
    
    return false;  // Calibration not yet complete
}

float VoiceActivityDetector::get_calibration_progress() const {
    return (float)calibration_sample_count / VAD_CALIBRATION_SAMPLES;
}

VoiceActivityDetector::VADResult VoiceActivityDetector::detect_activity(float* audio, int length) {
    TIMING_START();
    
    VADResult result = {false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
    
    if (!is_calibrated) {
        DEBUG_WARN("VAD not calibrated\n");
        result.processing_time_us = micros() - start_time;
        return result;
    }
    
    // Extract features
    result.energy = extract_energy(audio, length);
    result.spectral_centroid = extract_spectral_centroid(audio, length);
    result.zcr = extract_zero_crossing_rate(audio, length);
    result.mfcc_variance = extract_mfcc_variance(audio, length);
    
    // Individual feature decisions
    bool energy_active = result.energy > energy_threshold;
    bool spectral_active = result.spectral_centroid > spectral_centroid_threshold;
    bool zcr_active = result.zcr > zero_crossing_threshold;
    bool mfcc_active = result.mfcc_variance > mfcc_variance_threshold;
    
    // Weighted decision
    float confidence = 0.0f;
    confidence += energy_active ? VAD_ENERGY_WEIGHT : 0.0f;
    confidence += spectral_active ? VAD_SPECTRAL_WEIGHT : 0.0f;
    confidence += zcr_active ? VAD_ZCR_WEIGHT : 0.0f;
    confidence += mfcc_active ? VAD_MFCC_WEIGHT : 0.0f;
    
    result.confidence = confidence;
    bool current_activity = confidence > 0.5f;  // Majority vote with weights
    
    // Temporal smoothing
    activity_buffer[activity_buffer_index] = current_activity;
    activity_buffer_index = (activity_buffer_index + 1) % VAD_ACTIVITY_BUFFER_SIZE;
    
    // Count recent activity
    int active_count = 0;
    for (int i = 0; i < VAD_ACTIVITY_BUFFER_SIZE; i++) {
        if (activity_buffer[i]) {
            active_count++;
        }
    }
    
    float recent_activity_ratio = (float)active_count / VAD_ACTIVITY_BUFFER_SIZE;
    result.is_activity = recent_activity_ratio > VAD_ACTIVITY_THRESHOLD;
    
    result.processing_time_us = micros() - start_time;
    
    DEBUG_DEBUG("VAD: E=%.4f(%.4f) SC=%.1f(%.1f) ZCR=%.4f(%.4f) MFCC=%.4f(%.4f) -> %s (%.2f)\n",
                result.energy, energy_threshold,
                result.spectral_centroid, spectral_centroid_threshold,
                result.zcr, zero_crossing_threshold,
                result.mfcc_variance, mfcc_variance_threshold,
                result.is_activity ? "ACTIVE" : "SILENT",
                result.confidence);
    
    return result;
}

void VoiceActivityDetector::print_thresholds() const {
    DEBUG_INFO("VAD Thresholds:\n");
    DEBUG_INFO("  Energy: %.6f\n", energy_threshold);
    DEBUG_INFO("  Spectral Centroid: %.1f Hz\n", spectral_centroid_threshold);
    DEBUG_INFO("  Zero Crossing Rate: %.6f\n", zero_crossing_threshold);
    DEBUG_INFO("  MFCC Variance: %.6f\n", mfcc_variance_threshold);
}

// Helper functions
float VoiceActivityDetector::compute_percentile(float* data, int n, float percentile) {
    // Create a copy for sorting
    float* sorted_data = (float*)malloc(n * sizeof(float));
    if (!sorted_data) {
        DEBUG_ERROR("Failed to allocate memory for percentile computation\n");
        return 0.0f;
    }
    
    memcpy(sorted_data, data, n * sizeof(float));
    insertion_sort(sorted_data, n);
    
    int index = (int)((percentile / 100.0f) * (n - 1));
    index = CLAMP(index, 0, n - 1);
    
    float result = sorted_data[index];
    free(sorted_data);
    
    return result;
}

void VoiceActivityDetector::insertion_sort(float* arr, int n) {
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}