/*
 * I2S Audio Capture for ESP32-CAM with INMP441 Microphone
 * =======================================================
 * 
 * Handles real-time audio capture using I2S DMA for low-latency streaming.
 * Optimized for ESP32-CAM with PSRAM support.
 */

#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include <Arduino.h>
#include <driver/i2s.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include "config.h"

class AudioCapture {
private:
    // I2S configuration
    i2s_config_t i2s_config;
    i2s_pin_config_t pin_config;
    bool is_initialized;
    bool is_capturing;
    
    // Audio buffers
    int32_t* raw_buffer;      // Raw I2S data (32-bit)
    float* audio_buffer;      // Converted audio data (-1.0 to 1.0)
    float* window_buffer;     // Sliding window for processing
    
    // Buffer management
    int buffer_write_index;
    int buffer_read_index;
    bool buffer_ready;
    
    // Threading
    TaskHandle_t capture_task_handle;
    QueueHandle_t audio_queue;
    SemaphoreHandle_t buffer_mutex;
    
    // Performance monitoring
    struct CaptureStats {
        unsigned long samples_captured;
        unsigned long buffer_overruns;
        unsigned long dma_errors;
        unsigned long conversion_time_us;
        float cpu_usage_percent;
    };
    
    CaptureStats stats;
    
    // Helper functions
    bool configure_i2s();
    void convert_samples(int32_t* raw_data, float* audio_data, size_t sample_count);
    static void capture_task(void* parameter);
    void capture_loop();
    
public:
    AudioCapture();
    ~AudioCapture();
    
    // Initialization
    bool init();
    void deinit();
    
    // Capture control
    bool start_capture();
    bool stop_capture();
    bool is_running() const { return is_capturing; }
    
    // Data access
    bool get_audio_window(float* output_buffer, size_t buffer_size);
    bool is_window_ready() const { return buffer_ready; }
    
    // Configuration
    void set_gain(float gain_db);
    bool test_microphone(int duration_ms = 1000);
    
    // Performance monitoring  
    CaptureStats get_stats() const { return stats; }
    void reset_stats();
    float get_signal_level() const;  // Current RMS level
    
    // Diagnostics
    void print_i2s_status() const;
    void print_buffer_status() const;
    bool self_test();
    
    // Error handling
    enum CaptureError {
        CAPTURE_OK = 0,
        CAPTURE_NOT_INITIALIZED,
        CAPTURE_I2S_ERROR,
        CAPTURE_MEMORY_ERROR,
        CAPTURE_TIMEOUT_ERROR,
        CAPTURE_BUFFER_OVERRUN
    };
    
    CaptureError get_last_error() const { return last_error; }
    const char* get_error_string(CaptureError error) const;
    
private:
    CaptureError last_error;
    float current_gain;
    unsigned long last_stats_update;
};

#endif // AUDIO_CAPTURE_H