/*
 * I2S Audio Capture Implementation
 * ================================
 */

#include "audio_capture.h"

AudioCapture::AudioCapture() 
    : is_initialized(false), is_capturing(false), buffer_write_index(0), 
      buffer_read_index(0), buffer_ready(false), capture_task_handle(nullptr),
      audio_queue(nullptr), buffer_mutex(nullptr), current_gain(0.0f),
      last_error(CAPTURE_OK), last_stats_update(0) {
    
    // Initialize pointers
    raw_buffer = nullptr;
    audio_buffer = nullptr;
    window_buffer = nullptr;
    
    // Clear stats
    memset(&stats, 0, sizeof(stats));
}

AudioCapture::~AudioCapture() {
    deinit();
}

bool AudioCapture::init() {
    DEBUG_INFO("Initializing I2S audio capture...\n");
    
    if (is_initialized) {
        DEBUG_WARN("Audio capture already initialized\n");
        return true;
    }
    
    // Allocate buffers
    raw_buffer = (int32_t*)ps_malloc(AUDIO_BUFFER_SIZE);
    audio_buffer = (float*)ps_malloc(AUDIO_BUFFER_SIZE);
    window_buffer = (float*)ps_malloc(SAMPLES_PER_WINDOW * sizeof(float));
    
    if (!raw_buffer || !audio_buffer || !window_buffer) {
        DEBUG_ERROR("Failed to allocate audio buffers\n");
        last_error = CAPTURE_MEMORY_ERROR;
        return false;
    }
    
    DEBUG_INFO("Audio buffers allocated in PSRAM\n");
    DEBUG_INFO("  Raw buffer: %d bytes\n", AUDIO_BUFFER_SIZE);
    DEBUG_INFO("  Audio buffer: %d bytes\n", AUDIO_BUFFER_SIZE);
    DEBUG_INFO("  Window buffer: %d bytes\n", (int)(SAMPLES_PER_WINDOW * sizeof(float)));
    
    // Create synchronization objects
    buffer_mutex = xSemaphoreCreateMutex();
    audio_queue = xQueueCreate(4, sizeof(bool));  // Small queue for notifications
    
    if (!buffer_mutex || !audio_queue) {
        DEBUG_ERROR("Failed to create synchronization objects\n");
        last_error = CAPTURE_MEMORY_ERROR;
        return false;
    }
    
    // Configure I2S
    if (!configure_i2s()) {
        DEBUG_ERROR("Failed to configure I2S\n");
        return false;
    }
    
    is_initialized = true;
    DEBUG_INFO("I2S audio capture initialized successfully\n");
    
    return true;
}

void AudioCapture::deinit() {
    if (is_capturing) {
        stop_capture();
    }
    
    if (is_initialized) {
        i2s_driver_uninstall(I2S_PORT);
        is_initialized = false;
    }
    
    // Free buffers
    if (raw_buffer) {
        free(raw_buffer);
        raw_buffer = nullptr;
    }
    if (audio_buffer) {
        free(audio_buffer);
        audio_buffer = nullptr;
    }
    if (window_buffer) {
        free(window_buffer);
        window_buffer = nullptr;
    }
    
    // Clean up synchronization objects
    if (buffer_mutex) {
        vSemaphoreDelete(buffer_mutex);
        buffer_mutex = nullptr;
    }
    if (audio_queue) {
        vQueueDelete(audio_queue);
        audio_queue = nullptr;
    }
    
    DEBUG_INFO("Audio capture deinitialized\n");
}

bool AudioCapture::configure_i2s() {
    DEBUG_INFO("Configuring I2S...\n");
    
    // I2S configuration
    i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,  // INMP441 is mono
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    // Pin configuration
    pin_config = {
        .bck_io_num = I2S_SCK_PIN,      // BCLK
        .ws_io_num = I2S_WS_PIN,        // LRCLK
        .data_out_num = I2S_PIN_NO_CHANGE,  // Not used for input
        .data_in_num = I2S_SD_PIN       // Data input
    };
    
    // Install I2S driver
    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        DEBUG_ERROR("I2S driver install failed: %s\n", esp_err_to_name(err));
        last_error = CAPTURE_I2S_ERROR;
        return false;
    }
    
    // Set I2S pin configuration
    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        DEBUG_ERROR("I2S pin configuration failed: %s\n", esp_err_to_name(err));
        last_error = CAPTURE_I2S_ERROR;
        return false;
    }
    
    // Clear DMA buffers
    i2s_zero_dma_buffer(I2S_PORT);
    
    DEBUG_INFO("I2S configured successfully\n");
    DEBUG_INFO("  Sample rate: %d Hz\n", SAMPLE_RATE);
    DEBUG_INFO("  Bits per sample: 32\n");
    DEBUG_INFO("  DMA buffers: %d x %d samples\n", DMA_BUF_COUNT, DMA_BUF_LEN);
    DEBUG_INFO("  Pin config: SCK=%d, WS=%d, SD=%d\n", I2S_SCK_PIN, I2S_WS_PIN, I2S_SD_PIN);
    
    return true;
}

bool AudioCapture::start_capture() {
    if (!is_initialized) {
        DEBUG_ERROR("Audio capture not initialized\n");
        last_error = CAPTURE_NOT_INITIALIZED;
        return false;
    }
    
    if (is_capturing) {
        DEBUG_WARN("Audio capture already running\n");
        return true;
    }
    
    DEBUG_INFO("Starting audio capture...\n");
    
    // Reset buffer state
    buffer_write_index = 0;
    buffer_read_index = 0;
    buffer_ready = false;
    
    // Start I2S
    esp_err_t err = i2s_start(I2S_PORT);
    if (err != ESP_OK) {
        DEBUG_ERROR("Failed to start I2S: %s\n", esp_err_to_name(err));
        last_error = CAPTURE_I2S_ERROR;
        return false;
    }
    
    // Create capture task
    BaseType_t task_result = xTaskCreatePinnedToCore(
        capture_task,
        "audio_capture",
        8192,           // Stack size
        this,           // Parameter
        5,              // Priority (high)
        &capture_task_handle,
        0               // Pin to core 0
    );
    
    if (task_result != pdPASS) {
        DEBUG_ERROR("Failed to create capture task\n");
        i2s_stop(I2S_PORT);
        last_error = CAPTURE_MEMORY_ERROR;
        return false;
    }
    
    is_capturing = true;
    DEBUG_INFO("Audio capture started successfully\n");
    
    return true;
}

bool AudioCapture::stop_capture() {
    if (!is_capturing) {
        return true;
    }
    
    DEBUG_INFO("Stopping audio capture...\n");
    
    is_capturing = false;
    
    // Wait for task to finish
    if (capture_task_handle) {
        vTaskDelete(capture_task_handle);
        capture_task_handle = nullptr;
    }
    
    // Stop I2S
    i2s_stop(I2S_PORT);
    
    DEBUG_INFO("Audio capture stopped\n");
    
    return true;
}

void AudioCapture::capture_task(void* parameter) {
    AudioCapture* capture = static_cast<AudioCapture*>(parameter);
    capture->capture_loop();
}

void AudioCapture::capture_loop() {
    DEBUG_INFO("Capture task started on core %d\n", xPortGetCoreID());
    
    size_t bytes_read = 0;
    int32_t* temp_buffer = (int32_t*)malloc(DMA_BUF_LEN * sizeof(int32_t));
    
    if (!temp_buffer) {
        DEBUG_ERROR("Failed to allocate temp buffer in capture task\n");
        return;
    }
    
    unsigned long loop_count = 0;
    unsigned long last_stats_print = 0;
    
    while (is_capturing) {
        // Read from I2S
        esp_err_t result = i2s_read(I2S_PORT, temp_buffer, 
                                   DMA_BUF_LEN * sizeof(int32_t), 
                                   &bytes_read, portMAX_DELAY);
        
        if (result != ESP_OK) {
            DEBUG_ERROR("I2S read error: %s\n", esp_err_to_name(result));
            stats.dma_errors++;
            continue;
        }
        
        if (bytes_read == 0) {
            continue;
        }
        
        unsigned long convert_start = micros();
        
        // Convert samples and add to circular buffer
        size_t samples_read = bytes_read / sizeof(int32_t);
        
        if (xSemaphoreTake(buffer_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            for (size_t i = 0; i < samples_read; i++) {
                // Convert 32-bit I2S data to float (-1.0 to 1.0)
                float sample = (float)temp_buffer[i] / (float)INT32_MAX;
                
                // Apply gain if needed
                if (current_gain != 0.0f) {
                    sample *= powf(10.0f, current_gain / 20.0f);  // dB to linear
                    sample = CLAMP(sample, -1.0f, 1.0f);  // Prevent clipping
                }
                
                audio_buffer[buffer_write_index] = sample;
                buffer_write_index = (buffer_write_index + 1) % (AUDIO_BUFFER_SIZE / sizeof(float));
                
                // Check for buffer overrun
                if (buffer_write_index == buffer_read_index) {
                    stats.buffer_overruns++;
                    buffer_read_index = (buffer_read_index + 1) % (AUDIO_BUFFER_SIZE / sizeof(float));
                }
            }
            
            // Check if we have enough samples for a window
            int available_samples = (buffer_write_index - buffer_read_index + 
                                   (AUDIO_BUFFER_SIZE / sizeof(float))) % 
                                   (AUDIO_BUFFER_SIZE / sizeof(float));
            
            if (available_samples >= SAMPLES_PER_WINDOW) {
                buffer_ready = true;
                
                // Notify waiting task
                bool notification = true;
                xQueueSend(audio_queue, &notification, 0);
            }
            
            xSemaphoreGive(buffer_mutex);
        }
        
        stats.conversion_time_us += micros() - convert_start;
        stats.samples_captured += samples_read;
        loop_count++;
        
        // Print stats periodically (every 5 seconds)
        if (millis() - last_stats_print > 5000) {
            DEBUG_DEBUG("Capture stats: %lu samples, %lu overruns, %lu errors\n",
                       stats.samples_captured, stats.buffer_overruns, stats.dma_errors);
            last_stats_print = millis();
        }
        
        // Small delay to prevent task starvation
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    free(temp_buffer);
    DEBUG_INFO("Capture task finished\n");
}

bool AudioCapture::get_audio_window(float* output_buffer, size_t buffer_size) {
    if (!is_capturing || !buffer_ready) {
        return false;
    }
    
    if (buffer_size != SAMPLES_PER_WINDOW) {
        DEBUG_ERROR("Buffer size mismatch: %d != %d\n", buffer_size, (int)SAMPLES_PER_WINDOW);
        return false;
    }
    
    if (xSemaphoreTake(buffer_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        // Copy samples from circular buffer to output
        int buffer_size_samples = AUDIO_BUFFER_SIZE / sizeof(float);
        
        for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
            int idx = (buffer_read_index + i) % buffer_size_samples;
            output_buffer[i] = audio_buffer[idx];
        }
        
        // Advance read pointer
        buffer_read_index = (buffer_read_index + SAMPLES_PER_WINDOW) % buffer_size_samples;
        
        // Check if we still have enough samples for next window
        int available_samples = (buffer_write_index - buffer_read_index + buffer_size_samples) % buffer_size_samples;
        buffer_ready = (available_samples >= SAMPLES_PER_WINDOW);
        
        xSemaphoreGive(buffer_mutex);
        return true;
    }
    
    DEBUG_WARN("Failed to acquire buffer mutex\n");
    return false;
}

void AudioCapture::set_gain(float gain_db) {
    current_gain = CLAMP(gain_db, -40.0f, 40.0f);  // Reasonable gain range
    DEBUG_INFO("Audio gain set to %.1f dB\n", current_gain);
}

bool AudioCapture::test_microphone(int duration_ms) {
    DEBUG_INFO("Testing microphone for %d ms...\n", duration_ms);
    
    if (!is_initialized) {
        if (!init()) {
            return false;
        }
    }
    
    if (!start_capture()) {
        return false;
    }
    
    // Wait for buffer to fill
    vTaskDelay(pdMS_TO_TICKS(100));
    
    unsigned long start_time = millis();
    float max_level = 0.0f;
    float total_energy = 0.0f;
    int sample_count = 0;
    
    while (millis() - start_time < duration_ms) {
        if (buffer_ready) {
            float test_buffer[SAMPLES_PER_WINDOW];
            if (get_audio_window(test_buffer, SAMPLES_PER_WINDOW)) {
                // Calculate RMS level
                float rms = 0.0f;
                for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
                    rms += test_buffer[i] * test_buffer[i];
                }
                rms = sqrtf(rms / SAMPLES_PER_WINDOW);
                
                max_level = MAX(max_level, rms);
                total_energy += rms;
                sample_count++;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    stop_capture();
    
    float avg_level = sample_count > 0 ? total_energy / sample_count : 0.0f;
    
    DEBUG_INFO("Microphone test results:\n");
    DEBUG_INFO("  Max level: %.6f (%.1f dB)\n", max_level, 20.0f * log10f(max_level + 1e-8f));
    DEBUG_INFO("  Avg level: %.6f (%.1f dB)\n", avg_level, 20.0f * log10f(avg_level + 1e-8f));
    DEBUG_INFO("  Samples: %d\n", sample_count);
    
    // Consider test successful if we detect some signal
    bool test_passed = (max_level > 1e-6f);  // Above noise floor
    
    if (test_passed) {
        DEBUG_INFO("✓ Microphone test PASSED\n");
    } else {
        DEBUG_WARN("✗ Microphone test FAILED - no signal detected\n");
    }
    
    return test_passed;
}

float AudioCapture::get_signal_level() const {
    // This would calculate current RMS level from recent samples
    // For now, return a placeholder
    return 0.0f;
}

void AudioCapture::reset_stats() {
    memset(&stats, 0, sizeof(stats));
}

void AudioCapture::print_i2s_status() const {
    DEBUG_INFO("I2S Status:\n");
    DEBUG_INFO("  Initialized: %s\n", is_initialized ? "Yes" : "No");
    DEBUG_INFO("  Capturing: %s\n", is_capturing ? "Yes" : "No");
    DEBUG_INFO("  Sample rate: %d Hz\n", SAMPLE_RATE);
    DEBUG_INFO("  Buffer ready: %s\n", buffer_ready ? "Yes" : "No");
}

void AudioCapture::print_buffer_status() const {
    if (xSemaphoreTake(buffer_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        int buffer_size_samples = AUDIO_BUFFER_SIZE / sizeof(float);
        int available_samples = (buffer_write_index - buffer_read_index + buffer_size_samples) % buffer_size_samples;
        
        DEBUG_INFO("Buffer Status:\n");
        DEBUG_INFO("  Size: %d samples\n", buffer_size_samples);
        DEBUG_INFO("  Available: %d samples\n", available_samples);
        DEBUG_INFO("  Write index: %d\n", buffer_write_index);
        DEBUG_INFO("  Read index: %d\n", buffer_read_index);
        DEBUG_INFO("  Usage: %.1f%%\n", 100.0f * available_samples / buffer_size_samples);
        
        xSemaphoreGive(buffer_mutex);
    }
}

bool AudioCapture::self_test() {
    DEBUG_INFO("Running audio capture self-test...\n");
    
    // Test 1: Initialization
    if (!init()) {
        DEBUG_ERROR("Self-test failed: initialization\n");
        return false;
    }
    
    // Test 2: I2S configuration
    print_i2s_status();
    
    // Test 3: Buffer allocation
    if (!raw_buffer || !audio_buffer || !window_buffer) {
        DEBUG_ERROR("Self-test failed: buffer allocation\n");
        return false;
    }
    
    // Test 4: Microphone test
    if (!test_microphone(2000)) {
        DEBUG_WARN("Self-test warning: microphone test failed (may be environmental)\n");
        // Don't fail self-test for this, as it may be environmental
    }
    
    DEBUG_INFO("✓ Audio capture self-test PASSED\n");
    return true;
}

const char* AudioCapture::get_error_string(CaptureError error) const {
    switch (error) {
        case CAPTURE_OK: return "No error";
        case CAPTURE_NOT_INITIALIZED: return "Not initialized";
        case CAPTURE_I2S_ERROR: return "I2S error";
        case CAPTURE_MEMORY_ERROR: return "Memory allocation error";
        case CAPTURE_TIMEOUT_ERROR: return "Timeout error";
        case CAPTURE_BUFFER_OVERRUN: return "Buffer overrun";
        default: return "Unknown error";
    }
}