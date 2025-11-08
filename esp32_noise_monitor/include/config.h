/*
 * ESP32-CAM Noise Monitor Configuration
 * ====================================
 * 
 * Hardware configuration and constants for the ESP32-CAM audio classifier
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ================================
// HARDWARE CONFIGURATION
// ================================

// I2S Microphone Pins (INMP441)
#define I2S_WS_PIN      15    // LRCLK (Word Select)
#define I2S_SCK_PIN     14    // BCLK (Bit Clock)
#define I2S_SD_PIN      13    // DOUT (Data Out)
#define I2S_PORT        I2S_NUM_0

// Audio Configuration
#define SAMPLE_RATE     16000
#define BITS_PER_SAMPLE 32
#define CHANNELS        1
#define DURATION_SEC    1.0f
#define SAMPLES_PER_WINDOW  (SAMPLE_RATE * DURATION_SEC)

// I2S DMA Configuration
#define DMA_BUF_COUNT   4
#define DMA_BUF_LEN     512

// ================================
// MODEL CONFIGURATION
// ================================

// Feature extraction parameters (must match training)
#define N_MEL_BANDS     40
#define N_TIME_FRAMES   100
#define N_FFT           512
#define HOP_LENGTH      160     // 10ms at 16kHz
#define FRAME_LENGTH    400     // 25ms at 16kHz

// Model input/output
#define MODEL_INPUT_SIZE    (N_TIME_FRAMES * N_MEL_BANDS)
#define NUM_CLASSES         4

// Class names (must match training order)
const char* const CLASS_NAMES[NUM_CLASSES] = {
    "Whispering",
    "Typing", 
    "Phone_ringing",
    "Loud_talking"
};

// Class emojis for display
const char* const CLASS_EMOJIS[NUM_CLASSES] = {
    "💬",  // Whispering
    "⌨️",   // Typing
    "📞",  // Phone_ringing  
    "🗣️"   // Loud_talking
};

// ================================
// VAD CONFIGURATION
// ================================

// VAD calibration
#define VAD_CALIBRATION_DURATION_SEC    3.5f
#define VAD_CALIBRATION_SAMPLES         (int)(VAD_CALIBRATION_DURATION_SEC / DURATION_SEC)

// VAD thresholds and margins
#define VAD_ENERGY_MARGIN       1.3f
#define VAD_SPECTRAL_MARGIN     1.2f  
#define VAD_ZCR_MARGIN          1.3f
#define VAD_MFCC_MARGIN         1.4f

// VAD feature weights
#define VAD_ENERGY_WEIGHT       0.4f
#define VAD_SPECTRAL_WEIGHT     0.25f
#define VAD_ZCR_WEIGHT          0.2f
#define VAD_MFCC_WEIGHT         0.15f

// VAD temporal smoothing
#define VAD_ACTIVITY_BUFFER_SIZE    10
#define VAD_ACTIVITY_THRESHOLD      0.4f    // 40% of recent samples must be active

// ================================
// PERFORMANCE CONFIGURATION  
// ================================

// Memory allocation
#define AUDIO_BUFFER_SIZE       (SAMPLES_PER_WINDOW * 4)    // 4 bytes per sample (float)
#define FEATURE_BUFFER_SIZE     (MODEL_INPUT_SIZE * 4)      // 4 bytes per float
#define TFLITE_ARENA_SIZE       (200 * 1024)               // 200KB for TFLite

// Performance targets
#define MAX_INFERENCE_TIME_MS   200
#define MAX_VAD_TIME_MS         50
#define MAX_FEATURE_TIME_MS     100

// Timing and monitoring
#define PERFORMANCE_WINDOW      100     // Average over last 100 inferences
#define MEMORY_CHECK_INTERVAL   50      // Check memory every 50 inferences

// ================================
// COMMUNICATION CONFIGURATION
// ================================

// Serial output
#define SERIAL_BAUD_RATE        115200
#define CSV_OUTPUT_HEADER       "timestamp,class,confidence,inference_ms,vad_confidence,heap_free,psram_free"

// WiFi (optional - for remote logging)
#define WIFI_SSID               "YOUR_WIFI_SSID"
#define WIFI_PASSWORD           "YOUR_WIFI_PASSWORD"
#define ENABLE_WIFI_LOGGING     false

// HTTP logging endpoint (if WiFi enabled)
#define LOG_SERVER_URL          "http://192.168.1.100:8000/log"

// ================================
// DEBUG CONFIGURATION
// ================================

// Debug levels
#define DEBUG_LEVEL_NONE        0
#define DEBUG_LEVEL_ERROR       1
#define DEBUG_LEVEL_WARN        2
#define DEBUG_LEVEL_INFO        3
#define DEBUG_LEVEL_DEBUG       4

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL             DEBUG_LEVEL_INFO
#endif

// Debug macros
#if DEBUG_LEVEL >= DEBUG_LEVEL_ERROR
#define DEBUG_ERROR(x...)       Serial.printf("[ERROR] " x)
#else
#define DEBUG_ERROR(x...)
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_WARN
#define DEBUG_WARN(x...)        Serial.printf("[WARN] " x)
#else
#define DEBUG_WARN(x...)
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
#define DEBUG_INFO(x...)        Serial.printf("[INFO] " x)
#else
#define DEBUG_INFO(x...)
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
#define DEBUG_DEBUG(x...)       Serial.printf("[DEBUG] " x)
#else
#define DEBUG_DEBUG(x...)
#endif

// ================================
// ERROR HANDLING
// ================================

// Error codes
enum ErrorCode {
    ERROR_NONE = 0,
    ERROR_I2S_INIT = 1,
    ERROR_MEMORY_ALLOCATION = 2,
    ERROR_MODEL_LOAD = 3,
    ERROR_INFERENCE = 4,
    ERROR_VAD_CALIBRATION = 5,
    ERROR_AUDIO_CAPTURE = 6
};

// Watchdog timer
#define WATCHDOG_TIMEOUT_SEC    30

// ================================
// UTILITY MACROS
// ================================

#define MIN(a, b)               ((a) < (b) ? (a) : (b))
#define MAX(a, b)               ((a) > (b) ? (a) : (b))
#define CLAMP(x, min, max)      (MAX(MIN(x, max), min))

// Memory alignment for PSRAM
#define ALIGN_32(x)             (((x) + 31) & ~31)

// Performance timing
#define TIMING_START()          unsigned long start_time = micros()
#define TIMING_END(name)        DEBUG_DEBUG("%s took %lu us\n", name, micros() - start_time)

#endif // CONFIG_H