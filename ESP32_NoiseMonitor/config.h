#ifndef CONFIG_H
#define CONFIG_H

// ═══════════════════════════════════════════════════════════
//  ESP32-CAM STANDALONE NOISE MONITOR CONFIGURATION
// ═══════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────
//  WIFI CONFIGURATION
// ───────────────────────────────────────────────────────────
#define WIFI_SSID       "Rudr's Room"       // ⚠️ CHANGE THIS
#define WIFI_PASSWORD   "741085246-+"   // ⚠️ CHANGE THIS
#define WIFI_TIMEOUT_MS 20000                // 20 second timeout

// Server configuration for data transmission
#define SERVER_IP       "192.168.1.100"      // ⚠️ Your laptop IP
#define SERVER_PORT     8888                 // UDP port

// Alternative: Use HTTP POST instead of UDP
#define USE_HTTP_POST   true                // Set true for HTTP
#define HTTP_ENDPOINT   "https://noise2.futurixai.com/data"

// ───────────────────────────────────────────────────────────
//  DEVICE IDENTIFICATION
// ───────────────────────────────────────────────────────────
#define DEVICE_ID       "ESP32_Node_01"      // Unique ID per device
#define DEVICE_LOCATION "Library_Floor1_NE"  // Physical location

// ───────────────────────────────────────────────────────────
//  I²S MICROPHONE PINS (INMP441)
// ───────────────────────────────────────────────────────────
#define I2S_WS_PIN      15    // Word Select (LRCLK) → GPIO15
#define I2S_SCK_PIN     14    // Serial Clock (BCLK) → GPIO14
#define I2S_SD_PIN      13    // Serial Data (DOUT)  → GPIO13
#define I2S_PORT        I2S_NUM_0

// ───────────────────────────────────────────────────────────
//  AUDIO PARAMETERS
// ───────────────────────────────────────────────────────────
#define SAMPLE_RATE     16000      // 16 kHz sampling
#define AUDIO_BUFFER_SIZE 16000    // 1 second of audio
#define FRAME_LENGTH    400        // 25ms @ 16kHz
#define HOP_LENGTH      160        // 10ms @ 16kHz
#define N_FFT           512        // FFT size
#define N_MEL_BANDS     32         // Mel filter banks (reduced from 40)
#define N_FRAMES        80         // Time frames in spectrogram (reduced from 100)

// ───────────────────────────────────────────────────────────
//  MODEL PARAMETERS
// ───────────────────────────────────────────────────────────
#define NUM_CLASSES     4          // Whispering, Typing, Phone, Loud_talking
#define TFLITE_ARENA_SIZE (60 * 1024)   // 60KB for TFLite (reduced)

// ───────────────────────────────────────────────────────────
//  VAD PARAMETERS
// ───────────────────────────────────────────────────────────
#define VAD_CALIBRATION_SECONDS 2.5  // Reduced from 3.5 to save memory
#define VAD_ENERGY_MARGIN       1.3
#define VAD_SPECTRAL_MARGIN     1.2
#define VAD_ZCR_MARGIN          1.3

// ───────────────────────────────────────────────────────────
//  TRANSMISSION SETTINGS
// ───────────────────────────────────────────────────────────
#define SEND_INTERVAL_MS    1000   // Send every prediction
#define BATCH_SIZE          1      // Send individual predictions
#define ENABLE_LED_FEEDBACK true   // Blink LED on activity

// LED Pin (ESP32-CAM has built-in LED on GPIO4)
#define LED_PIN             4      // Flash LED
#define LED_BLINK_MS        100    // Blink duration

// ───────────────────────────────────────────────────────────
//  DEBUG SETTINGS
// ───────────────────────────────────────────────────────────
#define ENABLE_SERIAL_DEBUG false  // Set false for production
#define SERIAL_BAUD         115200

// ───────────────────────────────────────────────────────────
//  MEMORY ALLOCATION
// ───────────────────────────────────────────────────────────
#define USE_PSRAM       true       // Use external PSRAM for buffers

// ───────────────────────────────────────────────────────────
//  CLASS NAMES
// ───────────────────────────────────────────────────────────
const char* const CLASS_NAMES[NUM_CLASSES] = {
  "Whispering",
  "Typing",
  "Phone_ringing",
  "Loud_talking"
};

#endif // CONFIG_H