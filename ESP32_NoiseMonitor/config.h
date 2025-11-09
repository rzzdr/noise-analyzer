#ifndef CONFIG_H
#define CONFIG_H

// ═══════════════════════════════════════════════════════════
//  ESP32 AUDIO STREAMER CONFIGURATION
//  Sends raw audio data to Flask server for processing
// ═══════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────
//  WIFI CONFIGURATION
// ───────────────────────────────────────────────────────────
#define WIFI_SSID "noise"        // ⚠️ CHANGE THIS
#define WIFI_PASSWORD "12345678" // ⚠️ CHANGE THIS
#define WIFI_TIMEOUT_MS 20000    // 20 second timeout

// Server configuration for data transmission
#define SERVER_URL "http://4.240.35.54:6002/predict"      // Flask prediction endpoint
#define CALIBRATE_URL "http://4.240.35.54:6002/calibrate" // Flask calibration endpoint

// ───────────────────────────────────────────────────────────
//  DEVICE IDENTIFICATION
// ───────────────────────────────────────────────────────────
#define DEVICE_ID "ESP32_Node_01"           // Unique ID per device
#define DEVICE_LOCATION "Library_Floor1_NE" // Physical location

// ───────────────────────────────────────────────────────────
//  HW-484 ANALOG MICROPHONE CONFIGURATION
// ───────────────────────────────────────────────────────────
#define MIC_ANALOG_PIN 36 // A0 pin → GPIO36 (ADC1_CH0, VP)
                          // Wiring: A0→GPIO36, G→GND, +→3.3V
#define ADC_RESOLUTION 12 // 12-bit ADC (0-4095)
#define ADC_VREF 3.3      // Reference voltage

// ───────────────────────────────────────────────────────────
//  AUDIO PARAMETERS
// ───────────────────────────────────────────────────────────
#define SAMPLE_RATE 16000       // 16 kHz sampling (to match Flask server)
#define AUDIO_BUFFER_SIZE 16000 // 1 second of audio
#define SAMPLING_PERIOD_US 62.5 // 1000000/16000 = 62.5 microseconds

// ───────────────────────────────────────────────────────────
//  CALIBRATION SETTINGS
// ───────────────────────────────────────────────────────────
#define CALIBRATION_SAMPLES 4 // Number of 1-second samples for VAD calibration (3.5s total)

// ───────────────────────────────────────────────────────────
//  TRANSMISSION SETTINGS
// ───────────────────────────────────────────────────────────
#define SEND_INTERVAL_MS 1000    // Send every 1 second
#define ENABLE_LED_FEEDBACK true // Blink LED on transmission

// LED Pin (ESP32-CAM has built-in LED on GPIO4)
#define LED_PIN 4       // Flash LED
#define LED_BLINK_MS 50 // Blink duration

// ───────────────────────────────────────────────────────────
//  DEBUG SETTINGS
// ───────────────────────────────────────────────────────────
#define ENABLE_SERIAL_DEBUG true // Enable debug output
#define SERIAL_BAUD 115200

// ───────────────────────────────────────────────────────────
//  MEMORY ALLOCATION
// ───────────────────────────────────────────────────────────
#define USE_PSRAM true // Use external PSRAM for buffers

// ═══════════════════════════════════════════════════════════
//  LEGACY PARAMETERS (for compatibility with old header files)
//  Note: These are NOT used in the streaming architecture
// ═══════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────
//  FEATURE EXTRACTION PARAMETERS (Legacy)
// ───────────────────────────────────────────────────────────
#define FRAME_LENGTH 512 // ~32 ms @ 16 kHz
#define HOP_LENGTH 160   // 10 ms hop
#define N_FFT 512        // FFT size
#define N_MEL_BANDS 40   // Mel filter banks
#define N_FRAMES 96      // Time frames in spectrogram

// ───────────────────────────────────────────────────────────
//  MODEL PARAMETERS (Legacy)
// ───────────────────────────────────────────────────────────
#define NUM_CLASSES 4                  // Number of noise classes
#define TFLITE_ARENA_SIZE (320 * 1024) // TFLite memory arena

// Class names (legacy)
static const char *CLASS_NAMES[NUM_CLASSES] = {
    "Whispering",
    "Typing",
    "Phone_ringing",
    "Loud_talking"};

// ───────────────────────────────────────────────────────────
//  VAD PARAMETERS (Legacy)
// ───────────────────────────────────────────────────────────
#define VAD_CALIBRATION_SECONDS 3.0f // Calibration duration
#define VAD_ENERGY_MARGIN 1.5f       // Energy threshold margin
#define VAD_SPECTRAL_MARGIN 1.8f     // Spectral flux threshold margin
#define VAD_ZCR_MARGIN 1.2f          // Zero-crossing rate margin

// ───────────────────────────────────────────────────────────
//  NETWORK PARAMETERS (Legacy)
// ───────────────────────────────────────────────────────────
#define SERVER_IP "4.240.35.54"                 // Legacy server IP
#define SERVER_PORT 6002                        // Legacy server port
#define HTTP_ENDPOINT "http://4.240.35.54:6002" // Legacy endpoint

#endif // CONFIG_H
