# ESP32-CAM Noise Monitor Troubleshooting Guide

## 🚨 Common Issues & Solutions

### 🔋 Power & Boot Issues

#### Issue: ESP32-CAM won't boot / No serial output
**Symptoms:**
- No output on serial monitor
- LED doesn't flash during boot
- No response to programming attempts

**Diagnostic Steps:**
1. Check power supply voltage and current capacity
2. Verify all ground connections
3. Test with known-good USB-UART adapter

**Solutions:**
```bash
# Check power:
- Use 5V supply (NOT 3.3V) to ESP32-CAM 5V pin
- Minimum 1A current capacity required
- Measure actual voltage at ESP32-CAM pins

# Check UART:
- TX/RX lines must be crossed: ESP32 TX → UART RX, ESP32 RX → UART TX
- Baud rate: 115200 (default)
- Try different USB-UART adapter if available

# Check GPIO0:
- GPIO0 must be floating (not connected) for normal boot
- Connect to GND only during programming
```

#### Issue: Boot loop / Continuous resets
**Symptoms:**
- ESP32 starts booting but resets repeatedly
- Brownout detector messages
- Watchdog timeout errors

**Solutions:**
```cpp
// 1. Check power supply stability
// Measure voltage under load - should stay >4.8V

// 2. Disable brownout detector temporarily (for debugging only)
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout
    // ... rest of setup
}

// 3. Increase watchdog timeout
esp_task_wdt_init(60, true);  // 60 second timeout
```

#### Issue: Programming fails / Can't upload firmware
**Symptoms:**
- "Failed to connect to ESP32" error
- Upload times out
- Verification fails

**Solutions:**
```bash
# Enter programming mode correctly:
1. Connect GPIO0 to GND
2. Press and release RESET button
3. Release GPIO0 connection
4. Start upload within 10 seconds

# Check connections:
- ESP32 IO3 ← UART TX
- ESP32 IO1 → UART RX  
- Common GND connection
- Stable 3.3V or 5V power

# Alternative method:
- Hold GPIO0 to GND during entire upload process
- Some boards need GPIO0 held throughout programming
```

---

### 🎤 Audio Capture Issues

#### Issue: "I2S driver install failed"
**Symptoms:**
- Error during I2S initialization
- Cannot start audio capture
- DMA allocation failures

**Diagnostic Steps:**
```cpp
// Check available pins
Serial.println("Testing I2S pin configuration...");

// Try alternative pin configuration
#define I2S_WS_PIN      4    // Instead of 15
#define I2S_SCK_PIN     2    // Instead of 14  
#define I2S_SD_PIN      12   // Instead of 13
```

**Solutions:**
```cpp
// 1. Check for pin conflicts
// Avoid pins used by camera or SD card

// 2. Verify DMA buffer settings
i2s_config_t i2s_config = {
    .dma_buf_count = 2,      // Reduce if memory issues
    .dma_buf_len = 256,      // Reduce if memory issues
    // ... other settings
};

// 3. Alternative I2S port
#define I2S_PORT        I2S_NUM_1  // Try port 1 instead of 0
```

#### Issue: "Microphone test FAILED - no signal detected"
**Symptoms:**
- Audio input always zero or very low
- VAD calibration never completes
- No response to sound sources

**Diagnostic Steps:**
```cpp
// Raw I2S data test
void test_raw_i2s() {
    int32_t raw_samples[256];
    size_t bytes_read;
    
    i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, 1000);
    
    for (int i = 0; i < 10; i++) {
        Serial.printf("Raw[%d]: %d (0x%08X)\n", i, raw_samples[i], raw_samples[i]);
    }
}
```

**Solutions:**
```cpp
// 1. Check INMP441 wiring
/*
VDD → ESP32 3V3 (NOT 5V!)
GND → ESP32 GND  
L/R → ESP32 GND (for left channel)
SCK → ESP32 IO14
WS  → ESP32 IO15
SD  → ESP32 IO13
*/

// 2. Verify power supply
// INMP441 requires clean 3.3V, measure at microphone pins

// 3. Check I2S format
i2s_config_t i2s_config = {
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,  // INMP441 is mono
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,  // Standard I2S
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,  // INMP441 outputs 24-bit in 32-bit frame
};

// 4. Add decoupling capacitors
// 0.1µF ceramic + 10µF electrolytic near INMP441 VDD-GND
```

#### Issue: Noisy/distorted audio
**Symptoms:**
- High background noise
- Clipped or distorted signals
- Inconsistent audio levels

**Solutions:**
```cpp
// 1. Improve power supply filtering
// Add capacitors: 0.1µF ceramic + 100µF electrolytic

// 2. Shorten I2S wires
// Keep all I2S connections <15cm
// Use twisted pairs for longer runs

// 3. Add series resistors (for long wires)
// 33Ω resistors in series with SCK, WS, SD lines

// 4. Adjust sample format
i2s_config_t i2s_config = {
    .use_apll = true,        // Use Audio PLL for better clock stability
    .fixed_mclk = 0,         // Let driver calculate MCLK
};

// 5. Software filtering
void apply_high_pass_filter(float* audio, int length) {
    static float prev_input = 0, prev_output = 0;
    const float alpha = 0.95f;  // High-pass cutoff
    
    for (int i = 0; i < length; i++) {
        float output = alpha * (prev_output + audio[i] - prev_input);
        prev_input = audio[i];
        prev_output = output;
        audio[i] = output;
    }
}
```

---

### 🧠 VAD & Classification Issues

#### Issue: VAD calibration never completes
**Symptoms:**
- Calibration stuck at partial progress
- Unrealistic threshold values
- "Calibration timeout" errors

**Diagnostic Steps:**
```cpp
// Check VAD feature extraction
void debug_vad_features(float* audio, int length) {
    float energy = extract_energy(audio, length);
    float spectral_centroid = extract_spectral_centroid(audio, length);
    float zcr = extract_zero_crossing_rate(audio, length);
    
    Serial.printf("VAD Features: E=%.6f, SC=%.1f, ZCR=%.4f\n", 
                  energy, spectral_centroid, zcr);
}
```

**Solutions:**
```cpp
// 1. Ensure quiet environment for calibration
// Move to quieter location, avoid air conditioning, fans

// 2. Adjust calibration parameters
#define VAD_CALIBRATION_SAMPLES    20   // Reduce if environment is challenging
#define VAD_ENERGY_MARGIN         2.0f  // Increase for noisier environments

// 3. Manual threshold setting (if automatic fails)
void set_manual_vad_thresholds() {
    energy_threshold = 0.005f;           // Typical office environment
    spectral_centroid_threshold = 800.0f; 
    zero_crossing_threshold = 0.01f;
    mfcc_variance_threshold = 0.1f;
}

// 4. Validate audio input during calibration
if (audio_level < 1e-8f) {
    Serial.println("WARNING: No audio signal during calibration");
}
```

#### Issue: Poor classification accuracy
**Symptoms:**
- All sounds classified as one class
- Very low confidence scores
- Random/inconsistent classifications

**Diagnostic Steps:**
```cpp
// Check feature extraction
void debug_features(float* features) {
    Serial.println("Feature stats:");
    
    float min_val = features[0], max_val = features[0];
    float mean = 0;
    
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        min_val = MIN(min_val, features[i]);
        max_val = MAX(max_val, features[i]);
        mean += features[i];
    }
    mean /= MODEL_INPUT_SIZE;
    
    Serial.printf("Min: %.3f, Max: %.3f, Mean: %.3f\n", min_val, max_val, mean);
    
    // Check for NaN or infinite values
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        if (!isfinite(features[i])) {
            Serial.printf("Invalid feature[%d]: %.3f\n", i, features[i]);
        }
    }
}
```

**Solutions:**
```cpp
// 1. Verify normalization parameters
// Ensure model_normalization.h matches training data

// 2. Check model quantization
// Compare float vs quantized model outputs

// 3. Validate input preprocessing
void validate_preprocessing(float* audio) {
    // Check audio amplitude range
    float min_amp = audio[0], max_amp = audio[0];
    for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
        min_amp = MIN(min_amp, audio[i]);
        max_amp = MAX(max_amp, audio[i]);
    }
    
    if (max_amp - min_amp < 1e-6f) {
        Serial.println("WARNING: Audio signal too quiet for classification");
    }
    if (max_amp > 0.9f || min_amp < -0.9f) {
        Serial.println("WARNING: Audio signal may be clipped");
    }
}

// 4. Retrain model with ESP32-specific data
// Collect audio samples directly from ESP32 for retraining
```

---

### 💾 Memory Issues

#### Issue: "Failed to allocate memory" / Memory allocation errors
**Symptoms:**
- Crashes during initialization
- "ps_malloc failed" errors
- System becomes unresponsive

**Diagnostic Steps:**
```cpp
void check_memory_status() {
    Serial.printf("=== Memory Status ===\n");
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
    Serial.printf("Heap size: %d bytes\n", ESP.getHeapSize());
    Serial.printf("PSRAM size: %d bytes\n", ESP.getPsramSize());
    Serial.printf("Min free heap: %d bytes\n", ESP.getMinFreeHeap());
    
    // Check for fragmentation
    Serial.printf("Largest free block: %d bytes\n", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
}
```

**Solutions:**
```cpp
// 1. Use PSRAM for large buffers
float* audio_buffer = (float*)ps_malloc(AUDIO_BUFFER_SIZE);
if (!audio_buffer) {
    Serial.println("Failed to allocate audio buffer in PSRAM");
    // Try regular malloc as fallback
    audio_buffer = (float*)malloc(AUDIO_BUFFER_SIZE);
}

// 2. Reduce buffer sizes if memory constrained
#define AUDIO_BUFFER_SIZE       (SAMPLES_PER_WINDOW * 2 * sizeof(float))  // Reduce multiplier
#define TFLITE_ARENA_SIZE       (150 * 1024)  // Reduce from 200KB

// 3. Free unused memory early
if (calibration_complete) {
    free(calibration_buffer);  // Free calibration data after use
    calibration_buffer = nullptr;
}

// 4. Enable memory defragmentation
heap_caps_malloc_extmem_enable(16);  // Use external memory for >16 byte allocations
```

#### Issue: Memory leaks / Gradually increasing memory usage
**Symptoms:**
- Available memory decreases over time
- System crashes after extended operation
- Performance degrades gradually

**Solutions:**
```cpp
// 1. Check for memory leaks
void* test_malloc = malloc(1000);
Serial.printf("Allocated test memory at: %p\n", test_malloc);
free(test_malloc);

// Add leak detection
#define DEBUG_MALLOC 1
#if DEBUG_MALLOC
void* debug_malloc(size_t size, const char* file, int line) {
    void* ptr = malloc(size);
    Serial.printf("MALLOC: %p (%d bytes) at %s:%d\n", ptr, size, file, line);
    return ptr;
}
#define malloc(size) debug_malloc(size, __FILE__, __LINE__)
#endif

// 2. Proper cleanup in destructors
AudioCapture::~AudioCapture() {
    stop_capture();          // Stop threads first
    if (audio_buffer) {
        free(audio_buffer);
        audio_buffer = nullptr;
    }
    // Clean up all allocated resources
}

// 3. Monitor memory usage
static unsigned long last_memory_check = 0;
if (millis() - last_memory_check > 10000) {  // Every 10 seconds
    check_memory_status();
    last_memory_check = millis();
}
```

---

### ⚡ Performance Issues

#### Issue: Inference time too slow (>300ms)
**Symptoms:**
- Classification takes too long
- Audio buffer overruns
- Real-time performance degraded

**Solutions:**
```cpp
// 1. CPU frequency optimization
void optimize_cpu_frequency() {
    setCpuFrequencyMhz(240);  // Maximum performance during inference
    
    // Profile different frequencies
    for (int freq : {80, 160, 240}) {
        setCpuFrequencyMhz(freq);
        
        unsigned long start = micros();
        run_inference();
        unsigned long time_us = micros() - start;
        
        Serial.printf("Freq %dMHz: %lu us\n", freq, time_us);
    }
}

// 2. Core affinity optimization
void pin_tasks_to_cores() {
    // Pin audio capture to Core 0 (protocol CPU)
    xTaskCreatePinnedToCore(audio_task, "audio", 8192, NULL, 5, NULL, 0);
    
    // Pin inference to Core 1 (application CPU)
    xTaskCreatePinnedToCore(inference_task, "inference", 8192, NULL, 4, NULL, 1);
}

// 3. Reduce model complexity if needed
// Use smaller TensorFlow Lite arena
#define TFLITE_ARENA_SIZE       (100 * 1024)  // Reduce from 200KB

// 4. Skip unnecessary processing
if (vad_confidence < 0.3f) {
    // Skip classification for very low VAD confidence
    continue;
}
```

#### Issue: Audio buffer overruns / Dropouts
**Symptoms:**
- "Buffer overrun" warnings
- Choppy or intermittent audio
- Missed classifications

**Solutions:**
```cpp
// 1. Increase buffer sizes
#define DMA_BUF_COUNT           8    // Increase from 4
#define DMA_BUF_LEN             1024 // Increase from 512

// 2. Higher priority for audio task
xTaskCreatePinnedToCore(audio_capture_task, "audio", 8192, NULL, 
                       configMAX_PRIORITIES - 1,  // Highest priority
                       NULL, 0);

// 3. Reduce other processing during audio capture
void yield_for_audio() {
    if (audio_capture_active) {
        vTaskDelay(1);  // Yield CPU time to audio task
    }
}

// 4. Optimize I2S DMA configuration
i2s_config_t i2s_config = {
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,  // High interrupt priority
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = true,  // Use Audio PLL for stable timing
};
```

---

### 🌐 Communication Issues

#### Issue: Serial communication problems
**Symptoms:**
- Garbled text in serial monitor
- Missing characters or lines
- Communication drops out

**Solutions:**
```cpp
// 1. Check baud rate matching
#define SERIAL_BAUD_RATE        115200  // Must match logger settings

// 2. Add flow control if needed
Serial.setTxTimeoutMs(1000);

// 3. Ensure proper line endings
Serial.println("Message");  // Adds proper line ending

// 4. Handle buffer overflows
if (Serial.availableForWrite() < 100) {
    Serial.flush();  // Wait for transmission to complete
}

// 5. Alternative debugging via LED patterns
void debug_led_pattern(int error_code) {
    for (int i = 0; i < error_code; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(200);
        digitalWrite(LED_PIN, LOW);
        delay(200);
    }
}
```

#### Issue: Python logger can't connect to ESP32
**Symptoms:**
- "Failed to connect" error in Python script
- Port not found or access denied
- No data received

**Solutions:**
```bash
# 1. Check port permissions (Linux/Mac)
sudo chmod 666 /dev/ttyUSB0
# or add user to dialout group:
sudo usermod -a -G dialout $USER

# 2. Find correct port
python -c "import serial.tools.list_ports; [print(p.device, p.description) for p in serial.tools.list_ports.comports()]"

# 3. Check if port is already in use
lsof /dev/ttyUSB0  # Linux/Mac
# Close other applications using the port

# 4. Test with simple serial tool first
screen /dev/ttyUSB0 115200  # Linux/Mac
# or use PuTTY on Windows

# 5. Update Python logger port detection
# Modify esp32_logger.py to try multiple ports automatically
```

---

### 🔧 Development & Debugging Issues

#### Issue: Can't see debug messages
**Symptoms:**
- No debug output despite DEBUG_LEVEL setting
- Missing detailed logging
- Hard to trace execution flow

**Solutions:**
```cpp
// 1. Verify debug level setting
#define DEBUG_LEVEL             DEBUG_LEVEL_DEBUG  // Maximum verbosity

// 2. Check serial initialization timing
void setup() {
    Serial.begin(115200);
    delay(2000);  // Wait for serial connection to stabilize
    
    DEBUG_INFO("Debug system initialized\n");
    DEBUG_INFO("Free heap: %d bytes\n", ESP.getFreeHeap());
}

// 3. Add custom debug macros with timestamps
#define DEBUG_TIMESTAMP(level, format, ...) \
    Serial.printf("[%8lu] " level " " format, millis(), ##__VA_ARGS__)

#define DEBUG_ERROR_TS(format, ...)   DEBUG_TIMESTAMP("ERROR", format, ##__VA_ARGS__)
#define DEBUG_INFO_TS(format, ...)    DEBUG_TIMESTAMP("INFO ", format, ##__VA_ARGS__)

// 4. Visual debugging with LED
void debug_blink(int count, int delay_ms = 200) {
    for (int i = 0; i < count; i++) {
        digitalWrite(LED_BUILTIN, HIGH);
        delay(delay_ms);
        digitalWrite(LED_BUILTIN, LOW);
        delay(delay_ms);
    }
}
```

#### Issue: Compilation errors / Missing dependencies
**Symptoms:**
- "No such file or directory" errors
- Undefined reference errors
- Library version conflicts

**Solutions:**
```bash
# 1. Update PlatformIO libraries
pio lib update

# 2. Clear build cache  
pio run --target clean

# 3. Reinstall problematic libraries
pio lib uninstall "ArduinoFFT"
pio lib install "kosme/arduinoFFT@^1.6"

# 4. Check library compatibility
pio lib show "kosme/arduinoFFT"

# 5. Manual library installation if needed
# Download library and place in lib/ folder

# 6. Verify platformio.ini configuration
[env:esp32cam]
lib_deps = 
    kosme/arduinoFFT@^1.6.0
    bblanchon/ArduinoJson@^6.21.3
```

---

## 🔍 Diagnostic Tools

### Built-in Diagnostics

```cpp
// Add to main.cpp for comprehensive system diagnostics
void run_system_diagnostics() {
    Serial.println("\n=== SYSTEM DIAGNOSTICS ===");
    
    // Hardware info
    Serial.printf("Chip: %s\n", ESP.getChipModel());
    Serial.printf("Revision: %d\n", ESP.getChipRevision());
    Serial.printf("CPU Freq: %d MHz\n", ESP.getCpuFreqMHz());
    Serial.printf("Flash: %d KB\n", ESP.getFlashChipSize() / 1024);
    Serial.printf("PSRAM: %d KB\n", ESP.getPsramSize() / 1024);
    
    // Memory status
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
    Serial.printf("Min free heap: %d bytes\n", ESP.getMinFreeHeap());
    
    // GPIO status
    Serial.println("GPIO Status:");
    Serial.printf("  IO13: %d\n", digitalRead(13));
    Serial.printf("  IO14: %d\n", digitalRead(14));
    Serial.printf("  IO15: %d\n", digitalRead(15));
    
    // I2S test
    Serial.println("I2S Test:");
    bool i2s_ok = test_i2s_functionality();
    Serial.printf("  Status: %s\n", i2s_ok ? "OK" : "FAILED");
    
    // Audio test
    Serial.println("Audio Test:");
    bool audio_ok = test_microphone_connectivity();
    Serial.printf("  Status: %s\n", audio_ok ? "OK" : "FAILED");
    
    Serial.println("=========================\n");
}
```

### External Tools

```bash
# 1. ESP32 Flash Tool (for low-level debugging)
esptool.py --port COM3 flash_id
esptool.py --port COM3 read_flash 0x1000 0x1000 flash_dump.bin

# 2. Logic analyzer (for I2S signal debugging)
# Connect to SCK, WS, SD pins to verify timing

# 3. Oscilloscope (for power supply analysis)
# Check for voltage ripple, brownout conditions

# 4. Multimeter (for continuity and voltage testing)
# Verify all connections, measure power consumption
```

---

## 📞 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide** - covers 90% of common issues
2. **Run system diagnostics** - gather comprehensive system information
3. **Test with minimal configuration** - remove optional features
4. **Document the exact error messages** - copy/paste complete output
5. **Note your hardware configuration** - board model, power supply, wiring

### Information to Include in Support Requests

```
Hardware Configuration:
- ESP32-CAM model: [AI-Thinker/Other]
- INMP441 microphone: [Yes/No]
- Power supply: [Voltage/Current rating]
- USB-UART adapter: [Model]

Software Configuration:
- PlatformIO/Arduino IDE version: [Version]
- Firmware version: [Git commit/date]
- Library versions: [List key libraries]

Error Details:
- Exact error message: [Copy/paste]
- When error occurs: [Boot/Runtime/Programming]
- Steps to reproduce: [Detailed steps]
- Serial output: [Complete log]

Troubleshooting Attempted:
- [List what you've already tried]
```

### Support Channels

- **GitHub Issues:** Technical bugs and feature requests
- **ESP32 Forum:** Hardware and low-level issues
- **PlatformIO Community:** Build and library issues
- **Project Wiki:** Additional examples and tutorials

---

*Remember: Most issues are power supply, wiring, or configuration related. Double-check the basics first!*