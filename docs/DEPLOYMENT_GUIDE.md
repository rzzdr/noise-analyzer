# ESP32-CAM Noise Monitor Deployment Guide

## 📋 Overview

This guide provides complete step-by-step instructions for deploying the 4-class audio noise classifier to ESP32-CAM hardware with INMP441 I²S microphone.

**System Architecture:**
- **Phase 1:** Voice Activity Detection (VAD) - Binary silence detection
- **Phase 2:** Neural Network Classification - 4-class noise classification
- **Classes:** Whispering, Typing, Phone_ringing, Loud_talking

---

## 🛠️ Hardware Requirements

### Required Components

| Component | Specification | Quantity | Notes |
|-----------|---------------|----------|-------|
| **ESP32-CAM** | AI-Thinker or equivalent | 1 | Must have PSRAM |
| **INMP441** | I²S MEMS Microphone | 1 | 3.3V only |
| **Power Supply** | 5V/1A+ USB adapter | 1 | Stable power required |
| **USB-UART Bridge** | CP2102, CH340, FTDI | 1 | For programming |
| **Jumper Wires** | Male-to-female | 10+ | For connections |
| **Breadboard** | Half-size minimum | 1 | Optional but recommended |
| **Capacitors** | 0.1µF, 10µF | 2 each | For power decoupling |

### Optional Components
- **Series Resistors** (33Ω) - For longer I²S wires (>15cm)
- **Pull-up Resistors** (10kΩ) - If connection issues occur
- **Enclosure** - For deployment protection

---

## 🔌 Wiring Diagram

### ESP32-CAM to INMP441 Connections

```
INMP441 Pin    →    ESP32-CAM Pin    →    GPIO Function
─────────────────────────────────────────────────────────
VDD (3.3V)     →    3V3              →    Power supply
GND            →    GND              →    Ground
SCK/BCLK       →    IO14             →    I²S Bit Clock
WS/LRCLK       →    IO15             →    I²S Word Select  
SD (Data)      →    IO13             →    I²S Data Input
L/R            →    GND              →    Left channel select
```

### Programming Connections (USB-UART)

```
USB-UART       →    ESP32-CAM
─────────────────────────────
3.3V           →    3V3
GND            →    GND  
TX             →    U0RXD (GPIO3)
RX             →    U0TXD (GPIO1)
```

### Power Connection

```
Power Supply   →    ESP32-CAM
─────────────────────────────
5V+            →    5V pin
GND            →    GND
```

### Visual Wiring Diagram

```
                    ESP32-CAM
                  ┌─────────────┐
                  │    RESET    │
           3V3 ←──┤             │
           GND ←──┤             │
    (I²S WS) ←────┤ IO15    IO14│────→ (I²S SCK)
  (I²S Data) ←────┤ IO13    IO12│
                  │ IO4     IO2 │
                  │ IO16    IO1 │────→ (UART TX)
                  │ VCC     IO3 │────→ (UART RX)
        5V+ ──────┤ 5V      GND │────→ GND
                  └─────────────┘
                        │
                        │ USB-UART for programming
                        │
                  ┌─────────────┐
                  │   INMP441   │
                  │     MIC     │
                  ├─────────────┤
                  │ VDD → 3V3   │
                  │ GND → GND   │
                  │ L/R → GND   │
                  │ SCK → IO14  │
                  │ WS  → IO15  │
                  │ SD  → IO13  │
                  └─────────────┘
```

### Important Wiring Notes

⚠️ **Critical Guidelines:**
1. **Power:** Always use 5V supply to ESP32-CAM's 5V pin (not 3.3V pin)
2. **INMP441 Power:** Connect INMP441 VDD to ESP32's 3V3 output (regulated)
3. **L/R Pin:** Connect INMP441 L/R to GND for left channel
4. **Wire Length:** Keep I²S wires short (<15cm) to avoid noise
5. **Decoupling:** Add 0.1µF + 10µF capacitors near INMP441 VDD-GND

---

## 💻 Software Setup

### Step 1: Install Development Environment

#### Option A: PlatformIO (Recommended)

1. **Install VS Code:**
   ```bash
   # Download from https://code.visualstudio.com/
   ```

2. **Install PlatformIO Extension:**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search "PlatformIO IDE"
   - Click Install

3. **Clone Project:**
   ```bash
   git clone <your-repo>
   cd noise-analyzer/esp32_noise_monitor
   ```

#### Option B: Arduino IDE

1. **Install Arduino IDE 2.0+**
2. **Add ESP32 Board Package:**
   - File → Preferences
   - Additional Board Manager URLs: 
     ```
     https://dl.espressif.com/dl/package_esp32_index.json
     ```
   - Tools → Board → Boards Manager
   - Search "ESP32" and install

3. **Install Required Libraries:**
   - ArduinoFFT
   - ArduinoJson
   - TensorFlowLite_ESP32

### Step 2: Model Conversion

1. **Convert Keras Model to TensorFlow Lite:**
   ```bash
   cd noise-analyzer
   python convert_to_tflite.py --model models/best_model.h5
   ```

2. **Extract Normalization Parameters:**
   ```bash
   python extract_normalization.py --params models/model_params.npz --output esp32_deployment
   ```

3. **Copy Files to ESP32 Project:**
   ```bash
   cp esp32_deployment/model_normalization.h esp32_noise_monitor/include/
   cp models/noise_classifier_quantized.tflite esp32_noise_monitor/data/
   ```

### Step 3: Compile and Flash

#### Using PlatformIO:

1. **Open Project:**
   ```bash
   cd esp32_noise_monitor
   code .  # Opens in VS Code
   ```

2. **Build Project:**
   - Click PlatformIO icon in sidebar
   - Click "Build" (✓ symbol)

3. **Upload to ESP32:**
   - Connect ESP32-CAM via USB-UART
   - Put ESP32 in flash mode:
     - Connect GPIO0 to GND
     - Press RESET button
     - Release GPIO0
   - Click "Upload" (→ symbol)

#### Using Arduino IDE:

1. **Open main.cpp as .ino file**
2. **Select Board:** Tools → Board → ESP32 → AI Thinker ESP32-CAM
3. **Select Port:** Tools → Port → (your COM port)
4. **Upload:** Sketch → Upload

### Step 4: Verify Installation

1. **Check Serial Output:**
   ```bash
   # PlatformIO
   pio device monitor --baud 115200
   
   # Arduino IDE
   # Use Serial Monitor at 115200 baud
   ```

2. **Expected Boot Sequence:**
   ```
   ESP32-CAM Noise Monitor v1.0
   =====================================
   Initializing system components...
   ✓ Audio capture initialized
   ✓ VAD initialized  
   ✓ Feature extractor initialized
   ✓ Model loaded
   🎙️  VAD CALIBRATION PHASE
   Please ensure silence for 3.5 seconds...
   Progress: .................... ✓
   🚀 REAL-TIME CLASSIFICATION STARTED
   ```

---

## 🎯 Testing & Validation

### Phase 1: Hardware Validation

#### Test 1: Power and Boot
```bash
# Expected output
ESP32-CAM Noise Monitor v1.0
Chip: ESP32-D0WD-V3
CPU Freq: 240 MHz  
Flash: 4096 KB
PSRAM: 4096 KB
```

**Troubleshooting:**
- No output → Check power supply and UART connections
- Boot loop → Check GPIO0 during programming
- Low PSRAM → Verify ESP32-CAM model has PSRAM

#### Test 2: I²S Microphone
```bash
# Check for this output:
[INFO] I2S configured successfully
  Sample rate: 16000 Hz
  Pin config: SCK=14, WS=15, SD=13
✓ Microphone test PASSED
  Max level: 0.012345 (-38.2 dB)
  Avg level: 0.003456 (-49.2 dB)
```

**Troubleshooting:**
- "Microphone test FAILED" → Check wiring, try different pins
- No audio signal → Verify INMP441 power (3.3V), check L/R pin
- Noisy signal → Add decoupling capacitors, shorten wires

#### Test 3: VAD Calibration
```bash
# Should complete automatically:
VAD Thresholds:
  Energy: 0.003456
  Spectral Centroid: 1234.5 Hz
  Zero Crossing Rate: 0.012345
  MFCC Variance: 0.067890
```

**Troubleshooting:**
- Calibration stuck → Environment too noisy, move to quieter location
- Unrealistic thresholds → Check microphone sensitivity

### Phase 2: Classification Testing

#### Test 1: Silence Detection
- Remain quiet for 30 seconds
- Should output: `[timestamp],Silence,0.xxx,0,[heap],[psram]`

#### Test 2: Known Sound Sources

| Test Sound | Expected Classification | Confidence Target |
|------------|------------------------|-------------------|
| Whisper "hello" | Whispering | >0.6 |
| Type on keyboard | Typing | >0.7 |
| Phone ringtone | Phone_ringing | >0.6 |
| Normal conversation | Loud_talking | >0.5 |

#### Test 3: Performance Validation

**Target Performance:**
- VAD processing: <50ms
- Feature extraction: <100ms  
- Neural inference: <150ms
- Total pipeline: <300ms

**Memory Usage:**
- Free heap: >50KB
- Free PSRAM: >3MB

### Phase 3: Deployment Testing

#### Long-term Stability Test
```bash
# Run for 1+ hours continuously
python esp32_logger.py --output stability_test.csv
```

**Success Criteria:**
- No crashes or resets
- Consistent memory usage
- Stable inference times
- Reasonable classification distribution

---

## 📊 Data Collection

### Real-time Monitoring

1. **Start ESP32 Logger:**
   ```bash
   python esp32_logger.py --manual-labeling
   ```

2. **Logger Features:**
   - Auto-detects ESP32 COM port
   - Real-time console display with emojis
   - CSV logging with timestamps
   - Manual ground truth labeling
   - Confusion matrix generation

3. **Example Output:**
   ```
   [14:32:15] ⌨️  TYPING (Conf: 0.91, Time: 178ms)
   📊 Current Probabilities:
      ⌨️  Typing        0.91 |████████████████████████████████████████|
      💬 Whispering    0.05 |██████                                  |
      📞 Phone_ringing 0.02 |████                                    |
      🗣️  Loud_talking  0.01 |██                                      |
   ⚡ Stats (last 100): Avg=165ms, Min=142ms, Max=195ms
   💾 Memory: Heap=87KB, PSRAM=3.8MB
   ```

### Data Analysis

#### CSV Output Format
```csv
actual_timestamp,esp32_timestamp,class_name,confidence,inference_time_ms,vad_confidence,heap_free_kb,psram_free_kb
2025-11-06T14:32:15.123,1234567,Typing,0.91,178,0.87,87,3800
```

#### Analysis Scripts
```bash
# Generate performance report
python analyze_performance.py esp32_log_20251106_143215.csv

# Create confusion matrix (if manual labeling was used)  
python confusion_matrix.py esp32_log_20251106_143215_confusion.json
```

---

## 🚀 Deployment Strategies

### Option 1: Standalone Operation

**Configuration:**
- Power via 5V wall adapter
- Serial output to local logger
- Enclosure for protection

**Use Case:** Single-point monitoring with local data collection

### Option 2: WiFi Integration (Future Enhancement)

**Configuration:**
```cpp
#define ENABLE_WIFI_LOGGING     true
#define WIFI_SSID               "LibraryNetwork"
#define WIFI_PASSWORD           "password123"
#define LOG_SERVER_URL          "http://server.local:8000/log"
```

**Use Case:** Multiple nodes reporting to central server

### Option 3: Edge Computing Network

**Configuration:**
- Multiple ESP32-CAM nodes
- Centralized data aggregation
- Real-time dashboard

**Use Case:** Building-wide noise monitoring system

---

## 🔧 Maintenance & Updates

### Regular Maintenance

#### Daily Checks
- [ ] System uptime and stability
- [ ] Memory usage trends
- [ ] Classification accuracy spot checks

#### Weekly Maintenance  
- [ ] Download and analyze CSV logs
- [ ] Check for firmware updates
- [ ] Verify microphone functionality

#### Monthly Reviews
- [ ] Performance trend analysis
- [ ] Model accuracy assessment
- [ ] Hardware inspection

### Firmware Updates

#### Over-the-Air (OTA) Updates
```cpp
#include <ArduinoOTA.h>

void setup_ota() {
    ArduinoOTA.setHostname("esp32-noise-monitor");
    ArduinoOTA.setPassword("update123");
    ArduinoOTA.begin();
}
```

#### Manual Updates
1. Connect USB-UART programmer
2. Put ESP32 in flash mode
3. Upload new firmware via PlatformIO/Arduino IDE

### Model Updates

#### Retraining Process
1. Collect misclassified samples from deployment
2. Retrain model with augmented dataset
3. Convert to TensorFlow Lite
4. Validate quantized model accuracy
5. Deploy via firmware update

---

## 📈 Performance Optimization

### Memory Optimization

#### PSRAM Usage
- Store large buffers (audio, features) in PSRAM
- Keep TensorFlow Lite arena in DRAM for speed
- Monitor fragmentation

#### Heap Management
```cpp
// Regular memory checks
void check_memory() {
    size_t heap_free = ESP.getFreeHeap();
    size_t psram_free = ESP.getFreePsram();
    
    if (heap_free < 50000) {  // 50KB threshold
        DEBUG_WARN("Low heap memory: %d bytes\n", heap_free);
    }
}
```

### CPU Optimization

#### Task Pinning
```cpp
// Pin audio capture to Core 0
xTaskCreatePinnedToCore(audio_task, "audio", 8192, NULL, 5, NULL, 0);

// Pin inference to Core 1  
xTaskCreatePinnedToCore(inference_task, "inference", 8192, NULL, 4, NULL, 1);
```

#### Frequency Scaling
```cpp
// Boost CPU during inference
setCpuFrequencyMhz(240);  // Max performance

// Reduce during idle
setCpuFrequencyMhz(80);   // Power saving
```

---

## ❗ Troubleshooting

### Common Issues

#### Issue: "Model inference failed"
**Symptoms:** Classification errors, "inference failed" messages
**Solutions:**
1. Check TensorFlow Lite arena size (increase if needed)
2. Verify model file integrity
3. Ensure normalization parameters are correct
4. Check input feature dimensions

#### Issue: "Audio capture failed" 
**Symptoms:** No audio input, microphone test fails
**Solutions:**
1. Verify I²S pin connections (especially clock pins)
2. Check INMP441 power supply (must be 3.3V)
3. Ensure L/R pin is connected to GND
4. Try different GPIO pins if conflicts exist

#### Issue: "VAD calibration stuck"
**Symptoms:** Calibration never completes, unrealistic thresholds
**Solutions:**
1. Move to quieter environment for calibration
2. Check microphone sensitivity (may need gain adjustment)
3. Verify audio signal levels during calibration
4. Reset and try different calibration duration

#### Issue: High memory usage
**Symptoms:** Low heap warnings, system crashes
**Solutions:**
1. Reduce buffer sizes if possible
2. Move more data to PSRAM
3. Check for memory leaks in custom code
4. Increase heap size in partition table

#### Issue: Poor classification accuracy
**Symptoms:** Wrong classifications, low confidence scores
**Solutions:**
1. Recalibrate VAD thresholds
2. Check feature extraction normalization
3. Verify model quantization didn't degrade performance
4. Collect additional training data for problematic classes

### Debug Techniques

#### Serial Debug Output
```cpp
#define DEBUG_LEVEL DEBUG_LEVEL_DEBUG  // Enable verbose logging
```

#### Memory Monitoring
```cpp
void print_memory_usage() {
    Serial.printf("Heap: %d, PSRAM: %d, Stack: %d\n",
                  ESP.getFreeHeap(), 
                  ESP.getFreePsram(),
                  uxTaskGetStackHighWaterMark(NULL));
}
```

#### Performance Profiling
```cpp
#define TIMING_START() unsigned long start = micros()
#define TIMING_END(name) Serial.printf("%s: %lu us\n", name, micros() - start)
```

---

## 🆘 Support Resources

### Documentation
- [ESP32-CAM Technical Reference](https://docs.espressif.com/)
- [INMP441 Datasheet](https://invensense.tdk.com/products/digital/inmp441/)
- [TensorFlow Lite Micro Guide](https://www.tensorflow.org/lite/microcontrollers)

### Community Support
- [ESP32 Arduino Forum](https://www.esp32.com/viewforum.php?f=19)
- [PlatformIO Community](https://community.platformio.org/)
- [TensorFlow Lite Issues](https://github.com/tensorflow/tensorflow/issues)

### Project Repository
- Code: `https://github.com/your-repo/noise-analyzer`
- Issues: Report bugs and request features
- Wiki: Additional examples and tutorials

---

## 📄 Appendix

### Pin Reference

| ESP32-CAM GPIO | Function | Used For | Notes |
|----------------|----------|----------|-------|
| GPIO0 | Boot mode | Programming | Pull to GND for flash mode |
| GPIO1 | U0TXD | UART TX | Serial output |
| GPIO3 | U0RXD | UART RX | Serial input |
| GPIO13 | HSPI MOSI | I²S Data | Can conflict with SD card |
| GPIO14 | HSPI SCK | I²S Clock | Can conflict with SD card |
| GPIO15 | HSPI SS | I²S WS | Can conflict with SD card |

### Memory Layout

| Region | Size | Usage |
|--------|------|-------|
| Flash | 4MB | Firmware + model |
| DRAM | ~300KB | Program execution + TFLite arena |
| PSRAM | 4MB | Audio buffers + features |

### Performance Targets

| Metric | Target | Typical | Maximum |
|--------|--------|---------|---------|
| VAD Time | <50ms | 30ms | 80ms |
| Feature Time | <100ms | 70ms | 150ms |
| Inference Time | <150ms | 120ms | 200ms |
| Total Pipeline | <300ms | 220ms | 430ms |
| Free Heap | >50KB | 80KB | N/A |
| Free PSRAM | >3MB | 3.5MB | N/A |

---

*Last updated: November 2025*
*Version: 1.0*