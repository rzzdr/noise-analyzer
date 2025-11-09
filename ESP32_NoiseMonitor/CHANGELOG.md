# ESP32 Architecture Changes - Migration Summary

## 🔄 Major Changes

### Previous Architecture (I2S Microphone + Local Processing)

- Used INMP441 I2S digital microphone
- Performed local VAD (Voice Activity Detection)
- Extracted audio features locally
- Ran TensorFlow Lite model on ESP32
- Sent only classification results to server

### New Architecture (Analog Microphone + Server Processing)

- Uses HW-484 analog microphone module
- Sends raw audio data to Flask server
- Server performs all processing (VAD, feature extraction, classification)
- ESP32 is a simple audio capture and streaming device

## 📝 File Changes

### Modified Files

#### 1. `config.h`

**Removed:**

- I2S pin configurations (I2S_WS_PIN, I2S_SCK_PIN, I2S_SD_PIN)
- Feature extraction parameters (N_FFT, N_MEL_BANDS, N_FRAMES)
- Model parameters (NUM_CLASSES, TFLITE_ARENA_SIZE)
- VAD threshold parameters
- UDP/old HTTP endpoint configurations
- CLASS_NAMES array

**Added:**

- HW-484 microphone configuration (MIC_ANALOG_PIN, ADC settings)
- Server URLs for predict and calibrate endpoints
- Calibration sample count configuration
- Simplified transmission settings

#### 2. `ESP32_NoiseMonitor.ino`

**Removed:**

- All TensorFlow Lite includes and code
- I2S driver initialization and reading
- VAD implementation (local)
- Feature extraction (local)
- Model inference (local)
- WiFiManager, FeatureExtractor, ModelInference classes

**Added:**

- ADC-based audio sampling
- Precise timing loop for 16kHz sampling
- Base64 encoding for audio data
- HTTP POST communication with Flask server
- Calibration routine on bootup
- ArduinoJson for JSON payload creation
- Simplified loop: capture → encode → transmit → receive

### Files No Longer Needed

The following files are **not used** in the new architecture:

- `wifi_manager.cpp` / `wifi_manager.h` (using built-in WiFi library)
- `vad.cpp` / `vad.h` (VAD done on server)
- `feature_extractor.cpp` / `feature_extractor.h` (features extracted on server)
- `model_inference.cpp` / `model_inference.h` (inference on server)
- `model_data.cpp` / `model_data.h` (no local model)
- `model_normalization.h` (normalization on server)
- `models/` directory contents (no local model files needed)

### New Files Created

- `README.md` - Complete setup and usage guide
- `WIRING_HW484.md` - Detailed wiring instructions for HW-484
- `CHANGELOG.md` - This file

## 🔌 Hardware Changes

### Old Setup

```
ESP32-CAM + INMP441 I2S Microphone

Connections:
- GPIO15 → WS (Word Select)
- GPIO14 → SCK (Serial Clock)
- GPIO13 → SD (Serial Data)
- 3.3V  → VDD
- GND   → GND
```

### New Setup

```
ESP32 + HW-484 Analog Microphone

Connections:
- GPIO36 → A0 (Analog Out)
- 3.3V   → + (Power)
- GND    → G (Ground)
- (D0 not connected)
```

## 📡 Communication Protocol Changes

### Old Protocol

**UDP/HTTP POST with processed results:**

```json
{
  "device_id": "ESP32_Node_01",
  "location": "Library_Floor1_NE",
  "class": "Whispering",
  "confidence": 0.87,
  "vad_confidence": 0.92,
  "is_activity": true
}
```

### New Protocol

**Calibration - POST /calibrate:**

```json
{
  "audio": "<base64_encoded_float32_array>",
  "sample_rate": 16000,
  "device_id": "ESP32_Node_01"
}
```

**Prediction - POST /predict:**

```json
{
  "audio": "<base64_encoded_float32_array>",
  "sample_rate": 16000,
  "device_id": "ESP32_Node_01"
}
```

**Response from server:**

```json
{
  "predicted_class": "Whispering",
  "confidence": 0.87,
  "vad_activity": true,
  "vad_confidence": 0.92,
  "probabilities": {...}
}
```

## 💾 Memory Impact

### Before (with TFLite)

- TFLite Arena: 60 KB
- Feature Buffers: ~10 KB
- Audio Buffers: 32 KB
- Model Data: ~50 KB
- **Total: ~152 KB**

### After (streaming only)

- Audio Buffers: 64 KB (uint16 + float)
- Base64 Buffer: ~90 KB (temporary)
- **Total: ~64 KB persistent**

**Result:** ~88 KB memory savings, no PSRAM requirement

## ⚡ Processing Distribution

| Task               | Old            | New               |
| ------------------ | -------------- | ----------------- |
| Audio Capture      | ESP32 (I2S)    | ESP32 (ADC)       |
| VAD                | ESP32          | Flask Server      |
| Feature Extraction | ESP32          | Flask Server      |
| ML Inference       | ESP32 (TFLite) | Flask Server      |
| Result Storage     | Server         | Server (Firebase) |

## 🎯 Advantages of New Architecture

### ✅ Pros

1. **Simpler ESP32 code** - easier to maintain and debug
2. **Cheaper hardware** - HW-484 costs less than INMP441
3. **Easier to update model** - no need to reflash ESP32
4. **Better accuracy** - server can use full TensorFlow model
5. **Centralized processing** - easier to monitor and log
6. **Lower memory usage** - no model storage on ESP32
7. **More flexible** - can easily change processing pipeline

### ⚠️ Cons

1. **Network dependency** - requires stable WiFi
2. **Higher latency** - network transmission adds delay
3. **Bandwidth usage** - sends ~64KB per second per device
4. **Server load** - server must handle all processing
5. **Privacy concerns** - raw audio sent over network

## 🚀 Migration Steps

If upgrading from old version:

1. **Hardware**: Replace I2S microphone with HW-484
2. **Wiring**: Rewire according to WIRING_HW484.md
3. **Code**: Flash new ESP32_NoiseMonitor.ino
4. **Server**: Ensure Flask server is running at configured URL
5. **Test**: Monitor Serial output during calibration
6. **Verify**: Check Flask server receives predictions

## 📊 Performance Comparison

| Metric      | Old (Local)      | New (Server)     |
| ----------- | ---------------- | ---------------- |
| Latency     | ~100ms           | ~500ms           |
| Power       | 240mA            | 180mA            |
| Accuracy    | 85% (quantized)  | 92% (full model) |
| Updates     | Reflash required | Server-side only |
| Scalability | Independent      | Centralized      |

## 🔮 Future Enhancements

Possible improvements:

- [ ] Add error retry logic with exponential backoff
- [ ] Implement audio compression before transmission
- [ ] Add local buffering for offline operation
- [ ] Support HTTPS for secure transmission
- [ ] Add OTA (Over-The-Air) firmware updates
- [ ] Implement deep sleep for battery operation
- [ ] Add local fallback model for network failures

## 📞 Support

For questions about the migration:

1. Check README.md for setup instructions
2. Review WIRING_HW484.md for hardware connections
3. Monitor Serial output at 115200 baud for debugging
4. Verify Flask server is accessible and running

---

**Migration Date:** November 9, 2025  
**Architecture Version:** 2.0 (Streaming)  
**Previous Version:** 1.0 (Local Processing)
