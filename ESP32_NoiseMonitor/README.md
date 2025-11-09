# ESP32 Audio Streamer for HW-484 Microphone

This Arduino sketch captures audio from a HW-484 analog microphone module and streams it to a Flask server for real-time noise classification.

## 🎯 Overview

The ESP32 acts as an audio capture device that:

1. **Samples audio** at 16kHz using the HW-484 analog microphone
2. **Calibrates** the VAD on bootup by sending silent samples to the server
3. **Streams audio** to the Flask server every second for classification
4. **Receives predictions** back from the server

## 🔧 Hardware Requirements

- **ESP32 Development Board** (ESP32-CAM or any ESP32)
- **HW-484 Analog Microphone Module**
- **Jumper wires**
- **USB cable** for programming and power

## 📋 Wiring

See [WIRING_HW484.md](WIRING_HW484.md) for detailed wiring instructions.

**Quick Reference:**

```
HW-484 → ESP32
A0     → GPIO36 (VP)
G      → GND
+      → 3.3V
D0     → Not connected
```

## ⚙️ Configuration

Edit `config.h` to customize settings:

### WiFi Settings

```cpp
#define WIFI_SSID       "Your_WiFi_Name"
#define WIFI_PASSWORD   "Your_Password"
```

### Server Settings

```cpp
#define SERVER_URL      "http://4.240.35.54:6002/predict"
#define CALIBRATE_URL   "http://4.240.35.54:6002/calibrate"
```

### Device Identification

```cpp
#define DEVICE_ID       "ESP32_Node_01"      // Change for each device
#define DEVICE_LOCATION "Library_Floor1_NE"  // Physical location
```

## 📚 Required Libraries

Install these libraries via Arduino IDE Library Manager:

1. **WiFi** (built-in with ESP32 board support)
2. **HTTPClient** (built-in with ESP32 board support)
3. **ArduinoJson** by Benoit Blanchon (v6.x)
4. **Base64** by Densaugeo

### Installing ESP32 Board Support

1. Open Arduino IDE
2. Go to **File → Preferences**
3. Add to **Additional Board Manager URLs**:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to **Tools → Board → Boards Manager**
5. Search for "ESP32" and install "ESP32 by Espressif Systems"

## 🚀 Upload Instructions

1. **Connect ESP32** to your computer via USB
2. **Open** `ESP32_NoiseMonitor.ino` in Arduino IDE
3. **Select Board**:
   - Go to **Tools → Board → ESP32 Arduino**
   - Choose your board (e.g., "ESP32 Dev Module" or "AI Thinker ESP32-CAM")
4. **Select Port**:
   - Go to **Tools → Port**
   - Choose the COM port where ESP32 is connected
5. **Configure Upload Settings** (for ESP32-CAM):
   - Partition Scheme: **Huge APP (3MB No OTA/1MB SPIFFS)**
   - Upload Speed: **115200**
6. **Upload**: Click the Upload button (→)

### For ESP32-CAM with CH340 Programmer:

- Connect IO0 to GND before uploading
- Press RESET button on programmer
- Click Upload
- Disconnect IO0 from GND after upload
- Press RESET to run the program

## 📊 Operation Flow

### 1. Bootup Sequence

```
1. Initialize Serial (115200 baud)
2. Allocate audio buffers (PSRAM if available)
3. Connect to WiFi
4. Initialize ADC for microphone
5. CALIBRATION PHASE (stay silent!)
   - Collect 4 samples of 1 second each
   - Send to /calibrate endpoint
   - Wait for server confirmation
6. Begin normal operation
```

### 2. Normal Operation Loop

```
Every 1 second:
1. Capture 16,000 audio samples (1 second @ 16kHz)
2. Convert ADC values to normalized float [-1, 1]
3. Base64 encode audio data
4. Send POST request to /predict endpoint
5. Receive and display classification result
6. Blink LED to indicate activity
```

## 🔍 Serial Monitor Output

Set Serial Monitor to **115200 baud** to see debug output:

```
╔═══════════════════════════════════════╗
║  ESP32 Audio Streamer (HW-484)       ║
║  Analog Microphone → Flask Server    ║
╚═══════════════════════════════════════╝

Device ID: ESP32_Node_01
Location: Library_Floor1_NE
Server: http://4.240.35.54:6002/predict

🌐 Connecting to WiFi...
   SSID: Your_WiFi
✅ WiFi connected!
   IP Address: 192.168.1.xxx
   RSSI: -45 dBm

🎤 Initializing ADC for HW-484 microphone...
✅ ADC initialized successfully
   Pin: GPIO36 (ADC1_CH0)
   Resolution: 12-bit (0-4095)
   Sample Rate: 16000 Hz
   Test Reading: 2048

🎙️  CALIBRATION PHASE
┌─────────────────────────────────────┐
│  Please remain SILENT for 4 sec    │
│  Sending calibration samples...    │
└─────────────────────────────────────┘
📊 Starting calibration...
   Collecting sample 1/4...
      Progress: 25.0% - in_progress
   ✅ Sample 1 sent
   Collecting sample 2/4...
      Progress: 50.0% - in_progress
   ✅ Sample 2 sent
   ...
✅ Calibration completed successfully!

✅ All systems ready!

┌─────────────────────────────────────┐
│    AUDIO STREAMING STARTED          │
└─────────────────────────────────────┘

📊 Capturing audio...
✅ Audio sent successfully
   → Prediction: Silence (98.50%)

📊 SYSTEM STATUS REPORT
Total Transmissions:   10
Successful:            100.0%
Errors:                0.0%
Capture Time:          1001 ms
Transmit Time:         234 ms
Free Heap:             145 KB
Free PSRAM:            3072 KB
WiFi RSSI:             -45 dBm
Local IP:              192.168.1.xxx
```

## 🔧 Troubleshooting

### WiFi Connection Issues

- **Problem**: Cannot connect to WiFi
- **Solution**: Double-check SSID and password in `config.h`
- **Solution**: Ensure WiFi is 2.4GHz (ESP32 doesn't support 5GHz)

### Audio Quality Issues

- **Problem**: ADC readings always around 2048 (no variation)
- **Solution**: Check A0 connection to GPIO36
- **Solution**: Adjust sensitivity potentiometer on HW-484

### Server Communication Errors

- **Problem**: HTTP requests fail
- **Solution**: Verify server URL and port
- **Solution**: Check if Flask server is running
- **Solution**: Ensure ESP32 and server are on same network (or server has public IP)

### Calibration Fails

- **Problem**: Calibration samples fail to send
- **Solution**: Ensure environment is quiet during calibration
- **Solution**: Check server /calibrate endpoint is working
- **Solution**: Try resetting the ESP32

### Memory Issues

- **Problem**: Crashes or heap errors
- **Solution**: Use ESP32 with PSRAM
- **Solution**: Reduce AUDIO_BUFFER_SIZE if needed

## 📡 API Communication

### Calibration Endpoint

**POST** `/calibrate`

```json
{
  "audio": "base64_encoded_float32_array",
  "sample_rate": 16000,
  "device_id": "ESP32_Node_01"
}
```

**Response**:

```json
{
  "status": "in_progress",
  "progress": 25.0,
  "message": "Calibration 25.0% complete"
}
```

### Prediction Endpoint

**POST** `/predict`

```json
{
  "audio": "base64_encoded_float32_array",
  "sample_rate": 16000,
  "device_id": "ESP32_Node_01"
}
```

**Response**:

```json
{
  "predicted_class": "Whispering",
  "confidence": 0.87,
  "vad_activity": true,
  "vad_confidence": 0.92,
  "probabilities": {
    "Silence": 0.02,
    "Whispering": 0.87,
    "Typing": 0.05,
    "Phone_ringing": 0.01,
    "Loud_talking": 0.05
  }
}
```

## 📈 Performance

- **Sample Rate**: 16,000 Hz
- **Sample Period**: 1 second
- **Capture Time**: ~1000ms
- **Transmission Time**: 200-500ms (depends on WiFi)
- **Total Cycle**: ~1.5 seconds per prediction

## 🔋 Power Consumption

- **Active (WiFi + Sampling)**: ~160-240 mA
- **Idle**: ~80 mA
- **Deep Sleep** (not implemented): ~10 μA

## 📝 License

This project is part of the noise-analyzer system.

## 🤝 Support

For issues and questions, refer to the main project documentation.
