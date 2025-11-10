# ESP32-CAM Setup Guide for Audio Monitoring

## Hardware Requirements

### Components Needed:
1. **ESP32-CAM Module** (with ESP32-S chip)
2. **HW-484 Analog Microphone Module**
3. **FTDI USB-to-Serial Adapter** (3.3V) for programming
4. **Jumper wires** for connections
5. **Breadboard** (optional, for prototyping)
6. **Stable 3.3V Power Supply** (600mA minimum)

### Tools Required:
- Arduino IDE (with ESP32 board package)
- Serial monitor software
- Multimeter (for voltage verification)

## Software Setup

### 1. Arduino IDE Setup
```bash
1. Install Arduino IDE (latest version)
2. Add ESP32 board package:
   - File → Preferences → Additional Board Manager URLs
   - Add: https://dl.espressif.com/dl/package_esp32_index.json
3. Install ESP32 boards:
   - Tools → Board → Boards Manager
   - Search "ESP32" and install "ESP32 by Espressif Systems"
4. Select board: "AI Thinker ESP32-CAM"
```

### 2. Required Libraries
Install these libraries via Arduino Library Manager:
- **WiFi** (built-in with ESP32)
- **HTTPClient** (built-in with ESP32)
- **ArduinoJson** (by Benoit Blanchon)
- **base64** (by Densaugeo)

### 3. Board Configuration
```
Board: "AI Thinker ESP32-CAM"
Upload Speed: "115200"
CPU Frequency: "240MHz (WiFi/BT)"
Flash Frequency: "80MHz"
Flash Mode: "QIO"
Flash Size: "4MB (32Mb)"
Partition Scheme: "Default 4MB with spiffs"
Core Debug Level: "None"
PSRAM: "Enabled"
```

## Hardware Assembly

### Step 1: Prepare ESP32-CAM
1. Remove camera module if audio-only application
2. Locate GPIO12 pin on expansion connector
3. Ensure power supply can provide 600mA at 3.3V

### Step 2: Connect HW-484 Microphone
```
HW-484 Pin → ESP32-CAM Pin
A0 (Analog) → GPIO12
G (Ground)  → GND
+ (Power)   → 3.3V
D0 (Digital) → Not connected
```

### Step 3: Programming Connection (Temporary)
```
FTDI Adapter → ESP32-CAM
3.3V → 3.3V
GND → GND
TX → U0RXD (GPIO3)
RX → U0TXD (GPIO1)

For Programming Mode:
GPIO0 → GND (connect during upload only)
```

## Configuration

### 1. Update config.h
```cpp
// WiFi credentials
#define WIFI_SSID "your_wifi_name"
#define WIFI_PASSWORD "your_wifi_password"

// Server endpoints (update to your server)
#define SERVER_URL "http://your_server:port/predict"
#define CALIBRATE_URL "http://your_server:port/calibrate"

// Device identification
#define DEVICE_ID "ESP32CAM_Node_01"
#define DEVICE_LOCATION "Your_Location"
```

### 2. Pin Configuration (Already Set)
```cpp
#define MIC_ANALOG_PIN 12  // GPIO12 (ADC2_CH5)
#define LED_PIN 4          // Built-in flash LED
```

## Upload Process

### 1. Enter Programming Mode
1. Connect FTDI adapter to ESP32-CAM
2. Connect GPIO0 to GND
3. Power on or reset ESP32-CAM
4. LED should be dim (programming mode)

### 2. Upload Code
1. Select correct COM port in Arduino IDE
2. Click "Upload"
3. Wait for "Hard resetting via RTS pin..." message
4. **Immediately disconnect GPIO0 from GND**
5. Reset ESP32-CAM

### 3. Verify Operation
1. Open Serial Monitor (115200 baud)
2. Reset ESP32-CAM
3. Look for startup messages and WiFi connection
4. Flash LED should blink during audio transmission

## Expected Serial Output

```
╔═══════════════════════════════════════╗
║  ESP32-CAM Audio Streamer (HW-484)   ║
║  Analog Microphone → Flask Server    ║
╚═══════════════════════════════════════╝

Device ID: ESP32CAM_Node_01
Location: Your_Location
Server: http://your_server:port/predict
PSRAM: Available
Free Heap: 250 KB
Free PSRAM: 4000 KB

✓ Audio buffers allocated in PSRAM
🌐 Connecting to WiFi...
✅ WiFi connected!
   IP Address: 192.168.1.100
   RSSI: -45 dBm
🎤 Initializing ADC for HW-484 microphone (ESP32-CAM)...
✅ ADC initialized successfully
   Pin: GPIO12 (ADC2_CH5) - ESP32-CAM
   ⚠️  Note: ADC2 performance may vary with WiFi activity

🎙️  CALIBRATION PHASE
┌─────────────────────────────────────┐
│  Please remain SILENT for 4 sec    │
│  Sending calibration samples...    │
└─────────────────────────────────────┘

✅ All systems ready!
┌─────────────────────────────────────┐
│    AUDIO STREAMING STARTED          │
└─────────────────────────────────────┘

📊 Capturing audio...
✅ Audio sent successfully
   → Prediction: Whispering (85.23%)
```

## Troubleshooting

### Common Issues and Solutions:

#### 1. Upload Fails
- **Symptom**: "Failed to connect to ESP32"
- **Solution**: 
  - Ensure GPIO0 is connected to GND during upload
  - Check FTDI voltage (must be 3.3V)
  - Try different upload speed (9600)
  - Press reset while uploading

#### 2. No Serial Output
- **Symptom**: Blank serial monitor
- **Solution**:
  - Disconnect GPIO0 from GND after upload
  - Reset ESP32-CAM
  - Check baud rate (115200)
  - Verify TX/RX connections

#### 3. WiFi Connection Issues
- **Symptom**: "WiFi connection failed!"
- **Solution**:
  - Check SSID and password in config.h
  - Ensure 2.4GHz WiFi (ESP32 doesn't support 5GHz)
  - Check power supply (WiFi needs more current)
  - Move closer to router

#### 4. Poor Audio Quality
- **Symptom**: Erratic ADC readings or noise
- **Solution**:
  - Use stable 3.3V power supply
  - Keep microphone wires short
  - Add 100µF capacitor across power rails
  - Try GPIO13 instead of GPIO12

#### 5. Memory Issues
- **Symptom**: Crashes or failed audio buffer allocation
- **Solution**:
  - Ensure PSRAM is enabled in board settings
  - Check available heap memory
  - Reduce buffer size if needed

#### 6. ADC2 WiFi Interference
- **Symptom**: Zero or inconsistent ADC readings
- **Solution**:
  - Code already includes retry mechanism
  - Consider external ADC module for critical applications
  - Monitor WiFi signal strength

## Performance Optimization

### For Better Audio Quality:
1. **Power Supply**: Use regulated 3.3V with low noise
2. **Connections**: Keep analog connections short and shielded
3. **Placement**: Position away from WiFi antenna area
4. **Decoupling**: Add 100µF + 10µF capacitors near power pins

### For Better WiFi Performance:
1. **Antenna**: Ensure good antenna connection
2. **Signal**: Maintain strong WiFi signal (-50 dBm or better)
3. **Power**: Use adequate current capacity (600mA minimum)

## Status Indicators

### LED Patterns:
- **5 fast blinks**: System startup
- **3 blinks**: WiFi connected successfully
- **10 blinks**: WiFi connection failed
- **Single blink**: Audio transmission in progress
- **5 short blinks**: Calibration phase

### Serial Messages:
- **✅**: Success indicators
- **❌**: Error indicators  
- **⚠️**: Warning messages
- **📊**: Status reports
- **🎤**: Audio system messages
- **🌐**: Network messages

## Production Deployment

### Enclosure Considerations:
1. Ventilation for heat dissipation
2. Access to microphone (avoid obstruction)
3. LED visibility for status indication
4. Reset button accessibility

### Power Options:
1. **USB Power Bank**: Portable, 5V → 3.3V regulator needed
2. **Wall Adapter**: 3.3V regulated, 1A minimum
3. **Battery**: 3.7V Li-ion with voltage regulator
4. **PoE**: With 48V → 3.3V converter

### Monitoring:
1. Check serial output for error patterns
2. Monitor WiFi RSSI levels
3. Track transmission success rates
4. Watch for memory leaks or crashes