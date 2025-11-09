# ✅ ESP32 Audio Streamer - Setup Complete

## 📋 What Has Been Done

### Architecture Changed ✅

- **Old**: ESP32 with I2S microphone + local TFLite processing
- **New**: ESP32 with HW-484 analog microphone + server-side processing

### Code Rewritten ✅

All ESP32 code has been completely rewritten to:

1. ✅ Read audio from HW-484 analog microphone (GPIO36)
2. ✅ Sample at 16kHz using precise ADC timing
3. ✅ Send calibration data on bootup (4 samples to `/calibrate`)
4. ✅ Stream 1-second audio buffers to Flask server (`/predict`)
5. ✅ Receive and display predictions from server

### Files Created ✅

- `ESP32_NoiseMonitor.ino` - Main Arduino sketch (rewritten)
- `config.h` - Configuration file (updated)
- `README.md` - Complete documentation
- `WIRING_HW484.md` - Detailed wiring guide
- `QUICKSTART.md` - Quick setup reference
- `CHANGELOG.md` - Migration details

## 🔌 Hardware Wiring

```
HW-484 Microphone → ESP32
────────────────────────────
A0 (Analog Out)   → GPIO36 (VP)
G  (Ground)       → GND
+  (VCC)          → 3.3V
D0 (Digital)      → Not connected
```

## ⚙️ Configuration Required

Before uploading, edit `config.h`:

```cpp
// 1. Update WiFi credentials
#define WIFI_SSID       "Your_WiFi_Name"
#define WIFI_PASSWORD   "Your_Password"

// 2. Verify server URLs (already set)
#define SERVER_URL      "http://4.240.35.54:6002/predict"
#define CALIBRATE_URL   "http://4.240.35.54:6002/calibrate"

// 3. Set unique device ID (if deploying multiple)
#define DEVICE_ID       "ESP32_Node_01"
```

## 📚 Required Arduino Libraries

Install via **Library Manager** (Sketch → Include Library → Manage Libraries):

| Library             | Version | Author          |
| ------------------- | ------- | --------------- |
| ArduinoJson         | 6.x     | Benoit Blanchon |
| Base64              | Latest  | Densaugeo       |
| ESP32 Board Support | 2.x+    | Espressif       |

## 🚀 Upload Steps

1. **Open Arduino IDE**
2. **Load sketch**: Open `ESP32_NoiseMonitor.ino`
3. **Select board**: Tools → Board → ESP32 Dev Module
4. **Select port**: Tools → Port → (your COM port)
5. **Partition**: Tools → Partition Scheme → Huge APP (3MB)
6. **Upload**: Click Upload button (→)

### For ESP32-CAM:

- Connect IO0 to GND before upload
- Press RESET on programmer
- Upload
- Disconnect IO0 from GND
- Press RESET to run

## 🔍 Testing

### 1. Open Serial Monitor

- Set baud rate to **115200**
- You should see:

```
╔═══════════════════════════════════════╗
║  ESP32 Audio Streamer (HW-484)       ║
║  Analog Microphone → Flask Server    ║
╚═══════════════════════════════════════╝

Device ID: ESP32_Node_01
Location: Library_Floor1_NE
Server: http://4.240.35.54:6002/predict

🌐 Connecting to WiFi...
✅ WiFi connected!
   IP Address: 192.168.x.x

🎤 Initializing ADC for HW-484 microphone...
✅ ADC initialized successfully
   Test Reading: 2048

🎙️  CALIBRATION PHASE
┌─────────────────────────────────────┐
│  Please remain SILENT for 4 sec    │
│  Sending calibration samples...    │
└─────────────────────────────────────┘

   ✅ Sample 1 sent - Progress: 25.0%
   ✅ Sample 2 sent - Progress: 50.0%
   ✅ Sample 3 sent - Progress: 75.0%
   ✅ Sample 4 sent - Progress: 100.0%

✅ Calibration completed successfully!
✅ All systems ready!

┌─────────────────────────────────────┐
│    AUDIO STREAMING STARTED          │
└─────────────────────────────────────┘

📊 Capturing audio...
✅ Audio sent successfully
   → Prediction: Silence (98.50%)
```

### 2. Verify Flask Server Receives Data

Check your Flask server logs - you should see:

```
✅ Pushed to Firebase: Silence (0.985)
```

## 📊 Operation Flow

```
┌─────────────────────────────────────────────────┐
│                    BOOTUP                       │
├─────────────────────────────────────────────────┤
│ 1. Connect to WiFi                              │
│ 2. Initialize ADC (GPIO36)                      │
│ 3. CALIBRATION: Send 4 silent samples          │
│    to /calibrate endpoint                       │
│ 4. Wait for VAD calibration complete           │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              CONTINUOUS LOOP                    │
├─────────────────────────────────────────────────┤
│ Every 1 second:                                 │
│ 1. Capture 16,000 ADC samples @ 16kHz          │
│ 2. Normalize to float [-1, 1]                  │
│ 3. Base64 encode                                │
│ 4. POST to /predict endpoint                   │
│ 5. Receive classification result               │
│ 6. Display prediction                           │
└─────────────────────────────────────────────────┘
```

## 🎯 Expected Behavior

### LED Indicators

- **5 fast blinks** → Startup
- **3 blinks** → WiFi connected
- **LED on during transmission** → Sending data
- **10 rapid blinks** → Error

### Serial Output Every 10 Transmissions

```
📊 SYSTEM STATUS REPORT
Total Transmissions:   10
Successful:            100.0%
Errors:                0.0%
Capture Time:          1001 ms
Transmit Time:         245 ms
Free Heap:             145 KB
WiFi RSSI:             -45 dBm
```

## ✅ Verification Checklist

- [ ] ESP32 connects to WiFi successfully
- [ ] ADC test reading shows ~2048 (center value)
- [ ] All 4 calibration samples sent successfully
- [ ] Audio predictions received every second
- [ ] Flask server logs show incoming requests
- [ ] Predictions make sense (Silence when quiet)
- [ ] LED blinks during transmission
- [ ] No memory errors or crashes

## 🔧 Troubleshooting

### WiFi Connection Fails

```cpp
// Check in config.h:
- Correct SSID and password
- WiFi is 2.4GHz (not 5GHz)
- WiFi network is accessible
```

### Calibration Fails

```
✅ Solution: Ensure environment is QUIET
   - No talking during calibration
   - No background music/TV
   - Restart ESP32 to try again
```

### No Audio Variation (Always ~2048)

```
✅ Check wiring:
   - A0 connected to GPIO36?
   - Ground connected?
   - Power to HW-484?
   - Turn sensitivity potentiometer
```

### HTTP Errors

```
✅ Verify:
   - Flask server is running
   - Server URL is correct
   - Port 6002 is open
   - ESP32 can reach server IP
```

## 📞 Next Steps

1. **Test with different sounds**

   - Speak near microphone
   - Type on keyboard
   - Play phone ringing sound
   - Verify predictions change

2. **Adjust microphone sensitivity**

   - Turn potentiometer on HW-484
   - Clockwise = more sensitive
   - Counter-clockwise = less sensitive

3. **Monitor Firebase data**

   - Check Flask server's `/stats` endpoint
   - View real-time predictions
   - Analyze classification accuracy

4. **Deploy multiple nodes**
   - Change `DEVICE_ID` for each
   - Track different locations
   - Compare noise levels

## 🎉 Success Criteria

Your system is working correctly when:

✅ ESP32 boots without errors  
✅ WiFi connects automatically  
✅ Calibration completes (4 samples)  
✅ Predictions arrive every second  
✅ Flask server processes requests  
✅ Firebase stores data  
✅ Predictions match actual sounds

## 📖 Documentation

- **Quick Start**: See `QUICKSTART.md`
- **Wiring Details**: See `WIRING_HW484.md`
- **Full Guide**: See `README.md`
- **Changes Made**: See `CHANGELOG.md`

---

**Status**: ✅ Ready to Upload and Test  
**Date**: November 9, 2025  
**Version**: 2.0 (Server-based Processing)
