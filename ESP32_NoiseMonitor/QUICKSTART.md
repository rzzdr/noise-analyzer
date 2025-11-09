# ESP32 Quick Setup Guide - HW-484 Microphone

## ⚡ Quick Start (5 Minutes)

### 1️⃣ Install Arduino IDE & Libraries

```
Arduino IDE → Library Manager → Install:
• ArduinoJson (v6.x)
• Base64 (by Densaugeo)
• ESP32 Board Support (via Board Manager)
```

### 2️⃣ Wire Hardware

```
HW-484    →    ESP32
  A0      →    GPIO36 (VP)
  G       →    GND
  +       →    3.3V
  D0      →    (leave unconnected)
```

### 3️⃣ Configure WiFi & Server

Edit `config.h`:

```cpp
#define WIFI_SSID       "YourWiFiName"
#define WIFI_PASSWORD   "YourPassword"
#define SERVER_URL      "http://4.240.35.54:6002/predict"
```

### 4️⃣ Upload to ESP32

```
Arduino IDE:
1. Tools → Board → ESP32 Dev Module
2. Tools → Port → (select your COM port)
3. Tools → Partition → Huge APP (3MB)
4. Upload (Ctrl+U)
```

### 5️⃣ Test

```
1. Open Serial Monitor (115200 baud)
2. **Stay silent during calibration** (4 seconds)
3. Watch for predictions every second
```

---

## 🔧 Troubleshooting Quick Fixes

| Problem                | Quick Fix                                    |
| ---------------------- | -------------------------------------------- |
| **WiFi won't connect** | Check SSID/password, use 2.4GHz WiFi only    |
| **No audio variation** | Check A0 wire, adjust sensitivity knob       |
| **Calibration fails**  | Ensure quiet environment, restart ESP32      |
| **HTTP errors**        | Verify server URL, check Flask is running    |
| **Compilation errors** | Install missing libraries, update ESP32 core |

---

## 📋 Checklist Before First Run

- [ ] HW-484 connected to GPIO36, GND, 3.3V
- [ ] WiFi credentials updated in config.h
- [ ] Server URL points to Flask server
- [ ] ArduinoJson library installed
- [ ] Base64 library installed
- [ ] ESP32 board support installed
- [ ] Correct board selected in Arduino IDE
- [ ] Serial Monitor set to 115200 baud
- [ ] Flask server running and accessible

---

## 📊 Expected Serial Output

```
✅ WiFi connected!
   IP Address: 192.168.1.xxx

✅ ADC initialized successfully
   Test Reading: 2048

🎙️  CALIBRATION PHASE
   ✅ Sample 1 sent
   ✅ Sample 2 sent
   ✅ Sample 3 sent
   ✅ Sample 4 sent

✅ All systems ready!

📊 Capturing audio...
✅ Audio sent successfully
   → Prediction: Silence (98.50%)
```

---

## 🎯 Pin Reference

| ESP32 Pin   | Function | Connect To              |
| ----------- | -------- | ----------------------- |
| GPIO36 (VP) | ADC1_CH0 | HW-484 A0               |
| GND         | Ground   | HW-484 G                |
| 3.3V        | Power    | HW-484 +                |
| GPIO4       | LED      | (built-in on ESP32-CAM) |

**⚠️ IMPORTANT:**

- Only use ADC1 pins (32-39) when WiFi is active
- Do NOT exceed 3.3V on ADC pins
- GPIO36 is input-only (perfect for ADC)

---

## 🌐 Network Requirements

- **WiFi**: 2.4GHz (ESP32 doesn't support 5GHz)
- **Bandwidth**: ~64KB/sec upload per device
- **Server**: Must be accessible from ESP32 network
- **Ports**: TCP 6002 (or your configured port)

---

## 🔋 Power Options

| Method              | Voltage | Notes                     |
| ------------------- | ------- | ------------------------- |
| USB                 | 5V      | Easiest for development   |
| Battery (regulated) | 3.3-5V  | Use voltage regulator     |
| Power bank          | 5V      | Good for portable testing |

**Current Draw:** ~180-240mA during operation

---

## 📦 Bill of Materials

| Item              | Quantity | Cost (approx) |
| ----------------- | -------- | ------------- |
| ESP32 Dev Board   | 1        | $6-10         |
| HW-484 Microphone | 1        | $2-4          |
| Jumper Wires      | 3        | $0.50         |
| Micro USB Cable   | 1        | $2-5          |
| **Total**         |          | **$10-20**    |

---

## 🚀 Next Steps After Setup

1. **Test different sounds** - speaking, typing, phone ringing
2. **Monitor predictions** - check accuracy in real-time
3. **Adjust sensitivity** - turn potentiometer on HW-484
4. **Deploy multiple nodes** - change DEVICE_ID for each
5. **View Firebase data** - check server's `/stats` endpoint

---

## 📞 Getting Help

1. **Serial Monitor** - Check debug output at 115200 baud
2. **LED Patterns** - Different blink patterns indicate status
3. **Server Logs** - Check Flask server for error messages
4. **Documentation** - See README.md for detailed info

---

**Setup Time:** ~5 minutes  
**Difficulty:** Beginner  
**Skills Required:** Basic Arduino, basic wiring
