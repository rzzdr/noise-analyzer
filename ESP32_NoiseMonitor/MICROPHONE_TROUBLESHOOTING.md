# ESP32-CAM Microphone Troubleshooting Guide

## Problem: All ADC Readings Are Zero

Based on your debug output showing `Sample ADC values: 0, 0, 0`, here are the solutions:

### Immediate Actions

#### 1. Check Physical Connections
```
HW-484 Pin → ESP32-CAM Pin
A0 (Analog) → GPIO12
G (Ground)  → GND  
+ (Power)   → 3.3V
D0 (Digital)→ Not connected
```

**Verification Steps:**
- Use a multimeter to check continuity
- Verify 3.3V is present at the + pin
- Ensure GND connection is solid

#### 2. Test with Diagnostic Sketch
Upload the `ESP32_MIC_TEST.ino` file first:
1. Open `ESP32_MIC_TEST.ino` in Arduino IDE
2. Upload to your ESP32-CAM
3. Open Serial Monitor (115200 baud)
4. Follow the test results

### Hardware Solutions

#### Option A: Fix Current Setup (GPIO12)
The updated code now includes:
- Enhanced ADC initialization with multiple retries
- Better WiFi/ADC2 conflict handling
- Detailed diagnostics

#### Option B: Try Alternative Pins
If GPIO12 doesn't work, modify `config.h`:

```cpp
// Change this line in config.h:
#define MIC_ANALOG_PIN 2   // Try GPIO2 instead of GPIO12
// OR
#define MIC_ANALOG_PIN 13  // Try GPIO13 instead of GPIO12
```

**Pin Options:**
- `GPIO2` (ADC2_CH2) - Good alternative
- `GPIO13` (ADC2_CH4) - Another option  
- `GPIO15` (ADC2_CH3) - Use carefully (camera related)

#### Option C: Power Supply Check
- ESP32-CAM needs stable 3.3V at ~500mA
- Try powering from external 3.3V regulator
- Check if 5V power improves microphone signal

### Software Solutions

#### 1. Enable Test Tone
In `config.h`, change:
```cpp
#define ENABLE_TEST_TONE true  // Generate test tone if mic fails
```
This will generate a 1kHz test tone when microphone readings are mostly zero.

#### 2. Increase Debugging
The updated code now shows:
- Zero reading percentage
- Connection verification warnings  
- Heap usage monitoring

### Common Issues & Fixes

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **No Wiring** | All zeros, no variation | Check physical connections |
| **ADC2/WiFi Conflict** | Intermittent zeros | Use different pin or enhanced retry logic |
| **Power Issues** | Unstable readings | Use external 3.3V regulator |
| **Faulty Microphone** | Constant reading | Test with multimeter or different mic |

### Testing Procedure

1. **Upload Test Sketch**
   ```
   Arduino IDE → Open ESP32_MIC_TEST.ino → Upload
   ```

2. **Check Results**
   - If >90% zeros → Wiring problem
   - If <50% zeros → ADC2/WiFi issue  
   - If low range → Microphone problem

3. **Try Alternative Pin**
   ```cpp
   // In ESP32_MIC_TEST.ino, change:
   #define MIC_PIN 2  // Try GPIO2
   ```

4. **Verify with Multimeter**
   - Measure 3.3V at HW-484 + pin
   - Check continuity of A0 → GPIO12
   - Verify GND connection

### Expected Behavior After Fix

When working correctly, you should see:
```
Sample ADC values: 2048, 2055, 2041  // Values around 2048 (center)
Sample float values: 0.000, 0.003, -0.003  // Small variations around 0
Zero readings: 23/16000 (0.1%)  // Very few zeros
```

### Next Steps

1. **First**: Upload and run `ESP32_MIC_TEST.ino`
2. **If test passes**: The main code ADC2/WiFi handling should now work
3. **If test fails**: Try alternative GPIO pin
4. **Still failing**: Check hardware (power, wiring, microphone)

### Alternative Hardware Options

If none of the above work:
- Use external ADC (ADS1115) via I2C
- Use I2S microphone instead of analog
- Use ESP32 development board instead of ESP32-CAM

The updated main code should handle the ADC2/WiFi conflict much better now. Upload the modified version and test!