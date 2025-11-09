# HW-484 Microphone Wiring Guide for ESP32

## Hardware Requirements

- ESP32 Development Board (ESP32-CAM or similar)
- HW-484 Analog Microphone Module
- Jumper wires
- USB cable for programming

## HW-484 Module Pinout

```
┌─────────────────┐
│    HW-484       │
│  Microphone     │
├─────────────────┤
│ A0  - Analog Out│  ← Main audio signal
│ G   - Ground    │  ← GND
│ +   - VCC       │  ← Power (3.3V or 5V)
│ D0  - Digital   │  ← Not used
└─────────────────┘
```

## Wiring Connections

### ESP32 ↔ HW-484

| HW-484 Pin      | ESP32 Pin     | Description                     |
| --------------- | ------------- | ------------------------------- |
| A0 (Analog Out) | GPIO36 (VP)   | ADC1_CH0 - Analog audio signal  |
| G (Ground)      | GND           | Common ground                   |
| + (VCC)         | 3.3V          | Power supply                    |
| D0 (Digital)    | Not connected | Not needed for this application |

### Visual Wiring Diagram

```
ESP32                          HW-484 Microphone
┌──────────────┐              ┌──────────────┐
│              │              │              │
│   GPIO36(VP) │◄─────────────┤ A0           │
│              │   Analog     │              │
│   GND        │◄─────────────┤ G            │
│              │   Ground     │              │
│   3.3V       │◄─────────────┤ +            │
│              │   Power      │              │
│              │              │ D0 (unused)  │
│              │              │              │
└──────────────┘              └──────────────┘
```

## Important Notes

### 1. ADC Pin Selection

- **Use GPIO36 (VP)**: This is ADC1_CH0, one of the best ADC pins on ESP32
- Other suitable ADC1 pins: GPIO32-39 (if GPIO36 is not available)
- **Do NOT use ADC2 pins** (GPIO0, 2, 4, 12-15, 25-27) when WiFi is active

### 2. Power Supply

- The HW-484 can work with 3.3V or 5V
- For ESP32, use **3.3V** to match ADC input range
- If using 5V, add a voltage divider for A0 output

### 3. Signal Quality

- Keep wires as short as possible to reduce noise
- Use shielded cable for A0 if experiencing interference
- Place microphone away from WiFi antenna and power supply

### 4. Sensitivity Adjustment

- The HW-484 has a potentiometer to adjust sensitivity
- Turn clockwise to increase sensitivity
- Turn counter-clockwise to decrease sensitivity
- Adjust based on your noise environment

## Testing the Connection

1. **Power Test**: After connecting, the power LED on HW-484 should light up

2. **Signal Test**: Use Serial Monitor to check ADC readings:

   - Should read around 2048 in silence (center value for 12-bit ADC)
   - Should fluctuate 1800-2300 with normal ambient noise
   - Should spike above 2500 with loud sounds

3. **Audio Quality**:
   - If readings are always near 0 or 4095, check wiring
   - If readings don't change with sound, adjust sensitivity pot
   - If readings are very noisy, check ground connection

## Code Configuration

The pin is configured in `config.h`:

```cpp
#define MIC_ANALOG_PIN  36    // A0 pin → GPIO36 (ADC1_CH0, VP)
```

If you need to use a different pin, update this value to any ADC1 pin (32-39).

## Troubleshooting

| Problem              | Solution                                           |
| -------------------- | -------------------------------------------------- |
| No readings          | Check power (3.3V) and ground connections          |
| Readings always 0    | Check A0 wire connection                           |
| Readings always 4095 | Check if A0 is shorted to 3.3V                     |
| No response to sound | Adjust sensitivity potentiometer clockwise         |
| Too sensitive        | Adjust sensitivity potentiometer counter-clockwise |
| Noisy readings       | Ensure good ground connection, shorten wires       |

## Safety Warnings

⚠️ **Do not connect 5V to ESP32 ADC pins** - Maximum input is 3.3V
⚠️ **Use only ADC1 pins** - ADC2 conflicts with WiFi
⚠️ **Avoid static discharge** - Touch ground before handling components
