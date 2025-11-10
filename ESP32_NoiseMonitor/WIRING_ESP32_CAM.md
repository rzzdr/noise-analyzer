# ESP32-CAM Wiring Guide for HW-484 Analog Microphone

## Overview
This guide shows how to wire the HW-484 analog microphone module to the ESP32-CAM for audio monitoring and noise classification.

## ESP32-CAM Pin Layout

```
                          ESP32-CAM Module
                        ┌─────────────────────┐
                        │                     │
                    5V  │●                   ●│  GND
                   GND  │●                   ●│  GPIO3/U0RXD
                  GPIO2 │●                   ●│  GPIO1/U0TXD
            VCC (3.3V)  │●                   ●│  GND
                  GPIO4 │●                   ●│  GPIO16  
                 GPIO0  │●   [ESP32-CAM]     ●│  GPIO0
                 GPIO2  │●     [CAMERA]      ●│  GPIO0
                        │●                   ●│  GPIO15
                        │●                   ●│  GPIO14
                        │●                   ●│  GPIO12  ← MICROPHONE PIN
                        │●                   ●│  GPIO13
                        └─────────────────────┘
                                Front View
```

## HW-484 Microphone Module Pinout

```
                      HW-484 Analog Microphone Module
                        ┌─────────────────────┐
                        │                     │
                        │   [MICROPHONE]      │
                        │       ●             │
                        │                     │
                        └─────────────────────┘
                             A0  G  +  D0
                             │   │  │   │
                           Analog │ Power │
                           Output │  3.3V │
                              Ground   Digital
                                     (not used)
```

## Wiring Connections

### Power Supply
- **ESP32-CAM VCC (3.3V)** → **HW-484 "+" pin**
- **ESP32-CAM GND** → **HW-484 "G" pin**

### Audio Signal
- **ESP32-CAM GPIO12** → **HW-484 "A0" pin**
- **HW-484 "D0" pin** → **Not connected** (not used)

### Complete Wiring Table

| HW-484 Pin | ESP32-CAM Pin | Description |
|------------|---------------|-------------|
| A0 (Analog)| GPIO12        | Audio signal input (ADC2_CH5) |
| G (Ground) | GND           | Power ground |
| + (Power)  | 3.3V          | Power supply (3.3V) |
| D0 (Digital)| Not connected | Digital output (unused) |

## Visual Wiring Diagram

```
HW-484 Microphone              ESP32-CAM Module
┌─────────────────┐           ┌──────────────────┐
│                 │           │                  │
│   [MICROPHONE]  │           │                  │
│       ●         │           │                  │
│                 │           │                  │
└─────────────────┘           │                  │
     A0 G  +  D0              │                  │
     │  │  │  │               │                  │
     │  │  │  └─── (unused)   │                  │
     │  │  │                  │                  │
     │  │  └────────────────→ │ 3.3V             │
     │  │                     │                  │
     │  └─────────────────────→ │ GND              │
     │                        │                  │
     └──────────────────────→ │ GPIO12 (ADC2_CH5)│
                              │                  │
                              │     [STATUS LED] │
                              │       GPIO4      │
                              └──────────────────┘
```

## Programming Connection (FTDI/USB-Serial)

For programming the ESP32-CAM, you'll need an FTDI adapter:

```
FTDI USB-Serial          ESP32-CAM Module
┌──────────────┐        ┌──────────────────┐
│              │        │                  │
│ VCC (3.3V)   │──────→ │ 3.3V             │
│ GND          │──────→ │ GND              │
│ TX           │──────→ │ U0RXD (GPIO3)    │
│ RX           │──────→ │ U0TXD (GPIO1)    │
│              │        │                  │
└──────────────┘        │ GPIO0 ──────────→│ GND (for programming)
                        │                  │
                        └──────────────────┘
```

**Programming Steps:**
1. Connect GPIO0 to GND
2. Power on the ESP32-CAM
3. Upload code
4. Disconnect GPIO0 from GND
5. Reset the module

## Power Requirements

- **ESP32-CAM**: 3.3V, ~500mA (during WiFi transmission)
- **HW-484 Microphone**: 3.3V, ~5mA
- **Total**: ~505mA at 3.3V

**Important**: Use a stable 3.3V power supply capable of at least 600mA for reliable operation.

## Status LED

The ESP32-CAM has a built-in flash LED on GPIO4 that will blink to indicate:
- **Fast blinks (5x)**: System startup
- **3 blinks**: WiFi connected
- **10 blinks**: WiFi connection failed
- **Single blink**: Audio transmission in progress

## Pin Availability Notes

### ESP32-CAM Pin Usage:
- **GPIO0**: Boot mode selection (pulled up for normal operation)
- **GPIO1**: Serial TX (UART)
- **GPIO2**: Camera data
- **GPIO3**: Serial RX (UART)
- **GPIO4**: Flash LED (built-in)
- **GPIO12**: **Available for microphone** ✅
- **GPIO13**: Available
- **GPIO14**: Camera clock
- **GPIO15**: Camera power
- **GPIO16**: PSRAM

### Why GPIO12 for Microphone?
- GPIO12 is ADC2_CH5 - supports analog reading
- Available on expansion connector
- Not used by camera module
- Good for audio sampling (though ADC2 has WiFi limitations)

## Troubleshooting

### Common Issues:
1. **No audio readings**: Check power supply voltage and connections
2. **Erratic readings**: Ensure stable 3.3V power supply
3. **WiFi connection issues**: Check antenna connection and power supply
4. **Programming issues**: Ensure GPIO0 is connected to GND during upload

### ADC2 WiFi Limitation:
ADC2 channels (including GPIO12) may have reduced performance when WiFi is active. This is a known ESP32 limitation. The code handles this gracefully, but for best results:
- Use a stable power supply
- Ensure good WiFi signal strength
- Consider using an external ADC for critical applications

## Alternative Pin Options

If GPIO12 doesn't work well for your setup, you can try:
- **GPIO13** (ADC2_CH4) - Also available
- **GPIO15** (ADC2_CH3) - May conflict with camera, use carefully

Update the `MIC_ANALOG_PIN` definition in `config.h` to change the pin.

## Safety Notes

- Never exceed 3.3V on any ESP32-CAM pin
- Use proper ESD protection when handling the module
- Ensure stable power supply to prevent brown-out resets
- Keep connections short to minimize noise interference