# ESP32-CAM Wiring Guide

## 📐 Physical Connections

### Component Layout
```
                     ESP32-CAM Board Layout
                  ┌─────────────────────────┐
          RST ○───┤                         │
          3V3 ○───┤                         │
          GND ○───┤      [ESP32-CAM]        │
         IO15 ○───┤        Module           │───○ IO14
         IO13 ○───┤                         │───○ IO12  
          IO4 ○───┤      [MicroSD Slot]     │───○ IO2
         IO16 ○───┤        (unused)         │───○ IO1 (TX)
          VCC ○───┤                         │───○ IO3 (RX)
           5V ○───┤    [Camera Connector]   │───○ GND
                  │       (unused)          │
                  └─────────────────────────┘
                              │
                         [Flash LED]
```

### INMP441 Microphone Pinout
```
          INMP441 I²S MEMS Microphone
        ┌─────────────────────────────┐
        │          [●●●]              │ ← Acoustic holes
        │       Microphone            │
        │                             │
        ├─────────────────────────────┤
        │ VDD  GND  L/R  SCK  WS  SD  │ ← Pin headers
        └─────────────────────────────┘
          │    │    │    │    │    │
          │    │    │    │    │    └── Data Output
          │    │    │    │    └─────── Word Select (LRCLK)
          │    │    │    └──────────── Serial Clock (BCLK)
          │    │    └─────────────── Left/Right Channel Select
          │    └──────────────────── Ground
          └───────────────────────── Power (3.3V only!)
```

## 🔌 Connection Tables

### Main I²S Audio Connections

| INMP441 Pin | Wire Color | ESP32-CAM Pin | GPIO | Function |
|-------------|------------|---------------|------|----------|
| VDD | Red | 3V3 | - | 3.3V Power |
| GND | Black | GND | - | Ground |
| L/R | Black | GND | - | Left Channel |
| SCK/BCLK | Blue | IO14 | GPIO14 | Bit Clock |
| WS/LRCLK | Green | IO15 | GPIO15 | Word Select |
| SD/DOUT | Yellow | IO13 | GPIO13 | Data Out |

### Programming Connections (USB-UART Adapter)

| USB-UART | Wire Color | ESP32-CAM Pin | Notes |
|----------|------------|---------------|-------|
| VCC (3.3V) | Red | 3V3 | Logic level only |
| GND | Black | GND | Common ground |
| TXD | Green | IO3 (RX) | UART transmit |
| RXD | White | IO1 (TX) | UART receive |
| DTR/RTS | - | - | Not used |

### Power Supply Connection

| Power Supply | ESP32-CAM Pin | Notes |
|--------------|---------------|-------|
| +5V | 5V | Use onboard regulator |
| GND | GND | Stable ground connection |

**⚠️ Important:** Use 5V supply to ESP32-CAM's 5V pin, NOT 3.3V pin!

## 📸 Step-by-Step Wiring Photos

### Step 1: Prepare Components
```
Required Components:
□ ESP32-CAM module
□ INMP441 I²S microphone  
□ USB-UART programmer (CP2102/CH340)
□ 6x female-to-male jumper wires
□ 2x male-to-male jumper wires
□ Breadboard (optional)
□ 5V power supply
```

### Step 2: ESP32-CAM to INMP441 Wiring

**Pin Mapping Visual:**
```
ESP32-CAM Side View:
┌─────────────┐
│ 3V3 ● GND   │ ← Power pins
│ IO15● IO14  │ ← I²S Word Select & Clock  
│ IO13● IO12  │ ← I²S Data & unused
│ IO4 ● IO2   │
│ IO16● IO1   │ ← UART TX
│ VCC ● IO3   │ ← Power & UART RX
│ 5V  ● GND   │ ← Main power input
└─────────────┘

INMP441 Connections:
VDD (Red)    → ESP32-CAM 3V3
GND (Black)  → ESP32-CAM GND  
L/R (Black)  → ESP32-CAM GND
SCK (Blue)   → ESP32-CAM IO14
WS (Green)   → ESP32-CAM IO15
SD (Yellow)  → ESP32-CAM IO13
```

### Step 3: Programming Interface Wiring

**USB-UART to ESP32-CAM:**
```
USB-UART Adapter          ESP32-CAM
┌─────────────┐          ┌─────────────┐
│ VCC (3.3V)  │ ────Red──→ │ 3V3         │
│ GND         │ ──Black──→ │ GND         │  
│ TXD         │ ──Green──→ │ IO3 (RX)    │
│ RXD         │ ──White──→ │ IO1 (TX)    │
│ DTR/RTS     │            │             │ (not used)
└─────────────┘          └─────────────┘
```

### Step 4: Power Supply Connection

**5V Power to ESP32-CAM:**
```
5V Wall Adapter           ESP32-CAM
┌─────────────┐          ┌─────────────┐
│ +5V (Red)   │ ────Red──→ │ 5V          │
│ GND (Black) │ ──Black──→ │ GND         │
└─────────────┘          └─────────────┘
```

## 🔧 Assembly Instructions

### Physical Assembly

1. **Mount ESP32-CAM on breadboard** (optional)
   - Use breadboard for easier prototyping
   - Keep camera facing outward (not needed for audio)

2. **Position INMP441 microphone**
   - Place away from ESP32-CAM module (reduce RF noise)
   - Ensure acoustic holes are not blocked
   - Consider orientation for directional pickup

3. **Route I²S wires**
   - Keep wires as short as possible (<15cm ideal)
   - Separate power and signal wires
   - Avoid parallel runs with power cables

4. **Add decoupling capacitors** (recommended)
   - 0.1µF ceramic near INMP441 VDD-GND
   - 10µF electrolytic near ESP32-CAM power

### Wiring Best Practices

#### Do ✅
- Use different colors for each signal
- Keep I²S wires short and direct
- Add strain relief for permanent installations
- Double-check connections before power-on
- Use twisted pairs for longer I²S lines

#### Don't ❌
- Connect INMP441 VDD to 5V (will damage!)  
- Use long jumper wires for I²S signals
- Route I²S wires parallel to switching power supplies
- Forget ground connections
- Mix up TX/RX connections (crossed cable)

## 🧪 Testing Connections

### Continuity Testing

**Before applying power, test with multimeter:**

1. **Power connections:**
   ```
   ESP32-CAM 3V3 ↔ INMP441 VDD  
   ESP32-CAM GND ↔ INMP441 GND
   ESP32-CAM GND ↔ INMP441 L/R
   ESP32-CAM 5V  ↔ Power supply +5V
   ESP32-CAM GND ↔ Power supply GND
   ```

2. **I²S signal connections:**
   ```
   ESP32-CAM IO14 ↔ INMP441 SCK
   ESP32-CAM IO15 ↔ INMP441 WS  
   ESP32-CAM IO13 ↔ INMP441 SD
   ```

3. **UART connections:**
   ```
   ESP32-CAM IO1 ↔ USB-UART RXD
   ESP32-CAM IO3 ↔ USB-UART TXD
   ESP32-CAM GND ↔ USB-UART GND
   ```

### Power-On Testing

1. **Initial power test:**
   - Connect 5V power to ESP32-CAM
   - LED should light up briefly
   - No smoke or excessive heat

2. **UART communication test:**
   - Connect USB-UART adapter
   - Open serial monitor at 115200 baud
   - Should see boot messages

3. **I²S functionality test:**
   - Upload firmware with I²S test code
   - Should see "I2S configured successfully"
   - Microphone test should pass

## 🚨 Common Wiring Mistakes

### Mistake 1: Wrong Power Voltage
**Symptom:** INMP441 doesn't work, may be damaged
**Cause:** Connected INMP441 VDD to 5V instead of 3.3V
**Fix:** INMP441 requires exactly 3.3V, use ESP32-CAM's 3V3 output

### Mistake 2: Crossed UART Lines  
**Symptom:** No serial communication, can't program
**Cause:** Connected TX to TX, RX to RX (should be crossed)
**Fix:** ESP32 RX ↔ USB-UART TX, ESP32 TX ↔ USB-UART RX

### Mistake 3: Missing Ground Connections
**Symptom:** Intermittent operation, noise in audio
**Cause:** INMP441 L/R pin not connected to ground
**Fix:** Connect INMP441 L/R pin to ESP32-CAM GND

### Mistake 4: I²S Pin Conflicts
**Symptom:** "I2S driver install failed" error
**Cause:** Pins already used by other peripherals
**Fix:** Use different GPIO pins, avoid camera/SD card pins

### Mistake 5: Long I²S Wires
**Symptom:** Noisy audio, clock signal issues
**Cause:** I²S wires too long (>20cm)
**Fix:** Shorten wires, add series resistors, use twisted pairs

## 🔌 Alternative Pin Configurations

### If Default Pins Don't Work

**Alternative I²S Configuration:**
```cpp
// In config.h, change these values:
#define I2S_WS_PIN      4    // Alternative WS pin
#define I2S_SCK_PIN     2    // Alternative SCK pin  
#define I2S_SD_PIN      12   // Alternative SD pin
```

**Available GPIO Pins on ESP32-CAM:**
- **Safe to use:** GPIO2, GPIO4, GPIO12, GPIO13, GPIO14, GPIO15
- **Avoid:** GPIO0 (boot), GPIO16 (PSRAM), GPIO1/3 (UART)
- **Camera pins:** GPIO0, GPIO5, GPIO18, GPIO19, GPIO21, GPIO22, GPIO23, GPIO25, GPIO26, GPIO27

### Pin Selection Guidelines

**For I²S SCK (Clock):** 
- Use pins capable of high-frequency output
- GPIO14, GPIO2, GPIO4 recommended

**For I²S WS (Word Select):**
- Any GPIO pin works
- GPIO15, GPIO12, GPIO13 recommended  

**For I²S Data Input:**
- Any input-capable GPIO
- GPIO13, GPIO12, GPIO35 recommended

## 📷 Visual Reference

### Completed Wiring Example
```
Final Assembly View:

      [USB-UART]
          ↓ (programming only)
    ┌─────────────┐
    │  ESP32-CAM  │
    │   Module    │ ────I²S wires───→ [INMP441 MIC]
    └─────────────┘                      ↑
          ↓                        (audio input)
    [5V Power Supply]
```

### Pin Identification Guide
```
ESP32-CAM Pin Locations (looking at component side):

         [Camera Connector]
              (unused)
                │
    ┌───────────┼───────────┐
    │ RST                   │
    │ 3V3 ●   [ESP32]   ● ? │
    │ GND ●    CHIP     ● ? │  
    │ IO15●             ●IO14│ ← I²S pins
    │ IO13●             ●IO12│ ← I²S data pin
    │ IO4 ●             ●IO2 │
    │ IO16●             ●IO1 │ ← UART TX
    │ VCC ●             ●IO3 │ ← UART RX  
    │ 5V  ●  [MicroSD]  ●GND │ ← Power pins
    └───────────┼───────────┘
                │
        (SD slot unused)
```

---

## ⚡ Quick Reference Card

### Pin Summary
| Function | ESP32-CAM Pin | INMP441 Pin |
|----------|---------------|-------------|
| 🔴 Power | 3V3 | VDD |
| ⚫ Ground | GND | GND |
| ⚫ Left Ch | GND | L/R |
| 🔵 Clock | IO14 | SCK |
| 🟢 Word Sel | IO15 | WS |
| 🟡 Data | IO13 | SD |

### Programming Mode
1. Connect GPIO0 to GND
2. Press RESET button  
3. Release GPIO0
4. Upload firmware
5. Remove GPIO0 connection
6. Press RESET to run

### Power Requirements
- **ESP32-CAM:** 5V @ 200-300mA
- **INMP441:** 3.3V @ 1.4mA
- **Total:** 5V @ 350mA (minimum 1A supply recommended)

---

*Always double-check connections before applying power!*