# ESP32 Audio Streaming - Issue Resolution

## Issue Description
The ESP32-CAM was successfully connecting to WiFi and completing calibration, but experiencing HTTP 500 errors during main audio streaming with the error message:
```
"argument should be a bytes-like object or ASCII string, not 'NoneType'"
```

## Root Cause
The issue was caused by the `base64.h` library's `base64::encode()` function returning `None` or failing to properly encode the audio data, likely due to memory allocation issues when handling large audio buffers (64KB of audio data).

## Solutions Implemented

### 1. Enhanced Input Validation
- Added null pointer checks for audio data
- Added length validation
- Added basic audio data sanity checks

### 2. Improved Base64 Encoding
- **Replaced** `#include <base64.h>` with `#include "mbedtls/base64.h"`
- **Switched** from `base64::encode()` to ESP32's built-in `mbedtls_base64_encode()`
- **Added** proper error handling for base64 encoding failures
- **Added** memory allocation checks for base64 output buffer

### 3. Enhanced Debugging
- Added detailed logging for buffer validation
- Added sample value logging for debugging
- Added base64 encoding status reporting
- Added memory usage monitoring

### 4. Calibration Improvements (Previously Fixed)
- Reduced calibration samples from 4 to 2
- Shortened calibration duration from 1.0s to 0.5s per sample
- Added retry logic for failed calibration attempts
- Improved error reporting with server response details

## Technical Details

### Memory Usage Before:
- Audio buffer: 16,000 floats × 4 bytes = 64KB
- Base64 encoded: ~85KB string
- JSON payload: ~90KB total

### Memory Usage After:
- Same audio buffer size for main streaming
- Calibration samples: 8,000 floats × 4 bytes = 32KB (reduced)
- More robust memory allocation with proper cleanup

### Key Code Changes:

#### Base64 Encoding (Before):
```cpp
String base64Audio = base64::encode((uint8_t *)audioData, byteLength);
```

#### Base64 Encoding (After):
```cpp
size_t outputLen = 0;
int ret = mbedtls_base64_encode(NULL, 0, &outputLen, (uint8_t*)audioData, byteLength);
uint8_t* base64Buffer = (uint8_t*)malloc(outputLen + 1);
ret = mbedtls_base64_encode(base64Buffer, outputLen, &outputLen, (uint8_t*)audioData, byteLength);
base64Buffer[outputLen] = '\0';
String base64Audio = String((char*)base64Buffer);
free(base64Buffer);
```

## Expected Results
- ✅ Calibration should complete successfully (already working)
- ✅ Main audio streaming should work without HTTP 500 errors
- ✅ Base64 encoding should be reliable and handle large payloads
- ✅ Better error reporting for debugging future issues

## Next Steps
1. Upload the modified code to ESP32-CAM
2. Monitor serial output for successful audio streaming
3. Verify predictions are received from Flask server
4. Check memory usage remains stable over time

## Files Modified
- `ESP32_NoiseMonitor.ino` - Main streaming logic and base64 encoding
- `config.h` - Calibration settings (reduced samples and duration)
- `test_server.py` - Created for server connectivity testing (confirmed server works)