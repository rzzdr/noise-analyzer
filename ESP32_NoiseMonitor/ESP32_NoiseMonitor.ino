/*
 * ESP32-CAM Audio Streamer for HW-484 Analog Microphone
 * Captures audio from analog microphone and streams to Flask server
 *
 * Hardware: ESP32-CAM + HW-484 Analog Microphone Module
 * Wiring (ESP32-CAM):
 *   - A0 (Analog Out) → GPIO12 (ADC2_CH5) - Use expansion pin
 *   - G  (Ground)     → GND
 *   - +  (VCC)        → 3.3V (or 5V if available)
 *   - D0 (Digital)    → Not connected
 *
 * Communication: WiFi (HTTP POST) to Flask server
 * Note: ESP32-CAM has 4MB PSRAM for large audio buffers
 *
 * Author: Audio Monitoring System
 * Date: 2025
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "mbedtls/base64.h"  // Use ESP32's built-in base64
#include <math.h>            // For sin, PI functions
#include "config.h"

#ifndef PI
#define PI 3.14159265359f
#endif

// Global objects
HTTPClient http;
WiFiClient wifiClient;

// Audio buffers - will be allocated in setup() to use PSRAM
uint16_t *adcBuffer = nullptr; // Raw ADC readings
float *audioFloat = nullptr;   // Normalized float audio [-1, 1]

// Statistics
unsigned long totalTransmissions = 0;
unsigned long successfulTransmissions = 0;
unsigned long transmissionErrors = 0;

// Timing
hw_timer_t *samplingTimer = NULL;
volatile bool bufferReady = false;
volatile uint16_t bufferIndex = 0;

void setup()
{
#if ENABLE_SERIAL_DEBUG
  Serial.begin(SERIAL_BAUD);
  delay(1000);
#endif

  // Allocate audio buffers in PSRAM if available
  if (psramFound())
  {
    adcBuffer = (uint16_t *)ps_malloc(AUDIO_BUFFER_SIZE * sizeof(uint16_t));
    audioFloat = (float *)ps_malloc(AUDIO_BUFFER_SIZE * sizeof(float));
#if ENABLE_SERIAL_DEBUG
    Serial.println("✓ Audio buffers allocated in PSRAM");
#endif
  }
  else
  {
    adcBuffer = (uint16_t *)malloc(AUDIO_BUFFER_SIZE * sizeof(uint16_t));
    audioFloat = (float *)malloc(AUDIO_BUFFER_SIZE * sizeof(float));
#if ENABLE_SERIAL_DEBUG
    Serial.println("✓ Audio buffers allocated in heap");
#endif
  }

  if (!adcBuffer || !audioFloat)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("❌ Failed to allocate audio buffers! Halting.");
#endif
    while (1)
    {
      delay(1000);
    }
  }

  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

#if ENABLE_SERIAL_DEBUG
  Serial.println("\n╔═══════════════════════════════════════╗");
  Serial.println("║  ESP32-CAM Audio Streamer (HW-484)   ║");
  Serial.println("║  Analog Microphone → Flask Server    ║");
  Serial.println("╚═══════════════════════════════════════╝\n");
  Serial.printf("Device ID: %s\n", DEVICE_ID);
  Serial.printf("Location: %s\n", DEVICE_LOCATION);
  Serial.printf("Server: %s\n", SERVER_URL);
  Serial.printf("PSRAM: %s\n", psramFound() ? "Available" : "Not found");
  Serial.printf("Free Heap: %d KB\n", ESP.getFreeHeap() / 1024);
  if (psramFound()) {
    Serial.printf("Free PSRAM: %d KB\n\n", ESP.getFreePsram() / 1024);
  }
#endif

  // Connect to WiFi
  blinkLED(5, 200); // Startup blink pattern
  connectToWiFi();

  // Initialize ADC for analog microphone
  if (!initADC())
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("❌ ADC initialization failed! Halting.");
#endif
    while (1)
    {
      blinkLED(3, 500);
      delay(2000);
    }
  }

// Perform VAD calibration by sending silent samples to server
#if ENABLE_SERIAL_DEBUG
  Serial.println("\n🎙️  CALIBRATION PHASE");
  Serial.println("┌─────────────────────────────────────┐");
  Serial.println("│  Please remain SILENT for 4 sec    │");
  Serial.println("│  Sending calibration samples...    │");
  Serial.println("└─────────────────────────────────────┘");
#endif

  if (!performCalibration())
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("❌ Calibration failed! Continuing anyway...");
#endif
    blinkLED(5, 100);
  }
  else
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("✅ Calibration completed successfully!");
#endif
    blinkLED(3, 100);
  }

#if ENABLE_SERIAL_DEBUG
  Serial.println("\n✅ All systems ready!");
  Serial.println("\n┌─────────────────────────────────────┐");
  Serial.println("│    AUDIO STREAMING STARTED          │");
  Serial.println("└─────────────────────────────────────┘\n");
#endif

  delay(500);
}

void loop()
{
  unsigned long loopStart = millis();

  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("⚠️  WiFi disconnected, reconnecting...");
#endif
    connectToWiFi();
  }

  // Collect 1 second of audio samples
#if ENABLE_SERIAL_DEBUG && (totalTransmissions % 10 == 0)
  Serial.println("📊 Capturing audio...");
#endif

  unsigned long captureStart = micros();
  int zeroReadings = 0;
  
  for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
  {
    unsigned long sampleStart = micros();

    // Read ADC value (0-4095 for 12-bit)
    // Enhanced retry mechanism for ADC2/WiFi conflict
    uint16_t adcValue = 0;
    bool validReading = false;
    
    for (int retry = 0; retry < 5; retry++)
    {
      adcValue = analogRead(MIC_ANALOG_PIN);
      
      // Consider reading valid if it's not 0 or 4095 (saturation)
      if (adcValue > 0 && adcValue < 4095) {
        validReading = true;
        break;
      }
      
      // Even if we get 0, it might be legitimate silence, so accept after retries
      if (retry >= 3) {
        validReading = true;
        break;
      }
      
      delayMicroseconds(2); // Slightly longer delay between retries
    }
    
    adcBuffer[i] = adcValue;
    if (adcValue == 0) zeroReadings++;

    // Wait for next sample period
    while (micros() - sampleStart < SAMPLING_PERIOD_US)
    {
      // Busy wait for precise timing
    }
  }
  unsigned long captureTime = (micros() - captureStart) / 1000;

  // Convert ADC values to normalized float [-1, 1]
  // ADC range: 0-4095, center at ~2048
  // Normalize: (value - 2048) / 2048.0
  for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
  {
    audioFloat[i] = (adcBuffer[i] - 2048.0f) / 2048.0f;
  }

#if ENABLE_TEST_TONE
  // If we're getting mostly zeros, generate a test tone for debugging
  if (zeroReadings > AUDIO_BUFFER_SIZE * 0.8f) {
    Serial.println("   🔧 Generating test tone (microphone issue detected)");
    float amplitude = 0.3f; // 30% amplitude
    float freq = TEST_TONE_FREQ; // Test tone frequency
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
      float t = (float)i / SAMPLE_RATE;
      audioFloat[i] = amplitude * sin(2.0f * PI * freq * t);
    }
  }
#endif

#if ENABLE_SERIAL_DEBUG && (totalTransmissions % 10 == 0)
  // Debug: Check buffer validity and sample some values
  Serial.printf("   Buffer check: adcBuffer=%p, audioFloat=%p\n", adcBuffer, audioFloat);
  Serial.printf("   Sample ADC values: %d, %d, %d\n", adcBuffer[0], adcBuffer[100], adcBuffer[1000]);
  Serial.printf("   Sample float values: %.3f, %.3f, %.3f\n", audioFloat[0], audioFloat[100], audioFloat[1000]);
  Serial.printf("   Zero readings: %d/%d (%.1f%%)\n", zeroReadings, AUDIO_BUFFER_SIZE, 
                (float)zeroReadings * 100.0f / AUDIO_BUFFER_SIZE);
  Serial.printf("   Free heap: %d KB\n", ESP.getFreeHeap() / 1024);
  
  if (zeroReadings > AUDIO_BUFFER_SIZE * 0.9f) {
    Serial.println("   ❌ ERROR: >90% zero readings - check microphone connection!");
    Serial.println("      Verify wiring: A0→GPIO12, +→3.3V, G→GND");
  }
#endif

// Send to Flask server
#if ENABLE_LED_FEEDBACK
  digitalWrite(LED_PIN, HIGH);
#endif

  unsigned long transmitStart = millis();
  bool success = sendAudioToServer(audioFloat, AUDIO_BUFFER_SIZE);
  unsigned long transmitTime = millis() - transmitStart;

#if ENABLE_LED_FEEDBACK
  digitalWrite(LED_PIN, LOW);
#endif

  totalTransmissions++;
  if (success)
  {
    successfulTransmissions++;
  }
  else
  {
    transmissionErrors++;
  }

// Periodic status report
#if ENABLE_SERIAL_DEBUG
  if (totalTransmissions % 10 == 0)
  {
    printStatusReport(captureTime, transmitTime);
  }
#endif

  // Maintain ~1 second loop time
  unsigned long loopTime = millis() - loopStart;
  if (loopTime < SEND_INTERVAL_MS)
  {
    delay(SEND_INTERVAL_MS - loopTime);
  }
}

bool sendAudioToServer(float *audioData, int length)
{
  if (WiFi.status() != WL_CONNECTED)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("❌ Cannot send: WiFi not connected");
#endif
    return false;
  }

  // Validate input parameters
  if (!audioData || length <= 0)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ Invalid audio data: ptr=%p, length=%d\n", audioData, length);
#endif
    return false;
  }

  // Convert float array to bytes for base64 encoding
  size_t byteLength = length * sizeof(float);

  // Validate audio data before encoding
  bool hasValidData = false;
  for (int i = 0; i < min(10, length); i++) {
    if (audioData[i] != 0.0f) {
      hasValidData = true;
      break;
    }
  }

#if ENABLE_SERIAL_DEBUG
  if (!hasValidData) {
    Serial.println("⚠️  Warning: Audio data appears to be all zeros");
  }
#endif

  // Base64 encode the audio data using ESP32's built-in mbedtls
#if ENABLE_SERIAL_DEBUG
  Serial.printf("   Encoding %d bytes to base64...\n", byteLength);
#endif

  // Calculate required output buffer size
  size_t outputLen = 0;
  int ret = mbedtls_base64_encode(NULL, 0, &outputLen, (uint8_t*)audioData, byteLength);
  
  if (ret != MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL) {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ Base64 size calculation failed: %d\n", ret);
#endif
    return false;
  }

  // Allocate output buffer
  uint8_t* base64Buffer = (uint8_t*)malloc(outputLen + 1);
  if (!base64Buffer) {
#if ENABLE_SERIAL_DEBUG
    Serial.println("❌ Failed to allocate base64 buffer");
#endif
    return false;
  }

  // Perform the actual encoding
  ret = mbedtls_base64_encode(base64Buffer, outputLen, &outputLen, (uint8_t*)audioData, byteLength);
  
  if (ret != 0) {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ Base64 encoding failed: %d\n", ret);
#endif
    free(base64Buffer);
    return false;
  }

  // Null terminate and create String
  base64Buffer[outputLen] = '\0';
  String base64Audio = String((char*)base64Buffer);
  free(base64Buffer);

#if ENABLE_SERIAL_DEBUG
  Serial.printf("   Base64 encoded: %d characters (mbedtls)\n", base64Audio.length());
#endif

  // Create JSON payload - need large buffer for base64 encoded audio
  // 16000 floats = 64000 bytes → ~85KB base64 → need DynamicJsonDocument
  DynamicJsonDocument doc(200000); // ~90KB to hold base64 audio + metadata
  doc["audio"] = base64Audio;
  doc["sample_rate"] = SAMPLE_RATE;
  doc["device_id"] = DEVICE_ID;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Test basic connectivity first
  WiFiClient testClient;
  Serial.printf("   Testing connection to %s...\n", "4.240.35.54");
  if (!testClient.connect("4.240.35.54", 6002)) {
    Serial.printf("   ❌ Cannot connect to server (network issue)\n");
    Serial.printf("   Local IP: %s, Gateway: %s\n", 
                  WiFi.localIP().toString().c_str(), 
                  WiFi.gatewayIP().toString().c_str());
    return false;
  }
  testClient.stop();
  Serial.printf("   ✅ Basic TCP connection successful\n");

  // Send HTTP POST request
  http.begin(wifiClient, SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(15000); // Increase timeout for large payload

  int httpResponseCode = http.POST(jsonPayload);

  bool success = false;
  if (httpResponseCode > 0)
  {
#if ENABLE_SERIAL_DEBUG
    String response = http.getString();
    if (httpResponseCode == 200)
    {
      Serial.println("✅ Audio sent successfully");

      // Parse response to show prediction
      StaticJsonDocument<512> responseDoc;
      DeserializationError error = deserializeJson(responseDoc, response);

      if (!error)
      {
        const char *predictedClass = responseDoc["predicted_class"];
        float confidence = responseDoc["confidence"];
        Serial.printf("   → Prediction: %s (%.2f%%)\n",
                      predictedClass, confidence * 100);
      }
      success = true;
    }
    else
    {
      Serial.printf("⚠️  HTTP Error: %d\n", httpResponseCode);
      Serial.println("   Response: " + response);
    }
#else
    success = (httpResponseCode == 200);
#endif
  }
  else
  {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ HTTP Request failed: %s\n", http.errorToString(httpResponseCode).c_str());
#endif
  }

  http.end();
  return success;
}

void connectToWiFi()
{
#if ENABLE_SERIAL_DEBUG
  Serial.println("🌐 Connecting to WiFi...");
  Serial.printf("   SSID: %s\n", WIFI_SSID);
#endif

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < WIFI_TIMEOUT_MS)
  {
    delay(500);
#if ENABLE_SERIAL_DEBUG
    Serial.print(".");
#endif
  }

  if (WiFi.status() == WL_CONNECTED)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("\n✅ WiFi connected!");
    Serial.printf("   IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("   Gateway: %s\n", WiFi.gatewayIP().toString().c_str());
    Serial.printf("   DNS: %s\n", WiFi.dnsIP().toString().c_str());
    Serial.printf("   Subnet: %s\n", WiFi.subnetMask().toString().c_str());
    Serial.printf("   RSSI: %d dBm\n", WiFi.RSSI());
    
    // Test internet connectivity
    Serial.println("\n🌐 Testing internet connectivity...");
    WiFiClient testClient;
    if (testClient.connect("8.8.8.8", 53)) {
      Serial.println("   ✅ Internet connectivity: OK");
      testClient.stop();
    } else {
      Serial.println("   ❌ Internet connectivity: FAILED");
    }
    
    // Test server connectivity
    Serial.printf("   Testing server %s:%d...\n", "4.240.35.54", 6002);
    if (testClient.connect("4.240.35.54", 6002)) {
      Serial.println("   ✅ Server connectivity: OK");
      testClient.stop();
    } else {
      Serial.println("   ❌ Server connectivity: FAILED");
      Serial.println("   💡 Server may be down or network route blocked");
    }
#endif
    blinkLED(3, 100);
  }
  else
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("\n❌ WiFi connection failed!");
#endif
    blinkLED(10, 100);
  }
}

bool initADC()
{
#if ENABLE_SERIAL_DEBUG
  Serial.println("🎤 Initializing ADC for HW-484 microphone (ESP32-CAM)...");
#endif

  // Configure ADC - try to optimize for ADC2/WiFi coexistence
  analogReadResolution(ADC_RESOLUTION); // 12-bit resolution
  analogSetAttenuation(ADC_11db);       // Full range 0-3.3V
  // Note: analogSetCycles/analogSetSamples not available in all ESP32 core versions

  // Set ADC pin mode
  pinMode(MIC_ANALOG_PIN, INPUT);

  // Test read multiple times (ADC2 may have issues with WiFi active)
  uint16_t testValue = 0;
  int validReadings = 0;
  uint32_t readingSum = 0;
  
  for (int i = 0; i < 10; i++) {
    uint16_t reading = analogRead(MIC_ANALOG_PIN);
    if (reading > 0) {
      validReadings++;
      readingSum += reading;
    }
    delay(10);
  }
  
  if (validReadings > 0) {
    testValue = readingSum / validReadings;
  }

#if ENABLE_SERIAL_DEBUG
  Serial.println("✅ ADC initialized successfully");
  Serial.printf("   Pin: GPIO%d (ADC2_CH5) - ESP32-CAM\n", MIC_ANALOG_PIN);
  Serial.printf("   Resolution: %d-bit (0-%d)\n", ADC_RESOLUTION, (1 << ADC_RESOLUTION) - 1);
  Serial.printf("   Sample Rate: %d Hz\n", SAMPLE_RATE);
  Serial.printf("   Test Reading: %d (avg of %d valid readings)\n", testValue, validReadings);
  
  if (validReadings == 0) {
    Serial.println("   ❌ WARNING: No valid ADC readings! Check microphone wiring:");
    Serial.println("      - HW-484 A0 pin → ESP32 GPIO12");
    Serial.println("      - HW-484 + pin → ESP32 3.3V");
    Serial.println("      - HW-484 G pin → ESP32 GND");
    Serial.println("   ⚠️  ADC2/WiFi conflict may affect readings");
    return false;
  } else if (validReadings < 5) {
    Serial.printf("   ⚠️  Only %d/10 ADC readings valid - check connections\n", validReadings);
  }
  
  Serial.println("   ⚠️  Note: ADC2 performance may vary with WiFi activity");
#endif

  return validReadings > 0;
}

void printStatusReport(unsigned long captureMs, unsigned long transmitMs)
{
  float successRate = (successfulTransmissions * 100.0f) / totalTransmissions;
  float errorRate = (transmissionErrors * 100.0f) / totalTransmissions;

  Serial.println("\n📊 SYSTEM STATUS REPORT");
  Serial.printf("Total Transmissions:   %lu\n", totalTransmissions);
  Serial.printf("Successful:            %.1f%%\n", successRate);
  Serial.printf("Errors:                %.1f%%\n", errorRate);
  Serial.printf("Capture Time:          %lu ms\n", captureMs);
  Serial.printf("Transmit Time:         %lu ms\n", transmitMs);
  Serial.printf("Free Heap:             %d KB\n", ESP.getFreeHeap() / 1024);
  if (psramFound())
  {
    Serial.printf("Free PSRAM:            %d KB\n", ESP.getFreePsram() / 1024);
  }
  Serial.printf("WiFi RSSI:             %d dBm\n", WiFi.RSSI());
  Serial.printf("Local IP:              %s\n", WiFi.localIP().toString().c_str());
  Serial.println();
}

bool performCalibration()
{
  // Skip calibration if disabled
  if (CALIBRATION_SAMPLES == 0) {
#if ENABLE_SERIAL_DEBUG
    Serial.println("📊 Calibration disabled (CALIBRATION_SAMPLES = 0)");
#endif
    return true;
  }

#if ENABLE_SERIAL_DEBUG
  Serial.println("📊 Starting calibration...");
#endif

  bool allSuccess = true;

  for (int sample = 0; sample < CALIBRATION_SAMPLES; sample++)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("   Collecting sample %d/%d...\n", sample + 1, CALIBRATION_SAMPLES);
#endif

    // Collect shorter calibration sample (0.5 seconds)
    int calibrationSamples = (SAMPLE_RATE * CALIBRATION_DURATION_MS) / 1000;
    
    for (int i = 0; i < calibrationSamples; i++)
    {
      unsigned long sampleStart = micros();
      
      // ADC2 retry mechanism for ESP32-CAM stability
      uint16_t adcValue = 0;
      for (int retry = 0; retry < 3; retry++)
      {
        adcValue = analogRead(MIC_ANALOG_PIN);
        if (adcValue != 0 || retry == 2) break;
        delayMicroseconds(1);
      }
      adcBuffer[i] = adcValue;

      while (micros() - sampleStart < SAMPLING_PERIOD_US)
      {
        // Precise timing
      }
    }

    // Convert to normalized float
    for (int i = 0; i < calibrationSamples; i++)
    {
      audioFloat[i] = (adcBuffer[i] - 2048.0f) / 2048.0f;
    }

    // Send calibration sample to server with retry
    bool success = false;
    for (int retry = 0; retry < 2 && !success; retry++)
    {
      if (retry > 0)
      {
#if ENABLE_SERIAL_DEBUG
        Serial.printf("   🔄 Retrying sample %d (attempt %d)...\n", sample + 1, retry + 1);
#endif
        delay(1000); // Wait before retry
      }
      success = sendCalibrationSample(audioFloat, calibrationSamples);
    }

    if (!success)
    {
      allSuccess = false;
#if ENABLE_SERIAL_DEBUG
      Serial.printf("   ❌ Sample %d failed after retries\n", sample + 1);
#endif
    }
    else
    {
#if ENABLE_SERIAL_DEBUG
      Serial.printf("   ✅ Sample %d sent successfully\n", sample + 1);
#endif
    }

    // Small delay between samples
    delay(200);
  }

  return allSuccess;
}

bool sendCalibrationSample(float *audioData, int length)
{
  if (WiFi.status() != WL_CONNECTED)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.println("❌ Cannot calibrate: WiFi not connected");
#endif
    return false;
  }

  // Convert float array to bytes for base64 encoding
  size_t byteLength = length * sizeof(float);

  // Base64 encode the audio data using ESP32's built-in mbedtls
  size_t outputLen = 0;
  int ret = mbedtls_base64_encode(NULL, 0, &outputLen, (uint8_t*)audioData, byteLength);
  
  if (ret != MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL) {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ Calibration base64 size calculation failed: %d\n", ret);
#endif
    return false;
  }

  uint8_t* base64Buffer = (uint8_t*)malloc(outputLen + 1);
  if (!base64Buffer) {
#if ENABLE_SERIAL_DEBUG
    Serial.println("❌ Failed to allocate calibration base64 buffer");
#endif
    return false;
  }

  ret = mbedtls_base64_encode(base64Buffer, outputLen, &outputLen, (uint8_t*)audioData, byteLength);
  
  if (ret != 0) {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ Calibration base64 encoding failed: %d\n", ret);
#endif
    free(base64Buffer);
    return false;
  }

  base64Buffer[outputLen] = '\0';
  String base64Audio = String((char*)base64Buffer);
  free(base64Buffer);

  // Create JSON payload - need large buffer for base64 encoded audio
  DynamicJsonDocument doc(90000); // ~90KB to hold base64 audio + metadata
  doc["audio"] = base64Audio;
  doc["sample_rate"] = SAMPLE_RATE;
  doc["device_id"] = DEVICE_ID;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Send HTTP POST request to calibration endpoint
  http.begin(wifiClient, CALIBRATE_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(15000); // Increase timeout for large payload

  int httpResponseCode = http.POST(jsonPayload);

  bool success = false;
  if (httpResponseCode > 0)
  {
    String response = http.getString();

    if (httpResponseCode == 200)
    {
#if ENABLE_SERIAL_DEBUG
      // Parse response to show progress
      StaticJsonDocument<256> responseDoc;
      DeserializationError error = deserializeJson(responseDoc, response);

      if (!error)
      {
        const char *status = responseDoc["status"];
        float progress = responseDoc["progress"];
        Serial.printf("      Progress: %.1f%% - %s\n", progress, status);
      }
#endif
      success = true;
    }
    else
    {
#if ENABLE_SERIAL_DEBUG
      Serial.printf("⚠️  Calibration HTTP Error: %d\n", httpResponseCode);
      if (response.length() > 0) {
        Serial.printf("      Server Response: %s\n", response.c_str());
      }
#endif
    }
  }
  else
  {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ Calibration request failed: %s\n", http.errorToString(httpResponseCode).c_str());
#endif
  }

  http.end();
  return success;
}

void blinkLED(int times, int delayMs)
{
#if ENABLE_LED_FEEDBACK
  for (int i = 0; i < times; i++)
  {
    digitalWrite(LED_PIN, HIGH);
    delay(delayMs);
    digitalWrite(LED_PIN, LOW);
    delay(delayMs);
  }
#endif
}