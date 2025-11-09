/*
 * ESP32 Audio Streamer for HW-484 Analog Microphone
 * Captures audio from analog microphone and streams to Flask server
 *
 * Hardware: ESP32 + HW-484 Analog Microphone Module
 * Wiring:
 *   - A0 (Analog Out) → GPIO36 (VP/ADC1_CH0)
 *   - G  (Ground)     → GND
 *   - +  (VCC)        → 3.3V
 *   - D0 (Digital)    → Not connected
 *
 * Communication: WiFi (HTTP POST) to Flask server
 *
 * Author: Audio Monitoring System
 * Date: 2025
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <base64.h>
#include "config.h"

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
  Serial.println("║  ESP32 Audio Streamer (HW-484)       ║");
  Serial.println("║  Analog Microphone → Flask Server    ║");
  Serial.println("╚═══════════════════════════════════════╝\n");
  Serial.printf("Device ID: %s\n", DEVICE_ID);
  Serial.printf("Location: %s\n", DEVICE_LOCATION);
  Serial.printf("Server: %s\n\n", SERVER_URL);
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
  for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
  {
    unsigned long sampleStart = micros();

    // Read ADC value (0-4095 for 12-bit)
    adcBuffer[i] = analogRead(MIC_ANALOG_PIN);

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

  // Convert float array to bytes
  uint8_t *audioBytes = (uint8_t *)audioData;
  int byteLength = length * sizeof(float);

  // Base64 encode the audio data
  String base64Audio = base64::encode(audioBytes, byteLength);

  // Create JSON payload
  StaticJsonDocument<200> doc;
  doc["audio"] = base64Audio;
  doc["sample_rate"] = SAMPLE_RATE;
  doc["device_id"] = DEVICE_ID;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Send HTTP POST request
  http.begin(wifiClient, SERVER_URL);
  http.addHeader("Content-Type", "application/json");

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
    Serial.printf("   RSSI: %d dBm\n", WiFi.RSSI());
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
  Serial.println("🎤 Initializing ADC for HW-484 microphone...");
#endif

  // Configure ADC
  analogReadResolution(ADC_RESOLUTION); // 12-bit resolution
  analogSetAttenuation(ADC_11db);       // Full range 0-3.3V

  // Set ADC pin mode
  pinMode(MIC_ANALOG_PIN, INPUT);

  // Test read
  uint16_t testValue = analogRead(MIC_ANALOG_PIN);

#if ENABLE_SERIAL_DEBUG
  Serial.println("✅ ADC initialized successfully");
  Serial.printf("   Pin: GPIO%d (ADC1_CH0)\n", MIC_ANALOG_PIN);
  Serial.printf("   Resolution: %d-bit (0-%d)\n", ADC_RESOLUTION, (1 << ADC_RESOLUTION) - 1);
  Serial.printf("   Sample Rate: %d Hz\n", SAMPLE_RATE);
  Serial.printf("   Test Reading: %d\n", testValue);
#endif

  return true;
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
#if ENABLE_SERIAL_DEBUG
  Serial.println("📊 Starting calibration...");
#endif

  bool allSuccess = true;

  for (int sample = 0; sample < CALIBRATION_SAMPLES; sample++)
  {
#if ENABLE_SERIAL_DEBUG
    Serial.printf("   Collecting sample %d/%d...\n", sample + 1, CALIBRATION_SAMPLES);
#endif

    // Collect 1 second of audio
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
    {
      unsigned long sampleStart = micros();
      adcBuffer[i] = analogRead(MIC_ANALOG_PIN);

      while (micros() - sampleStart < SAMPLING_PERIOD_US)
      {
        // Precise timing
      }
    }

    // Convert to normalized float
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
    {
      audioFloat[i] = (adcBuffer[i] - 2048.0f) / 2048.0f;
    }

    // Send calibration sample to server
    bool success = sendCalibrationSample(audioFloat, AUDIO_BUFFER_SIZE);

    if (!success)
    {
      allSuccess = false;
#if ENABLE_SERIAL_DEBUG
      Serial.printf("   ⚠️  Sample %d failed\n", sample + 1);
#endif
    }
    else
    {
#if ENABLE_SERIAL_DEBUG
      Serial.printf("   ✅ Sample %d sent\n", sample + 1);
#endif
    }

    // Small delay between samples
    delay(100);
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

  // Convert float array to bytes
  uint8_t *audioBytes = (uint8_t *)audioData;
  int byteLength = length * sizeof(float);

  // Base64 encode the audio data
  String base64Audio = base64::encode(audioBytes, byteLength);

  // Create JSON payload
  StaticJsonDocument<200> doc;
  doc["audio"] = base64Audio;
  doc["sample_rate"] = SAMPLE_RATE;
  doc["device_id"] = DEVICE_ID;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Send HTTP POST request to calibration endpoint
  http.begin(wifiClient, CALIBRATE_URL);
  http.addHeader("Content-Type", "application/json");

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