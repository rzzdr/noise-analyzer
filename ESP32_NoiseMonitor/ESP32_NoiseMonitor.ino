#include <arduinoFFT.h>

#include <TensorFlowLite_ESP32.h>

/*
 * ESP32-CAM Standalone Library Noise Monitor (WiFi Version)
 * Two-stage audio classification: VAD → 4-class classifier
 * 
 * Hardware: ESP32-CAM + INMP441 I²S Microphone + ESP32-CAM-MB
 * Communication: WiFi (UDP/HTTP) to laptop server
 * 
 * Author: Generated for Library Noise Monitoring
 * Date: 2025
 */

#include <driver/i2s.h>
#include "config.h"
#include "wifi_manager.h"
#include "vad.h"
#include "feature_extractor.h"
#include "model_inference.h"

// Global objects
WiFiManager wifiManager;
VoiceActivityDetector vad;
FeatureExtractor featureExtractor;
ModelInference modelInference;

// Audio buffers - will be allocated in setup() to use PSRAM
int16_t* audioBuffer = nullptr;
float* audioFloat = nullptr;

// Statistics
unsigned long totalInferences = 0;
unsigned long vadDetections = 0;
unsigned long classifierInferences = 0;
unsigned long transmissionErrors = 0;

// LED feedback
unsigned long lastLEDBlink = 0;

void setup() {
  #if ENABLE_SERIAL_DEBUG
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  #endif
  
  // Allocate audio buffers in PSRAM if available
  if (psramFound()) {
    audioBuffer = (int16_t*)ps_malloc(AUDIO_BUFFER_SIZE * sizeof(int16_t));
    audioFloat = (float*)ps_malloc(AUDIO_BUFFER_SIZE * sizeof(float));
    Serial.println("✓ Audio buffers allocated in PSRAM");
  } else {
    audioBuffer = (int16_t*)malloc(AUDIO_BUFFER_SIZE * sizeof(int16_t));
    audioFloat = (float*)malloc(AUDIO_BUFFER_SIZE * sizeof(float));
    Serial.println("✓ Audio buffers allocated in heap");
  }
  
  if (!audioBuffer || !audioFloat) {
    Serial.println("❌ Failed to allocate audio buffers! Halting.");
    while (1) {
      delay(1000);
    }
  }
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  #if ENABLE_SERIAL_DEBUG
  Serial.println("\n╔═══════════════════════════════════════╗");
  Serial.println("║  ESP32-CAM Noise Monitor (WiFi)      ║");
  Serial.println("║  Standalone System with VAD          ║");
  Serial.println("╚═══════════════════════════════════════╝\n");
  Serial.printf("Device ID: %s\n", DEVICE_ID);
  Serial.printf("Location: %s\n\n", DEVICE_LOCATION);
  #endif
  
  // Connect to WiFi
  blinkLED(5, 200);  // Startup blink pattern
  if (!wifiManager.begin()) {
    #if ENABLE_SERIAL_DEBUG
    Serial.println("⚠️  Starting in offline mode...");
    #endif
    blinkLED(10, 100);  // Error pattern
  }
  
  // Initialize I2S microphone
  if (!initI2S()) {
    #if ENABLE_SERIAL_DEBUG
    Serial.println("❌ I2S initialization failed! Halting.");
    #endif
    while (1) { 
      blinkLED(3, 500);
      delay(2000);
    }
  }
  
  // Initialize VAD
  #if ENABLE_SERIAL_DEBUG
  Serial.println("🎙️  Initializing Voice Activity Detector...");
  Serial.println("\n┌─────────────────────────────────────┐");
  Serial.println("│  VAD CALIBRATION - Stay Silent!    │");
  Serial.println("│  Collecting baseline noise (3.5s)  │");
  Serial.println("└─────────────────────────────────────┘");
  #endif
  
  vad.begin(SAMPLE_RATE);
  
  // VAD Calibration Phase
  while (!vad.isCalibrated()) {
    size_t bytesRead;
    i2s_read(I2S_PORT, audioBuffer, sizeof(audioBuffer), &bytesRead, portMAX_DELAY);
    
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
      audioFloat[i] = audioBuffer[i] / 32768.0f;
    }
    
    vad.addCalibrationSample(audioFloat, AUDIO_BUFFER_SIZE);
    
    #if ENABLE_SERIAL_DEBUG
    int progress = vad.getCalibrationProgress();
    if (progress % 25 == 0) {
      Serial.printf("  Calibration: %d%%\n", progress);
    }
    #endif
  }
  
  #if ENABLE_SERIAL_DEBUG
  Serial.println("✅ VAD calibrated successfully!\n");
  #endif
  
  blinkLED(3, 100);  // Calibration complete
  
  // Initialize feature extractor
  #if ENABLE_SERIAL_DEBUG
  Serial.println("🔧 Initializing Feature Extractor...");
  #endif
  featureExtractor.begin();
  
  // Initialize TFLite model
  #if ENABLE_SERIAL_DEBUG
  Serial.println("🧠 Loading TensorFlow Lite Model...");
  #endif
  if (!modelInference.begin()) {
    #if ENABLE_SERIAL_DEBUG
    Serial.println("❌ Model initialization failed! Halting.");
    #endif
    while (1) {
      blinkLED(3, 500);
      delay(2000);
    }
  }
  
  #if ENABLE_SERIAL_DEBUG
  Serial.println("✅ All systems ready!");
  Serial.println("\n┌─────────────────────────────────────┐");
  Serial.println("│    MONITORING STARTED (WiFi Mode)   │");
  Serial.println("└─────────────────────────────────────┘\n");
  #endif
  
  blinkLED(5, 50);  // Ready signal
  delay(500);
}

void loop() {
  unsigned long loopStart = millis();
  
  // Check WiFi connection
  if (!wifiManager.isConnected()) {
    wifiManager.reconnect();
  }
  
  // Read 1 second of audio (16000 samples)
  size_t bytesRead;
  i2s_read(I2S_PORT, audioBuffer, sizeof(audioBuffer), &bytesRead, portMAX_DELAY);
  
  // Convert to float [-1, 1]
  for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
    audioFloat[i] = audioBuffer[i] / 32768.0f;
  }
  
  // STAGE 1: Voice Activity Detection
  unsigned long vadStart = millis();
  bool isActivity = vad.detectActivity(audioFloat, AUDIO_BUFFER_SIZE);
  float vadConfidence = vad.getConfidence();
  unsigned long vadTime = millis() - vadStart;
  
  totalInferences++;
  
  String predictedClass;
  float classConfidence;
  unsigned long inferenceTime = 0;
  
  if (isActivity) {
    // STAGE 2: Noise Classification
    vadDetections++;
    
    #if ENABLE_LED_FEEDBACK
    digitalWrite(LED_PIN, HIGH);  // Turn on LED during activity
    #endif
    
    // Extract features
    unsigned long featureStart = millis();
    float* features = featureExtractor.extractFeatures(audioFloat, AUDIO_BUFFER_SIZE);
    unsigned long featureTime = millis() - featureStart;
    
    if (features != nullptr) {
      // Run TFLite inference
      unsigned long inferStart = millis();
      int classIndex = modelInference.predict(features, &classConfidence);
      inferenceTime = millis() - inferStart;
      
      predictedClass = modelInference.getClassName(classIndex);
      classifierInferences++;
      
      #if ENABLE_SERIAL_DEBUG
      if (classifierInferences % 10 == 0) {
        Serial.printf("⏱️  Timing - VAD: %lums, Features: %lums, Inference: %lums\n", 
                     vadTime, featureTime, inferenceTime);
      }
      #endif
    } else {
      predictedClass = "ERROR";
      classConfidence = 0.0f;
    }
    
    #if ENABLE_LED_FEEDBACK
    digitalWrite(LED_PIN, LOW);
    #endif
  } else {
    // No activity detected
    predictedClass = "Silence";
    classConfidence = vadConfidence;
    inferenceTime = vadTime;
  }
  
  // Prepare JSON data packet
  String jsonData = createDataPacket(
    loopStart,
    predictedClass,
    classConfidence,
    inferenceTime,
    vadConfidence,
    isActivity
  );
  
  // Transmit data
  bool transmitSuccess = false;
  
  #if USE_HTTP_POST
  transmitSuccess = wifiManager.sendHTTP(jsonData.c_str());
  #else
  transmitSuccess = wifiManager.sendUDP(jsonData.c_str());
  #endif
  
  if (!transmitSuccess) {
    transmissionErrors++;
  }
  
  // Periodic status report
  #if ENABLE_SERIAL_DEBUG
  if (totalInferences % 30 == 0) {
    printStatusReport();
  }
  #endif
  
  // Maintain ~1 second loop time
  unsigned long loopTime = millis() - loopStart;
  if (loopTime < 1000) {
    delay(1000 - loopTime);
  }
}

String createDataPacket(unsigned long timestamp, String className, 
                       float confidence, unsigned long inferenceMs,
                       float vadConf, bool activity) {
  // JSON format for easy parsing
  String json = "{";
  json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  json += "\"location\":\"" + String(DEVICE_LOCATION) + "\",";
  json += "\"timestamp\":" + String(timestamp) + ",";
  json += "\"class\":\"" + className + "\",";
  json += "\"confidence\":" + String(confidence, 3) + ",";
  json += "\"inference_ms\":" + String(inferenceMs) + ",";
  json += "\"vad_confidence\":" + String(vadConf, 3) + ",";
  json += "\"is_activity\":" + String(activity ? "true" : "false") + ",";
  json += "\"rssi\":" + String(wifiManager.getRSSI());
  json += "}";
  
  return json;
}

void printStatusReport() {
  float vadRate = (vadDetections * 100.0f) / totalInferences;
  float errorRate = (transmissionErrors * 100.0f) / totalInferences;
  
  // Serial.println("\n" + String("=") * 60);
  Serial.println("📊 SYSTEM STATUS REPORT");
  // Serial.println(String("=") * 60);
  Serial.printf("Total Inferences:      %lu\n", totalInferences);
  Serial.printf("Activity Detected:     %.1f%%\n", vadRate);
  Serial.printf("Classifier Runs:       %lu\n", classifierInferences);
  Serial.printf("Transmission Errors:   %.1f%%\n", errorRate);
  Serial.printf("Free Heap:             %d KB\n", ESP.getFreeHeap() / 1024);
  Serial.printf("Free PSRAM:            %d KB\n", ESP.getFreePsram() / 1024);
  Serial.printf("WiFi RSSI:             %d dBm\n", wifiManager.getRSSI());
  Serial.printf("Local IP:              %s\n", wifiManager.getLocalIP().c_str());
  // Serial.println(String("=") * 60 + "\n");
}

bool initI2S() {
  #if ENABLE_SERIAL_DEBUG
  Serial.println("🎤 Initializing I2S Microphone...");
  #endif
  
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN
  };
  
  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    #if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ I2S driver install failed: %d\n", err);
    #endif
    return false;
  }
  
  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    #if ENABLE_SERIAL_DEBUG
    Serial.printf("❌ I2S pin config failed: %d\n", err);
    #endif
    return false;
  }
  
  #if ENABLE_SERIAL_DEBUG
  Serial.println("✅ I2S initialized successfully");
  Serial.printf("  Sample Rate: %d Hz\n", SAMPLE_RATE);
  Serial.printf("  Pins - SCK:%d, WS:%d, SD:%d\n", I2S_SCK_PIN, I2S_WS_PIN, I2S_SD_PIN);
  #endif
  
  return true;
}

void blinkLED(int times, int delayMs) {
  #if ENABLE_LED_FEEDBACK
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(delayMs);
    digitalWrite(LED_PIN, LOW);
    delay(delayMs);
  }
  #endif
}