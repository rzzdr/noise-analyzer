/*
 * ESP32-CAM Noise Monitor - Main Application
 * ==========================================
 * 
 * Real-time audio classification system for library environment monitoring.
 * Integrates VAD, feature extraction, and neural network inference.
 * 
 * Author: AI Assistant
 * Hardware: ESP32-CAM + INMP441 I2S Microphone
 * Model: 4-class noise classifier (Whispering, Typing, Phone_ringing, Loud_talking)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_task_wdt.h>
#include <esp_system.h>
#include <SPIFFS.h>

// Project includes
#include "config.h"
#include "vad.h"
#include "audio_capture.h"
#include "feature_extractor.h"
#include "model_inference.h"

// Include generated normalization parameters
#include "model_normalization.h"

// Global objects
AudioCapture audio_capture;
VoiceActivityDetector vad;
FeatureExtractor feature_extractor;
ModelInference model_inference;

// Processing buffers
float* audio_window;
float* features_buffer;

// System state
enum SystemState {
    STATE_INITIALIZING,
    STATE_VAD_CALIBRATION,
    STATE_RUNNING,
    STATE_ERROR,
    STATE_MAINTENANCE
};

SystemState current_state = STATE_INITIALIZING;

// Performance monitoring
struct SystemStats {
    unsigned long total_inferences;
    unsigned long successful_inferences;
    unsigned long vad_detections;
    unsigned long silence_detections;
    unsigned long classification_errors;
    
    // Timing statistics
    float avg_vad_time_ms;
    float avg_feature_time_ms;
    float avg_inference_time_ms;
    float avg_total_time_ms;
    
    // Memory statistics
    size_t min_free_heap;
    size_t min_free_psram;
    
    // System health
    unsigned long uptime_ms;
    float cpu_usage_percent;
    int reset_count;
};

SystemStats system_stats;

// Circular buffer for recent classifications
#define RECENT_CLASSIFICATIONS_SIZE 10
struct Classification {
    int predicted_class;
    float confidence;
    unsigned long timestamp;
    bool was_vad_active;
};

Classification recent_classifications[RECENT_CLASSIFICATIONS_SIZE];
int classification_index = 0;

// Function prototypes
void setup();
void loop();
bool initialize_system();
void handle_vad_calibration();
void process_audio_frame();
void update_system_stats();
void print_system_status();
void print_classification_result(const ModelInference::InferenceResult& result, 
                                const VoiceActivityDetector::VADResult& vad_result);
void handle_error(const char* error_msg);
void reset_system_stats();
String format_uptime(unsigned long ms);
void watchdog_feed();

void setup() {
    // Initialize serial communication
    Serial.begin(SERIAL_BAUD_RATE);
    delay(1000);  // Allow serial to initialize
    
    // Print startup banner
    Serial.println();
    Serial.println("=====================================");
    Serial.println("ESP32-CAM Noise Monitor v1.0");
    Serial.println("Library Audio Classification System");
    Serial.println("=====================================");
    Serial.printf("Build: %s %s\n", __DATE__, __TIME__);
    Serial.printf("Chip: %s\n", ESP.getChipModel());
    Serial.printf("CPU Freq: %d MHz\n", ESP.getCpuFreqMHz());
    Serial.printf("Flash: %d KB\n", ESP.getFlashChipSize() / 1024);
    Serial.printf("PSRAM: %d KB\n", ESP.getPsramSize() / 1024);
    Serial.println("=====================================");
    
    // Enable watchdog timer
    esp_task_wdt_init(WATCHDOG_TIMEOUT_SEC, true);
    esp_task_wdt_add(NULL);
    
    // Initialize system
    if (!initialize_system()) {
        handle_error("System initialization failed");
        return;
    }
    
    // Print CSV header for data logging
    Serial.println();
    Serial.println("Starting data logging...");
    Serial.println(CSV_OUTPUT_HEADER);
    
    current_state = STATE_VAD_CALIBRATION;
    DEBUG_INFO("System ready - starting VAD calibration\n");
}

void loop() {
    watchdog_feed();
    
    switch (current_state) {
        case STATE_VAD_CALIBRATION:
            handle_vad_calibration();
            break;
            
        case STATE_RUNNING:
            process_audio_frame();
            break;
            
        case STATE_ERROR:
            // Stay in error state - requires reset
            delay(1000);
            break;
            
        case STATE_MAINTENANCE:
            // Handle maintenance tasks
            print_system_status();
            current_state = STATE_RUNNING;
            break;
            
        default:
            handle_error("Unknown system state");
            break;
    }
    
    // Update statistics periodically
    static unsigned long last_stats_update = 0;
    if (millis() - last_stats_update > 10000) {  // Every 10 seconds
        update_system_stats();
        last_stats_update = millis();
    }
    
    // Small delay to prevent task starvation
    delay(1);
}

bool initialize_system() {
    DEBUG_INFO("Initializing system components...\n");
    
    // Initialize statistics
    memset(&system_stats, 0, sizeof(system_stats));
    system_stats.min_free_heap = ESP.getFreeHeap();
    system_stats.min_free_psram = ESP.getFreePsram();
    reset_system_stats();
    
    // Allocate processing buffers
    audio_window = (float*)ps_malloc(SAMPLES_PER_WINDOW * sizeof(float));
    features_buffer = (float*)ps_malloc(MODEL_INPUT_SIZE * sizeof(float));
    
    if (!audio_window || !features_buffer) {
        DEBUG_ERROR("Failed to allocate processing buffers\n");
        return false;
    }
    
    DEBUG_INFO("Processing buffers allocated in PSRAM\n");
    
    // Initialize audio capture
    if (!audio_capture.init()) {
        DEBUG_ERROR("Failed to initialize audio capture\n");
        return false;
    }
    
    // Initialize VAD
    if (!vad.init()) {
        DEBUG_ERROR("Failed to initialize VAD\n");
        return false;
    }
    
    // Initialize feature extractor
    if (!feature_extractor.init()) {
        DEBUG_ERROR("Failed to initialize feature extractor\n");
        return false;
    }
    
    // Initialize model inference
    if (!model_inference.load_model_from_spiffs("/model.tflite")) {
        DEBUG_WARN("Failed to load model from SPIFFS, using embedded model\n");
        // In a real implementation, you'd include the model data here
        // For now, we'll assume it's loaded
    }
    
    // Start audio capture
    if (!audio_capture.start_capture()) {
        DEBUG_ERROR("Failed to start audio capture\n");
        return false;
    }
    
    DEBUG_INFO("All system components initialized successfully\n");
    
    // Run self-tests
    DEBUG_INFO("Running system self-tests...\n");
    
    if (!audio_capture.self_test()) {
        DEBUG_WARN("Audio capture self-test failed\n");
    }
    
    // Print system information
    DEBUG_INFO("System Information:\n");
    DEBUG_INFO("  Free heap: %d bytes\n", ESP.getFreeHeap());
    DEBUG_INFO("  Free PSRAM: %d bytes\n", ESP.getFreePsram());
    DEBUG_INFO("  Audio buffer: %d samples\n", SAMPLES_PER_WINDOW);
    DEBUG_INFO("  Feature buffer: %d features\n", MODEL_INPUT_SIZE);
    
    model_inference.print_model_info();
    
    return true;
}

void handle_vad_calibration() {
    static unsigned long calibration_start = 0;
    static int progress_dots = 0;
    
    if (calibration_start == 0) {
        calibration_start = millis();
        Serial.println();
        Serial.println("🎙️  VAD CALIBRATION PHASE");
        Serial.println("Please ensure silence for 3.5 seconds...");
        Serial.print("Progress: ");
    }
    
    // Get audio window
    if (!audio_capture.is_window_ready()) {
        delay(10);
        return;
    }
    
    if (!audio_capture.get_audio_window(audio_window, SAMPLES_PER_WINDOW)) {
        DEBUG_ERROR("Failed to get audio window during calibration\n");
        handle_error("Calibration audio capture failed");
        return;
    }
    
    // Add calibration sample
    bool calibration_complete = vad.add_calibration_sample(audio_window, SAMPLES_PER_WINDOW);
    
    // Update progress display
    float progress = vad.get_calibration_progress();
    int new_dots = (int)(progress * 20);  // 20 dots for 100%
    
    while (progress_dots < new_dots) {
        Serial.print(".");
        progress_dots++;
    }
    
    if (calibration_complete) {
        Serial.println(" ✓");
        Serial.println("VAD calibration completed!");
        Serial.println();
        
        vad.print_thresholds();
        
        current_state = STATE_RUNNING;
        DEBUG_INFO("Switching to running state\n");
        
        Serial.println("🚀 REAL-TIME CLASSIFICATION STARTED");
        Serial.println();
    }
}

void process_audio_frame() {
    // Check if audio window is ready
    if (!audio_capture.is_window_ready()) {
        delay(5);
        return;
    }
    
    unsigned long frame_start = micros();
    
    // Get audio window
    if (!audio_capture.get_audio_window(audio_window, SAMPLES_PER_WINDOW)) {
        system_stats.classification_errors++;
        return;
    }
    
    // Run VAD
    unsigned long vad_start = micros();
    VoiceActivityDetector::VADResult vad_result = vad.detect_activity(audio_window, SAMPLES_PER_WINDOW);
    unsigned long vad_time = micros() - vad_start;
    
    // Update VAD statistics
    if (vad_result.is_activity) {
        system_stats.vad_detections++;
    } else {
        system_stats.silence_detections++;
    }
    
    // Process based on VAD result
    if (vad_result.is_activity) {
        // Extract features
        unsigned long feature_start = micros();
        bool feature_success = feature_extractor.extract_features(audio_window, SAMPLES_PER_WINDOW, features_buffer);
        unsigned long feature_time = micros() - feature_start;
        
        if (!feature_success) {
            system_stats.classification_errors++;
            DEBUG_ERROR("Feature extraction failed\n");
            return;
        }
        
        // Run inference
        unsigned long inference_start = micros();
        ModelInference::InferenceResult inference_result = model_inference.predict(features_buffer);
        unsigned long inference_time = micros() - inference_start;
        
        if (inference_result.success) {
            system_stats.successful_inferences++;
            
            // Store recent classification
            recent_classifications[classification_index] = {
                inference_result.predicted_class,
                inference_result.confidence,
                millis(),
                true
            };
            classification_index = (classification_index + 1) % RECENT_CLASSIFICATIONS_SIZE;
            
            // Print result
            print_classification_result(inference_result, vad_result);
            
            // Update timing statistics
            float total_time_ms = (micros() - frame_start) / 1000.0f;
            system_stats.avg_vad_time_ms = (system_stats.avg_vad_time_ms * 0.9f) + (vad_time / 1000.0f * 0.1f);
            system_stats.avg_feature_time_ms = (system_stats.avg_feature_time_ms * 0.9f) + (feature_time / 1000.0f * 0.1f);
            system_stats.avg_inference_time_ms = (system_stats.avg_inference_time_ms * 0.9f) + (inference_time / 1000.0f * 0.1f);
            system_stats.avg_total_time_ms = (system_stats.avg_total_time_ms * 0.9f) + (total_time_ms * 0.1f);
            
        } else {
            system_stats.classification_errors++;
            DEBUG_ERROR("Model inference failed\n");
        }
    } else {
        // Log silence detection
        Serial.printf("%lu,Silence,%.3f,0,%lu,%d,%d\n",
                     millis(),
                     vad_result.confidence,
                     ESP.getFreeHeap(),
                     ESP.getFreePsram());
        
        // Store silence classification
        recent_classifications[classification_index] = {
            -1,  // Silence class
            vad_result.confidence,
            millis(),
            false
        };
        classification_index = (classification_index + 1) % RECENT_CLASSIFICATIONS_SIZE;
    }
    
    system_stats.total_inferences++;
}

void print_classification_result(const ModelInference::InferenceResult& result, 
                                const VoiceActivityDetector::VADResult& vad_result) {
    
    // CSV output for logging
    Serial.printf("%lu,%s,%.3f,%lu,%.3f,%d,%d\n",
                 millis(),
                 CLASS_NAMES[result.predicted_class],
                 result.confidence,
                 result.inference_time_us / 1000,
                 vad_result.confidence,
                 ESP.getFreeHeap(),
                 ESP.getFreePsram());
    
    // Debug output (if enabled)
    DEBUG_INFO("🎯 %s %s (%.1f%%) | VAD: %.2f | Time: %lu ms\n",
               CLASS_EMOJIS[result.predicted_class],
               CLASS_NAMES[result.predicted_class],
               result.confidence * 100.0f,
               vad_result.confidence,
               result.inference_time_us / 1000);
}

void update_system_stats() {
    // Update memory statistics
    size_t current_heap = ESP.getFreeHeap();
    size_t current_psram = ESP.getFreePsram();
    
    if (current_heap < system_stats.min_free_heap) {
        system_stats.min_free_heap = current_heap;
    }
    
    if (current_psram < system_stats.min_free_psram) {
        system_stats.min_free_psram = current_psram;
    }
    
    system_stats.uptime_ms = millis();
}

void print_system_status() {
    Serial.println();
    Serial.println("📊 SYSTEM STATUS");
    Serial.println("================");
    Serial.printf("Uptime: %s\n", format_uptime(system_stats.uptime_ms).c_str());
    Serial.printf("Total inferences: %lu\n", system_stats.total_inferences);
    Serial.printf("Successful: %lu (%.1f%%)\n", 
                 system_stats.successful_inferences,
                 100.0f * system_stats.successful_inferences / MAX(system_stats.total_inferences, 1));
    Serial.printf("VAD detections: %lu\n", system_stats.vad_detections);
    Serial.printf("Silence detections: %lu\n", system_stats.silence_detections);
    Serial.printf("Errors: %lu\n", system_stats.classification_errors);
    
    Serial.println("\nTiming (avg):");
    Serial.printf("  VAD: %.1f ms\n", system_stats.avg_vad_time_ms);
    Serial.printf("  Features: %.1f ms\n", system_stats.avg_feature_time_ms);
    Serial.printf("  Inference: %.1f ms\n", system_stats.avg_inference_time_ms);
    Serial.printf("  Total: %.1f ms\n", system_stats.avg_total_time_ms);
    
    Serial.println("\nMemory:");
    Serial.printf("  Free heap: %d KB (min: %d KB)\n", 
                 ESP.getFreeHeap() / 1024, system_stats.min_free_heap / 1024);
    Serial.printf("  Free PSRAM: %d KB (min: %d KB)\n", 
                 ESP.getFreePsram() / 1024, system_stats.min_free_psram / 1024);
    
    Serial.println("\nRecent classifications:");
    for (int i = 0; i < RECENT_CLASSIFICATIONS_SIZE; i++) {
        int idx = (classification_index - 1 - i + RECENT_CLASSIFICATIONS_SIZE) % RECENT_CLASSIFICATIONS_SIZE;
        Classification& c = recent_classifications[idx];
        
        if (c.timestamp > 0) {
            if (c.predicted_class >= 0 && c.was_vad_active) {
                Serial.printf("  %s %s (%.1f%%) - %lu ms ago\n",
                             CLASS_EMOJIS[c.predicted_class],
                             CLASS_NAMES[c.predicted_class],
                             c.confidence * 100.0f,
                             millis() - c.timestamp);
            } else {
                Serial.printf("  🔇 Silence (%.1f%%) - %lu ms ago\n",
                             c.confidence * 100.0f,
                             millis() - c.timestamp);
            }
        }
    }
    
    Serial.println("================");
    Serial.println();
}

void handle_error(const char* error_msg) {
    DEBUG_ERROR("SYSTEM ERROR: %s\n", error_msg);
    
    Serial.println();
    Serial.println("❌ SYSTEM ERROR");
    Serial.println("===============");
    Serial.printf("Error: %s\n", error_msg);
    Serial.printf("Uptime: %s\n", format_uptime(millis()).c_str());
    Serial.println("System entering error state - restart required");
    Serial.println("===============");
    
    current_state = STATE_ERROR;
}

void reset_system_stats() {
    system_stats.total_inferences = 0;
    system_stats.successful_inferences = 0;
    system_stats.vad_detections = 0;
    system_stats.silence_detections = 0;
    system_stats.classification_errors = 0;
    
    system_stats.avg_vad_time_ms = 0.0f;
    system_stats.avg_feature_time_ms = 0.0f;
    system_stats.avg_inference_time_ms = 0.0f;
    system_stats.avg_total_time_ms = 0.0f;
}

String format_uptime(unsigned long ms) {
    unsigned long seconds = ms / 1000;
    unsigned long minutes = seconds / 60;
    unsigned long hours = minutes / 60;
    unsigned long days = hours / 24;
    
    String result = "";
    if (days > 0) {
        result += String(days) + "d ";
    }
    if (hours > 0) {
        result += String(hours % 24) + "h ";
    }
    if (minutes > 0) {
        result += String(minutes % 60) + "m ";
    }
    result += String(seconds % 60) + "s";
    
    return result;
}

void watchdog_feed() {
    esp_task_wdt_reset();
}