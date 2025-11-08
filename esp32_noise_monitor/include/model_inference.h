/*
 * TensorFlow Lite Model Inference for ESP32-CAM
 * =============================================
 * 
 * Wrapper for TensorFlow Lite Micro inference with INT8 quantization support.
 * Handles model loading, input preprocessing, and output postprocessing.
 */

#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <Arduino.h>
#include "config.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

class ModelInference {
private:
    // TensorFlow Lite components
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::AllOpsResolver resolver;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    
    // Memory arena for TensorFlow Lite
    uint8_t* tensor_arena;
    
    // Input/output tensors
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    
    // Model metadata
    bool model_loaded;
    int input_size;
    int output_size;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    
    // Performance monitoring
    struct InferenceStats {
        unsigned long preprocessing_time_us;
        unsigned long inference_time_us;
        unsigned long postprocessing_time_us;
        unsigned long total_time_us;
        int inference_count;
    };
    
    InferenceStats stats;
    InferenceStats running_stats;
    
    // Helper functions
    bool setup_tensors();
    void quantize_input(float* input_data, int8_t* quantized_data, int size);
    void dequantize_output(int8_t* quantized_data, float* output_data, int size);
    
public:
    ModelInference();
    ~ModelInference();
    
    // Initialization
    bool init(const unsigned char* model_data, size_t model_size);
    bool load_model_from_spiffs(const char* model_path);
    
    // Inference
    struct InferenceResult {
        int predicted_class;
        float confidence;
        float class_probabilities[NUM_CLASSES];
        unsigned long inference_time_us;
        bool success;
    };
    
    InferenceResult predict(float* features);
    
    // Model information
    bool is_loaded() const { return model_loaded; }
    int get_input_size() const { return input_size; }
    int get_output_size() const { return output_size; }
    
    // Performance monitoring
    InferenceStats get_stats() const { return stats; }
    InferenceStats get_running_stats() const { return running_stats; }
    void reset_stats();
    
    // Memory usage
    size_t get_arena_size() const { return TFLITE_ARENA_SIZE; }
    size_t get_arena_used() const;
    
    // Debug and diagnostics
    void print_model_info() const;
    void print_tensor_info() const;
};

#endif // MODEL_INFERENCE_H