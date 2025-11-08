/*
 * TensorFlow Lite Model Inference Implementation
 * =============================================
 */

#include "model_inference.h"

// Model data (will be included from generated file)
extern const unsigned char noise_classifier_model_data[];
extern const size_t noise_classifier_model_data_len;

ModelInference::ModelInference() 
    : model(nullptr), interpreter(nullptr), tensor_arena(nullptr),
      input_tensor(nullptr), output_tensor(nullptr), model_loaded(false) {
    
    // Initialize stats
    memset(&stats, 0, sizeof(stats));
    memset(&running_stats, 0, sizeof(running_stats));
    
    input_size = 0;
    output_size = 0;
    input_scale = 1.0f;
    input_zero_point = 0;
    output_scale = 1.0f;
    output_zero_point = 0;
}

ModelInference::~ModelInference() {
    if (interpreter) {
        delete interpreter;
    }
    if (tensor_arena) {
        free(tensor_arena);
    }
}

bool ModelInference::init(const unsigned char* model_data, size_t model_size) {
    DEBUG_INFO("Initializing TensorFlow Lite model...\n");
    
    // Allocate tensor arena
    tensor_arena = (uint8_t*)ps_malloc(TFLITE_ARENA_SIZE);  // Use PSRAM
    if (!tensor_arena) {
        DEBUG_ERROR("Failed to allocate tensor arena (%d bytes)\n", TFLITE_ARENA_SIZE);
        return false;
    }
    
    DEBUG_INFO("Tensor arena allocated: %d bytes in PSRAM\n", TFLITE_ARENA_SIZE);
    
    // Load model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        DEBUG_ERROR("Model version %d not supported (expected %d)\n", 
                   model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    DEBUG_INFO("Model loaded, version: %d\n", model->version());
    
    // Create interpreter
    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, TFLITE_ARENA_SIZE, &micro_error_reporter);
    
    if (!interpreter) {
        DEBUG_ERROR("Failed to create interpreter\n");
        return false;
    }
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        DEBUG_ERROR("Failed to allocate tensors\n");
        return false;
    }
    
    DEBUG_INFO("Tensors allocated successfully\n");
    
    // Setup input/output tensors
    if (!setup_tensors()) {
        DEBUG_ERROR("Failed to setup tensors\n");
        return false;
    }
    
    model_loaded = true;
    DEBUG_INFO("Model initialized successfully\n");
    print_model_info();
    
    return true;
}

bool ModelInference::load_model_from_spiffs(const char* model_path) {
    // This would load from SPIFFS in a real implementation
    // For now, use the embedded model data
    DEBUG_INFO("Loading embedded model data...\n");
    return init(noise_classifier_model_data, noise_classifier_model_data_len);
}

bool ModelInference::setup_tensors() {
    // Get input tensor
    input_tensor = interpreter->input(0);
    if (!input_tensor) {
        DEBUG_ERROR("Failed to get input tensor\n");
        return false;
    }
    
    // Get output tensor
    output_tensor = interpreter->output(0);
    if (!output_tensor) {
        DEBUG_ERROR("Failed to get output tensor\n");
        return false;
    }
    
    // Validate input dimensions
    input_size = 1;
    for (int i = 0; i < input_tensor->dims->size; i++) {
        input_size *= input_tensor->dims->data[i];
    }
    
    if (input_size != MODEL_INPUT_SIZE) {
        DEBUG_ERROR("Input size mismatch: expected %d, got %d\n", MODEL_INPUT_SIZE, input_size);
        return false;
    }
    
    // Validate output dimensions
    output_size = 1;
    for (int i = 0; i < output_tensor->dims->size; i++) {
        output_size *= output_tensor->dims->data[i];
    }
    
    if (output_size != NUM_CLASSES) {
        DEBUG_ERROR("Output size mismatch: expected %d, got %d\n", NUM_CLASSES, output_size);
        return false;
    }
    
    // Get quantization parameters
    if (input_tensor->type == kTfLiteInt8) {
        input_scale = input_tensor->params.scale;
        input_zero_point = input_tensor->params.zero_point;
        DEBUG_INFO("Input quantization: scale=%.8f, zero_point=%d\n", input_scale, input_zero_point);
    } else {
        DEBUG_INFO("Input tensor type: float32\n");
        input_scale = 1.0f;
        input_zero_point = 0;
    }
    
    if (output_tensor->type == kTfLiteInt8) {
        output_scale = output_tensor->params.scale;
        output_zero_point = output_tensor->params.zero_point;
        DEBUG_INFO("Output quantization: scale=%.8f, zero_point=%d\n", output_scale, output_zero_point);
    } else {
        DEBUG_INFO("Output tensor type: float32\n");
        output_scale = 1.0f;
        output_zero_point = 0;
    }
    
    return true;
}

ModelInference::InferenceResult ModelInference::predict(float* features) {
    TIMING_START();
    
    InferenceResult result;
    result.success = false;
    result.predicted_class = -1;
    result.confidence = 0.0f;
    result.inference_time_us = 0;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        result.class_probabilities[i] = 0.0f;
    }
    
    if (!model_loaded) {
        DEBUG_ERROR("Model not loaded\n");
        return result;
    }
    
    // Preprocessing
    unsigned long preprocess_start = micros();
    
    if (input_tensor->type == kTfLiteInt8) {
        // Quantize input
        int8_t* input_data = input_tensor->data.int8;
        quantize_input(features, input_data, input_size);
    } else {
        // Copy float input directly
        float* input_data = input_tensor->data.f;
        memcpy(input_data, features, input_size * sizeof(float));
    }
    
    unsigned long preprocess_time = micros() - preprocess_start;
    
    // Inference
    unsigned long inference_start = micros();
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        DEBUG_ERROR("Model inference failed\n");
        return result;
    }
    
    unsigned long inference_time = micros() - inference_start;
    
    // Postprocessing
    unsigned long postprocess_start = micros();
    
    if (output_tensor->type == kTfLiteInt8) {
        // Dequantize output
        int8_t* output_data = output_tensor->data.int8;
        dequantize_output(output_data, result.class_probabilities, output_size);
    } else {
        // Copy float output directly
        float* output_data = output_tensor->data.f;
        memcpy(result.class_probabilities, output_data, output_size * sizeof(float));
    }
    
    // Apply softmax to get probabilities
    float max_logit = result.class_probabilities[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (result.class_probabilities[i] > max_logit) {
            max_logit = result.class_probabilities[i];
        }
    }
    
    float exp_sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        result.class_probabilities[i] = expf(result.class_probabilities[i] - max_logit);
        exp_sum += result.class_probabilities[i];
    }
    
    // Normalize and find best class
    result.confidence = 0.0f;
    result.predicted_class = 0;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        result.class_probabilities[i] /= exp_sum;
        
        if (result.class_probabilities[i] > result.confidence) {
            result.confidence = result.class_probabilities[i];
            result.predicted_class = i;
        }
    }
    
    unsigned long postprocess_time = micros() - postprocess_start;
    unsigned long total_time = micros() - start_time;
    
    // Update statistics
    stats.preprocessing_time_us = preprocess_time;
    stats.inference_time_us = inference_time;
    stats.postprocessing_time_us = postprocess_time;
    stats.total_time_us = total_time;
    stats.inference_count++;
    
    // Update running statistics
    running_stats.preprocessing_time_us += preprocess_time;
    running_stats.inference_time_us += inference_time;
    running_stats.postprocessing_time_us += postprocess_time;
    running_stats.total_time_us += total_time;
    running_stats.inference_count++;
    
    result.inference_time_us = total_time;
    result.success = true;
    
    DEBUG_DEBUG("Inference: %s (%.3f) in %lu us (pre: %lu, inf: %lu, post: %lu)\n",
                CLASS_NAMES[result.predicted_class], result.confidence,
                total_time, preprocess_time, inference_time, postprocess_time);
    
    return result;
}

void ModelInference::quantize_input(float* input_data, int8_t* quantized_data, int size) {
    for (int i = 0; i < size; i++) {
        float quantized_value = input_data[i] / input_scale + input_zero_point;
        quantized_value = CLAMP(quantized_value, -128, 127);
        quantized_data[i] = (int8_t)roundf(quantized_value);
    }
}

void ModelInference::dequantize_output(int8_t* quantized_data, float* output_data, int size) {
    for (int i = 0; i < size; i++) {
        output_data[i] = (quantized_data[i] - output_zero_point) * output_scale;
    }
}

void ModelInference::reset_stats() {
    memset(&running_stats, 0, sizeof(running_stats));
}

size_t ModelInference::get_arena_used() const {
    if (!interpreter) {
        return 0;
    }
    
    // This is an approximation - TFLite Micro doesn't provide exact used memory
    return interpreter->arena_used_bytes();
}

void ModelInference::print_model_info() const {
    if (!model_loaded) {
        DEBUG_INFO("Model not loaded\n");
        return;
    }
    
    DEBUG_INFO("Model Information:\n");
    DEBUG_INFO("  Input size: %d (%d x %d)\n", input_size, N_TIME_FRAMES, N_MEL_BANDS);
    DEBUG_INFO("  Output size: %d classes\n", output_size);
    DEBUG_INFO("  Input type: %s\n", input_tensor->type == kTfLiteInt8 ? "INT8" : "FLOAT32");
    DEBUG_INFO("  Output type: %s\n", output_tensor->type == kTfLiteInt8 ? "INT8" : "FLOAT32");
    DEBUG_INFO("  Arena size: %d bytes\n", TFLITE_ARENA_SIZE);
    DEBUG_INFO("  Arena used: %zu bytes\n", get_arena_used());
    
    DEBUG_INFO("  Classes:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        DEBUG_INFO("    %d: %s %s\n", i, CLASS_EMOJIS[i], CLASS_NAMES[i]);
    }
}

void ModelInference::print_tensor_info() const {
    if (!model_loaded) {
        return;
    }
    
    DEBUG_INFO("Input Tensor:\n");
    DEBUG_INFO("  Dims: ");
    for (int i = 0; i < input_tensor->dims->size; i++) {
        Serial.printf("%d ", input_tensor->dims->data[i]);
    }
    Serial.printf("\n");
    DEBUG_INFO("  Type: %d\n", input_tensor->type);
    DEBUG_INFO("  Bytes: %d\n", input_tensor->bytes);
    
    DEBUG_INFO("Output Tensor:\n");
    DEBUG_INFO("  Dims: ");
    for (int i = 0; i < output_tensor->dims->size; i++) {
        Serial.printf("%d ", output_tensor->dims->data[i]);
    }
    Serial.printf("\n");
    DEBUG_INFO("  Type: %d\n", output_tensor->type);
    DEBUG_INFO("  Bytes: %d\n", output_tensor->bytes);
}