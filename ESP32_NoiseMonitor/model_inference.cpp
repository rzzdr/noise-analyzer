#include "model_inference.h"

ModelInference::ModelInference() {
  initialized_ = false;
  errorReporter_ = nullptr;
  model_ = nullptr;
  interpreter_ = nullptr;
  input_ = nullptr;
  output_ = nullptr;
  tensorArena_ = nullptr;
}

ModelInference::~ModelInference() {
  if (interpreter_) delete interpreter_;
  if (tensorArena_) free(tensorArena_);
}

bool ModelInference::begin() {
  Serial.println("  Loading TFLite model...");
  
  // Allocate tensor arena (use PSRAM if available)
  if (psramFound()) {
    tensorArena_ = (uint8_t*)ps_malloc(TFLITE_ARENA_SIZE);
    Serial.println("  ✓ Using PSRAM for tensor arena");
  } else {
    tensorArena_ = (uint8_t*)malloc(TFLITE_ARENA_SIZE);
    Serial.println("  ✓ Using heap for tensor arena");
  }
  
  if (!tensorArena_) {
    Serial.println("  ❌ Failed to allocate tensor arena!");
    return false;
  }
  
  // Setup error reporter
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter_ = &microErrorReporter;
  
  // Load model
  model_ = tflite::GetModel(noise_classifier_model_data);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("  ❌ Model schema mismatch! Expected %d, got %d\n",
                 TFLITE_SCHEMA_VERSION, model_->version());
    return false;
  }
  
  Serial.printf("  ✓ Model loaded (size: %d bytes)\n", noise_classifier_model_data_len);
  
  // Setup resolver with all ops
  static tflite::AllOpsResolver resolver;
  
  // Build interpreter
  static tflite::MicroInterpreter staticInterpreter(
    model_, resolver, tensorArena_, TFLITE_ARENA_SIZE, errorReporter_);
  interpreter_ = &staticInterpreter;
  
  // Allocate tensors
  TfLiteStatus allocateStatus = interpreter_->AllocateTensors();
  if (allocateStatus != kTfLiteOk) {
    Serial.println("  ❌ Failed to allocate tensors!");
    return false;
  }
  
  // Get input and output tensors
  input_ = interpreter_->input(0);
  output_ = interpreter_->output(0);
  
  // Validate input dimensions
  if (input_->dims->size != 3 || 
      input_->dims->data[1] != N_FRAMES || 
      input_->dims->data[2] != N_MEL_BANDS) {
    Serial.printf("  ❌ Input dimension mismatch! Expected (1,%d,%d), got (%d,%d,%d)\n",
                 N_FRAMES, N_MEL_BANDS,
                 input_->dims->data[0], input_->dims->data[1], input_->dims->data[2]);
    return false;
  }
  
  // Validate output dimensions
  if (output_->dims->data[1] != NUM_CLASSES) {
    Serial.printf("  ❌ Output dimension mismatch! Expected %d classes, got %d\n",
                 NUM_CLASSES, output_->dims->data[1]);
    return false;
  }
  
  Serial.println("✅ TFLite model ready");
  printModelInfo();
  
  initialized_ = true;
  return true;
}

int ModelInference::predict(float* features, float* confidence) {
  if (!initialized_) {
    Serial.println("❌ Model not initialized!");
    *confidence = 0.0f;
    return -1;
  }
  
  // Check input tensor type
  if (input_->type == kTfLiteInt8) {
    // Quantized input - convert float to INT8
    int8_t* inputData = input_->data.int8;
    float inputScale = input_->params.scale;
    int32_t inputZeroPoint = input_->params.zero_point;
    
    for (int i = 0; i < N_FRAMES * N_MEL_BANDS; i++) {
      int32_t quantizedValue = (int32_t)(features[i] / inputScale) + inputZeroPoint;
      quantizedValue = max<int32_t>(-128, min<int32_t>(127, quantizedValue));
      inputData[i] = (int8_t)quantizedValue;
    }
  } else {
    // Float input
    float* inputData = input_->data.f;
    memcpy(inputData, features, N_FRAMES * N_MEL_BANDS * sizeof(float));
  }
  
  // Run inference
  TfLiteStatus invokeStatus = interpreter_->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("❌ Inference failed!");
    *confidence = 0.0f;
    return -1;
  }
  
  // Get output probabilities
  float probabilities[NUM_CLASSES];
  
  if (output_->type == kTfLiteInt8) {
    // Dequantize INT8 output
    int8_t* outputData = output_->data.int8;
    float outputScale = output_->params.scale;
    int32_t outputZeroPoint = output_->params.zero_point;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
      probabilities[i] = (outputData[i] - outputZeroPoint) * outputScale;
    }
  } else {
    // Float output
    float* outputData = output_->data.f;
    memcpy(probabilities, outputData, NUM_CLASSES * sizeof(float));
  }
  
  // Find class with highest probability
  int maxIndex = 0;
  float maxProb = probabilities[0];
  
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }
  
  *confidence = maxProb;
  return maxIndex;
}

const char* ModelInference::getClassName(int classIndex) {
  if (classIndex < 0 || classIndex >= NUM_CLASSES) {
    return "Unknown";
  }
  return CLASS_NAMES[classIndex];
}

void ModelInference::printModelInfo() {
  Serial.println("\n┌─────────────────────────────────────┐");
  Serial.println("│       TFLite Model Information      │");
  Serial.println("├─────────────────────────────────────┤");
  Serial.printf("│ Model Size:      %6d bytes     │\n", noise_classifier_model_data_len);
  Serial.printf("│ Arena Size:      %6d KB        │\n", TFLITE_ARENA_SIZE / 1024);
  Serial.printf("│ Input Shape:     (1,%d,%d)      │\n", N_FRAMES, N_MEL_BANDS);
  Serial.printf("│ Input Type:      %-15s │\n", 
               input_->type == kTfLiteInt8 ? "INT8" : "FLOAT32");
  Serial.printf("│ Output Shape:    (1,%d)           │\n", NUM_CLASSES);
  Serial.printf("│ Output Type:     %-15s │\n",
               output_->type == kTfLiteInt8 ? "INT8" : "FLOAT32");
  Serial.println("├─────────────────────────────────────┤");
  Serial.println("│ Classes:                            │");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.printf("│  %d. %-30s │\n", i, CLASS_NAMES[i]);
  }
  Serial.println("└─────────────────────────────────────┘\n");
}