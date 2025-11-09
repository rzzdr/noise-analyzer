#include <TensorFlowLite_ESP32.h>

#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "config.h"
#include "model_data.h"

class ModelInference {
public:
  ModelInference();
  ~ModelInference();
  
  bool begin();
  int predict(float* features, float* confidence);
  const char* getClassName(int classIndex);
  void printModelInfo();
  
private:
  bool setupInterpreter();
  void quantizeInput(float* input, int8_t* output, int size);
  void dequantizeOutput(int8_t* input, float* output, int size);
  
  // TFLite objects
  const tflite::Model* model_;
  tflite::MicroInterpreter* interpreter_;
  tflite::MicroErrorReporter* errorReporter_;
  tflite::AllOpsResolver* resolver_;
  
  // Tensor arena
  uint8_t* tensorArena_;
  
  // Input/output tensors
  TfLiteTensor* input_;
  TfLiteTensor* output_;
  
  // Quantization parameters
  float input_scale_;
  int32_t input_zero_point_;
  float output_scale_;
  int32_t output_zero_point_;
  
  bool initialized_;
};

#endif // MODEL_INFERENCE_H
