# TensorFlow Lite Conversion Report

**Generated:** 2025-11-08 15:42:46
**Model:** app/models/best_model.h5
**Output:** app/models/noise_classifier_quantized.tflite

## Conversion Summary

| Metric | Original | Quantized | Change |
|--------|----------|-----------|---------|
| **Model Size** | 289.4 KB | 76.0 KB | 3.8x compression |
| **Accuracy** | 0.9300 | 0.9350 | -0.0050 |
| **Conversion Time** | - | 2.7s | - |

## Per-Class Performance

| Class | Original F1 | Quantized F1 | Delta |
|-------|-------------|--------------|-------|
| Whispering | 0.9020 | 0.9126 | +0.0107 |
| Typing | 0.9263 | 0.9263 | +0.0000 |
| Phone_ringing | 0.9583 | 0.9583 | +0.0000 |
| Loud_talking | 0.9346 | 0.9434 | +0.0088 |

## Inference Performance

| Metric | Value |
|--------|-------|
| **Average Time** | 0.2ms |
| **Median Time** | 0.2ms |
| **Min Time** | 0.1ms |
| **Max Time** | 18.0ms |
| **Std Deviation** | 1.3ms |

## Deployment Readiness

| Requirement | Status |
|-------------|--------|
| **Model Size < 100KB** | ✅ PASS (76.0 KB) |
| **Accuracy Drop < 5%** | ✅ PASS (-0.50%) |
| **Inference < 200ms** | ✅ PASS (0.2ms avg) |

## Files Generated

- `app/models/noise_classifier_quantized.tflite` - Quantized TFLite model
- `app/models/conversion_report/per_class_comparison.csv` - Detailed class metrics
- `app/models/conversion_report/confusion_matrices.png` - Confusion matrix comparison
- `app/models/conversion_report/inference_times.png` - Inference time analysis

## ESP32 Deployment Notes

1. **Memory Requirements:**
   - Model size: 76.0 KB flash storage
   - Estimated arena size: ~152 KB RAM

2. **Performance Expectations:**
   - Expected inference time on ESP32: 1-1ms
   - Recommended minimum heap: 228 KB

3. **Integration Requirements:**
   - Use INT8 input/output tensors
   - Apply same normalization as training (mean/std from model_params.npz)
   - Handle quantization scaling in preprocessing

## Validation Status

🟢 **READY FOR DEPLOYMENT**

The quantized model meets all deployment criteria and is ready for ESP32-CAM deployment.
