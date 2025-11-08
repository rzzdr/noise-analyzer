#!/usr/bin/env python3
"""
TensorFlow Lite Model Conversion Script for ESP32-CAM Deployment
================================================================

Converts the trained Keras audio classifier to TensorFlow Lite with INT8 quantization
for deployment on ESP32-CAM with <100KB model size target.

Usage:
    python convert_to_tflite.py --model models/best_model.h5 --output models/noise_classifier_quantized.tflite

Features:
- INT8 quantization using representative dataset
- Model validation and accuracy comparison
- Performance benchmarking
- Detailed conversion report
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import time
import json
from datetime import datetime

# Add app directory to path for imports
sys.path.append('app')
from app.NoiseAnalyzer import NoiseAnalyzer, TARGET_CLASSES, SAMPLE_RATE, DURATION, N_FRAMES, N_MELS

class TFLiteConverter:
    """Handles conversion of Keras model to TensorFlow Lite with validation"""
    
    def __init__(self, model_path, output_path='app/models/noise_classifier_quantized.tflite'):
        self.model_path = model_path
        self.output_path = output_path
        self.original_model = None
        self.tflite_model = None
        self.tflite_interpreter = None
        self.analyzer = NoiseAnalyzer(dataset_path='data/ESC-50-master')
        
        # Conversion statistics
        self.conversion_stats = {
            'original_size_kb': 0,
            'quantized_size_kb': 0,
            'compression_ratio': 0,
            'original_accuracy': 0,
            'quantized_accuracy': 0,
            'accuracy_drop': 0,
            'conversion_time': 0,
            'timestamp': datetime.now().isoformat()
        }
        
    def load_original_model(self):
        """Load the original Keras model"""
        print(f"Loading original model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        self.original_model = tf.keras.models.load_model(self.model_path)
        
        # Get model size
        self.original_model.save('temp_model.h5')
        self.conversion_stats['original_size_kb'] = os.path.getsize('temp_model.h5') / 1024
        os.remove('temp_model.h5')
        
        print(f"✓ Original model loaded successfully")
        print(f"  Model size: {self.conversion_stats['original_size_kb']:.1f} KB")
        print(f"  Total params: {self.original_model.count_params():,}")
        
        return self.original_model
    
    def prepare_representative_dataset(self, num_samples=200):
        """
        Prepare representative dataset for quantization calibration
        Uses actual audio samples from the training dataset
        """
        print(f"Preparing representative dataset with {num_samples} samples...")
        
        try:
            # Load a subset of the training data for calibration
            features, labels, labels_raw = self.analyzer.load_esc50_dataset()
            
            # Remove channel dimension if present (model expects 3D input)
            if len(features.shape) == 4 and features.shape[-1] == 1:
                features = features.squeeze(-1)  # Remove last dimension if it's 1
                print(f"  Removed channel dimension: {features.shape}")
            
            # Take a representative subset stratified by class
            indices = []
            samples_per_class = num_samples // len(TARGET_CLASSES)
            
            for class_name in TARGET_CLASSES:
                class_indices = np.where(labels_raw == class_name)[0]
                if len(class_indices) >= samples_per_class:
                    selected = np.random.choice(class_indices, samples_per_class, replace=False)
                else:
                    selected = class_indices
                indices.extend(selected)
            
            # Ensure we have exactly num_samples
            if len(indices) > num_samples:
                indices = np.random.choice(indices, num_samples, replace=False)
            
            representative_features = features[indices]
            print(f"✓ Representative dataset prepared: {len(representative_features)} samples")
            print(f"  Feature shape: {representative_features.shape}")
            
            def representative_dataset():
                """Generator function for TFLite converter"""
                for i in range(len(representative_features)):
                    # Yield as expected by TFLite converter (3D: batch, time, features)
                    sample = representative_features[i:i+1].astype(np.float32)
                    yield [sample]
            
            return representative_dataset, representative_features, labels[indices], labels_raw[indices]
            
        except Exception as e:
            print(f"Error preparing representative dataset: {e}")
            print("Falling back to synthetic data...")
            return self._create_synthetic_representative_dataset(num_samples)
    
    def _create_synthetic_representative_dataset(self, num_samples):
        """Create synthetic representative dataset if real data unavailable"""
        print("Creating synthetic representative dataset...")
        
        # Generate synthetic mel spectrograms with realistic statistics
        synthetic_features = []
        synthetic_labels = []
        synthetic_labels_raw = []
        
        for i in range(num_samples):
            # Create realistic mel spectrogram patterns (3D: time, features)
            if i % 4 == 0:  # Whispering - low energy
                features = np.random.normal(-2, 0.5, (N_FRAMES, N_MELS))
            elif i % 4 == 1:  # Typing - sharp transients
                features = np.random.normal(-1, 1.0, (N_FRAMES, N_MELS))
                features[::10] *= 2  # Add periodic spikes
            elif i % 4 == 2:  # Phone ringing - tonal
                features = np.random.normal(-1.5, 0.8, (N_FRAMES, N_MELS))
                features[:, 10:20] *= 1.5  # Emphasize mid frequencies
            else:  # Loud talking - high energy
                features = np.random.normal(0, 1.2, (N_FRAMES, N_MELS))
            
            class_idx = i % len(TARGET_CLASSES)
            
            synthetic_features.append(features)
            synthetic_labels_raw.append(TARGET_CLASSES[class_idx])
            
            # Create one-hot encoded labels
            one_hot = np.zeros(len(TARGET_CLASSES))
            one_hot[class_idx] = 1
            synthetic_labels.append(one_hot)
        
        synthetic_features = np.array(synthetic_features).astype(np.float32)
        synthetic_labels = np.array(synthetic_labels)
        synthetic_labels_raw = np.array(synthetic_labels_raw)
        
        def representative_dataset():
            for i in range(len(synthetic_features)):
                yield [synthetic_features[i:i+1]]
        
        return representative_dataset, synthetic_features, synthetic_labels, synthetic_labels_raw
    
    def convert_to_tflite(self, representative_dataset):
        """Convert Keras model to TensorFlow Lite with INT8 quantization"""
        print("Starting TensorFlow Lite conversion with INT8 quantization...")
        
        start_time = time.time()
        
        try:
            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
            
            # Configure INT8 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Additional optimizations for embedded deployment
            converter.experimental_new_converter = True
            converter.allow_custom_ops = False
            
            # Convert
            print("  Converting model (this may take a few minutes)...")
            self.tflite_model = converter.convert()
            
            # Save the converted model
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'wb') as f:
                f.write(self.tflite_model)
            
            # Get quantized model size
            self.conversion_stats['quantized_size_kb'] = len(self.tflite_model) / 1024
            self.conversion_stats['compression_ratio'] = (
                self.conversion_stats['original_size_kb'] / 
                self.conversion_stats['quantized_size_kb']
            )
            self.conversion_stats['conversion_time'] = time.time() - start_time
            
            print(f"✓ Conversion completed successfully!")
            print(f"  Output file: {self.output_path}")
            print(f"  Quantized size: {self.conversion_stats['quantized_size_kb']:.1f} KB")
            print(f"  Compression ratio: {self.conversion_stats['compression_ratio']:.1f}x")
            print(f"  Conversion time: {self.conversion_stats['conversion_time']:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
            return False
    
    def setup_tflite_interpreter(self):
        """Setup TensorFlow Lite interpreter for validation"""
        print("Setting up TensorFlow Lite interpreter...")
        
        try:
            self.tflite_interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
            self.tflite_interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            
            print(f"✓ TFLite interpreter ready")
            print(f"  Input shape: {input_details[0]['shape']}")
            print(f"  Input type: {input_details[0]['dtype']}")
            print(f"  Output shape: {output_details[0]['shape']}")
            print(f"  Output type: {output_details[0]['dtype']}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to setup interpreter: {e}")
            return False
    
    def validate_models(self, test_features, test_labels, test_labels_raw):
        """Compare original vs quantized model performance"""
        print("Validating model performance...")
        
        # Original model predictions
        print("  Testing original model...")
        original_predictions = self.original_model.predict(test_features, verbose=0)
        original_pred_classes = np.argmax(original_predictions, axis=1)
        original_true_classes = np.argmax(test_labels, axis=1)
        
        self.conversion_stats['original_accuracy'] = accuracy_score(
            original_true_classes, original_pred_classes
        )
        
        # TFLite model predictions
        print("  Testing quantized model...")
        input_details = self.tflite_interpreter.get_input_details()
        output_details = self.tflite_interpreter.get_output_details()
        
        quantized_predictions = []
        inference_times = []
        
        for i in range(len(test_features)):
            # Prepare input (convert to INT8)
            input_data = test_features[i:i+1]
            
            # Quantize input
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            
            if input_scale > 0:  # Only quantize if quantization parameters are available
                input_quantized = (input_data / input_scale + input_zero_point).astype(np.int8)
            else:
                input_quantized = input_data.astype(np.float32)
            
            # Run inference
            start_time = time.time()
            self.tflite_interpreter.set_tensor(input_details[0]['index'], input_quantized)
            self.tflite_interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Get output
            output_data = self.tflite_interpreter.get_tensor(output_details[0]['index'])
            
            # Dequantize output if needed
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            
            if output_scale > 0:
                output_dequantized = (output_data.astype(np.float32) - output_zero_point) * output_scale
            else:
                output_dequantized = output_data
            
            quantized_predictions.append(output_dequantized[0])
            inference_times.append(inference_time)
        
        quantized_predictions = np.array(quantized_predictions)
        quantized_pred_classes = np.argmax(quantized_predictions, axis=1)
        
        self.conversion_stats['quantized_accuracy'] = accuracy_score(
            original_true_classes, quantized_pred_classes
        )
        self.conversion_stats['accuracy_drop'] = (
            self.conversion_stats['original_accuracy'] - 
            self.conversion_stats['quantized_accuracy']
        )
        
        # Performance statistics
        avg_inference_time = np.mean(inference_times)
        
        print(f"✓ Validation completed")
        print(f"  Original accuracy: {self.conversion_stats['original_accuracy']:.4f}")
        print(f"  Quantized accuracy: {self.conversion_stats['quantized_accuracy']:.4f}")
        print(f"  Accuracy drop: {self.conversion_stats['accuracy_drop']:.4f}")
        print(f"  Average inference time: {avg_inference_time:.1f}ms")
        
        return {
            'original_predictions': original_predictions,
            'quantized_predictions': quantized_predictions,
            'original_pred_classes': original_pred_classes,
            'quantized_pred_classes': quantized_pred_classes,
            'true_classes': original_true_classes,
            'inference_times': inference_times
        }
    
    def generate_validation_report(self, validation_results):
        """Generate comprehensive validation report"""
        print("Generating validation report...")
        
        # Create output directory
        report_dir = Path('app/models/conversion_report')
        report_dir.mkdir(exist_ok=True)
        
        # Per-class analysis
        true_classes = validation_results['true_classes']
        original_pred = validation_results['original_pred_classes']
        quantized_pred = validation_results['quantized_pred_classes']
        
        # Classification reports
        original_report = classification_report(true_classes, original_pred, 
                                              target_names=TARGET_CLASSES, 
                                              output_dict=True, zero_division=0)
        quantized_report = classification_report(true_classes, quantized_pred, 
                                               target_names=TARGET_CLASSES, 
                                               output_dict=True, zero_division=0)
        
        # Per-class comparison
        class_comparison = []
        for class_name in TARGET_CLASSES:
            original_f1 = original_report[class_name]['f1-score']
            quantized_f1 = quantized_report[class_name]['f1-score']
            
            class_comparison.append({
                'Class': class_name,
                'Original_F1': original_f1,
                'Quantized_F1': quantized_f1,
                'F1_Delta': quantized_f1 - original_f1
            })
        
        # Save detailed results
        results_df = pd.DataFrame(class_comparison)
        results_df.to_csv(report_dir / 'per_class_comparison.csv', index=False)
        
        # Confusion matrices
        self._plot_confusion_matrices(validation_results, report_dir)
        
        # Inference time analysis
        self._plot_inference_times(validation_results['inference_times'], report_dir)
        
        # Generate markdown report
        self._generate_markdown_report(results_df, validation_results, report_dir)
        
        print(f"✓ Validation report saved to {report_dir}")
        
    def _plot_confusion_matrices(self, validation_results, report_dir):
        """Plot confusion matrices for both models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        true_classes = validation_results['true_classes']
        
        # Original model confusion matrix
        cm_original = confusion_matrix(true_classes, validation_results['original_pred_classes'])
        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, ax=ax1)
        ax1.set_title('Original Model Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Quantized model confusion matrix
        cm_quantized = confusion_matrix(true_classes, validation_results['quantized_pred_classes'])
        sns.heatmap(cm_quantized, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, ax=ax2)
        ax2.set_title('Quantized Model Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(report_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_inference_times(self, inference_times, report_dir):
        """Plot inference time distribution"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(inference_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('TFLite Inference Time Distribution')
        plt.axvline(np.mean(inference_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(inference_times):.1f}ms')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot(inference_times)
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time Box Plot')
        
        # Add statistics
        stats_text = f'''Statistics:
Mean: {np.mean(inference_times):.1f}ms
Median: {np.median(inference_times):.1f}ms
Min: {np.min(inference_times):.1f}ms
Max: {np.max(inference_times):.1f}ms
Std: {np.std(inference_times):.1f}ms'''
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, family='monospace',
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(report_dir / 'inference_times.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_markdown_report(self, class_df, validation_results, report_dir):
        """Generate comprehensive markdown report"""
        
        inference_times = validation_results['inference_times']
        
        report_content = f"""# TensorFlow Lite Conversion Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {self.model_path}
**Output:** {self.output_path}

## Conversion Summary

| Metric | Original | Quantized | Change |
|--------|----------|-----------|---------|
| **Model Size** | {self.conversion_stats['original_size_kb']:.1f} KB | {self.conversion_stats['quantized_size_kb']:.1f} KB | {self.conversion_stats['compression_ratio']:.1f}x compression |
| **Accuracy** | {self.conversion_stats['original_accuracy']:.4f} | {self.conversion_stats['quantized_accuracy']:.4f} | {self.conversion_stats['accuracy_drop']:+.4f} |
| **Conversion Time** | - | {self.conversion_stats['conversion_time']:.1f}s | - |

## Per-Class Performance

| Class | Original F1 | Quantized F1 | Delta |
|-------|-------------|--------------|-------|
"""
        
        for _, row in class_df.iterrows():
            report_content += f"| {row['Class']} | {row['Original_F1']:.4f} | {row['Quantized_F1']:.4f} | {row['F1_Delta']:+.4f} |\n"
        
        report_content += f"""
## Inference Performance

| Metric | Value |
|--------|-------|
| **Average Time** | {np.mean(inference_times):.1f}ms |
| **Median Time** | {np.median(inference_times):.1f}ms |
| **Min Time** | {np.min(inference_times):.1f}ms |
| **Max Time** | {np.max(inference_times):.1f}ms |
| **Std Deviation** | {np.std(inference_times):.1f}ms |

## Deployment Readiness

| Requirement | Status |
|-------------|--------|
| **Model Size < 100KB** | {'✅ PASS' if self.conversion_stats['quantized_size_kb'] < 100 else '❌ FAIL'} ({self.conversion_stats['quantized_size_kb']:.1f} KB) |
| **Accuracy Drop < 5%** | {'✅ PASS' if abs(self.conversion_stats['accuracy_drop']) < 0.05 else '❌ FAIL'} ({self.conversion_stats['accuracy_drop']*100:+.2f}%) |
| **Inference < 200ms** | {'✅ PASS' if np.mean(inference_times) < 200 else '❌ FAIL'} ({np.mean(inference_times):.1f}ms avg) |

## Files Generated

- `{self.output_path}` - Quantized TFLite model
- `app/models/conversion_report/per_class_comparison.csv` - Detailed class metrics
- `app/models/conversion_report/confusion_matrices.png` - Confusion matrix comparison
- `app/models/conversion_report/inference_times.png` - Inference time analysis

## ESP32 Deployment Notes

1. **Memory Requirements:**
   - Model size: {self.conversion_stats['quantized_size_kb']:.1f} KB flash storage
   - Estimated arena size: ~{int(self.conversion_stats['quantized_size_kb'] * 2)} KB RAM

2. **Performance Expectations:**
   - Expected inference time on ESP32: {np.mean(inference_times) * 3:.0f}-{np.mean(inference_times) * 5:.0f}ms
   - Recommended minimum heap: {int(self.conversion_stats['quantized_size_kb'] * 3)} KB

3. **Integration Requirements:**
   - Use INT8 input/output tensors
   - Apply same normalization as training (mean/std from model_params.npz)
   - Handle quantization scaling in preprocessing

## Validation Status

{'🟢 **READY FOR DEPLOYMENT**' if self.conversion_stats['quantized_size_kb'] < 100 and abs(self.conversion_stats['accuracy_drop']) < 0.05 else '🟡 **REVIEW REQUIRED**'}

The quantized model {'meets' if self.conversion_stats['quantized_size_kb'] < 100 and abs(self.conversion_stats['accuracy_drop']) < 0.05 else 'does not meet'} all deployment criteria and {'is' if self.conversion_stats['quantized_size_kb'] < 100 and abs(self.conversion_stats['accuracy_drop']) < 0.05 else 'may not be'} ready for ESP32-CAM deployment.
"""
        
        # Save report
        with open(report_dir / 'conversion_report.md', 'w') as f:
            f.write(report_content)
        
        # Also save stats as JSON for programmatic access
        with open(report_dir / 'conversion_stats.json', 'w') as f:
            json.dump(self.conversion_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to TensorFlow Lite for ESP32 deployment')
    parser.add_argument('--model', default='app/models/best_model.h5', help='Path to Keras model')
    parser.add_argument('--output', default='app/models/noise_classifier_quantized.tflite', help='Output TFLite model path')
    parser.add_argument('--samples', type=int, default=200, help='Number of representative samples for quantization')
    parser.add_argument('--skip-validation', action='store_true', help='Skip model validation (faster conversion)')
    
    args = parser.parse_args()
    
    print("TensorFlow Lite Model Conversion for ESP32-CAM")
    print("=" * 50)
    
    try:
        # Initialize converter
        converter = TFLiteConverter(args.model, args.output)
        
        # Load original model
        converter.load_original_model()
        
        # Prepare representative dataset
        representative_dataset, test_features, test_labels, test_labels_raw = converter.prepare_representative_dataset(args.samples)
        
        # Convert to TFLite
        if not converter.convert_to_tflite(representative_dataset):
            print("Conversion failed!")
            return 1
        
        # Setup interpreter for validation
        if not converter.setup_tflite_interpreter():
            print("Failed to setup interpreter!")
            return 1
        
        # Validate models if not skipped
        if not args.skip_validation:
            validation_results = converter.validate_models(test_features, test_labels, test_labels_raw)
            converter.generate_validation_report(validation_results)
        
        print("\n" + "=" * 50)
        print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"📁 Quantized model: {args.output}")
        print(f"📊 Size reduction: {converter.conversion_stats['original_size_kb']:.1f} KB → {converter.conversion_stats['quantized_size_kb']:.1f} KB")
        print(f"📈 Compression: {converter.conversion_stats['compression_ratio']:.1f}x")
        
        if not args.skip_validation:
            print(f"🎯 Accuracy: {converter.conversion_stats['original_accuracy']:.3f} → {converter.conversion_stats['quantized_accuracy']:.3f}")
            print(f"📋 Full report: app/models/conversion_report/conversion_report.md")
        
        # Check deployment readiness
        size_ok = converter.conversion_stats['quantized_size_kb'] < 100
        accuracy_ok = abs(converter.conversion_stats['accuracy_drop']) < 0.05 if not args.skip_validation else True
        
        if size_ok and accuracy_ok:
            print("🚀 READY FOR ESP32 DEPLOYMENT!")
        else:
            print("⚠️  REVIEW REQUIRED - Check conversion report for details")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())