#include <arduinoFFT.h>

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <Arduino.h>
#include "config.h"
#include "model_normalization.h"

class FeatureExtractor {
public:
  FeatureExtractor();
  ~FeatureExtractor();
  
  void begin();
  float* extractFeatures(float* audio, int length);
  
private:
  void computeMelSpectrogram(float* audio, int length);
  void applyMelFilterbank(float* powerSpectrum);
  float hzToMel(float hz);
  float melToHz(float mel);
  
  ArduinoFFT<float>* fft_;
  
  // Buffers
  float* melSpectrogram_;    // (N_FRAMES x N_MEL_BANDS)
  float* fftReal_;          // Changed from double to float
  float* fftImag_;          // Changed from double to float
  float* melFilterbank_;     // (N_MEL_BANDS x (N_FFT/2 + 1))
  
  bool initialized_;
};

#endif // FEATURE_EXTRACTOR_H
