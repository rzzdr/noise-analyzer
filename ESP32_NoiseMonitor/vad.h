#ifndef VAD_H
#define VAD_H

#include <Arduino.h>
#include "config.h"

class VoiceActivityDetector {
public:
  VoiceActivityDetector();
  
  void begin(int sampleRate);
  bool isCalibrated();
  void addCalibrationSample(float* audio, int length);
  bool detectActivity(float* audio, int length);
  float getConfidence();
  int getCalibrationProgress();
  
private:
  // Feature extraction
  float calculateEnergy(float* audio, int length);
  float calculateSpectralCentroid(float* audio, int length);
  float calculateZeroCrossingRate(float* audio, int length);
  
  // Calibration
  int sampleRate_;
  bool isCalibrated_;
  float calibrationSamples_[100][4];  // Store feature vectors
  int calibrationCount_;
  int calibrationTarget_;
  
  // Thresholds
  float energyThreshold_;
  float spectralThreshold_;
  float zcrThreshold_;
  
  // Detection state
  float lastConfidence_;
  bool activityBuffer_[10];
  int bufferIndex_;
};

#endif // VAD_H