#include "vad.h"
#include <math.h>

VoiceActivityDetector::VoiceActivityDetector() {
  isCalibrated_ = false;
  calibrationCount_ = 0;
  lastConfidence_ = 0.0f;
  bufferIndex_ = 0;
  
  for (int i = 0; i < 10; i++) {
    activityBuffer_[i] = false;
  }
}

void VoiceActivityDetector::begin(int sampleRate) {
  sampleRate_ = sampleRate;
  calibrationTarget_ = (int)(VAD_CALIBRATION_SECONDS);  // ~3.5 samples
  
  // Initialize thresholds (will be set during calibration)
  energyThreshold_ = 0.005f;
  spectralThreshold_ = 1000.0f;
  zcrThreshold_ = 0.01f;
}

bool VoiceActivityDetector::isCalibrated() {
  return isCalibrated_;
}

int VoiceActivityDetector::getCalibrationProgress() {
  return (calibrationCount_ * 100) / calibrationTarget_;
}

void VoiceActivityDetector::addCalibrationSample(float* audio, int length) {
  if (isCalibrated_ || calibrationCount_ >= calibrationTarget_) {
    if (!isCalibrated_) {
      // Compute thresholds from collected samples
      float energySum = 0, spectralSum = 0, zcrSum = 0;
      
      for (int i = 0; i < calibrationCount_; i++) {
        energySum += calibrationSamples_[i][0];
        spectralSum += calibrationSamples_[i][1];
        zcrSum += calibrationSamples_[i][2];
      }
      
      float energyBaseline = energySum / calibrationCount_;
      float spectralBaseline = spectralSum / calibrationCount_;
      float zcrBaseline = zcrSum / calibrationCount_;
      
      // Set thresholds with safety margins
      energyThreshold_ = max<float>(energyBaseline * VAD_ENERGY_MARGIN, 0.002f);
      spectralThreshold_ = max<float>(spectralBaseline * VAD_SPECTRAL_MARGIN, 800.0f);
      zcrThreshold_ = max<float>(zcrBaseline * VAD_ZCR_MARGIN, 0.005f);
      
      isCalibrated_ = true;
      
      Serial.printf("\n✅ VAD Thresholds Set:\n");
      Serial.printf("  Energy: %.6f\n", energyThreshold_);
      Serial.printf("  Spectral: %.1f Hz\n", spectralThreshold_);
      Serial.printf("  ZCR: %.4f\n\n", zcrThreshold_);
    }
    return;
  }
  
  // Extract features
  float energy = calculateEnergy(audio, length);
  float spectral = calculateSpectralCentroid(audio, length);
  float zcr = calculateZeroCrossingRate(audio, length);
  
  // Store calibration sample
  if (calibrationCount_ < 100) {
    calibrationSamples_[calibrationCount_][0] = energy;
    calibrationSamples_[calibrationCount_][1] = spectral;
    calibrationSamples_[calibrationCount_][2] = zcr;
    calibrationCount_++;
  }
}

bool VoiceActivityDetector::detectActivity(float* audio, int length) {
  if (!isCalibrated_) {
    lastConfidence_ = 0.0f;
    return false;
  }
  
  // Extract features
  float energy = calculateEnergy(audio, length);
  float spectral = calculateSpectralCentroid(audio, length);
  float zcr = calculateZeroCrossingRate(audio, length);
  
  // Individual feature decisions
  bool energyActive = energy > energyThreshold_;
  bool spectralActive = spectral > spectralThreshold_;
  bool zcrActive = zcr > zcrThreshold_;
  
  // Weighted confidence score
  float confidence = 0.0f;
  confidence += energyActive ? 0.4f : 0.0f;      // 40% weight
  confidence += spectralActive ? 0.25f : 0.0f;   // 25% weight
  confidence += zcrActive ? 0.2f : 0.0f;         // 20% weight
  
  bool isActivity = confidence > 0.5f;
  
  // Temporal smoothing
  activityBuffer_[bufferIndex_] = isActivity;
  bufferIndex_ = (bufferIndex_ + 1) % 10;
  
  int activeCount = 0;
  for (int i = 0; i < 10; i++) {
    if (activityBuffer_[i]) activeCount++;
  }
  
  float smoothedConfidence = activeCount / 10.0f;
  
  // Require 40% recent activity for positive detection
  bool finalDecision = smoothedConfidence >= 0.4f;
  lastConfidence_ = finalDecision ? smoothedConfidence : (1.0f - smoothedConfidence);
  
  return finalDecision;
}

float VoiceActivityDetector::getConfidence() {
  return lastConfidence_;
}

float VoiceActivityDetector::calculateEnergy(float* audio, int length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    sum += audio[i] * audio[i];
  }
  return sqrtf(sum / length);
}

float VoiceActivityDetector::calculateSpectralCentroid(float* audio, int length) {
  // Simplified spectral centroid using frequency weighting
  float numerator = 0.0f;
  float denominator = 0.0f;
  
  for (int i = 1; i < length; i++) {
    float magnitude = fabsf(audio[i] - audio[i-1]);
    float frequency = (float)i * sampleRate_ / length;
    numerator += frequency * magnitude;
    denominator += magnitude;
  }
  
  return (denominator > 0.0001f) ? (numerator / denominator) : 0.0f;
}

float VoiceActivityDetector::calculateZeroCrossingRate(float* audio, int length) {
  int crossings = 0;
  for (int i = 1; i < length; i++) {
    if ((audio[i] >= 0 && audio[i-1] < 0) || (audio[i] < 0 && audio[i-1] >= 0)) {
      crossings++;
    }
  }
  return (float)crossings / length;
}