#include "feature_extractor.h"
#include <math.h>

FeatureExtractor::FeatureExtractor() {
  initialized_ = false;
  fft_ = nullptr;
  melSpectrogram_ = nullptr;
  fftReal_ = nullptr;
  fftImag_ = nullptr;
  melFilterbank_ = nullptr;
}

FeatureExtractor::~FeatureExtractor() {
  if (fft_) delete fft_;
  if (melSpectrogram_) free(melSpectrogram_);
  if (fftReal_) free(fftReal_);
  if (fftImag_) free(fftImag_);
  if (melFilterbank_) free(melFilterbank_);
}

void FeatureExtractor::begin() {
  Serial.println("  Allocating feature extraction buffers...");
  
  // Allocate buffers (use PSRAM if available)
  if (psramFound()) {
    melSpectrogram_ = (float*)ps_malloc(N_FRAMES * N_MEL_BANDS * sizeof(float));
    fftReal_ = (float*)ps_malloc(N_FFT * sizeof(float));
    fftImag_ = (float*)ps_malloc(N_FFT * sizeof(float));
    melFilterbank_ = (float*)ps_malloc(N_MEL_BANDS * (N_FFT/2 + 1) * sizeof(float));
    Serial.println("  ✓ Using PSRAM for buffers (float precision)");
  } else {
    melSpectrogram_ = (float*)malloc(N_FRAMES * N_MEL_BANDS * sizeof(float));
    fftReal_ = (float*)malloc(N_FFT * sizeof(float));
    fftImag_ = (float*)malloc(N_FFT * sizeof(float));
    melFilterbank_ = (float*)malloc(N_MEL_BANDS * (N_FFT/2 + 1) * sizeof(float));
    Serial.println("  ✓ Using heap for buffers (float precision)");
  }
  
  if (!melSpectrogram_ || !fftReal_ || !fftImag_ || !melFilterbank_) {
    Serial.println("  ❌ Buffer allocation failed!");
    return;
  }
  
  // Initialize FFT
  fft_ = new ArduinoFFT<float>(fftReal_, fftImag_, N_FFT, SAMPLE_RATE);
  
  // Precompute Mel filterbank
  float melMin = hzToMel(0);
  float melMax = hzToMel(SAMPLE_RATE / 2.0f);
  float melPoints[N_MEL_BANDS + 2];
  
  for (int i = 0; i < N_MEL_BANDS + 2; i++) {
    melPoints[i] = melMin + (melMax - melMin) * i / (N_MEL_BANDS + 1);
  }
  
  // Convert mel points to Hz and then to FFT bins
  int fftBins[N_MEL_BANDS + 2];
  for (int i = 0; i < N_MEL_BANDS + 2; i++) {
    float hz = melToHz(melPoints[i]);
    fftBins[i] = (int)((N_FFT + 1) * hz / SAMPLE_RATE);
  }
  
  // Compute triangular filters
  memset(melFilterbank_, 0, N_MEL_BANDS * (N_FFT/2 + 1) * sizeof(float));
  
  for (int m = 0; m < N_MEL_BANDS; m++) {
    int leftBin = fftBins[m];
    int centerBin = fftBins[m + 1];
    int rightBin = fftBins[m + 2];
    
    for (int k = leftBin; k < centerBin; k++) {
      melFilterbank_[m * (N_FFT/2 + 1) + k] = (float)(k - leftBin) / (centerBin - leftBin);
    }
    for (int k = centerBin; k < rightBin; k++) {
      melFilterbank_[m * (N_FFT/2 + 1) + k] = (float)(rightBin - k) / (rightBin - centerBin);
    }
  }
  
  initialized_ = true;
  Serial.println("✅ Feature extractor initialized");
}

float* FeatureExtractor::extractFeatures(float* audio, int length) {
  if (!initialized_) {
    Serial.println("❌ Feature extractor not initialized!");
    return nullptr;
  }
  
  // Compute mel spectrogram
  int frameCount = 0;
  
  for (int frameStart = 0; frameStart < length - FRAME_LENGTH && frameCount < N_FRAMES; frameStart += HOP_LENGTH) {
    // Copy frame to FFT buffer
    for (int i = 0; i < N_FFT; i++) {
      if (frameStart + i < length) {
        // Apply Hann window
        float window = 0.5f * (1.0f - cosf(2.0f * PI * i / N_FFT));
        fftReal_[i] = audio[frameStart + i] * window;
      } else {
        fftReal_[i] = 0.0;
      }
      fftImag_[i] = 0.0;
    }
    
    // Compute FFT
    fft_->compute(FFT_FORWARD);
    
    // Compute power spectrum
    float powerSpectrum[N_FFT/2 + 1];
    for (int i = 0; i <= N_FFT/2; i++) {
      powerSpectrum[i] = (fftReal_[i] * fftReal_[i] + fftImag_[i] * fftImag_[i]);
    }
    
    // Apply mel filterbank
    for (int m = 0; m < N_MEL_BANDS; m++) {
      float melEnergy = 0.0f;
      for (int k = 0; k <= N_FFT/2; k++) {
        melEnergy += melFilterbank_[m * (N_FFT/2 + 1) + k] * powerSpectrum[k];
      }
      
      // Convert to log scale (dB)
      melEnergy = 10.0f * log10f(melEnergy + 1e-10f);
      melSpectrogram_[frameCount * N_MEL_BANDS + m] = melEnergy;
    }
    
    frameCount++;
  }
  
  // Pad if necessary
  while (frameCount < N_FRAMES) {
    for (int m = 0; m < N_MEL_BANDS; m++) {
      melSpectrogram_[frameCount * N_MEL_BANDS + m] = melSpectrogram_[(frameCount-1) * N_MEL_BANDS + m];
    }
    frameCount++;
  }
  
  // Apply normalization using training parameters
  normalize_spectrogram(melSpectrogram_);
  
  return melSpectrogram_;
}

float FeatureExtractor::hzToMel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float FeatureExtractor::melToHz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}