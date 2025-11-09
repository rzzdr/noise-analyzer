# Flask Server for Real-Time Audio Classification

This Flask server receives audio data from clients, performs real-time noise classification using the trained model, and pushes results to Firebase Firestore.

## 🚀 Features

- **Real-time Audio Classification**: Processes 1-second audio chunks
- **Voice Activity Detection (VAD)**: Pre-filters silence before classification
- **Firebase Integration**: Automatically pushes predictions to Firestore
- **RESTful API**: Easy-to-use endpoints for calibration and prediction
- **Test Client**: Laptop-based test script to verify server functionality

## 📋 Prerequisites

1. **Trained Model**: You need a trained model at `app/models/best_model.h5`
2. **Firebase Credentials**: Service account key from Firebase Console
3. **Python 3.8+**

## 🔧 Setup Instructions

### 1. Install Dependencies

```powershell
pip install -r requirements_server.txt
```

### 2. Configure Firebase

#### Option A: Using Service Account Key (Recommended)

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: **hawties-2a013**
3. Navigate to: **Project Settings** > **Service Accounts**
4. Click **Generate New Private Key**
5. Save the downloaded JSON file as `firebase-credentials.json` in the project root

#### Option B: Using Template

1. Copy `firebase-credentials-template.json` to `firebase-credentials.json`
2. Fill in your credentials:
   - `private_key_id`: From Firebase service account
   - `private_key`: Full private key (including BEGIN/END markers)
   - `client_email`: Service account email
   - `client_id`: Client ID from service account
   - `client_x509_cert_url`: Certificate URL

### 3. Verify Model Path

Make sure your trained model exists at:

```
app/models/best_model.h5
```

If your model is elsewhere, update the `MODEL_PATH` variable in `flask_server.py`.

## 🎯 Running the Server

### Start the Flask Server

```powershell
python flask_server.py
```

The server will start on `http://0.0.0.0:5000` and display:

```
🎙️  NOISE ANALYZER FLASK SERVER
======================================================================
Model loaded: ✅
Firebase connected: ✅
VAD calibrated: ❌ (needs calibration)

Endpoints:
  GET  /health       - Health check
  POST /calibrate    - Calibrate VAD with silence samples
  POST /predict      - Predict audio class
  POST /reset_vad    - Reset VAD calibration
  GET  /stats        - Get prediction statistics from Firebase

Starting server on http://0.0.0.0:5000
======================================================================
```

## 🧪 Testing with Test Client

### Run the Test Client

The test client captures audio from your laptop's microphone and sends it to the server.

```powershell
python test_client.py
```

### Advanced Options

```powershell
# Use custom server URL
python test_client.py --server http://192.168.1.100:5000

# Use specific audio device
python test_client.py --device 1

# Skip VAD calibration (if already calibrated)
python test_client.py --skip-calibration

# Skip microphone test
python test_client.py --skip-test

# Custom device identifier
python test_client.py --device-id "laptop_kitchen"
```

### Test Client Workflow

1. **Server Health Check**: Verifies server is running
2. **List Audio Devices**: Shows available microphones
3. **Microphone Test**: 3-second audio level test
4. **VAD Calibration**: 3.5 seconds of silence for calibration
5. **Real-time Classification**: Continuous prediction with 1-second chunks

### Sample Output

```
[2025-01-09 14:32:15] Prediction #5
🔊 VAD: Activity
🎯 CLASSIFICATION: Loud_talking (Confidence: 0.876)
🎤 Audio Level: |████████████░░░░░░░░░░░░░░░░| RMS=0.0234
📈 All Class Probabilities:
   🔊 Loud_talking    : 0.876 |██████████████████████████░░░░|
   🔊 Whispering      : 0.089 |███░░░░░░░░░░░░░░░░░░░░░░░░░░|
   🔇 Silence         : 0.024 |█░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
   🔊 Typing          : 0.008 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
   🔊 Phone_ringing   : 0.003 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
✅ Pushed to Firebase (ID: abc123xyz)
--------------------------------------------------------------------------------
```

## 📡 API Endpoints

### 1. Health Check

**GET** `/health`

Check server status and configuration.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "vad_calibrated": true,
  "firebase_connected": true,
  "timestamp": "2025-01-09T14:32:15.123456"
}
```

### 2. VAD Calibration

**POST** `/calibrate`

Calibrate VAD with silence samples (required before predictions).

**Request:**

```json
{
  "audio": "base64_encoded_float32_array",
  "sample_rate": 16000
}
```

**Response:**

```json
{
  "status": "in_progress",
  "progress": 57.1,
  "message": "Calibration 57.1% complete"
}
```

### 3. Predict Audio

**POST** `/predict`

Classify audio and push to Firebase.

**Request:**

```json
{
  "audio": "base64_encoded_float32_array",
  "sample_rate": 16000,
  "device_id": "laptop_office"
}
```

**Response:**

```json
{
  "timestamp": "2025-01-09T14:32:15.123456",
  "device_id": "laptop_office",
  "predicted_class": "Loud_talking",
  "confidence": 0.876,
  "vad_activity": true,
  "vad_confidence": 0.923,
  "probabilities": {
    "Silence": 0.024,
    "Whispering": 0.089,
    "Typing": 0.008,
    "Phone_ringing": 0.003,
    "Loud_talking": 0.876
  },
  "audio_stats": {
    "rms": 0.0234,
    "max": 0.156
  },
  "firebase_id": "abc123xyz"
}
```

### 4. Reset VAD

**POST** `/reset_vad`

Reset VAD calibration.

**Response:**

```json
{
  "status": "success",
  "message": "VAD calibration reset. Please recalibrate with silence samples."
}
```

### 5. Get Statistics

**GET** `/stats`

Retrieve prediction statistics from Firebase.

**Response:**

```json
{
  "total_predictions": 150,
  "class_distribution": {
    "Silence": 45,
    "Whispering": 12,
    "Typing": 23,
    "Phone_ringing": 8,
    "Loud_talking": 62
  },
  "percentages": {
    "Silence": 30.0,
    "Whispering": 8.0,
    "Typing": 15.3,
    "Phone_ringing": 5.3,
    "Loud_talking": 41.3
  }
}
```

## 🔥 Firebase Data Structure

### Collection: `predictions`

Each prediction document contains:

```javascript
{
  timestamp: Timestamp,
  device_id: String,
  predicted_class: String,
  confidence: Number,
  vad_activity: Boolean,
  vad_confidence: Number,
  probabilities: {
    Silence: Number,
    Whispering: Number,
    Typing: Number,
    Phone_ringing: Number,
    Loud_talking: Number
  },
  audio_stats: {
    rms: Number,
    max: Number
  }
}
```

## 🐛 Troubleshooting

### Firebase Connection Issues

**Problem**: `Firebase not initialized` error

**Solutions**:

1. Verify `firebase-credentials.json` exists and is valid
2. Check Firebase project ID matches: `hawties-2a013`
3. Ensure service account has Firestore permissions

### Model Loading Issues

**Problem**: `Model not found` error

**Solutions**:

1. Verify model exists at `app/models/best_model.h5`
2. Train a model first: `python app/main.py` → Choose option 1
3. Update `MODEL_PATH` in `flask_server.py` if model is elsewhere

### Audio Device Issues

**Problem**: `No input devices found` in test client

**Solutions**:

1. Check microphone is connected and enabled
2. Run `python test_client.py` to list available devices
3. Specify device manually: `python test_client.py --device 0`

### VAD Calibration Issues

**Problem**: Calibration never completes

**Solutions**:

1. Ensure environment is quiet during calibration
2. Check audio is actually being captured (watch RMS values)
3. Reset and retry: `curl -X POST http://localhost:5000/reset_vad`

## 🔒 Security Notes

1. **Firebase Credentials**: Never commit `firebase-credentials.json` to version control
2. **Server Access**: Use firewall rules to restrict access in production
3. **HTTPS**: Use a reverse proxy (nginx, Apache) with SSL for production
4. **Authentication**: Add authentication middleware for production use

## 📊 Monitoring Firebase Data

### Using Firebase Console

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select project: **hawties-2a013**
3. Navigate to: **Firestore Database**
4. Browse the `predictions` collection

### Using Server Stats Endpoint

```powershell
curl http://localhost:5000/stats
```

## 🚀 Deployment Tips

### For Production

1. **Use Gunicorn** instead of Flask development server:

   ```powershell
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 flask_server:app
   ```

2. **Set up NGINX** as reverse proxy with SSL

3. **Use Environment Variables** for configuration:

   ```python
   import os
   SERVER_PORT = os.getenv('PORT', 5000)
   ```

4. **Enable Logging**:

   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

5. **Add Authentication** (e.g., API keys, JWT tokens)

## 📝 Notes

- Audio is sent as **base64-encoded float32 arrays**
- Default sample rate is **16kHz**
- Each prediction processes **1-second chunks**
- VAD calibration requires **3.5 seconds of silence**
- Server supports **CORS** for web client integration

## 🆘 Support

For issues or questions:

1. Check server logs for error messages
2. Verify all dependencies are installed
3. Test with the provided test client first
4. Check Firebase quota limits if pushes fail

## 📜 License

Same as the main project.
