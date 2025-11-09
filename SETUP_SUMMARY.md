# 🎙️ Flask Server & Test Client - Complete Solution

## 📁 Files Created

### Core Server Files

1. **`flask_server.py`** - Main Flask server with Firebase integration
2. **`test_client.py`** - Laptop test client for audio streaming
3. **`test_firebase_connection.py`** - Firebase credentials validator

### Configuration Files

4. **`firebase-credentials-template.json`** - Template for Firebase setup
5. **`requirements_server.txt`** - Python dependencies for server

### Documentation

6. **`README_SERVER.md`** - Complete API and deployment documentation
7. **`QUICKSTART_SERVER.md`** - Quick start guide (this file)
8. **`SETUP_SUMMARY.md`** - This summary file

## 🎯 What This Solution Does

### Flask Server (`flask_server.py`)

- ✅ Receives audio data via HTTP POST requests
- ✅ Performs Voice Activity Detection (VAD)
- ✅ Runs noise classification inference
- ✅ Pushes results to Firebase Firestore
- ✅ Provides REST API endpoints for:
  - Health checks
  - VAD calibration
  - Audio prediction
  - Statistics retrieval

### Test Client (`test_client.py`)

- ✅ Captures audio from laptop microphone
- ✅ Sends audio to server for processing
- ✅ Displays real-time classification results
- ✅ Shows colorful terminal output with:
  - VAD status (🔊 Activity / 🔇 Silence)
  - Classification results
  - Confidence scores
  - All class probabilities
  - Firebase push confirmation

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Dependencies

```powershell
pip install -r requirements_server.txt
```

### Step 2: Configure Firebase

```powershell
# Download service account key from Firebase Console
# Save as firebase-credentials.json
```

Or manually:

1. Visit: https://console.firebase.google.com/project/hawties-2a013/settings/serviceaccounts/adminsdk
2. Click "Generate new private key"
3. Save as `firebase-credentials.json`

### Step 3: Test Firebase (Optional but Recommended)

```powershell
python test_firebase_connection.py
```

## ▶️ Running the System

### Terminal 1: Start Server

```powershell
python flask_server.py
```

Wait for:

```
✅ Model loaded successfully
✅ Firebase initialized successfully
Starting server on http://0.0.0.0:5000
```

### Terminal 2: Run Test Client

```powershell
python test_client.py
```

The client will automatically:

1. Check server health
2. Test your microphone
3. Calibrate VAD (3.5 sec silence)
4. Start real-time predictions

## 📊 Data Flow

```
┌─────────────────┐
│  Laptop Mic     │
│  (Test Client)  │
└────────┬────────┘
         │ Base64 encoded
         │ audio (1-sec chunks)
         ▼
┌─────────────────┐
│  Flask Server   │
│  - VAD          │
│  - Inference    │
└────────┬────────┘
         │ Prediction
         │ results
         ▼
┌─────────────────┐
│  Firebase       │
│  Firestore DB   │
└─────────────────┘
```

## 🔌 API Endpoints

| Endpoint     | Method | Purpose                    |
| ------------ | ------ | -------------------------- |
| `/health`    | GET    | Check server status        |
| `/calibrate` | POST   | Calibrate VAD with silence |
| `/predict`   | POST   | Classify audio chunk       |
| `/reset_vad` | POST   | Reset VAD calibration      |
| `/stats`     | GET    | Get prediction statistics  |

## 📦 Request/Response Format

### Predict Request

```json
{
  "audio": "base64_encoded_float32_array",
  "sample_rate": 16000,
  "device_id": "laptop_office"
}
```

### Predict Response

```json
{
  "timestamp": "2025-01-09T14:32:15.123456",
  "predicted_class": "Loud_talking",
  "confidence": 0.876,
  "vad_activity": true,
  "probabilities": {
    "Silence": 0.024,
    "Whispering": 0.089,
    "Typing": 0.008,
    "Phone_ringing": 0.003,
    "Loud_talking": 0.876
  },
  "firebase_id": "abc123xyz"
}
```

## 🔥 Firebase Configuration

### Your Firebase Config

```javascript
const firebaseConfig = {
  apiKey: "AIzaSyDyH0LXgzkikiCTxgsw0ebEmjjQ0vkOl-w",
  authDomain: "hawties-2a013.firebaseapp.com",
  projectId: "hawties-2a013",
  storageBucket: "hawties-2a013.firebasestorage.app",
  messagingSenderId: "523392422092",
  appId: "1:523392422092:web:f63642ccc26c3888a1b269",
  measurementId: "G-JCWEDXQL6W",
};
```

### Firestore Structure

```
predictions/
  ├── {document_id_1}/
  │   ├── timestamp: Timestamp
  │   ├── device_id: "laptop_office"
  │   ├── predicted_class: "Loud_talking"
  │   ├── confidence: 0.876
  │   ├── vad_activity: true
  │   ├── vad_confidence: 0.923
  │   ├── probabilities: { ... }
  │   └── audio_stats: { rms, max }
  └── {document_id_2}/
      └── ...
```

## 🛠️ Test Client Options

### Basic Usage

```powershell
python test_client.py
```

### Advanced Options

```powershell
# Custom server URL (for remote testing)
python test_client.py --server http://192.168.1.100:5000

# Specific audio device
python test_client.py --device 1

# Skip tests (if already verified)
python test_client.py --skip-test --skip-calibration

# Custom device identifier
python test_client.py --device-id "office_laptop"
```

## 📈 Sample Output

### Server Console

```
✅ Pushed to Firebase: Loud_talking (0.876)
✅ Pushed to Firebase: Silence (0.923)
✅ Pushed to Firebase: Typing (0.654)
✅ Pushed to Firebase: Whispering (0.712)
```

### Client Console

```
[2025-01-09 14:32:15] Prediction #5
🔊 VAD: Activity
🎯 CLASSIFICATION: Loud_talking (Confidence: 0.876)
🎤 Audio Level: |████████████░░░░| RMS=0.0234
📈 All Class Probabilities:
   🔊 Loud_talking    : 0.876 |██████████████████████████░░░░|
   🔊 Whispering      : 0.089 |███░░░░░░░░░░░░░░░░░░░░░░░░░░|
   🔇 Silence         : 0.024 |█░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
   🔊 Typing          : 0.008 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
   🔊 Phone_ringing   : 0.003 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
✅ Pushed to Firebase (ID: abc123xyz)
```

## 🔍 Troubleshooting Guide

### Problem: Server won't start

**Check:**

- [ ] Model file exists: `app/models/best_model.h5`
- [ ] Firebase credentials exist: `firebase-credentials.json`
- [ ] Dependencies installed: `pip install -r requirements_server.txt`

**Test:**

```powershell
python test_firebase_connection.py
```

### Problem: Client can't connect

**Check:**

- [ ] Server is running in another terminal
- [ ] Server shows "Starting server on http://0.0.0.0:5000"
- [ ] Firewall allows port 5000

**Test:**

```powershell
curl http://localhost:5000/health
```

### Problem: No audio detected

**Check:**

- [ ] Microphone is connected
- [ ] Microphone is not muted
- [ ] Correct device selected

**Test:**

```powershell
python test_client.py --device 0  # Try different device numbers
```

### Problem: Firebase errors

**Check:**

- [ ] `firebase-credentials.json` has all fields filled
- [ ] Project ID is "hawties-2a013"
- [ ] Service account has Firestore permissions

**Test:**

```powershell
python test_firebase_connection.py
```

## 🌐 Remote Testing (Different Computers)

### On Server Machine:

```powershell
# Find IP address
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.100)

# Start server
python flask_server.py
```

### On Client Machine:

```powershell
# Install dependencies
pip install numpy sounddevice requests

# Connect to remote server
python test_client.py --server http://192.168.1.100:5000
```

## 📊 Viewing Results

### Firebase Console

1. Visit: https://console.firebase.google.com/project/hawties-2a013/firestore
2. Open `predictions` collection
3. See real-time predictions!

### Server Stats Endpoint

```powershell
curl http://localhost:5000/stats
```

Returns:

```json
{
  "total_predictions": 150,
  "class_distribution": {
    "Silence": 45,
    "Whispering": 12,
    "Typing": 23,
    "Phone_ringing": 8,
    "Loud_talking": 62
  }
}
```

## 🎓 Classification Classes

| Class             | Description           | Examples               |
| ----------------- | --------------------- | ---------------------- |
| **Silence**       | Background noise only | AC hum, quiet room     |
| **Whispering**    | Quiet speech          | Whispers, soft talking |
| **Typing**        | Keyboard sounds       | Typing, clicking       |
| **Phone_ringing** | Alert sounds          | Phone rings, alarms    |
| **Loud_talking**  | Normal to loud speech | Conversation, laughter |

## 🔒 Security Considerations

### For Development

- ✅ Server runs on localhost (127.0.0.1)
- ✅ CORS enabled for testing
- ✅ Firebase credentials in local file

### For Production

- ⚠️ Use HTTPS with SSL certificate
- ⚠️ Add authentication (API keys, JWT)
- ⚠️ Restrict CORS origins
- ⚠️ Use environment variables for secrets
- ⚠️ Deploy with Gunicorn + Nginx
- ⚠️ Set up firewall rules

## 📝 Complete File List

```
noise-analyzer/
├── flask_server.py                    # Main server
├── test_client.py                     # Test client
├── test_firebase_connection.py        # Firebase tester
├── requirements_server.txt            # Server dependencies
├── firebase-credentials.json          # Your credentials (DO NOT COMMIT!)
├── firebase-credentials-template.json # Template
├── README_SERVER.md                   # Full documentation
├── QUICKSTART_SERVER.md              # Quick start guide
├── SETUP_SUMMARY.md                  # This file
└── app/
    ├── NoiseAnalyzer.py              # Model inference
    ├── VAD.py                        # Voice Activity Detection
    └── models/
        └── best_model.h5             # Trained model
```

## ✅ Pre-flight Checklist

Before running, verify:

**Server Side:**

- [ ] Python 3.8+ installed
- [ ] `pip install -r requirements_server.txt` completed
- [ ] `firebase-credentials.json` exists and is valid
- [ ] Model file `app/models/best_model.h5` exists
- [ ] Port 5000 is available

**Client Side:**

- [ ] Microphone connected and working
- [ ] Can reach server (ping or curl)
- [ ] Dependencies installed (numpy, sounddevice, requests)

**Firebase:**

- [ ] Project ID is "hawties-2a013"
- [ ] Service account has Firestore read/write permissions
- [ ] Firestore database is created

## 🎉 Success Indicators

You'll know everything is working when:

1. **Server starts cleanly:**

   ```
   ✅ Model loaded successfully
   ✅ Firebase initialized successfully
   ```

2. **Client connects:**

   ```
   ✅ Server health check passed
   ✅ Microphone test passed!
   ✅ VAD calibration complete!
   ```

3. **Predictions flow:**

   ```
   [timestamp] Prediction #X
   🎯 CLASSIFICATION: ...
   ✅ Pushed to Firebase (ID: ...)
   ```

4. **Data appears in Firebase Console**

## 🚀 Next Steps

1. **Test locally first:**

   - Run server and client on same machine
   - Verify predictions appear in Firebase

2. **Try remote testing:**

   - Run server on one machine
   - Run client on another
   - Use `--server` flag with IP address

3. **Customize for your use case:**

   - Modify `device_id` in requests
   - Add more endpoints as needed
   - Integrate with web frontend

4. **Deploy to production:**
   - Use Gunicorn instead of Flask dev server
   - Add NGINX reverse proxy with SSL
   - Set up monitoring and logging

## 📚 Additional Resources

- **Full API Docs:** `README_SERVER.md`
- **Quick Start:** `QUICKSTART_SERVER.md`
- **Firebase Console:** https://console.firebase.google.com/project/hawties-2a013
- **Test Firebase:** `python test_firebase_connection.py`

## 🆘 Getting Help

If you encounter issues:

1. **Check server logs** - Look for error messages
2. **Test Firebase** - Run `test_firebase_connection.py`
3. **Verify model** - Check `app/models/best_model.h5` exists
4. **Test microphone** - Run test client with `--skip-calibration`
5. **Check firewall** - Ensure port 5000 is open

## 🎊 You're All Set!

You now have:

- ✅ Complete Flask server with VAD and inference
- ✅ Firebase integration for data persistence
- ✅ Test client for laptop-based testing
- ✅ Full documentation and troubleshooting guides

**Start testing now:**

```powershell
# Terminal 1
python flask_server.py

# Terminal 2
python test_client.py
```

Happy testing! 🎉
