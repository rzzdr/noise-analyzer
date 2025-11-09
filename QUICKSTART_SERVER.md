# Quick Start Guide for Flask Server

## 📦 Installation Steps

### 1. Install Server Dependencies

```powershell
pip install -r requirements_server.txt
```

### 2. Get Firebase Credentials

**Method A: Download from Firebase Console**

1. Visit: https://console.firebase.google.com/project/hawties-2a013/settings/serviceaccounts/adminsdk
2. Click "Generate new private key"
3. Save as `firebase-credentials.json` in project root

**Method B: Manual Setup**

1. Copy the template: `cp firebase-credentials-template.json firebase-credentials.json`
2. Fill in all the fields from your Firebase service account

### 3. Verify Model Exists

```powershell
# Check if model file exists
Test-Path app\models\best_model.h5
```

If it returns `False`, train a model first:

```powershell
cd app
python main.py
# Choose option 1 to train
```

## 🚀 Running the System

### Terminal 1: Start Flask Server

```powershell
python flask_server.py
```

Wait for the server to show:

```
✅ Model loaded successfully
✅ Firebase initialized successfully
Starting server on http://0.0.0.0:5000
```

### Terminal 2: Run Test Client

```powershell
python test_client.py
```

The test client will:

1. Check server health ✅
2. List your audio devices 🎤
3. Test your microphone 🔊
4. Calibrate VAD (3.5 seconds of silence) 🔇
5. Start real-time predictions 🎯

## 🎯 Expected Flow

### Server Terminal Output:

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

✅ Pushed to Firebase: Loud_talking (0.876)
✅ Pushed to Firebase: Silence (0.923)
✅ Pushed to Firebase: Typing (0.654)
```

### Client Terminal Output:

```
🎙️  AUDIO CLASSIFICATION TEST CLIENT
======================================================================

1. Checking server health...
✅ Server health check passed
   Model loaded: True
   VAD calibrated: False
   Firebase connected: True

2. Listing audio devices...
📱 Available audio input devices:
   0: Microphone Array (DEFAULT)
   ...

3. Testing microphone...
🎤 Testing microphone for 3 seconds...
   [1/3] |████████████░░░░| RMS=0.0123
   [2/3] |██████████░░░░░░| RMS=0.0098
   [3/3] |███████████░░░░░| RMS=0.0112
   ✅ Microphone test passed!

4. Calibrating VAD...
🔇 VAD Calibration Phase (keep quiet for 3.5 seconds)
   Progress: |████████████████████████████████████████| 100.0%
   ✅ VAD calibration complete!

5. Starting real-time classification...
🔊 Starting real-time audio classification
   Press Ctrl+C to stop

[2025-01-09 14:32:15] Prediction #1
🔊 VAD: Activity
🎯 CLASSIFICATION: Loud_talking (Confidence: 0.876)
🎤 Audio Level: |████████████░░░░| RMS=0.0234
📈 All Class Probabilities:
   🔊 Loud_talking    : 0.876 |██████████████████████████░░░░|
   🔊 Whispering      : 0.089 |███░░░░░░░░░░░░░░░░░░░░░░░░░░|
   🔇 Silence         : 0.024 |█░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
✅ Pushed to Firebase (ID: abc123xyz)
```

## 🔧 Troubleshooting

### Issue: "Cannot connect to server"

**Solution**: Make sure server is running in another terminal

### Issue: "Firebase not initialized"

**Solution**:

1. Check `firebase-credentials.json` exists
2. Verify all fields are filled (no "YOUR_XXX_HERE")
3. Check project ID is "hawties-2a013"

### Issue: "Model not found"

**Solution**: Train model first:

```powershell
cd app
python main.py
# Choose option 1
```

### Issue: "No audio detected"

**Solution**:

1. Check microphone is connected and unmuted
2. Try different device: `python test_client.py --device 1`
3. Check Windows audio settings

### Issue: "VAD calibration failed"

**Solution**:

1. Ensure room is quiet during calibration
2. Don't move or speak during 3.5 seconds
3. Reset and retry:
   ```powershell
   curl -X POST http://localhost:5000/reset_vad
   python test_client.py --skip-test
   ```

## 🌐 Testing from Another Computer

### On Server Computer:

```powershell
# Find your IP address
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.100)

# Start server
python flask_server.py
```

### On Client Computer:

```powershell
# Install client dependencies only
pip install numpy sounddevice requests

# Run client with server IP
python test_client.py --server http://192.168.1.100:5000
```

## 📊 View Results in Firebase

1. Go to: https://console.firebase.google.com/project/hawties-2a013/firestore
2. Open `predictions` collection
3. See real-time predictions appearing!

Each document shows:

- Timestamp
- Device ID
- Predicted class
- Confidence
- All probabilities
- Audio statistics

## 🎓 Understanding the Output

### VAD Status

- 🔊 Activity: Sound detected (will classify)
- 🔇 Silence: No significant sound (skips classification)

### Classification Classes

- **Silence**: Background noise only
- **Whispering**: Quiet speech/whispers
- **Typing**: Keyboard typing sounds
- **Phone_ringing**: Phone/alarm sounds
- **Loud_talking**: Normal to loud speech

### Confidence Levels

- **> 0.7**: High confidence (green)
- **0.5 - 0.7**: Medium confidence (yellow)
- **< 0.5**: Low confidence (may need recalibration)

## 🛑 Stopping the System

### Stop Test Client

Press `Ctrl+C` in client terminal

### Stop Flask Server

Press `Ctrl+C` in server terminal

## 📈 Advanced Usage

### Custom Device ID

```powershell
python test_client.py --device-id "office_mic_1"
```

### Skip Tests (if already verified)

```powershell
python test_client.py --skip-test --skip-calibration
```

### View Statistics

```powershell
curl http://localhost:5000/stats
```

### Reset VAD Calibration

```powershell
curl -X POST http://localhost:5000/reset_vad
```

## ✅ Checklist

Before starting, ensure:

- [ ] `flask_server.py` exists
- [ ] `test_client.py` exists
- [ ] `requirements_server.txt` dependencies installed
- [ ] `firebase-credentials.json` configured
- [ ] `app/models/best_model.h5` exists
- [ ] Microphone is connected and working
- [ ] Internet connection for Firebase

## 🎉 Success Indicators

You'll know it's working when you see:

1. Server shows "✅ Firebase initialized successfully"
2. Client completes VAD calibration
3. Predictions appear in real-time
4. Firebase IDs are shown (e.g., "abc123xyz")
5. Data appears in Firebase Console

## 📞 Getting Help

If you encounter issues:

1. Check both terminal outputs for errors
2. Verify microphone with `test_client.py --skip-calibration --skip-test`
3. Test server health: `curl http://localhost:5000/health`
4. Review `README_SERVER.md` for detailed documentation
