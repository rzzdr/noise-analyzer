# 5-Class Audio Classification System for Library Noise Monitoring

A deep learning system for real-time classification of library noise into 5 categories: Silence, Whispering, Typing, Phone_ringing, and Loud_talking.

## Features

- **Dataset**: Uses ESC-50 dataset with custom mapping to target classes
- **Architecture**: CNN-RNN hybrid model (Conv1D + GRU)
- **Data Augmentation**: Time stretching, pitch shifting, noise addition, time shifting
- **Real-time Classification**: Live microphone input processing
- **Comprehensive Evaluation**: Confusion matrix, F1-scores, training curves

## Quick Start

1. **Prerequisites**:
   - Python 3.11+
   - Poetry (install from https://python-poetry.org/docs/#installation)

2. **Setup Environment**:
   ```bash
   # Clone or navigate to project directory
   cd noise-analyzer
   
   # Install dependencies using Poetry
   poetry install
   ```

3. **Run the System**:
   ```bash
   # Activate Poetry virtual environment and run
   poetry run python main.py
   
   # Or activate shell and run directly
   poetry shell
   python main.py
   ```

4. **Choose Mode**:
   - Option 1: Train new model (requires ESC-50 dataset)
   - Option 2: Load existing model for real-time prediction

## Model Architecture

```
Input: (98, 40) - 1-second mel spectrogram
├── Conv1D(16) + MaxPool + Dropout(0.2)
├── Conv1D(32) + MaxPool + Dropout(0.2)  
├── Conv1D(64) + MaxPool + Dropout(0.2)
├── GRU(32) + Dropout(0.25)
├── Dense(64) + Dropout(0.25)
└── Dense(5, softmax)
```

## Audio Features

- **Sample Rate**: 16 kHz
- **Window Size**: 1 second
- **Mel Filters**: 40
- **Time Frames**: ~98 frames
- **FFT Size**: 512

## Class Mapping (ESC-50 → Target)

- **Silence**: breathing, silence, synthetic low-energy
- **Whispering**: sneezing (closest approximation)
- **Typing**: keyboard_typing, mouse_click
- **Phone_ringing**: phone_ringing, alarm_clock
- **Loud_talking**: laughing, coughing, crying_baby, snoring, children_playing

## Training Configuration

- **Split**: 70% train / 15% validation / 15% test
- **Batch Size**: 32
- **Epochs**: 50 (early stopping at 10 patience)
- **Optimizer**: Adam (lr=0.001)
- **Callbacks**: ModelCheckpoint, ReduceLROnPlateau, TensorBoard

## Real-time Prediction

- **Input**: Live microphone audio
- **Processing**: 1-second windows with 50% overlap
- **Output**: Class predictions with confidence scores
- **Logging**: Saves predictions to CSV for analysis

## Output Files

After training:
- `best_model.h5` - Best model checkpoint
- `noise_classifier_model.h5` - Final trained model
- `confusion_matrix.png` - Test set confusion matrix
- `training_curves.png` - Training/validation curves
- `real_time_predictions.csv` - Real-time prediction log

## Requirements

- Python 3.11+
- Poetry (for dependency management)
- Core dependencies managed by Poetry:
  - TensorFlow 2.20+
  - librosa 0.11+
  - sounddevice 0.5+
  - numpy 2.3+
  - pandas 2.3+
  - scikit-learn 1.7+
  - matplotlib 3.10+
  - seaborn 0.13+

All dependencies are automatically managed through `pyproject.toml`

## Usage Examples

### Training Mode
```bash
# Using Poetry
poetry run python main.py
# Choose option 1
# Wait for training to complete
# Optionally test real-time prediction

# Or in Poetry shell
poetry shell
python main.py
```

### Real-time Prediction Only
```bash
# Using Poetry
poetry run python main.py
# Choose option 2
# Enter model path (or use default)
# Speak into microphone to see classifications

# Or in Poetry shell
poetry shell
python main.py
```

## Model Performance

The system provides:
- Overall accuracy on test set
- Per-class F1-scores
- Detailed confusion matrix
- Real-time confidence scores

## Installation & Setup

### Prerequisites
1. **Install Poetry**:
   ```bash
   # On Linux/macOS
   curl -sSL https://install.python-poetry.org | python3 -
   
   # On Windows (PowerShell)
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```

2. **Install Dependencies**:
   ```bash
   # Navigate to project directory
   cd noise-analyzer
   
   # Install all dependencies
   poetry install
   
   # Verify installation
   poetry run python -c "import tensorflow, librosa, sounddevice; print('All dependencies installed successfully!')"
   ```

### ESC-50 Dataset Setup
1. Download ESC-50 dataset from https://github.com/karolpiczak/ESC-50
2. Extract to project directory as `data/ESC-50-master/`
3. Verify structure: `ESC-50-master/audio/` and `ESC-50-master/meta/esc50.csv` should exist

## Troubleshooting

1. **ESC-50 Dataset**: Ensure proper extraction in project root as `ESC-50-master/`
2. **Audio Issues**: Ensure microphone permissions and proper audio drivers
3. **Memory Issues**: Reduce `BATCH_SIZE` in `main.py` or use GPU acceleration
4. **Dependencies**: Run `poetry install` or `poetry update` to refresh dependencies
5. **Python Version**: Ensure Python 3.11+ is installed and accessible to Poetry

## Development with Poetry

### Common Commands
```bash
# Install dependencies
poetry install

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Activate virtual environment
poetry shell

# Run commands in Poetry environment
poetry run python main.py
poetry run pytest  # if tests are added

# Export requirements (if needed for compatibility)
poetry export -f requirements.txt --output requirements.txt
```

### Project Structure
```
noise-analyzer/
├── main.py                    # Main application entry point
├── pyproject.toml            # Poetry configuration & dependencies
├── README.md                 # Project documentation
├── model_params.npz          # Saved normalization parameters
├── real_time_predictions.csv # Real-time prediction logs
├── ESC-50-master/           # ESC-50 dataset (user provided)
│   ├── audio/               # Audio files
│   └── meta/esc50.csv       # Metadata
├── logs/                    # TensorBoard training logs
│   ├── train/
│   └── validation/
└── models/                  # Saved model files
    ├── best_model.h5
    └── noise_classifier_model.h5
```

## License

This project is for educational and research purposes.