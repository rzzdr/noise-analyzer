import os
from NoiseAnalyzer import NoiseAnalyzer
from RealTimeClassifier import RealTimeClassifier

def main():
    print("Voice Activity Detection + 4-Class Audio Classification System")
    print("for Library Noise Monitoring")
    print("="*70)
    
    # Initialize analyzer
    analyzer = NoiseAnalyzer()
    
    choice = input("Choose mode:\n1. Train new model\n2. Load existing model for real-time prediction\nEnter choice (1 or 2): ")
    
    if choice == '1':
        print("\nTraining mode selected")
        
        # Check if ESC-50 dataset exists
        if not os.path.exists(analyzer.dataset_path):
            print(f"ESC-50 dataset not found at {analyzer.dataset_path}")
            print("Please download ESC-50 dataset from: https://github.com/karolpiczak/ESC-50")
            print("Extract it to the current directory as 'ESC-50-master'")
            return
        
        try:
            # Load and prepare dataset
            features, labels, labels_raw = analyzer.load_esc50_dataset()
            
            # Train model
            history = analyzer.train_model(features, labels, labels_raw)
            
            # Ask if user wants to test real-time prediction
            test_realtime = input("\nWould you like to test real-time prediction? (y/n): ")
            if test_realtime.lower() == 'y':
                rt_classifier = RealTimeClassifier(analyzer)
                rt_classifier.start_real_time_prediction()
            
        except Exception as e:
            print(f"Error during training: {e}")
            return
    
    elif choice == '2':
        print("\nReal-time prediction mode selected")
        
        # Load pre-trained model
        model_path = input("Enter model path (default: models/best_model.h5): ").strip()
        if not model_path:
            model_path = 'models/best_model.h5'
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
        
        try:
            analyzer.load_model(model_path)
            
            # Initialize real-time classifier
            rt_classifier = RealTimeClassifier(analyzer)
            rt_classifier.start_real_time_prediction()
            
        except Exception as e:
            print(f"Error in real-time prediction: {e}")
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()