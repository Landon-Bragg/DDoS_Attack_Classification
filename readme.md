DDoS Attack Classification with 3D CNN and Adversarial Robustness

This project builds and evaluates a 3D Convolutional Neural Network (CNN) model to classify DDoS vs Normal network traffic based on preprocessed video-like sequences represented as 4D tensors. The model is trained and evaluated under clean, augmented, and adversarial (FGSM and PGD) attack conditions to test its robustness. 

<img width="500" height="150" alt="Screenshot 2025-07-23 at 3 20 43 PM" src="https://github.com/user-attachments/assets/991b9d4f-8f68-4a53-9409-65fe70df6542" />


Project Structure
- `run_experiments.py`: Trains models under clean and adversarial regimes and evaluates on clean, augmented, FGSM, and PGD conditions.
- `evaluate_timestamps.py`: Evaluates model accuracy at each individual frame (t0 to t7) on clean data.
- `evaluate_adv_timestamps.py`: Evaluates model accuracy at each frame under adversarial augmentations (AUG, PGD, FGSM).
- `evaluate_combined_attacks.py`: Applies augmentation, PGD, and FGSM sequentially to evaluate worst-case scenario accuracy across frames.
- `models/cnn3d.py`: Defines the 3D CNN architecture.
- `data/sequence_dataset.py`: Loads preprocessed `.pt` tensor datasets organized by class (Normal, DDoS).
- `attacks/adversarial_attacks.py`: Defines functions to create FGSM and PGD attack modules.
- `utils.py`: Helper utilities (e.g., reproducibility through random seeding).
- `config.py`: Hyperparameters and configuration for training and evaluation.
- `evaluate.py`: Simple script to read and print evaluation results.

Dataset
- Preprocessed dataset is expected under `preprocessed_dataset/train/Normal`, `preprocessed_dataset/train/DDoS`, `preprocessed_dataset/val/Normal`, and `preprocessed_dataset/val/DDoS`.
- Each file is a PyTorch tensor with shape `[C, D, H, W]`, where:
  - `C` = Channels
  - `D` = Depth (frames)
  - `H`, `W` = Height, Width

Key Features
- 3D CNN trained on temporal slices of traffic.
- Full adversarial training (FGSM + PGD + heavy augmentation).
- Timestamp-specific evaluation to see which frames are most predictive.
- Automatic visualization and result saving.
- Fully GPU-accelerated, including adversarial attack generation.

How to Run
1. Install dependencies:
   pip install -r requirements.txt


2. Train and evaluate:
    python run_experiments.py --config config.py

3. Evaluate clean frame-by-frame
    python evaluate_timestamps.py

4. Evaluate adversarial frame-by-frame
    python evaluate_adv_timestamps.py

Trained model weights: outputs/model_clean.pth, outputs/model_adversarial.pth

Evaluation results: outputs/results.csv

Metrics visualization: outputs/metrics_visualization.png

