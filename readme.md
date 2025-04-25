```
DDoS_Attack_Classification/
├── README.md
├── requirements.txt
├── config.py
├── preprocess_hiveplots.py
├── data/
│   └── hive_plot_dataset.py
├── models/
│   ├── cnn.py
│   └── transformer.py
├── attacks/
│   └── adversarial_attacks.py
├── utils.py
├── adversarial_train.py
└── evaluate.py
```
# Adversarially Robust DDoS Classification

This project implements and evaluates adversarially robust deep‑learning models to classify Distributed Denial-of-Service (DDoS) attacks using hive‑plot images of network traffic.

## Project Structure
See the tree above.

## Setup
pip install -r requirements.txt

Usage

Preprocess network flow graphs into hive‑plot images (skipped if dataset is ready):

python preprocess_hiveplots.py --input_dir raw_graphs/ --output_dir data/hive_images/

Train baseline model on clean data:
python train.py --config config.py

Train adversarially robust model:
python adversarial_train.py --config config.py

Evaluate models:
python evaluate.py --config config.py
