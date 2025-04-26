CONFIG = {
    # Paths
    "data_root": "preprocessed_dataset",
    "output_dir": "outputs",

    # Training 
    "batch_size": 16,
    "epochs": 3,
    "lr": 5e-5,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,

    # Adversarial parameters
    "fgsm_eps": 1.19,
    "pgd_eps": 1.225,
    "pgd_alpha": 1.1, 
    "pgd_steps": 40,
    "max_eval_batches": 100,
    "max_train_batches": None,

    # Model
    "num_classes": 2,

    # Random seed
    "seed": 42,
}