CONFIG = {
    # Paths
    "data_root": "preprocessed_dataset",
    "output_dir": "outputs",

    # Training DO NOT GO ABOVE 16
    "batch_size": 16,
    "epochs": 3,
    "lr": 5e-5,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,

    # Adversarial parameters
    "fgsm_eps": 1.2,
    "pgd_eps": 1.225,
    "pgd_alpha": 1.1, # look into this
    # The higher you make this value, the slower everything takes to run 
    "pgd_steps": 40,
    # For some reason if you run less batches, it does worse on clean data but better on augmented and pgd
    "max_eval_batches": 100,
    "max_train_batches": None,

    # Model
    "num_classes": 2,

    # Random seed
    "seed": 42,
}