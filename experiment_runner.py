import copy
import argparse
import pandas as pd
from train import Trainer, config as base_config
from torchattacks import FGSM

def run_experiment(mode, epochs):
    # Set up the config with the chosen mode.
    config = copy.deepcopy(base_config)
    config["epochs"] = epochs
    config["training_mode"] = mode  # "clean" or "augmented"

    print(f"\nTraining the model in {mode} training mode...")
    trainer = Trainer(config)
    trainer.run()

    # Always evaluate on clean data, regardless of training mode.
    print("\nEvaluating on clean data...")
    clean_metrics = trainer.evaluate_model(evaluation_mode="clean")


    # Evaluate on augmented data.
    print("\nEvaluating on augmented data...")
    aug_metrics = trainer.evaluate_augmented()

    # Evaluate on adversarial data using PGD.
    print("\nEvaluating on adversarial data (PGD)...")
    adv_metrics_pgd = trainer.evaluate_model(evaluation_mode="adversarial")

    # Evaluate on adversarial data using FGSM.
    from torchattacks import FGSM
    trainer.attacker = FGSM(trainer.model, eps=1.33)
    print("\nEvaluating on adversarial data (FGSM)...")
    adv_metrics_fgsm = trainer.evaluate_model(evaluation_mode="adversarial")

    # Prepare results.
    results = []
    if mode == "clean":
        results.append({
            "Experiment": "Clean Evaluation",
            "Accuracy": round(clean_metrics[0], 2),
            "Precision": round(clean_metrics[1], 2),
            "Recall": round(clean_metrics[2], 2)
        })
    results.append({
        "Experiment": "Augmented Evaluation",
        "Accuracy": round(aug_metrics[0], 2),
        "Precision": round(aug_metrics[1], 2),
        "Recall": round(aug_metrics[2], 2)
    })
    results.append({
        "Experiment": "Adversarial Evaluation (PGD)",
        "Accuracy": round(adv_metrics_pgd[0], 2),
        "Precision": round(adv_metrics_pgd[1], 2),
        "Recall": round(adv_metrics_pgd[2], 2)
    })
    results.append({
        "Experiment": "Adversarial Evaluation (FGSM)",
        "Accuracy": round(adv_metrics_fgsm[0], 2),
        "Precision": round(adv_metrics_fgsm[1], 2),
        "Recall": round(adv_metrics_fgsm[2], 2)
    })

    df = pd.DataFrame(results)
    print(f"\nFinal Research Table for {mode} training mode:")
    print(df)
    return df

def main():
    parser = argparse.ArgumentParser(description="Run experiments in different training modes.")
    parser.add_argument('--mode', type=str, default="both",
                        choices=["clean", "augmented", "both"],
                        help='Training mode: clean, augmented, or both.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for training.')
    args = parser.parse_args()

    if args.mode == "both":
        df_clean = run_experiment("clean", args.epochs)
        df_aug = run_experiment("augmented", args.epochs)
        # Optionally combine the results in one file or display both tables.
    else:
        run_experiment(args.mode, args.epochs)

if __name__ == "__main__":
    main()
