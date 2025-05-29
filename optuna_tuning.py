import optuna
import numpy as np
from main import runner

def main():
    # Define the objective function for Optuna
    def objective(trial: optuna.trial.Trial) -> float:
        # Define hyperparameter search space
        # Using broader ranges for Optuna to explore effectively
        base_lr = trial.suggest_float("base_lr", 1e-4, 1e-2, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3) # e.g., 1, 2, or 3 layers
        dropout = trial.suggest_float("dropout", 0.1, 0.8)
        out_channels = trial.suggest_categorical("out_channels", [16, 32, 64, 128, 256])

        current_config = {
            'base_lr': base_lr,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'dropout': dropout,
            'out_channels': out_channels,
            # Other fixed parameters from runner's CONFIG like weight_decay are not tuned here
            # but will be used by runner.
        }
        
        # runner calls seed_everything(42) internally, so each run for a given
        # config is deterministic. One run per trial is sufficient.
        run_results = runner(**current_config)
        final_val_auc = run_results['final_val']
        
        # Optuna will log basic trial info. You can add more logging here if needed.
        # print(f"Trial {trial.number} - Config: {current_config} -> Val AUC: {final_val_auc:.4f}")

        return final_val_auc

    # --- Optuna Study ---
    N_TRIALS = 50  # Number of hyperparameter combinations to try
    N_PARALLEL_JOBS = 16 # Number of trials to run in parallel. Adjust based on VRAM.
                        # (e.g., 6 jobs * 0.5GB/job = 3GB VRAM)

    print(f"Starting Optuna hyperparameter optimization with {N_TRIALS} trials, {N_PARALLEL_JOBS} parallel jobs.")
    
    # Create a study object. ` средством ` (storage) can be specified for distributed optimization.
    # For local parallel, the default in-memory storage with `n_jobs` is fine.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_PARALLEL_JOBS)

    # --- Get Best Results ---
    best_trial = study.best_trial
    best_config = best_trial.params
    best_val_auc_from_study = best_trial.value

    print("\nOptuna optimization finished.")
    print("Best trial:")
    print(f"  Value (Maximized Val AUC): {best_val_auc_from_study:.4f}")
    print("  Params: ")
    for key, value in best_config.items():
        print(f"    {key}: {value}")

    # --- Final Evaluation with Best Config ---
    print("\nRunning final evaluation with the best configuration found by Optuna...")
    final_results_metrics = {
        'final_val': [],
        'final_test': [],
        'untrained_val': [],
        'untrained_test': []
    }
    
    N_FINAL_RUNS = 5 # Number of times to run the best config for final stats
    print(f"Performing {N_FINAL_RUNS} final runs with best config...")
    for i in range(N_FINAL_RUNS):
        print(f"Final run {i+1}/{N_FINAL_RUNS}...")
        # runner will use its internal seed_everything(42) for each of these runs
        run_output = runner(**best_config)
        for metric_key in final_results_metrics.keys():
            final_results_metrics[metric_key].append(run_output[metric_key])
    
    # Print final statistics
    print(f"\nFinal Statistics over {N_FINAL_RUNS} runs with the best config:")
    print("-" * 50)
    for metric, values_list in final_results_metrics.items():
        values_np = np.array(values_list)
        mean_val = np.mean(values_np)
        std_val = np.std(values_np)
        print(f"{metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
    print("-" * 50)

    # Print best config in a more readable format
    print("\nBest configuration details (used for final evaluation):")
    print("-" * 30)
    for param, value in best_config.items():
        print(f"{param:17s}: {value}")
    print("-" * 30)

if __name__ == "__main__":
    main()