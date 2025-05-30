import optuna
import numpy as np
from main import runner, seed_everything
from dataset import OpenAlexGraphDataset

def main():
    dataset_builder = OpenAlexGraphDataset(
        num_authors=-1,
        # use_cache=True,
        use_cache=False,
        use_citation_count=True,
        use_work_count=True,
        use_institution_embedding=True
    )
    
    def objective(trial: optuna.trial.Trial) -> float:
        base_lr = trial.suggest_float("base_lr", 1e-4, 1e-2, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3) 
        dropout = trial.suggest_float("dropout", 0.1, 0.8)
        out_channels = trial.suggest_categorical("out_channels", [16, 32, 64, 128, 256])

        current_config = {
            'base_lr': base_lr,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'dropout': dropout,
            'out_channels': out_channels,
        }
        seed_everything(42)

        run_results = runner(dataset_builder, **current_config)
        final_val_auc = run_results['final_val']['roc_auc']
        final_dev_test_auc = run_results['final_dev_test']['roc_auc']
        g_score = final_dev_test_auc / final_val_auc
    
        if g_score < 0.95:  
            # Too low generalization 
            raise optuna.exceptions.TrialPruned()
        return final_val_auc

    # --- Optuna Study ---
    N_TRIALS = 100 
    N_PARALLEL_JOBS = 16 
    # N_TRIALS = 1
    # N_PARALLEL_JOBS = 1

    print(f"Starting Optuna hyperparameter optimization with {N_TRIALS} trials, {N_PARALLEL_JOBS} parallel jobs.")
    
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
    metrics = ['roc_auc', 'pr_auc']
    results = {
        f'untrained_val_{m}': [] for m in metrics
    }
    results.update({f'untrained_dev_test_{m}': [] for m in metrics})
    results.update({f'untrained_test_{m}': [] for m in metrics})
    results.update({f'final_val_{m}': [] for m in metrics})
    results.update({f'final_dev_test_{m}': [] for m in metrics})
    results.update({f'final_test_{m}': [] for m in metrics})

    N_FINAL_RUNS = 5 
    print(f"Performing {N_FINAL_RUNS} final runs with best config...")
    for i in range(N_FINAL_RUNS):
        print(f"Final run {i+1}/{N_FINAL_RUNS}...")
        seed_everything(i) 
        run_output = runner(dataset_builder, **best_config)
        for phase in ['untrained_val', 'untrained_dev_test', 'untrained_test', 'final_val', 'final_dev_test', 'final_test']:
            for metric in metrics:
                results[f'{phase}_{metric}'].append(run_output[phase][metric])
    
    # Print final statistics
    print(f"\nFinal Statistics over {N_FINAL_RUNS} runs with the best config:")
    print("-" * 50)
    for metric, values_list in results.items():
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