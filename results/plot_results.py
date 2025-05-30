import matplotlib.pyplot as plt
import numpy as np

# Data
authors = ['100', '200', '400', '800', '1600', 'All']
x = np.arange(len(authors))

# Colors
colors = {
    'val': ('lightcoral', 'red'),
    'dev': ('lightblue', 'blue'),
    'test': ('lightgreen', 'green')
}

# Untrained and trained values with standard deviations
def extract_metrics(data):
    means = [m for m, _ in data]
    stds = [s for _, s in data]
    return means, stds

# Parse the raw values
def parse(values):
    return [(float(x.split('±')[0]), float(x.split('±')[1])) for x in values]

# ROC AUC
untrained_roc = {
    'val': parse(['0.7701 ± 0.0858', '0.8832 ± 0.1029', '0.8458 ± 0.0685', '0.1485 ± 0.0643', '0.2783 ± 0.0700', '0.5966 ± 0.0647']),
    'dev': parse(['0.8424 ± 0.0996', '0.9095 ± 0.0824', '0.9518 ± 0.0635', '0.8251 ± 0.0442', '0.6412 ± 0.0477', '0.4006 ± 0.0522']),
    'test': parse(['0.7087 ± 0.0884', '0.8957 ± 0.0611', '0.9292 ± 0.0478', '0.8837 ± 0.0393', '0.4838 ± 0.0726', '0.5666 ± 0.0741'])
}

trained_roc = {
    'val': parse(['0.9017 ± 0.0202', '0.9483 ± 0.0055', '0.9667 ± 0.0030', '0.9286 ± 0.0185', '0.8771 ± 0.0200', '0.9450 ± 0.0102']),
    'dev': parse(['0.8813 ± 0.0227', '0.9700 ± 0.0042', '0.9845 ± 0.0050', '0.9689 ± 0.0075', '0.8883 ± 0.0350', '0.9013 ± 0.0210']),
    'test': parse(['0.9058 ± 0.0303', '0.9209 ± 0.0155', '0.9865 ± 0.0044', '0.9827 ± 0.0046', '0.7914 ± 0.0580', '0.8670 ± 0.0296'])
}

# PR AUC
untrained_pr = {
    'val': parse(['0.7813 ± 0.1261', '0.7773 ± 0.1223', '0.8259 ± 0.0833', '0.3544 ± 0.0143', '0.4322 ± 0.0264', '0.6840 ± 0.0716']),
    'dev': parse(['0.8255 ± 0.1428', '0.8161 ± 0.1146', '0.8927 ± 0.1158', '0.8352 ± 0.0963', '0.7199 ± 0.0742', '0.5275 ± 0.0398']),
    'test': parse(['0.7522 ± 0.1341', '0.8485 ± 0.0874', '0.9031 ± 0.0809', '0.8863 ± 0.0835', '0.5916 ± 0.0603', '0.6516 ± 0.0692'])
}

trained_pr = {
    'val': parse(['0.8540 ± 0.0586', '0.8573 ± 0.0128', '0.9205 ± 0.0082', '0.8377 ± 0.0343', '0.8333 ± 0.0171', '0.9251 ± 0.0114']),
    'dev': parse(['0.8353 ± 0.0605', '0.9197 ± 0.0141', '0.9444 ± 0.0163', '0.9438 ± 0.0119', '0.8827 ± 0.0135', '0.8624 ± 0.0205']),
    'test': parse(['0.8536 ± 0.0475', '0.9196 ± 0.0104', '0.9656 ± 0.0145', '0.9678 ± 0.0098', '0.8023 ± 0.0224', '0.8524 ± 0.0226'])
}

# Plotting function
def plot_metric(metric_untrained, metric_trained, title):
    plt.figure(figsize=(12, 6))
    for key in ['val', 'dev', 'test']:
        untrained_means, untrained_stds = extract_metrics(metric_untrained[key])
        trained_means, trained_stds = extract_metrics(metric_trained[key])
        light_color, dark_color = colors[key]

        plt.errorbar(x, untrained_means, yerr=untrained_stds, label=f'Untrained {key.title()}', color=light_color, linestyle='--', marker='o', capsize=4)
        plt.errorbar(x, trained_means, yerr=trained_stds, label=f'Trained {key.title()}', color=dark_color, linestyle='-', marker='s', capsize=4)

    plt.xticks(x, authors)
    plt.xlabel("Authors")
    plt.ylabel(title)
    plt.title(f"{title} over Authors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{title}.png")

# Plot both metrics
plot_metric(untrained_roc, trained_roc, "ROC AUC")
plot_metric(untrained_pr, trained_pr, "PR AUC")
