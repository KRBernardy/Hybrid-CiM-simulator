import os
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

def parse_stats_file(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if 'leakage_energy:' in line:
                metrics['leakage'] = float(line.split(':')[1].strip())
            elif 'dynamic_energy:' in line:
                metrics['dynamic'] = float(line.split(':')[1].strip())
            elif 'total_energy:' in line:
                metrics['total'] = float(line.split(':')[1].strip())
            elif 'time:' in line:
                metrics['latency'] = float(line.split(':')[1].strip())
    return metrics

# Models and their data
models = ['conv_layer', 'mlp_l4_mnist', 'LeNet_5', 'ResNet_20', 'parallel_CNN', 'DS_CNN']
valid_models = []
leakage_energies = []
dynamic_energies = []
total_energies = []
latencies = []

for model in models:
    filepath = '/HybridCiM/data/test/traces/{}/harwdare_stats.txt'.format(model)
    if os.path.exists(filepath):
        metrics = parse_stats_file(filepath)
        if metrics:
            valid_models.append(model)
            leakage_energies.append(metrics['leakage'])
            dynamic_energies.append(metrics['dynamic'])
            total_energies.append(metrics['total'])
            latencies.append(metrics['latency'])

# Plot only if we have data
if valid_models:
    # Energy plot
    x = np.arange(len(valid_models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, leakage_energies, width, label='Leakage Energy')
    ax.bar(x, dynamic_energies, width, label='Dynamic Energy')
    ax.bar(x + width, total_energies, width, label='Total Energy')

    ax.set_ylabel('Energy (J)')
    ax.set_title('Energy Consumption Across Different Models')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig('energy_comparison.png')
    print("Graph saved as energy_comparison.png")

    # Latency plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(valid_models, latencies)
    ax2.set_ylabel('Latency (s)')
    ax2.set_title('Latency Across Different Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('latency_comparison.png')
    print("Graph saved as latency_comparison.png")
else:
    print("No valid data found for plotting")