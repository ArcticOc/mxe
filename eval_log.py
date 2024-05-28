import matplotlib.pyplot as plt

data_best1 = {
    'MXE': {
        'l1_cp': {
            'Val': {'shot1_acc': 68.28, 'shot1_conf': 0.39, 'shot5_acc': 82.61, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 64.83, 'shot1_conf': 0.20, 'shot5_acc': 79.87, 'shot5_conf': 0.14},
        },
        'l1_wcp': {
            'Val': {'shot1_acc': 68.44, 'shot1_conf': 0.39, 'shot5_acc': 82.75, 'shot5_conf': 0.25},
            'Test': {'shot1_acc': 64.84, 'shot1_conf': 0.20, 'shot5_acc': 79.82, 'shot5_conf': 0.14},
        },
        'l2_cp': {
            'Val': {'shot1_acc': 67.99, 'shot1_conf': 0.41, 'shot5_acc': 81.80, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 62.90, 'shot1_conf': 0.21, 'shot5_acc': 77.65, 'shot5_conf': 0.15},
        },
        'l2_wcp': {
            'Val': {'shot1_acc': 67.50, 'shot1_conf': 0.40, 'shot5_acc': 81.72, 'shot5_conf': 0.27},
            'Test': {'shot1_acc': 63.54, 'shot1_conf': 0.21, 'shot5_acc': 78.16, 'shot5_conf': 0.15},
        },
    },
    'PN': {
        'l1_cp': {
            'Val': {'shot1_acc': 66.00, 'shot1_conf': 0.37, 'shot5_acc': 82.59, 'shot5_conf': 0.25},
            'Test': {'shot1_acc': 63.72, 'shot1_conf': 0.20, 'shot5_acc': 80.75, 'shot5_conf': 0.14},
        },
        'l1_wcp': {
            'Val': {'shot1_acc': 66.30, 'shot1_conf': 0.38, 'shot5_acc': 82.31, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 63.90, 'shot1_conf': 0.20, 'shot5_acc': 80.44, 'shot5_conf': 0.14},
        },
        'l2_cp': {
            'Val': {'shot1_acc': 66.31, 'shot1_conf': 0.39, 'shot5_acc': 81.98, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 63.52, 'shot1_conf': 0.20, 'shot5_acc': 79.79, 'shot5_conf': 0.14},
        },
        'l2_wcp': {
            'Val': {'shot1_acc': 66.47, 'shot1_conf': 0.38, 'shot5_acc': 81.88, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 63.42, 'shot1_conf': 0.20, 'shot5_acc': 79.52, 'shot5_conf': 0.14},
        },
    },
    'NCA': {
        'l1_cp': {
            'Val': {'shot1_acc': 65.48, 'shot1_conf': 0.38, 'shot5_acc': 81.49, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 62.97, 'shot1_conf': 0.20, 'shot5_acc': 78.89, 'shot5_conf': 0.15},
        },
        'l1_wcp': {
            'Val': {'shot1_acc': 66.64, 'shot1_conf': 0.39, 'shot5_acc': 81.65, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 63.25, 'shot1_conf': 0.20, 'shot5_acc': 78.92, 'shot5_conf': 0.15},
        },
        'l2_cp': {
            'Val': {'shot1_acc': 66.44, 'shot1_conf': 0.40, 'shot5_acc': 81.81, 'shot5_conf': 0.26},
            'Test': {'shot1_acc': 62.74, 'shot1_conf': 0.20, 'shot5_acc': 78.39, 'shot5_conf': 0.15},
        },
        'l2_wcp': {
            'Val': {'shot1_acc': 66.25, 'shot1_conf': 0.39, 'shot5_acc': 81.22, 'shot5_conf': 0.27},
            'Test': {'shot1_acc': 63.18, 'shot1_conf': 0.20, 'shot5_acc': 78.39, 'shot5_conf': 0.15},
        },
    },
}

data_best5 = {
    "MXE": {
        "l1_cp": {
            "Val": {"shot1_acc": 67.70, "shot1_conf": 0.40, "shot5_acc": 82.86, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 64.60, "shot1_conf": 0.20, "shot5_acc": 80.08, "shot5_conf": 0.14},
        },
        "l1_wcp": {
            "Val": {"shot1_acc": 68.33, "shot1_conf": 0.39, "shot5_acc": 83.15, "shot5_conf": 0.25},
            "Test": {"shot1_acc": 64.70, "shot1_conf": 0.20, "shot5_acc": 80.26, "shot5_conf": 0.14},
        },
        "l2_cp": {
            "Val": {"shot1_acc": 67.47, "shot1_conf": 0.39, "shot5_acc": 81.62, "shot5_conf": 0.27},
            "Test": {"shot1_acc": 63.41, "shot1_conf": 0.20, "shot5_acc": 78.25, "shot5_conf": 0.15},
        },
        "l2_wcp": {
            "Val": {"shot1_acc": 67.80, "shot1_conf": 0.40, "shot5_acc": 81.88, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 63.55, "shot1_conf": 0.20, "shot5_acc": 78.30, "shot5_conf": 0.15},
        },
    },
    "PN": {
        "l1_cp": {
            "Val": {"shot1_acc": 66.03, "shot1_conf": 0.38, "shot5_acc": 82.30, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 63.83, "shot1_conf": 0.20, "shot5_acc": 80.65, "shot5_conf": 0.14},
        },
        "l1_wcp": {
            "Val": {"shot1_acc": 66.13, "shot1_conf": 0.38, "shot5_acc": 82.15, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 63.79, "shot1_conf": 0.20, "shot5_acc": 80.54, "shot5_conf": 0.14},
        },
        "l2_cp": {
            "Val": {"shot1_acc": 66.03, "shot1_conf": 0.38, "shot5_acc": 81.77, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 63.03, "shot1_conf": 0.20, "shot5_acc": 79.24, "shot5_conf": 0.15},
        },
        "l2_wcp": {
            "Val": {"shot1_acc": 65.82, "shot1_conf": 0.38, "shot5_acc": 82.14, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 63.83, "shot1_conf": 0.20, "shot5_acc": 79.97, "shot5_conf": 0.15},
        },
    },
    "NCA": {
        "l1_cp": {
            "Val": {"shot1_acc": 65.66, "shot1_conf": 0.39, "shot5_acc": 81.75, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 62.52, "shot1_conf": 0.20, "shot5_acc": 78.62, "shot5_conf": 0.15},
        },
        "l1_wcp": {
            "Val": {"shot1_acc": 66.41, "shot1_conf": 0.40, "shot5_acc": 82.06, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 63.62, "shot1_conf": 0.20, "shot5_acc": 79.35, "shot5_conf": 0.15},
        },
        "l2_cp": {
            "Val": {"shot1_acc": 66.52, "shot1_conf": 0.38, "shot5_acc": 81.75, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 62.86, "shot1_conf": 0.20, "shot5_acc": 78.41, "shot5_conf": 0.15},
        },
        "l2_wcp": {
            "Val": {"shot1_acc": 66.44, "shot1_conf": 0.39, "shot5_acc": 81.45, "shot5_conf": 0.26},
            "Test": {"shot1_acc": 63.04, "shot1_conf": 0.20, "shot5_acc": 78.40, "shot5_conf": 0.15},
        },
    },
}


def plot_data(dataset_type):
    plt.figure(figsize=(10, 8))
    plt.title(f'shot5_{dataset_type}')

    labels = ['l1_cp', 'l1_wcp', 'l2_cp', 'l2_wcp']
    loss_types = ['PN', 'NCA', 'MXE']
    bar_width = 0.2
    colors = plt.get_cmap('Set1')(range(len(loss_types)))

    for i, loss_type in enumerate(loss_types):
        shot1_accs = [data_best5[loss_type][label][dataset_type]['shot5_acc'] for label in labels]
        positions = [x + (i * bar_width) for x in range(len(labels))]

        bars = plt.barh(positions, shot1_accs, height=bar_width, color=colors[i], alpha=0.7, label=f'{loss_type} loss')

        for bar, value in zip(bars, shot1_accs):  # noqa: B905
            plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', va='center')

    plt.yticks([x + bar_width for x in range(len(labels))], labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)  # Change legend position to upper right
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Norm')
    plt.savefig(f'{dataset_type}_Data.png', bbox_inches='tight', dpi=600)
    plt.show()


# Example usage
plot_data('Val')
plot_data('Test')
