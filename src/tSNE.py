import collections

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class TSNEVisualizer:
    def __init__(self, val_loader, model, args):
        self.val_loader = val_loader
        self.model = model
        self.args = args

    def extract_features(self):
        self.model.eval()
        output_dict = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(self.val_loader):
            inputs, labels = (
                inputs.to(self.args.device),
                labels.to(self.args.device),
            )
            # compute output
            outputs, _ = self.model(inputs)
            outputs = outputs.cpu().data.numpy()
            labels = labels.cpu().data.numpy()

            for out, label in zip(outputs, labels):  # noqa: B905
                output_dict[label.item()].append(out)

        return output_dict

    def visualize_with_tsne(self, title="t-SNE Visualization"):
        self.model.eval()
        output_dict_ = self.extract_features()
        features = np.concatenate(list(output_dict_.values()))
        labels = np.concatenate(
            [[label] * len(features) for label, features in output_dict_.items()]
        )

        tsne = TSNE(
            n_components=4,
            method="exact",
            n_iter=2000,
            random_state=0,
            perplexity=60,
            learning_rate=200,
        )
        features_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels)
        selected_labels = unique_labels[:6]
        for label in selected_labels:
            idx = labels == label
            plt.scatter(
                features_tsne[idx, 0], features_tsne[idx, 1], label=label, alpha=0.6
            )

        plt.title(title)
        plt.axis("off")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig("tsne/tsne_nca.png")
        plt.show()
