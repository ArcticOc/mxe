# import collections
# import os
# from datetime import datetime
# from itertools import product

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from matplotlib.backends.backend_pdf import PdfPages
# from rich.progress import track

# # from sklearn.manifold import TSNE
# from torch import distributed as dist
# from tsnecuda import TSNE


# class Filter:
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels

#     def filter_if(self, n):
#         from sklearn.ensemble import IsolationForest

#         iso_forest = IsolationForest(n_estimators=50, contamination=n)
#         outlier_flags = iso_forest.fit_predict(self.features)
#         mask = outlier_flags != -1
#         self.filtered_features = self.features[mask]
#         self.filtered_labels = self.labels[mask]
#         return self.filtered_features, self.filtered_labels

#     def filter_zscores(self, n):
#         from scipy.stats import zscore

#         z_scores = np.abs(zscore(self.features))
#         filtered_indices = (z_scores < n).all(axis=1)
#         self.filtered_features = self.features[filtered_indices]
#         self.filtered_labels = self.labels[filtered_indices]
#         return self.filtered_features, self.filtered_labels

#     def filter_lof(self, n):
#         from sklearn.neighbors import LocalOutlierFactor

#         lof = LocalOutlierFactor(n_neighbors=n, contamination=0.5)
#         outlier_flags = lof.fit_predict(self.features)
#         mask = outlier_flags != -1
#         self.filtered_features = self.features[mask]
#         self.filtered_labels = self.labels[mask]
#         return self.filtered_features, self.filtered_labels

#     def original(self):
#         return self.features, self.labels

#     def __call__(self, n=None, method="if"):
#         if method == "if":
#             return self.filter_if(n=n if n is not None else 0.1)
#         elif method == "zscore":
#             return self.filter_zscores(n=n if n is not None else 4)
#         elif method == "lof":
#             return self.filter_lof(n=n if n is not None else 15)
#         elif method == "original":
#             return self.original()
#         else:
#             raise ValueError("Invalid method")


# class TSNEVisualizer:
#     def __init__(self, val_loader, model, args):
#         self.val_loader = val_loader
#         self.model = model
#         self.args = args
#         if dist.is_initialized():
#             self.rank = dist.get_rank()
#             os.makedirs(f"tsne/mxe/rank_{self.rank}", exist_ok=True)
#         else:
#             self.rank = 0

#     def extract_features(self):
#         self.model.eval()
#         output_dict = collections.defaultdict(list)
#         for i, (inputs, labels) in enumerate(self.val_loader):
#             inputs, labels = (
#                 inputs.to(self.args.device),
#                 labels.to(self.args.device),
#             )
#             # compute output
#             outputs, _ = self.model(inputs)
#             outputs = outputs.cpu().data.numpy()
#             labels = labels.cpu().data.numpy()

#             for out, label in zip(outputs, labels):
#                 output_dict[label.item()].append(out)

#         return output_dict

#     def visualize_with_tsne(self, title=None):
#         self.model.eval()
#         base_title = self.args.loss if title is None else title
#         output_dict_ = self.extract_features()
#         new_features = []
#         new_labels = []
#         tsne_sample = 250  # 每个类别采样数
#         tsne_class = 5  # 可视化类别数

#         np.random.seed(42)

#         for label, feats in output_dict_.items():
#             feats = np.array(feats)
#             if len(feats) < tsne_sample:
#                 print(f"Label {label} only has {len(feats)} samples, which is less than tsne_sample ({tsne_sample}).")
#             indices = np.random.choice(len(feats), tsne_sample, replace=False)
#             selected_feats = feats[indices]
#             # 如果需要归一化，可取消下面代码的注释
#             # selected_feats = (selected_feats - np.mean(selected_feats, axis=0)) / (
#             #     np.std(selected_feats, axis=0) + 1e-8
#             # )
#             new_features.extend(selected_feats)
#             new_labels.extend([label] * tsne_sample)

#         features = np.array(new_features)
#         labels = np.array(new_labels)

#         # 使用 SelectKBest 选择最具区分性的特征
#         from sklearn.feature_selection import SelectKBest

#         selector = SelectKBest(k=192)
#         features = selector.fit_transform(features, labels)

#         # 定义 TSNE 参数网格
#         # perplexity_values = list(range(16, 60, 4))
#         # learning_rate_values = list(range(20, 200, 10))
#         # early_exaggerations = list(range(8, 52, 4))
#         perplexity_values = [60]
#         learning_rate_values = [20]
#         early_exaggerations = [15]

#         pdf_dir = f"tsne/mxe/rank_{self.rank}"
#         os.makedirs(pdf_dir, exist_ok=True)
#         pdf_filename = os.path.join(pdf_dir, f"tsne_plots_{base_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

#         # 构造所有参数组合
#         param_combinations = list(product(perplexity_values, learning_rate_values, early_exaggerations))

#         with PdfPages(pdf_filename) as pdf:
#             # 使用 rich.progress.track 显示进度条
#             for perplexity, learning_rate, early_exaggeration in track(
#                 param_combinations, description="Running TSNE..."
#             ):
#                 tsne = TSNE(
#                     n_components=2,
#                     n_iter=1500,
#                     perplexity=perplexity,
#                     early_exaggeration=early_exaggeration,
#                     learning_rate=learning_rate,
#                     verbose=False,
#                 )
#                 features_tsne = tsne.fit_transform(features)
#                 # 使用 LOF 过滤异常点
#                 filt = Filter(features_tsne, labels)
#                 filtered_features_tsne, filtered_labels = filt(n=60, method="lof")
#                 torch.cuda.empty_cache()

#                 # 绘图
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 cmap = plt.get_cmap('tab10')

#                 unique_labels = np.unique(filtered_labels)
#                 selected_labels = unique_labels[:tsne_class]
#                 for lbl in selected_labels:
#                     idx = filtered_labels == lbl
#                     ax.scatter(
#                         filtered_features_tsne[idx, 0],
#                         filtered_features_tsne[idx, 1],
#                         c=[cmap(lbl / len(selected_labels))],
#                         label=str(lbl),
#                         alpha=0.8,
#                     )

#                 # ax.set_xlabel("t-SNE feature 1")
#                 # ax.set_ylabel("t-SNE feature 2")
#                 ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#                 ax.set_title(
#                     f"{base_title}: TSNE (perplexity={perplexity}, lr={learning_rate}, early_exag={early_exaggeration})"
#                 )
#                 pdf.savefig(fig, bbox_inches='tight')
#                 plt.close(fig)
#             print(f"Saved all t-SNE plots in {pdf_filename}")
import collections
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# from openTSNE import TSNE
# from umap import UMAP
# from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from torch import distributed as dist


class Filter:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def filter_if(self, n):
        from sklearn.ensemble import IsolationForest

        iso_forest = IsolationForest(n_estimators=50, contamination=n)
        outlier_flags = iso_forest.fit_predict(self.features)
        mask = outlier_flags != -1
        self.filtered_features = self.features[mask]
        self.filtered_labels = self.labels[mask]
        return self.filtered_features, self.filtered_labels

    def filter_zscores(self, n):
        from scipy.stats import zscore

        z_scores = np.abs(zscore(self.features))
        filtered_indices = (z_scores < n).all(axis=1)
        self.filtered_features = self.features[filtered_indices]
        self.filtered_labels = self.labels[filtered_indices]
        return self.filtered_features, self.filtered_labels

    def filter_lof(self, n):
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=n, contamination=0.5)  # 100
        outlier_flags = lof.fit_predict(self.features)
        mask = outlier_flags != -1
        self.filtered_features = self.features[mask]
        self.filtered_labels = self.labels[mask]
        return self.filtered_features, self.filtered_labels

    def original(self):
        return self.features, self.labels

    def __call__(self, n=None, method="if"):
        if method == "if":
            return self.filter_if(n=n if n is not None else 0.1)
        elif method == "zscore":
            return self.filter_zscores(n=n if n is not None else 4)
        elif method == "lof":
            return self.filter_lof(n=n if n is not None else 15)
        elif method == "original":
            return self.original()
        else:
            raise ValueError("Invalid method")


class TSNEVisualizer:
    def __init__(self, val_loader, model, args):
        self.val_loader = val_loader
        self.model = model
        self.args = args
        if dist.is_initialized():
            self.rank = dist.get_rank()
            os.makedirs(f"tsne/mxe/rank_{self.rank}", exist_ok=True)
        else:
            self.rank = 0

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

    def visualize_with_tsne(self, title=None):
        self.model.eval()
        title = self.args.loss
        output_dict_ = self.extract_features()
        # features = np.concatenate(list(output_dict_.values()))
        # labels = np.concatenate(
        #     [[label] * len(feats) for label, feats in output_dict_.items()]
        # )
        new_features = []
        new_labels = []
        tsne_sample, tsne_class = 150, 5
        for label, feats in output_dict_.items():
            np.random.seed(1)
            indices = np.random.choice(len(feats), tsne_sample, replace=False)
            selected_feats = [feats[i] for i in indices]
            new_features.extend(selected_feats)
            new_labels.extend([label] * tsne_sample)
        features = np.array(new_features)
        labels = np.array(new_labels)

        for i in range(12, 13):
            for k in range(10, 11):
                # os.makedirs(f"tsne/comp_2/rank_{self.rank}/perp_{i*3}", exist_ok=True)
                # filename = f"tsne/comp_2/rank_{self.rank}/perp_{i*3}/comp_{k*3}.png"
                filename = f"tsne/mxe/rank_{self.rank}/lof_60_{datetime.now().strftime('%H%M%S')}.pdf"
                tsne = TSNE(
                    n_components=2,
                    method="exact",
                    n_iter=1500,
                    random_state=0,
                    perplexity=60,
                    early_exaggeration=15,
                    learning_rate=20,
                    n_jobs=24,
                    verbose=True,
                    init="pca",
                )

                features_tsne = tsne.fit_transform(features)
                filter = Filter(features_tsne, labels)
                filtered_features_tsne, filtered_labels = filter(n=60, method="lof")

                plt.figure(figsize=(10, 6))
                cmap = plt.get_cmap('tab20')

                unique_labels = np.unique(labels)
                selected_labels = unique_labels[:tsne_class]
                for label in selected_labels:
                    idx = filtered_labels == label
                    plt.scatter(
                        filtered_features_tsne[idx, 0],
                        filtered_features_tsne[idx, 1],
                        c=[cmap(label / len(selected_labels))],
                        label=label,
                        alpha=0.8,
                    )

                plt.title('PN')
                # plt.axis("off")
                # plt.xlabel("t-SNE feature 1")
                # plt.ylabel("t-SNE feature 2")
                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                plt.savefig(filename, bbox_inches="tight", dpi=600, format='pdf')
                plt.show()
                print(f"Saved t-SNE plot at {filename}")
