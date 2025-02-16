import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_pn = pd.read_csv("ppl2.csv")
df_nca = pd.read_csv("kkl2.csv")
df_gm = pd.read_csv("mkl2.csv")

df_pn["method"] = "PN"
df_nca["method"] = "AMK"
df_gm["method"] = "GMK"

df_all = pd.concat([df_pn, df_nca, df_gm], ignore_index=True)

print(df_all.head())

sns.set_style("white")
colors = ["#2878B5", "#9AC9DB", "#C82423"]  # Professional blue, light blue, red
# Alternative palette options:
# colors = ["#4C72B0", "#55A868", "#C44E52"]  # Seaborn default-like
# colors = ["#2096BA", "#4065B1", "#BA3C3C"]  # Blue dominant

plt.figure(figsize=(8, 5), dpi=600)

sns.lineplot(
    x="Step",
    y="Value",
    hue="method",
    data=df_all,
    palette=colors,
    linewidth=2,
)

plt.title("Mean Intra-Class Variance Comparison")
plt.xlabel("Iteration")
plt.ylabel("Mean Variance")

plt.legend(title=None, loc='upper left')
plt.tight_layout()
plt.savefig('line.pdf', bbox_inches='tight', format='pdf')
plt.show()
