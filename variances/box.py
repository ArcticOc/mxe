import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_nca = pd.read_csv("kkvar.csv")  # cls,variance
df_gm = pd.read_csv("mkvar.csv")  # cls,variance
df_pn = pd.read_csv("ppvar.csv")  # cls,variance

df_pn["Method"] = "PN"
df_nca["Method"] = "AMK"
df_gm["Method"] = "GMK"

df_all = pd.concat([df_pn, df_nca, df_gm], ignore_index=True)

# Set style and color palette
sns.set_style("white")
sns.set_palette("Set3")
plt.figure(figsize=(6, 4), dpi=600)

# Create boxplot with customized appearance
# sns.boxplot(
#     x="Method",
#     y="variance",
#     data=df_all,
#     medianprops=dict(color="black", linewidth=1.5),
#     # flierprops=dict(marker='o', markerfacecolor='gray', markersize=4),
#     showfliers=False,
#     # color="white",  # Set box fill color to white
#     # boxprops={'facecolor': 'white'},
# )
sns.violinplot(x="Method", y="variance", data=df_all, inner="box")

plt.xlabel("")
plt.title("Class-wise Intra-Class Variance", pad=15)
plt.tight_layout()

# Save as PDF with high quality
plt.savefig('variance_comparison.pdf', bbox_inches='tight', format='pdf')
plt.show()
