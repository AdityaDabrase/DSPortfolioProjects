# %% [markdown]
# # Seaborn Exercises

# %% [markdown]
# ## The Data
#
# will be working with a famous titanic data set for these exercise

# %%
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

# %%
sns.set_style("whitegrid")

# %%
titanic = sns.load_dataset("titanic")

# %%
titanic.head()

# %%
g = sns.jointplot(x="fare", y="age", data=titanic)
g.figure.savefig(ASSETS / "jointplot_fare_age.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)

# %%
fig, ax = plt.subplots()
sns.histplot(titanic["fare"], bins=30, kde=False, color="red", ax=ax)
fig.savefig(ASSETS / "fare_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
fig, ax = plt.subplots()
sns.boxplot(x="class", y="age", data=titanic, hue="class", palette="Set2", legend=False, ax=ax)
fig.savefig(ASSETS / "age_by_class.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
sns.swarmplot(x="class", y="age", data=titanic, hue="class", palette="Set2", legend=False, ax=ax)
fig.savefig(ASSETS / "age_swarm_by_class.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
fig, ax = plt.subplots()
sns.countplot(x="sex", data=titanic, ax=ax)
fig.savefig(ASSETS / "sex_count.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(titanic.select_dtypes(include="number").corr(), cmap="coolwarm", ax=ax)
ax.set_title("Numeric feature correlations")
fig.savefig(ASSETS / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
g = sns.FacetGrid(data=titanic, col="sex")
g.map(plt.hist, "age")
g.figure.savefig(ASSETS / "age_by_sex_hist.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)

print(f"Figures saved to {ASSETS}/")
