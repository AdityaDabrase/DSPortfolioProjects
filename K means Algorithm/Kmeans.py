# %% [markdown]
# # Customer Segmentation using K-Means Algorithm

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
customer_data = pd.read_csv('Mall_Customers.csv')

# %%
customer_data

# %% [markdown]
# # About Dataset
# 
# ## Context
# 
# This data set is created only for the learning purpose of the customer segmentation concepts , also known as market basket analysis. I will demonstrate this by using unsupervised ML technique (KMeans Clustering Algorithm) in the simplest form.
# 
# This is about a supermarket mall and through membership cards , they have some basic data about customers like Customer ID, age, gender, annual income and spending score.Spending Score is something assigned to the customer based on defined parameters like customer behavior and purchasing data.
# 
# ## Problem Statement
# 
# The business wants to understand the customers like who can be easily converted [Target Customers] so that the data can be given to marketing team and plan the strategy accordingly.
# 
# 

# %%

customer_data.drop('Gender', axis=1, inplace=True)


# %%
customer_data

# %%
features = ['Age', 'Annual Income (k$)'] 
X = customer_data[features]

# %%
features

# %%
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# %%
# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# %%
# Choose the optimal number of clusters and perform k-means clustering
optimal_k = 4  # Adjust this based on the Elbow Method graph
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_scaled)

# %%
# Add cluster labels to the original dataset
customer_data['Cluster'] = kmeans.labels_

# %%
# Visualize clusters using PCA (adjust dimensions based on your feature count)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# %%
# Create a scatter plot of the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customer_data['Cluster'], cmap='viridis', edgecolor='k', s=50)
plt.title('Customer Segmentation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set a custom color palette for better visibility of clusters
custom_palette = sns.color_palette("Set2", as_cmap=True)

# Create a scatter plot with cluster centers
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customer_data['Cluster'], cmap=custom_palette, edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

# Annotate points with customer IDs or any relevant information
for i, customer_id in enumerate(customer_data['CustomerID']):
    plt.annotate(customer_id, (X_pca[i, 0], X_pca[i, 1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')

plt.title('Customer Segmentation with Cluster Centers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# %%
# Explore cluster characteristics
cluster_means = customer_data.groupby('Cluster').mean()
print(cluster_means)


