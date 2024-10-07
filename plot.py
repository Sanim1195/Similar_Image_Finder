#%%

import matplotlib.pyplot as plt

#  X - independent features(excluding target variable)
# y - dependent variables, called (target).

# settimg data for plotting
x = [1,2,3,4,5]
y = [2,3,5,7,11]

# creating a plot 
plt.plot(x,y)
#  a title for the plot
plt.title("My plot") 
# labels for the plot:
plt.xlabel("I.V")
plt.ylabel("Target")

# show plot:
plt.show()
# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-means
kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the data points and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# %%
