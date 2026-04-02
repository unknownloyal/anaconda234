import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = StandardScaler().fit_transform(X)

cov = np.cov(X.T)
eig_vals, eig_vecs = np.linalg.eig(cov)

idx = np.argsort(eig_vals)[::-1]
eig_vecs = eig_vecs[:, idx]

X_manual = X @ eig_vecs[:, :2]

X_sklearn = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(X_manual[:,0], X_manual[:,1], c=y)
plt.title("Manual PCA")

plt.subplot(1,2,2)
plt.scatter(X_sklearn[:,0], X_sklearn[:,1], c=y)
plt.title("Sklearn PCA")

plt.show()

print(eig_vals[idx])
