import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import cv2
A = load_digits().images[0]
# A = cv2.imread('1.jpeg', cv2.IMREAD_GRAYSCALE)

U, S, Vt = np.linalg.svd(A, False)

def rec(k):
    return U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]

imgs = [A, rec(5), rec(10), rec(15)]
titles = ["Original", "k=5", "k=10", "k=15"]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()

print("Original:", A.size)
for k in [5,10,15]:
    print(f"k={k}:", U[:,:k].size + S[:k].size + Vt[:k,:].size)
