#Om Gole, Period 6, Gabor
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Olivetti faces dataset
dataset = fetch_olivetti_faces(shuffle=True, random_state=0)
faces = dataset.data

# Apply PCA
n_components = 50  # You can tweak this parameter
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(faces)


faces_pca = pca.transform(faces)
faces_inv_transform = pca.inverse_transform(faces_pca)
def plot_faces(faces, n_row, n_col):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(faces):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape((64, 64)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
n_row = 2
n_col = 3
#ORIGIONAL FACE
print("Original faces:")
plot_faces(faces[:n_row*n_col], n_row, n_col)
plt.show()
#AFTER PCA AND RECONSTRUCTION FROM BOOK
print("Reconstructed faces:")
plot_faces(faces_inv_transform[:n_row*n_col], n_row, n_col)
plt.show()
