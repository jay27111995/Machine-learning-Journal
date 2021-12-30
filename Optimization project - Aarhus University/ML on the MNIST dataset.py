from scipy.io import loadmat
import numpy as np
import pandas as pd
from my_classification_schemes import my_classification_schemes_comparison
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the MNIST dataset
mnist=loadmat('mnist_loaded.mat')

# Each column represents an image in the dataset,
# take transpose to feed into classifier as that is the regular format.
(MNIST_Xtrain, MNIST_train_lbls) = (np.transpose(mnist["train_images"]), mnist["train_labels"].reshape(-1,))
(MNIST_Xtest, MNIST_test_lbls) = (np.transpose(mnist["test_images"]), mnist["test_labels"].reshape(-1,))

# MNIST PCA Visualisation
def MNIST_PCA_2D_visualisation(principalComponents, lbls, ax):
    # Put PC1, PC2, and corresponding labels in a pd data frame
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    lblsDf = pd.DataFrame(data=lbls, columns=['label'])
    finalDf = pd.concat([principalDf, lblsDf], axis=1)

    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA for MNIST dataset', fontsize=20)

    labels = [str(i) for i in range(10)]
    colors = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "brown", "gray"]
    for label, color in zip(labels, colors):
        indicesToKeep = finalDf['label'] == int(label)
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c=color,
                   s=10)
    ax.legend(labels)

# MNIST TSNE Visualisation
def MNIST_TSNE_2D_visualisation(Samples, lbls, ax):
    # Data standardization
    x = StandardScaler().fit_transform(Samples)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(x)

    # Put PC1, PC2, and corresponding labels in a pd data frame
    tsneDf = pd.DataFrame(data=tsne_data,
                               columns=['Dimension 1', 'Dimension 2'])
    lblsDf = pd.DataFrame(data=lbls, columns=['label'])
    finalDf = pd.concat([tsneDf, lblsDf], axis=1)

    ax.set_xlabel('Dimension 1', fontsize=15)
    ax.set_ylabel('Dimension 2', fontsize=15)
    ax.set_title('t-SNE for MNIST dataset', fontsize=20)

    labels = [str(i) for i in range(10)]
    colors = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "brown", "gray"]
    for label, color in zip(labels, colors):
        indicesToKeep = finalDf['label'] == int(label)
        ax.scatter(finalDf.loc[indicesToKeep, 'Dimension 1'],
                   finalDf.loc[indicesToKeep, 'Dimension 2'],
                   c=color,
                   s=10)
    ax.legend(labels)

# Run all classification schemes
MNIST_obj = my_classification_schemes_comparison(MNIST_Xtrain, MNIST_train_lbls, MNIST_Xtest, MNIST_test_lbls)

# Visualize the MNIST Dataset
fig = plt.figure(figsize=(16,8))

ax1 = fig.add_subplot(1, 2, 1)
MNIST_PCA_2D_visualisation(MNIST_obj.Xtest_PCA, MNIST_test_lbls, ax1)

ax2 = fig.add_subplot(1, 2, 2)
MNIST_TSNE_2D_visualisation(MNIST_Xtest, MNIST_test_lbls, ax2)

plt.show()





