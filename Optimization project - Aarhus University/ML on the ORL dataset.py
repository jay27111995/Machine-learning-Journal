from scipy.io import loadmat
import numpy as np
import pandas as pd
from my_classification_schemes import my_classification_schemes_comparison
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors


# Load the ORL dataset
orl_data=np.transpose(loadmat('ORL/orl_data.mat')['data'])
orl_lbls=(loadmat('ORL/orl_lbls.mat')['lbls']).reshape(-1,)

# Split into train and test sets
ORL_Xtrain, ORL_Xtest, ORL_train_lbls, ORL_test_lbls = train_test_split(orl_data, orl_lbls, test_size=0.30)


# ORL PCA Visualisation
def ORL_PCA_2D_visualisation(principalComponents, lbls, ax):
    # Put PC1, PC2, and corresponding labels in a pd data frame
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    lblsDf = pd.DataFrame(data=lbls, columns=['label'])
    finalDf = pd.concat([principalDf, lblsDf], axis=1)

    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    labels = [str(i) for i in range(1, 41)]
    for label in labels:
        indicesToKeep = finalDf['label'] == int(label)
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c=list(mcolors.CSS4_COLORS)[int(label) * 3],
                   s=10)
    # ax.legend(labels)

# ORL TSNE Visualisation
def ORL_TSNE_2D_visualisation(Samples, lbls, ax):
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
    ax.set_title('t-SNE', fontsize=20)

    labels = [str(i) for i in range(1,41)]
    for label in labels:
        indicesToKeep = finalDf['label'] == int(label)
        ax.scatter(finalDf.loc[indicesToKeep, 'Dimension 1'],
                   finalDf.loc[indicesToKeep, 'Dimension 2'],
                   c=list(mcolors.CSS4_COLORS)[int(label) * 3],
                   s=10)
    # ax.legend(labels)

# Run all classification schemes
ORL_obj = my_classification_schemes_comparison(ORL_Xtrain, ORL_train_lbls, ORL_Xtest, ORL_test_lbls)

# Visualize the ORL Dataset
fig = plt.figure(figsize=(16,8))

ax1 = fig.add_subplot(1, 2, 1)
ORL_PCA_2D_visualisation(ORL_obj.Xtrain_PCA, ORL_train_lbls, ax1)

ax2 = fig.add_subplot(1, 2, 2)
ORL_TSNE_2D_visualisation(ORL_Xtrain, ORL_train_lbls, ax2)

plt.show()


