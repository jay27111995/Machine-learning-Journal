from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
import numpy as np


# 2D PCA transformation
def PCA_2D(Samples, labels):
    x = StandardScaler().fit_transform(Samples)
    # Reduce dimension to 2 with PCA
    pca = PCA(n_components=2)
    pca.fit(x)
    # Apply PCA
    return pca.transform(x)

# Nearest Centroid Classifier
def nc_classify(Xtrain, Xtest, train_lbls):
    NCC = NearestCentroid()
    NCC.fit(Xtrain, train_lbls)
    NCC_out = NCC.predict(Xtest)
    return NCC_out

# Nearest Neighbor Classifier
def nn_classify(Xtrain, K, Xtest, train_lbls):
    NN = KNeighborsClassifier(n_neighbors=K, weights='uniform')
    NN.fit(Xtrain, train_lbls)
    NN_out = NN.predict(Xtest)
    return NN_out

# Neural network trained over backpropogation
def perceptron_bp_classify(Xtrain, train_lbls, Xtest, eta):
    # Standardize the training data
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    SGD_hinge = SGDClassifier(loss="hinge", penalty="l2", max_iter=10)
    SGD_hinge.fit(Xtrain, train_lbls)
    SGD_hinge_out = SGD_hinge.predict(Xtest)
    return SGD_hinge_out

def perceptron_mse_classify(Xtrain, train_lbls, Xtest, eta):
    # Standardize the training data
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    SGD_mse = SGDClassifier(loss="squared_error", penalty="l2", max_iter=10)
    SGD_mse.fit(Xtrain, train_lbls)
    SGD_mse_out = SGD_mse.predict(Xtest)
    return SGD_mse_out

# Nearest Subclass CLassifier
def nsc_classify(Xtrain, K, Xtest, train_lbls):
    # From the training data, let's split the classes
    # Here, a dictionary for which the key is the label and the value is
    # the samples belonging to it.
    lbl_dic = {}
    cnt = 0
    for l in train_lbls:
        if l not in lbl_dic:
            lbl_dic[l] = Xtrain[cnt]
        else:
            lbl_dic[l] = np.vstack((lbl_dic[l], Xtrain[cnt]))
        cnt = cnt + 1

    # Apply k-means on each cluster splitting into number of subclasses
    cnt = 0
    for k in lbl_dic:
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(lbl_dic[k])
        if cnt == 0:
            centers = kmeans.cluster_centers_
        else:
            centers = np.vstack((centers, kmeans.cluster_centers_))
        for i in range(K):
            if cnt == 0 and i == 0:
                lbls = np.array([k])
            else:
                lbls = np.vstack((lbls, np.array([k])))
        cnt = cnt + 1

    NN = KNeighborsClassifier(n_neighbors=1, weights='distance')
    NN.fit(centers, lbls)
    NSC_out = NN.predict(Xtest)
    return NSC_out


# 5 Classification schemes wrapped into a class
class my_classification_schemes_comparison:
    Xtrain_PCA = None
    Xtest_PCA = None

    def __init__(self, Xtrain, train_lbls, Xtest, test_lbls):
        # PCA transformed samples
        self.Xtrain_PCA = PCA_2D(Xtrain, train_lbls)
        self.Xtest_PCA = PCA_2D(Xtest, test_lbls)

        # Nearest Subclass Centroid Classification, Neighbors = 2
        NSC_n2_lbls = nsc_classify(Xtrain, 2, Xtest, train_lbls)
        NSC_n2_PCA_lbls = nsc_classify(self.Xtrain_PCA, 2, self.Xtest_PCA, train_lbls)

        NSC_n2_accuracy = metrics.accuracy_score(test_lbls, NSC_n2_lbls)
        NSC_n2_PCA_accuracy = metrics.accuracy_score(test_lbls, NSC_n2_PCA_lbls)
        print("NSC n2 Accuracy: ", NSC_n2_accuracy)
        print("NSC_PCA n2 Accuracy: ", NSC_n2_PCA_accuracy)

        # Nearest Subclass Centroid Classification, Neighbors = 3
        NSC_n3_lbls = nsc_classify(Xtrain, 3, Xtest, train_lbls)
        NSC_n3_PCA_lbls = nsc_classify(self.Xtrain_PCA, 3, self.Xtest_PCA, train_lbls)

        NSC_n3_accuracy = metrics.accuracy_score(test_lbls, NSC_n3_lbls)
        NSC_n3_PCA_accuracy = metrics.accuracy_score(test_lbls, NSC_n3_PCA_lbls)
        print("NSC n3 Accuracy: ", NSC_n3_accuracy)
        print("NSC_PCA n3 Accuracy: ", NSC_n3_PCA_accuracy)

        # Nearest Subclass Centroid Classification, Neighbors = 5
        NSC_n5_lbls = nsc_classify(Xtrain, 5, Xtest, train_lbls)
        NSC_n5_PCA_lbls = nsc_classify(self.Xtrain_PCA, 5, self.Xtest_PCA, train_lbls)

        NSC_n5_accuracy = metrics.accuracy_score(test_lbls, NSC_n5_lbls)
        NSC_n5_PCA_accuracy = metrics.accuracy_score(test_lbls, NSC_n5_PCA_lbls)
        print("NSC n5 Accuracy: ", NSC_n5_accuracy)
        print("NSC_PCA n5 Accuracy: ", NSC_n5_PCA_accuracy)

        # Nearest Centroid Classification
        NCC_lbls = nc_classify(Xtrain, Xtest, train_lbls)
        NCC_PCA_lbls = nc_classify(self.Xtrain_PCA, self.Xtest_PCA, train_lbls)
        print(confusion_matrix(test_lbls, NCC_lbls))
        print(classification_report(test_lbls, NCC_lbls))

        # Measure accuracy
        NCC_accuracy = metrics.accuracy_score(test_lbls, NCC_lbls)
        NCC_PCA_accuracy = metrics.accuracy_score(test_lbls, NCC_PCA_lbls)
        print("NCC Accuracy: ", NCC_accuracy)
        print("NCC_PCA Accuracy: ", NCC_PCA_accuracy)

        # Nearest neighbor
        for n in range(1, 6):
            NN_lbls = nn_classify(Xtrain, n, Xtest, train_lbls)
            NN_PCA_lbls = nn_classify(self.Xtrain_PCA, n, self.Xtest_PCA, train_lbls)

            NN_accuracy = metrics.accuracy_score(test_lbls, NN_lbls)
            NN_PCA_accuracy = metrics.accuracy_score(test_lbls, NN_PCA_lbls)
            print("NN Accuracy for ", n, " neighbors: ", NN_accuracy)
            print("NN_PCA Accuracy for ", n, " neighbors: ", NN_PCA_accuracy)

        # Perceptron trained by backpropagation
        SGD_hinge_lbls = perceptron_bp_classify(Xtrain, train_lbls, Xtest, 0.001)
        SGD_hinge_PCA_lbls = perceptron_bp_classify(self.Xtrain_PCA, train_lbls, self.Xtest_PCA, 0.001)

        SGD_hinge_accuracy = metrics.accuracy_score(test_lbls, SGD_hinge_lbls)
        SGD_hinge_PCA_accuracy = metrics.accuracy_score(test_lbls, SGD_hinge_PCA_lbls)
        print("SGD with hinge loss, Accuracy: ", SGD_hinge_accuracy)
        print("SGD_PCA with hinge loss, Accuracy: ", SGD_hinge_PCA_accuracy)

        # Perceptron trained by MSE
        SGD_mse_lbls = perceptron_bp_classify(Xtrain, train_lbls, Xtest, 0.001)
        SGD_mse_PCA_lbls = perceptron_bp_classify(self.Xtrain_PCA, train_lbls, self.Xtest_PCA, 0.001)

        SGD_mse_accuracy = metrics.accuracy_score(test_lbls, SGD_mse_lbls)
        SGD_mse_PCA_accuracy = metrics.accuracy_score(test_lbls, SGD_mse_PCA_lbls)
        print("SGD with mse loss, Accuracy: ", SGD_mse_accuracy)
        print("SGD_PCA with mse loss, Accuracy: ", SGD_mse_PCA_accuracy)
