from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

# 2D array of (n_samples, n_features)
print("2D array of (n_samples, n_features):", digits.data)
# Output of samples in dataset
print("Output of samples in dataset:", digits.target)
# (8 * 8) image of each sample in dataset
print("(8 * 8) image of each sample in dataset:", digits.images[0])

# Create a classifier object of type SVM
clf = svm.SVC(gamma=0.001, C=100.)
# Train the classifier
clf.fit(digits.data[:-1], digits.target[:-1])
# Predict using the trained classifier
out = clf.predict(digits.data[-1:])
print("Predicted value using SVM:", out)
