import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
# Print the first few instances
print('Dataset:\n', data.head())

# Use only selected 3 features/attributes
X = data[[
    'buying',
    'maint',
    'safety']].values

y = data[['class']]
print("x, y:\n", X, y)

# Data transformation - converting data into usable format
# Method 1
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
print("Transformed X:\n", X)

# Method 2
label_maping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_maping)
y = np.array(y)
print("Transformed y:\n", y)

# KNN Model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)
print("Predictions:", prediction)
print("accuracy:", accuracy)

a = 100
print("Actual value:", y[a])
print("Predicted value:", knn.predict(X)[a])