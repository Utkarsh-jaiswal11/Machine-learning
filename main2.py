import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv('TShirt_size.csv')

# Separate the features (X) and the labels (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x=X.fillna(0)
X=X.values
y=y.fillna(0)
y=y.values
from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

#view transformed values
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Choose the value of k and create an instance of the KNN model
k = 5  # The number of neighbors to consider
model = KNeighborsClassifier(n_neighbors=k)

# Train the KNN model using the training data
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
