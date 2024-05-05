import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pickle

# Read data
users = pd.read_csv('users.csv')
fusers = pd.read_csv('fusers.csv')

# Assign target labels
users['Target'] = 1
fusers['Target'] = 0

# Replace zero values with mean
zero_not_accepted = ['fav_number', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count']

for df in [users, fusers]:
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.NaN)
        mean = int(df[column].mean(skipna=True))
        df[column] = df[column].replace(np.NaN, mean)

# Concatenate dataframes
df = pd.concat([users, fusers])

# Vectorize name feature
vectorizer_name = TfidfVectorizer()
X_name = vectorizer_name.fit_transform(df['name'])

# Combine features
X_numeric = df[['fav_number', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count']].values
X = np.hstack((X_numeric, X_name.toarray()))
y = df['Target'].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix :", confusion_matrix(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Recall score: ", recall_score(y_test, y_pred))

# Save model
pickle.dump(knn, open('knn_model.pkl', 'wb'))

# Save vectorizer
pickle.dump(vectorizer_name, open('vectorizer_name.pkl', 'wb'))

# Sample predictions
def label(i):
    if i == 0:
        return "Fake"
    else:
        return "Genuine"

# Test predictions
test_data = [
    [95, 606, 592, 101, 89, 78, 'gargiimittal'],
    [24, 50, 20, 630, 0, 0, 'Thomasen Frank']
]

for data in test_data:
    inp_numeric = np.array(data[:-1]).reshape(1, -1)
    inp_name = vectorizer_name.transform([data[-1]])
    inp = np.hstack((inp_numeric, inp_name.toarray()))
    print(f"The predicted outcome is {label(knn.predict(inp))}")
