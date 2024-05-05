from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder="templates")

# Load the trained model
model = pickle.load(open('knn_model.pkl', 'rb'))

# Load the vectorizer
vectorizer_name = pickle.load(open('vectorizer_name.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        data = [request.form['fav_number'],
                request.form['statuses_count'],
                request.form['followers_count'],
                request.form['friends_count'],
                request.form['favourites_count'],
                request.form['listed_count'],
                request.form['name']]
        
        # Vectorize the name feature
        X_name = vectorizer_name.transform([data[-1]])
        
        # Combine features
        X_numeric = np.array(data[:-1]).reshape(1, -1).astype(float)  # Convert to float
        X = np.hstack((X_numeric, X_name.toarray()))
        
        # Make prediction
        pred = model.predict(X)
        
        # Return the result
        return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)
