from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'b645420873a84ceb73ae98074aefd3bc3e77197770ce92c459bc440b25800a9a'

# Load the pretrained Random Forest model and TF-IDF Vectorizer
model = joblib.load('random_forest_model.pkl', mmap_mode='r')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Assuming you saved the vectorizer
job_data = pd.read_csv('job_descriptions.csv')

# Preprocess and vectorize job descriptions
job_descriptions = job_data['Job Description'].fillna('').tolist()
job_vectors = vectorizer.fit_transform(job_descriptions)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        if username:
            session['username'] = username
            return redirect(url_for('recommend'))
    return render_template('login.html')

# Recommendation Page
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    #education = ""
    #skills = ""
    #recommended_jobs = []

    if request.method == 'POST':
        education = request.form['education']
        user_skills = request.form['skills']
        
        # Vectorize the user input
        user_vector = vectorizer.transform([user_skills])
        similarities = cosine_similarity(user_vector, job_vectors)
        
        # Get the top 5 most similar job indices
        top_indices = similarities[0].argsort()[-5:][::-1]
        
        # Fetch the recommended job titles
        recommended_jobs = job_data.iloc[top_indices]['Job Title'].tolist()

        return render_template('recommend.html', username=username, jobs=recommended_jobs, education=education, skills=user_skills)
    
    return render_template('recommend.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
