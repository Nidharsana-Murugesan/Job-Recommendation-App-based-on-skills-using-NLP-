
# Job Recommendation System Based on Skills

## Introduction
The **Job Recommendation System Based on Skills** is a web-based application that leverages machine learning and natural language processing (NLP) techniques to recommend jobs to users based on their skills. The platform simplifies the job search process by providing personalized job recommendations, making it easier for users to find positions that align with their expertise. The system uses **TF-IDF vectorization** to analyze job descriptions and a **Random Forest Classifier** to predict the most suitable job titles based on user input.

---


## Tech Stack
- **Frontend**: HTML, CSS, JavaScript (for an intuitive and user-friendly interface).
- **Backend**: Python (Flask/Django for server-side logic).
- **Machine Learning**:
  - **Model**: Random Forest Classifier for job prediction.
  - **NLP**: TF-IDF vectorization for analyzing job descriptions.


## Libraries used :

For Model Building:
-- **Pandas**: For data manipulation and preprocessing.
-- **NumPy**: For numerical computations.
-- **Scikit-learn**: Random Forest Classifier for model training and prediction.
		     TF-IDF Vectorizer for natural language processing.
		     Train-test split for data splitting.
		     Performance metrics such as accuracy and classification report.
-- **FuzzyWuzzy** : Matching user-input skills to predefined skill sets in the job descriptions with approximate matching.
-- **Levenshtein** : It speeds up FuzzyWuzzy computations significantly because FuzzyWuzzy relies on Levenshtein distance under the hood for string similarity calculations.

For Model Loading:
-- **Joblib**: For saving and loading the machine learning model efficiently.
-- **Pickle**: For serialization and deserialization of the trained model.

Additional Libraries:
-- **Flask** : For backend web framework integration.
-- **HTML/CSS/JavaScript** : For frontend UI development.