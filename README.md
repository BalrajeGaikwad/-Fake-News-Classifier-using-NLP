# -Fake-News-Classifier-using-NLP


📌 Project Overview:
This project involves building a Fake News Classifier that can automatically detect whether a news article is real or fake using Natural Language Processing (NLP) and Machine Learning techniques. The system processes raw text data, extracts meaningful features, and trains a model to classify the news content accurately.

🧠 Technologies & Tools:
Python

Scikit-learn

Natural Language Toolkit (NLTK)

Pandas / NumPy

Matplotlib / Seaborn

Jupyter Notebook

🔧 Key Steps Involved:
Data Collection

Used a labeled dataset of real and fake news articles (e.g., from Kaggle).

Data Preprocessing

Removed punctuation, stopwords, and digits

Converted text to lowercase

Tokenization and Lemmatization (using NLTK or spaCy)

Feature Extraction

Used Bag of Words or TF-IDF Vectorization with CountVectorizer / TfidfVectorizer

Converted textual data into numerical vectors

Model Building

Trained models using algorithms like:

Multinomial Naive Bayes

Logistic Regression

(Optional) SVM or Random Forest for comparison

Evaluated using Accuracy, Precision, Recall, F1 Score

Confusion Matrix & Evaluation

Visualized model performance using confusion matrix and classification report

Prediction

Developed a function to input a custom news article and predict whether it is FAKE or REAL
