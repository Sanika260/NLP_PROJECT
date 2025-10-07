import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from textblob import TextBlob
import numpy as np

# Load datasets
fake = pd.read_csv('fake.csv')
true = pd.read_csv('true.csv')

# Label datasets
fake['category'] = 1
true['category'] = 0

# Combine datasets & reset index
df = pd.concat([fake, true]).reset_index(drop=True)

# Use only text and category columns
df = df[['text', 'category']]

# Remove rows with empty or whitespace-only text
blanks = [i for i, text in df['text'].items() if str(text).isspace()]
df.drop(blanks, inplace=True)

# Initialize lemmatizer and stopwords
lemma = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", ' ', text)

    words = [lemma.lemmatize(word) for word in text.split() if word not in stopwords_set]
    return ' '.join(words)


# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'],
                                                    test_size=0.33, random_state=42)

# Create pipeline: TF-IDF vectorizer + LinearSVC classifier
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Train model
text_clf.fit(X_train, y_train)

# Evaluate on test set
predictions = text_clf.predict(X_test)
print(metrics.classification_report(y_test, predictions))
print('Accuracy:', metrics.accuracy_score(y_test, predictions))


# Enhanced prediction function with extra features
def predict_news(text):
    cleaned_text = clean_text(text)
    pred = text_clf.predict([cleaned_text])[0]
    label = 'Fake' if pred == 1 else 'Real'

    confidence_score = text_clf.decision_function([cleaned_text])[0]
    confidence_percentage = np.tanh(abs(confidence_score)) * 100

    sentiment = TextBlob(text).sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    word_count = len(text.split())
    avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0

    print("\n--- News Verification Result ---")
    print(f"Prediction: {label}")
    print(f"Model Confidence: {confidence_percentage:.2f}%")
    print(f"Sentiment Polarity: {polarity:.2f} (-1 negative to +1 positive)")
    print(f"Sentiment Subjectivity: {subjectivity:.2f} (0 factual to 1 opinionated)")
    print(f"Text Word Count: {word_count}")
    print(f"Average Word Length: {avg_word_length:.2f}")

    if confidence_percentage < 30:
        print("Note: Model confidence is low. Interpret the result with caution.")
    if polarity < -0.5 and label == "Real":
        print("Warning: Negative sentiment detected but predicted as Real news.")
    elif polarity > 0.5 and label == "Fake":
        print("Warning: Positive sentiment detected but predicted as Fake news.")


# User input
user_input = input("Enter news text to check if it's Fake or Real: ")
predict_news(user_input)
