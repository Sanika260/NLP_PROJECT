import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# --------------------------
# Data preprocessing function
# --------------------------
lemma = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))

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

# --------------------------
# Load and prepare dataset
# --------------------------
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

fake['category'] = 1
true['category'] = 0

df = pd.concat([fake, true]).reset_index(drop=True)
df = df[['text', 'category']]

blanks = [i for i, text in df['text'].items() if str(text).isspace()]
df.drop(blanks, inplace=True)

df['text'] = df['text'].apply(clean_text)

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['category'], test_size=0.33, random_state=42
)

# --------------------------
# Define models
# --------------------------
models = {
    'Passive Aggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(max_iter=1000, random_state=42)
}

# --------------------------
# Train, evaluate and save best model
# --------------------------
results = {}

for name, model in models.items():
    print(f"\nTraining and evaluating {name} model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{metrics.classification_report(y_test, preds)}")

    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy
    }

# Pick best model by accuracy
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_pipeline = results[best_model_name]['pipeline']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Save the best model to disk
joblib.dump(best_pipeline, 'fake_news_best_model.pkl')
print(f"Best model saved as 'fake_news_best_model.pkl'")
