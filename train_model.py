import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load datasets
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake['category'] = 1
true['category'] = 0

df = pd.concat([fake, true]).reset_index(drop=True)

df = df[['text', 'category']]

blanks = [i for i, text in df['text'].items() if str(text).isspace()]
df.drop(blanks, inplace=True)

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


df['text'] = df['text'].apply(clean_text)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

pipeline.fit(df['text'], df['category'])

# Save the trained model
joblib.dump(pipeline, 'fake_news_model.pkl')

print("Model training complete and saved as fake_news_model.pkl")
