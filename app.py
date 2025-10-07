from flask import Flask, request, render_template
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import numpy as np
import joblib
import webbrowser
from threading import Timer

app = Flask(__name__)

# Load trained pipeline model
text_clf = joblib.load('models/fake_news_best_model.pkl')

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

def generate_fake_reason(text):
    reasons = []
    suspicious_keywords = ["shocking", "incredible", "viral", "exposed", "secret", "breaking", "unbelievable"]
    subjective_words = ["think", "feel", "believe", "rumor", "allegedly", "supposedly", "claimed"]

    for word in suspicious_keywords:
        if word in text.lower():
            reasons.append(f"It uses words like '{word}', which are often found in exaggerated or misleading news stories.")
    for word in subjective_words:
        if word in text.lower():
            reasons.append(f"It includes words like '{word}', which suggest that the news may be someone's opinion or a rumor, not proven fact.")
    if not reasons:
        reasons.append("The system found patterns or language that are common in news stories that may not be completely true or reliable.")
    return reasons

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['news_text']
        cleaned_text = clean_text(text)
        if text.strip() == '':
            result = "Please enter some news text."
        else:
            pred = text_clf.predict([cleaned_text])[0]
            label = 'Fake' if pred == 1 else 'Real'
            confidence_score = text_clf.decision_function([cleaned_text])[0]
            confidence_percentage = np.tanh(abs(confidence_score)) * 100
            sentiment = TextBlob(text).sentiment
            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity

            reason = []
            if label == "Fake":
                reason = generate_fake_reason(text)

            result = {
                'label': label,
                'confidence': f"{confidence_percentage:.2f}%",
                'polarity': f"{polarity:.2f}",
                'subjectivity': f"{subjectivity:.2f}",
                'text': text,
                'reason': reason
            }
    return render_template('index.html', result=result)

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    Timer(1, lambda: webbrowser.open(url)).start()
    app.run(debug=True, port=port)
