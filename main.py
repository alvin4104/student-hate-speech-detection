"""
Project: Hate Speech Detection in Student Online Forums
"""

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. DATASET (FIXED) ───────────────────────────────────────────────
comments = [
    # HATE (13)
    "You are so dumb, people like you should not be allowed in school.",
    "Stop embarrassing yourself in class, everyone laughs at you.",
    "That student is such a freak, nobody wants to sit near them.",
    "You will never pass any exam, you are just too stupid to learn.",
    "Losers like you do not deserve to be in this university at all.",
    "Go back to your village, you do not belong in this school.",
    "Why does the teacher even bother with students like you?",
    "You are the worst student in the class, everyone thinks so.",
    "People like you make this school look bad, just drop out already.",
    "Your presentation was pathetic, everyone was laughing behind your back.",
    "You are too ugly and stupid to have any friends here.",
    "Nobody wants to be in your study group, you bring everyone down.",
    "Students like you should be expelled, you ruin everything for others.",

    # NON-HATE (14)
    "Could anyone share the notes from today's lecture?",
    "Great presentation! I learned a lot.",
    "Anyone want to form a study group?",
    "I disagree but you made good points.",
    "The professor explained it very well.",
    "Can someone help me understand this?",
    "We should support each other.",
    "Your essay was well structured.",
    "Let’s meet at the library tomorrow.",
    "I found a helpful resource online.",
    "The seminar was interesting.",
    "Thanks for explaining that.",
    "Our class is hardworking.",
    "Good luck for the exam everyone!"
]

# 🔥 FIX: auto label (không bao giờ lệch)
labels = ['HATE'] * 13 + ['NON-HATE'] * 14

df = pd.DataFrame({
    'comment': comments,
    'label': labels
})

print("Dataset shape:", df.shape)

# ── 2. PREPROCESS ───────────────────────────────────────────────────
STOP_WORDS = set([
    'the','a','an','is','it','in','on','at','to','for','of','and',
    'or','but','this','that','was','are','be','have','has','had',
    'do','does','did','will','would','could','should','with','from',
    'by','about','as','its','were','been','their','they','we','i',
    'my','your','me','him','her','us','our','just','so','if','not',
    'very','really','too','also','even','still','already'
])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

df['clean'] = df['comment'].apply(preprocess)

# ── 3. SPLIT ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# ── 4. TF-IDF ────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)

# ── 5. MODELS ────────────────────────────────────────────────────────
models = {
    "NB": MultinomialNB(),
    "SVM": LinearSVC(),
    "LR": LogisticRegression(max_iter=200)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    results[name] = acc
    print(f"\n{name} Accuracy:", acc)
    print(classification_report(y_test, pred))

# ── 6. VISUAL ────────────────────────────────────────────────────────
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy")
plt.show()

# WordCloud
hate_text = " ".join(df[df['label']=="HATE"]['clean'])
wc = WordCloud().generate(hate_text)

plt.imshow(wc)
plt.axis("off")
plt.show()

# ── 7. CONFUSION MATRIX ──────────────────────────────────────────────
best_model = max(models, key=lambda x: results[x])
pred = list(models.values())[list(models.keys()).index(best_model)].predict(X_test)

cm = confusion_matrix(y_test, pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title(f"Confusion Matrix - {best_model}")
plt.show()