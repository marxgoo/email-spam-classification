import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib


def train_and_save_model():
    # 1. Load dataset
    df = pd.read_csv("C:\Users\dell\OneDrive\Bureau\AI &ML\spam.csv", encoding="utf-8")

    # 2. Keep only needed columns and rename
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # 3. Clean dataset
    df = df.dropna().drop_duplicates()

    # 4. Encode labels (ham=0, spam=1)
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])

    # 5. Split into train/test
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. TF-IDF transformation
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 7. Train Naive Bayes model
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    # 8. Evaluate
    y_pred = nb.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 9. Save model + vectorizer
    joblib.dump(nb, "spam_classifier_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("\n✅ Model and vectorizer saved successfully!")


def predict_new_messages(messages):
    # Load saved model + vectorizer
    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Transform input
    messages_tfidf = vectorizer.transform(messages)

    # Predict
    predictions = model.predict(messages_tfidf)

    # Show results
    for msg, pred in zip(messages, predictions):
        label = "Spam" if pred == 1 else "Ham"
        print(f"\nMessage: {msg}\nPrediction: {label}")


if __name__ == "__main__":
    # Train model (only once, or when retraining is needed)
    train_and_save_model()

    # Test with new messages
    test_messages = [
        "Congratulations! You won a free iPhone. Claim your prize now!!!",
        "Hey, are we still meeting for the project tomorrow?"
    ]
    predict_new_messages(test_messages)
