import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import GridSearchCV

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_dataset(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded with {len(df)} documents:")
    print(df['category'].value_counts())
    return df

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # text = re.sub(r'\\d+', '', text)

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Function to train and save models
def train_and_save_models(csv_path):
    df = load_dataset(csv_path)
    df['processed_text'] = df['text'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['category'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    smote = SMOTE(random_state=42)
    # X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_smote, y_train_smote)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_smote, y_train_smote)

    rf_model = RandomForestClassifier(n_estimators = 100, random_state=42)
    rf_model.fit(X_train_smote, y_train_smote)

    #Save the models and vectorizer
    with open('models/vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
    
    with open('models/nb_model.pkl', 'wb') as nb_file:
        pickle.dump(nb_model, nb_file)

    with open('models/lr_model.pkl', 'wb') as lr_file:
        pickle.dump(lr_model, lr_file)
    
    with open('models/rf_model.pkl', 'wb') as rf_file:
        pickle.dump(rf_model, rf_file)

    print("Models and vectorizer saved to disk.")

# Function to load the models and vectorizer
def load_models():
    with open('models/vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    with open('models/nb_model.pkl', 'rb') as nb_file:
        nb_model = pickle.load(nb_file)

    with open('models/lr_model.pkl', 'rb') as lr_file:
        lr_model = pickle.load(lr_file)

    with open('models/rf_model.pkl', 'rb') as rf_file:
        rf_model = pickle.load(rf_file)

    return vectorizer, nb_model, lr_model, rf_model

# Function to train and evaluate classification models
def train_models(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['category'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_predictions = nb_model.predict(X_test_tfidf)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)

    print("Naive Bayes Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, nb_predictions):.4f}")
    print(classification_report(y_test, nb_predictions))

    print("\nLogistic Regression Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, lr_predictions):.4f}")
    print(classification_report(y_test, lr_predictions))

    #Confusion matrices and plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    nb_cm = confusion_matrix(y_test, nb_predictions)
    sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['category'].unique()), yticklabels=sorted(df['category'].unique()))
    plt.title('Naive bayes Confusion Matrix')

    plt.subplot(1, 2, 2)
    lr_cm = confusion_matrix(y_test, lr_predictions)
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['category'].unique()), yticklabels=sorted(df['category'].unique()))
    plt.title('Logistic Regression Confusion Matrix')

    plt.tight_layout()
    plt.show()


    return vectorizer, (nb_model if accuracy_score(y_test, nb_predictions) > accuracy_score(y_test, lr_predictions) else lr_model)

# Function to classify new text
def classify_text(text, vectorizer, models):
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    predictions = {model_name: model.predict(text_tfidf)[0] for model_name, model in models.items()}
    probabilities = {model_name: model.predict_proba(text_tfidf)[0] for model_name, model in models.items()}

    prob_dict ={}
    for model_name, prob_array in probabilities.items():

        classes = models[model_name].classes_
 #   probabilities = model.predict_proba(text_tfidf)[0]
 #   categories = model.classes_
    prob_dict[model_name] = {classes[i]: prob_array[i] for i in range(len(classes))}
    return predictions, prob_dict
#Interactive classifier function
def interactive_classifier(vectorizer, models):
    print("\nInteractive Classifier: Type 'quit' to exit.")

    while True:
        text = input("Enter a document to classify: ")
        if text.lower() == 'quit':
            print("Exiting interactive classifier.")
            break

        #prediction, probabilities = classify_text(text, vectorizer, model)
        predictions,prob_dict = classify_text(text, vectorizer, models)
        print(f"Prediction: {predictions}")
        for model, prediction in predictions.items():
            print(f"{model}: {prediction}")
        print("-" * 50)
        print("Confidence Scores:")
        for model, category_probabilities in prob_dict.items():
            for category, prob in sorted(category_probabilities.items(), key=lambda x:x[1], reverse=True):
                print(f"{category}: {prob:4f} ({prob * 100:.2f}%)")

        # for category, probability in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        #     print(f"{category}:{probability:.4f} ({probability * 100:.2f}%)")

        print("-" * 50)

# Main function
def main(csv_path=None):
    if csv_path:
        train_and_save_models(csv_path)
        vectorizer, nb_model, lr_model, rf_model = load_models()
        models = {
            "Naive Bayes": nb_model,
            "Logistic Regression": lr_model,
            "Random Forest": rf_model
        }

    #Start interactive classifier
        interactive_classifier(vectorizer, models)
    else:
        print("No datset provided. Please provide a CSV path.")
    return

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # train_and_save_models(sys.argv[1])
        main(sys.argv[1])


    else:
        print("Please provide a path to the CSV dataset file: python document_classifier.py path/to/dataset.csv")


