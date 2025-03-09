from flask import Flask, request, render_template
import pickle
from document_classifier import load_models, classify_text, preprocess_text

app = Flask(__name__)

# Load models and vectorizer
vectorizer, nb_model, lr_model, rf_model = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')

    if not text:
        return render_template('result.html', error="No text provided!")

    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])

    nb_prediction = nb_model.predict(text_tfidf)[0]
    nb_probabilities = nb_model.predict_proba(text_tfidf)[0]

    lr_prediction = lr_model.predict(text_tfidf)[0]
    lr_probabilities = lr_model.predict_proba(text_tfidf)[0]

    rf_prediction = rf_model.predict(text_tfidf)[0]
    rf_probabilities = rf_model.predict_proba(text_tfidf)[0]

    categories = nb_model.classes_

    nb_prob_dict = {categories[i]: nb_probabilities[i] for i in range(len(categories))}
    lr_prob_dict = {categories[i]: lr_probabilities[i] for i in range(len(categories))}
    rf_prob_dict = {categories[i]: rf_probabilities[i] for i in range(len(categories))}

    model_results = {
        "Naive Bayes": {"prediction": nb_prediction, "scores": nb_prob_dict},
        "Logistic Regression": {"prediction": lr_prediction, "scores": lr_prob_dict}, 
        "Random Forest": {"prediction":rf_prediction, "scores": rf_prob_dict}
    }

    sorted_models = sorted(
        model_results.items(),
        key=lambda item: max(item[1]["scores"].values()),
        reverse=True
    )

    return render_template('result.html', 
                           sorted_models=sorted_models,
                           nb_prob_dict=nb_prob_dict,
                           lr_prob_dict=lr_prob_dict,
                           rf_prob_dict=rf_prob_dict
                           )

if __name__ == "__main__":
    app.run(debug=True)