# Document Classifier App

## Overview
This project is a **Document Classification Web Application** developed as part of **Task 2** for the **Intelligent Information Retrieval (STW7071CEM)** coursework. The application classifies input text into predefined categories using **Naïve Bayes** and **Logistic Regression** models.

## Features
- 📄 **User-friendly Web Interface** built with Flask & Bootstrap
- 🔍 **Text Classification** using **Naïve Bayes** and **Logistic Regression**
- 📊 **Confidence Scores** displayed for each prediction
- 🎨 **Enhanced UI** for a better user experience during demonstration and viva

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.7+** installed on your system.

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
python app.py
```
Access the application at **http://127.0.0.1:5000/** in your web browser.

## Project Structure
```
📁 DOCUMENT CLASSIFIER
│── 📁 __pycache__             # Compiled Python files
│── 📁 dataset                 # Dataset files for training/testing
│   │── classifier_data_20250308092100.csv
│   │── news_articles_20250308092100.csv
│── 📁 models                  # Saved machine learning models
│   │── lr_model.pkl           # Logistic Regression model
│   │── nb_model.pkl           # Naïve Bayes model
│   │── vectorizer.pkl         # TF-IDF vectorizer
│── 📁 templates               # HTML templates
│   │── index.html             # Input page
│   │── result.html            # Prediction results page
│── 📄 app.py                  # Main Flask application
│── 📄 document_classifier.py   # Model training & classification logic
│── 📄 Figure_1.png            # Visualization or report figure
│── 📄 Procfile                # Deployment configuration for Heroku
│── 📄 requirements.txt        # Required Python packages
│── 📄 rss_crawler.py          # Web crawler for data collection
│── 📄 runtime.txt             # Python runtime version for deployment
```

## Usage
1. Open the app in your browser.
2. Enter or paste text into the input field.
3. Click **Classify Text** to get predictions.
4. View classification results with confidence scores.
5. Click **Try Another Document** to classify a new text.

## Technologies Used
- **Flask** - Web framework
- **Bootstrap** - UI Styling
- **scikit-learn** - Machine learning models
- **HTML, CSS** - Frontend design

## Future Enhancements
- 📌 Implement **Deep Learning Models** for better accuracy
- 📌 Add **API endpoints** for classification
- 📌 Include **More Categories** for classification

## Author
Developed by: **[Your Name]**

## License
This project is licensed under the **MIT License**.

## Acknowledgments
- Coventry University - **Intelligent Information Retrieval Module**
- Lecturer: **Siddhartha Neupane**

