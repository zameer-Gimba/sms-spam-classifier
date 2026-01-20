# Neural Network SMS Spam Classifier

## Overview
This project implements a neural network–based SMS spam classifier using Natural Language Processing (NLP) techniques. Messages are transformed into numerical features using TF-IDF vectorization and classified as **spam** or **ham** using a feed-forward neural network built with TensorFlow/Keras.

This repository is part of my applied machine learning portfolio and reflects hands-on experience with text preprocessing, supervised learning, and model evaluation.


## Key Features
- TF-IDF vectorization for text representation
- Binary classification (Spam vs Ham)
- Neural network with dropout regularization
- Reproducible results via fixed random seeds
- Jupyter Notebook included for exploratory analysis and visualization


## Dataset
The dataset consists of SMS messages labeled as `spam` or `ham`.

Files:
- `data/train-data.tsv`
- `data/valid-data.tsv`


## How to Run (Python Script)

### 1. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the classifier
 ```
python src/sms_classifier.py
```
The model will train and output accuracy on the validation dataset.

### How to Run Tests
```
python tests/test.py
```
### Jupyter Notebook

The notebook NN_sms_text_classifier.ipynb contains:

Data loading

Model training

Evaluation

Example predictions

Note: Some plots and outputs may not render properly in GitHub’s notebook viewer.
For best results, open the notebook locally or in Google Colab.

### Technologies Used

Python

TensorFlow / Keras

Scikit-learn

Pandas

NumPy

Matplotlib

### Author

Muhammad Ibrahim Gimba
Machine Learning | Data Science | NLP | Remote-ready
