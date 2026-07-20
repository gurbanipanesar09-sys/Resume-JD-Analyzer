# Intelligent Resume–Job Description Matching System

## Developed By

Gurbani Panesar

## Project Overview

This project analyses the compatibility between a candidate's
resume and a job description using a Siamese Bidirectional
Long Short-Term Memory neural network.

The system predicts one of three categories:

- Weak Match
- Medium Match
- Strong Match

It also displays:

- Confidence score
- Matched skills
- Missing skills
- Candidate recommendation

## Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Joblib

## Dataset

The project uses a synthetically generated dataset containing
3,000 resume and job-description pairs.

The classes are balanced:

- 1,000 Weak Match examples
- 1,000 Medium Match examples
- 1,000 Strong Match examples

The dataset contains the following columns:

- resume
- job_description
- role
- matched_skills
- label
- match_level

## Model Architecture

The resume and job description pass through shared:

- Embedding layer
- Bidirectional LSTM layer
- Global Max Pooling layer

The two vectors are compared using:

- Original resume vector
- Original job-description vector
- Absolute vector difference
- Element-wise multiplication

The combined features pass through dense layers to predict
the final match category.

## Project Structure

```text
Gurbani_Panesar_Resume_JD_Analyzer/
│
├── data/
│   └── resume_jd_dataset.csv
│
├── models/
│   ├── resume_jd_model.keras
│   └── tokenizer.pkl
│
├── outputs/
│   ├── graphs/
│   │   ├── accuracy.png
│   │   ├── loss.png
│   │   └── confusion_matrix.png
│   │
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── generate_dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore