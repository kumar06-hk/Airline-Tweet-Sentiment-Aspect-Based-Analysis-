# Airline Tweet Sentiment & Aspect-Based Analysis

Sentiment analysis pipeline on the US Airline Twitter dataset, comparing lexicon-based, traditional machine learning, deep learning (BiLSTM), and transformer-based (DistilBERT) approaches, with an aspect-based sentiment analysis (ABSA) component to identify major service issues.

Group project for CDS6344 (Social Media Computing), Multimedia University.

## Overview

- 14,409 cleaned airline tweets
- Ground truth sentiment labels constructed via a majority-voting approach across multiple lexicon methods, since the dataset's original sentiment labels were intentionally withheld as part of the assignment
- Models compared: TextBlob, VADER, AFINN (lexicon-based), traditional ML (Logistic Regression, SVM with TF-IDF), BiLSTM, and DistilBERT
- SMOTE applied for class imbalance, k-fold cross-validation used for model evaluation
- Aspect-based sentiment analysis (ABSA) using SVO (subject-verb-object) triples to identify specific service issues (e.g. delays, staff, baggage)

## Key results

- **DistilBERT achieved the best performance: 95.42% test accuracy, 95.40% weighted F1**
- Full model comparison and per-model strengths/limitations are in `results/model_comparison.csv` and `results/model_strengths_limitations.csv`
- Aspect-level sentiment breakdown is in `results/absa_summary.csv`

## My contribution

I worked on the pipeline from preprocessing through traditional machine learning: data cleaning, ground truth sentiment label construction (majority-voting approach), lexicon-based methods (TextBlob, VADER, AFINN), and traditional ML models (Logistic Regression, SVM with TF-IDF) including the sentiment method output distribution analysis.

The deep learning components (BiLSTM, DistilBERT) were built by a teammate as part of this group project.

## Tech stack

Python, Pandas, Scikit-learn, TensorFlow/Keras (BiLSTM), Hugging Face Transformers (DistilBERT), NLTK/TextBlob/VADER

## Files

- `TT4L_G10_ASGN.ipynb` — full pipeline: preprocessing, ground truth construction, model training, evaluation, and ABSA
- `Tweets (3).csv` — raw dataset
- `results/` — evaluation outputs (ground truth labels, model comparison, ABSA summary, lexicon agreement)

## Note on trained models

Trained model files (BiLSTM weights, DistilBERT checkpoint, TF-IDF vectorizer, SVM/Logistic Regression pipelines, tokenizers) are not included due to file size. All results are reproducible by running the notebook, and evaluation outputs are documented in `results/` and within the notebook itself.
