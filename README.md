# Disease Prediction Using Classifiers

A desktop GUI application that predicts disease from a selection of up to 5 symptoms, comparing predictions across two machine learning classifiers.

## How It Works

1. User selects up to 5 symptoms from dropdown menus
2. Two classifiers run independently on the input
3. Predicted disease and classifier accuracy are displayed side by side

## Classifiers

- **Random Forest** — ensemble of decision trees, majority vote prediction
- **Naive Bayes** — probabilistic classifier using Gaussian likelihood

## Dataset

- Symptom-to-disease mapping across 100 disease classes
- Stratified sampling for train/test split per disease class

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

## Run
```bash
pip install scikit-learn pandas numpy customtkinter
python disease_predictor.py
```
