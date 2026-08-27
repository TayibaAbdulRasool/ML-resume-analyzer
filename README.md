# ML Resume Analyzer

An end-to-end Machine Learning project that analyzes candidate resume-related data, predicts candidate outcomes, evaluates multiple classification models, and generates candidate scores and rankings.

## Overview

**ML Resume Analyzer** is a machine learning application designed to assist in resume screening and candidate evaluation.

The project follows a complete ML workflow:

**Data Processing → Exploratory Analysis → Feature Engineering → Model Training → Model Evaluation → Fairness Analysis → Candidate Scoring → Streamlit Deployment**

The goal is to build a practical and interpretable machine learning solution that can help recruiters identify potentially suitable candidates while reducing repetitive manual screening.

## Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature encoding and scaling
* Multiple machine learning classification models
* Model comparison using standard evaluation metrics
* Confusion matrix analysis
* False-positive and false-negative analysis
* Candidate score generation
* Candidate ranking
* Basic fairness analysis
* Final model saving
* Interactive Streamlit interface

## Machine Learning Models

The project compares several classification algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Classifier (SVC)
* XGBoost

Models are evaluated and compared to identify the most suitable classifier for the dataset.

## Evaluation Metrics

The following metrics are used to evaluate model performance:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

False positives and false negatives are also analyzed to understand the types of prediction errors made by the model.

## Project Workflow

### 1. Data Preparation

The dataset is loaded and examined for:

* Missing values
* Duplicate records
* Data types
* Categorical features
* Numerical features
* Target distribution

### 2. Exploratory Data Analysis

EDA is performed to understand relationships between candidate attributes and the target variable.

Visualizations are used to identify:

* Feature distributions
* Candidate patterns
* Class imbalance
* Relationships between important features

### 3. Feature Engineering

Categorical variables are transformed into numerical representations using encoding techniques.

Numerical features are standardized where required using `StandardScaler`.

### 4. Model Training

Multiple classification algorithms are trained on the prepared dataset.

The models are then evaluated using the same evaluation metrics to enable a fair comparison.

### 5. Model Evaluation

Model performance is compared using:

```text
Accuracy
Precision
Recall
F1-Score
```

The confusion matrix is also used to examine correct and incorrect predictions.

### 6. Fairness Analysis

The project includes a basic fairness analysis to investigate whether model predictions differ across relevant candidate groups.

This helps highlight potential bias and encourages responsible use of machine learning in recruitment.

### 7. Candidate Scoring & Ranking

Candidates are assigned a machine-learning-based score.

Candidates can then be ranked according to their predicted suitability, helping demonstrate how a model could support the screening process.

> The generated score should be treated as a decision-support signal, not as the sole basis for hiring decisions.

## Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Core programming language      |
| Pandas           | Data manipulation              |
| NumPy            | Numerical computing            |
| Matplotlib       | Data visualization             |
| Scikit-learn     | Machine learning               |
| XGBoost          | Gradient boosting model        |
| Streamlit        | Interactive web application    |
| Jupyter Notebook | Model development and analysis |
| Git & GitHub     | Version control                |

## Project Structure

```text
ML-Resume-Analyzer/
│
├── ML Resume Analyzer.ipynb
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── final_model.pkl
│
├── data/
│   └── dataset.csv
│
└── .gitignore
```

The exact structure may vary depending on the final version of the project.

## UI

<img width="1132" height="742" alt="image" src="https://github.com/user-attachments/assets/eeb0d065-29d0-4686-bce0-9db8a69a4ef8" />


## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/ML-resume-analyzer.git
```

Move into the project directory:

```bash
cd ML-resume-analyzer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
ML Resume Analyzer.ipynb
```

Run the cells sequentially to reproduce the data analysis, model training, evaluation, and candidate ranking workflow.

## Running the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Example ML Pipeline

```text
Resume / Candidate Data
          ↓
    Data Cleaning
          ↓
    Feature Engineering
          ↓
    Encoding & Scaling
          ↓
    Train / Test Split
          ↓
    Model Training
          ↓
    Model Comparison
          ↓
    Performance Evaluation
          ↓
    Fairness Analysis
          ↓
 Candidate Score & Ranking
          ↓
    Streamlit Application
```

## Key Learning Outcomes

This project demonstrates practical experience with:

* End-to-end machine learning workflows
* Supervised classification
* Feature preprocessing
* Model comparison
* Classification evaluation
* Error analysis
* Fairness considerations in ML
* Candidate ranking systems
* Model persistence
* Streamlit application development

## Future Improvements

Possible improvements include:

* Resume PDF/DOCX parsing
* Natural Language Processing (NLP)
* Skill extraction from resumes
* Job-description matching
* Semantic similarity using embeddings
* Explainable AI using SHAP
* More comprehensive fairness metrics
* Cross-validation and hyperparameter tuning
* Candidate recommendation system
* Improved Streamlit dashboard
* Cloud deployment

## Disclaimer

This project is intended for **educational and demonstration purposes**.

Recruitment decisions can have significant consequences. A machine learning prediction should not be used as the sole criterion for hiring, rejection, or other employment decisions. Human review and appropriate fairness and compliance checks should remain part of the process.

## Author

**Tayiba Abdul Rasool**

