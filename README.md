# ML Resume Analyzer

A machine learning-based resume screening project that predicts whether a candidate is likely to be **shortlisted** based on structured resume and candidate-related features.

The project covers the complete machine learning workflow, from data exploration and preprocessing to model comparison, hyperparameter tuning, evaluation, fairness analysis, candidate scoring, and deployment through Streamlit.

## Live Application

Streamlit Application:[ML Resume Analyzer — Streamlit App](https://tayibaabdulrasool-ml-resume-analyzer-app-vxq5rs.streamlit.app/)

## Project Overview

The **ML Resume Analyzer** uses candidate information such as years of experience, skills match, education level, projects, resume length, and GitHub activity to predict the `shortlisted` outcome.

Instead of manually evaluating every candidate based on individual factors, the project uses supervised machine learning to learn patterns from historical candidate data and generate a shortlist prediction.

### Dataset Size

* **Rows:** 30,001
* **Columns:** 7
* **Target variable:** `shortlisted`
* **Problem type:** Binary Classification

## Dataset Columns

| Column               | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `years_experience`   | Number of years of professional experience of the candidate                     |
| `skills_match_score` | Score representing how closely the candidate's skills match the required skills |
| `education_level`    | Candidate's education qualification/level                                       |
| `project_count`      | Number of projects included by the candidate                                    |
| `resume_length`      | Length of the candidate's resume                                                |
| `github_activity`    | Measure of the candidate's GitHub activity                                      |
| `shortlisted`        | Target variable indicating whether the candidate was shortlisted                |

The first six columns are used as input features, while `shortlisted` is the target variable.

## How the Project Works

The project follows a complete supervised machine learning pipeline:

```text
Candidate Dataset
       ↓
Data Inspection & Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Preparation
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
Train-Test Split
       ↓
Multiple Model Training
       ↓
Model Comparison
       ↓
Cross-Validation
       ↓
Hyperparameter Tuning
       ↓
Final Model Evaluation
       ↓
Fairness & Error Analysis
       ↓
Candidate Score & Ranking
       ↓
Streamlit Application
```

## 1. Data Loading & Initial Analysis

The dataset is loaded using Pandas and examined to understand its structure.

The analysis includes:

* Dataset shape
* Data types
* Missing values
* Duplicate records
* Numerical and categorical features
* Target distribution

The dataset contains **30,001 candidate records and 7 columns**.

## 2. Exploratory Data Analysis

Exploratory analysis is performed to understand how candidate characteristics relate to the `shortlisted` outcome.

The analysis focuses on factors such as:

* Years of experience
* Skills match score
* Education level
* Number of projects
* Resume length
* GitHub activity
* Shortlisting distribution

The visualizations help identify patterns in the data before applying machine learning models.

## 3. Data Preprocessing

The candidate data contains both numerical and categorical variables.

Categorical features are converted into numerical values using **Label Encoding**.

Numerical features are standardized using **StandardScaler** so that features with different scales can be used effectively by the machine learning algorithms.

## 4. Train-Test Split

The processed dataset is divided into training and testing data.

The training data is used to learn patterns from the candidates, while the test data is used to evaluate how well the models perform on unseen candidates.

A stratified split is used so that the target-class distribution is maintained between the training and testing datasets.

## 5. Machine Learning Models

The project compares six classification algorithms:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**
5. **Support Vector Classifier (SVC)**
6. **XGBoost**

Comparing multiple models helps determine which algorithm performs best for this candidate-screening problem.

## 6. Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The project also performs **5-fold cross-validation** to obtain a more reliable estimate of model performance.

## 7. Hyperparameter Tuning

After comparing the initial models, hyperparameter tuning is performed using **GridSearchCV**.

The purpose of tuning is to find better model parameters and improve the final model's performance.

The tuned final model achieved approximately:

| Metric            |     Result |
| ----------------- | ---------: |
| **Test Accuracy** | **90.67%** |
| **F1-Score**      | **93.37%** |

The final model therefore provides strong classification performance on the test dataset.

## 8. Final Model

The final model is selected after comparing the different algorithms and performing hyperparameter tuning.

The final trained model is then saved so that it can be reused by the Streamlit application without retraining the model every time the application starts.

## 9. Confusion Matrix & Error Analysis

A confusion matrix is used to understand the model's predictions in more detail.

The project also analyzes:

* **False Positives** — candidates predicted as shortlisted when they were not.
* **False Negatives** — candidates predicted as not shortlisted when they were actually shortlisted.

This analysis is important because accuracy alone does not explain what types of mistakes the model makes.

## 10. Feature Importance

Feature importance is analyzed to understand which candidate characteristics contribute most to the model's predictions.

This provides better interpretability and helps answer questions such as:

* Does skills match have a strong influence?
* How important is experience?
* Does GitHub activity contribute to shortlisting?
* How much does education level affect the prediction?
* Do projects influence candidate selection?

This makes the project more interpretable than simply presenting a prediction from a black-box model.

## 11. Candidate Score & Ranking

In addition to predicting whether a candidate is shortlisted, the project generates a candidate score that can be used to rank candidates.

The ranking provides a way to prioritize candidates based on the model's predicted suitability.

Example:

```text
Candidate A → 94% → High Priority
Candidate B → 88% → High Priority
Candidate C → 72% → Medium Priority
Candidate D → 41% → Low Priority
```

The score and ranking are intended to support the screening process rather than replace human decision-making.

## 12. Fairness Analysis

The project includes a fairness analysis to examine whether model predictions behave differently across relevant candidate groups.

This is particularly important for a recruitment-related machine learning system because model performance should not be considered only from an accuracy perspective.

The analysis helps identify potential differences in prediction outcomes and encourages responsible use of the model.

## 13. Streamlit Application

The trained model is integrated into a **Streamlit web application**.

The application provides an interactive interface where candidate information can be entered.

The input features correspond to the dataset:

```text
Years of Experience
Skills Match Score
Education Level
Project Count
Resume Length
GitHub Activity
```

The model then provides the candidate's predicted shortlisting result and score.

### Application Flow

```text
Candidate Information
        ↓
Streamlit Input
        ↓
Preprocessing
        ↓
Saved ML Model
        ↓
Prediction
        ↓
Candidate Score
        ↓
Shortlist Result
```

## UI

![ML Resume Analyzer UI](https://github.com/user-attachments/assets/eeb0d065-29d0-4686-bce0-9db8a69a4ef8)

## Recommended Project Outputs

The following outputs represent the important ML work in this project and are useful to showcase on GitHub:

### 1. Model Comparison

Show a table or bar chart comparing:

```text
Logistic Regression
Decision Tree
Random Forest
KNN
SVC
XGBoost
```

using Accuracy, Precision, Recall, and F1-Score.

### 2. Confusion Matrix

Include the confusion matrix of the final tuned model.

This visually shows:

* Correct shortlisted predictions
* Correct non-shortlisted predictions
* False positives
* False negatives

### 3. Feature Importance

Include a feature-importance chart showing which of the six input features have the greatest influence on predictions.

### 4. Candidate Score / Ranking

Show a sample output demonstrating how candidates are assigned scores and ranked.

### 5. Streamlit UI

The Streamlit screenshot demonstrates the final interactive application.

These outputs are more valuable for the README than adding many unrelated charts because they directly demonstrate the machine learning pipeline and results.

## Project Structure

```text
ML-Resume-Analyzer/
│
├── ML Resume Analyzer.ipynb
│
├── app.py
│
├── requirements.txt
│
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

### File Description

| File / Folder              | Purpose                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `ML Resume Analyzer.ipynb` | Complete data analysis, preprocessing, model training, evaluation, tuning, fairness analysis, and candidate ranking |
| `app.py`                   | Streamlit application for interacting with the trained model                                                        |
| `requirements.txt`         | Python dependencies required to run the project                                                                     |
| `models/`                  | Stores the trained final machine learning model                                                                     |
| `data/`                    | Contains the candidate dataset                                                                                      |
| `README.md`                | Project documentation                                                                                               |
| `.gitignore`               | Prevents unnecessary or sensitive files from being committed                                                        |

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **XGBoost**
* **Jupyter Notebook**
* **Streamlit**
* **Git & GitHub**

## Results

The project evaluates multiple classification models and then applies hyperparameter tuning to improve the selected model.

The tuned final model achieved approximately:

**90.67% Test Accuracy**

with an **F1-Score of 93.37%**.

The project also goes beyond accuracy by including:

* Cross-validation
* Hyperparameter tuning
* Classification report
* Confusion matrix
* Feature importance
* False-positive analysis
* False-negative analysis
* Fairness analysis
* Candidate scoring and ranking

## Key Takeaway

The ML Resume Analyzer demonstrates how structured candidate information can be processed and used with supervised machine learning to predict shortlisting outcomes.

Rather than relying on a single model or metric, the project compares multiple algorithms, tunes the final model, analyzes prediction errors, examines feature importance, and presents the result through an interactive Streamlit application.

## Disclaimer

This project is developed for educational and demonstration purposes. The model should be used as a decision-support tool and not as the sole basis for real-world hiring or rejection decisions.

## Author

**Tayiba Abdul Rasool**

Software Engineering Student | Machine Learning | Data Analytics | Generative AI

