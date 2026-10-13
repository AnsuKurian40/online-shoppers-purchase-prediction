# Online Shoppers Purchase Prediction

## Project Overview

This project uses Machine Learning to predict whether an online shopping session will result in a purchase. The goal is to identify patterns in visitor behavior and understand which session characteristics are associated with purchasing intention.

The project includes data cleaning, exploratory data analysis (EDA), feature preprocessing, model training, hyperparameter tuning, and model evaluation.

## Objectives

- Analyze online shopping session behavior.
- Identify patterns associated with purchase completion.
- Handle missing values, duplicate records, and class imbalance where appropriate.
- Train and compare multiple classification models.
- Evaluate model performance using accuracy, precision, recall, and F1-score.

## Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

- Original records: 12,330
- Records after duplicate removal: 12,205
- Target variable: `Revenue`
- Problem type: Binary classification

The target variable indicates whether a shopping session resulted in a purchase.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. Data loading and inspection
2. Data cleaning and duplicate removal
3. Exploratory data analysis and visualization
4. Feature and target separation
5. Categorical encoding and numerical feature scaling
6. Training and evaluating classification models
7. Hyperparameter tuning using RandomizedSearchCV
8. Comparing model performance and examining feature importance

## Models Implemented

- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Hyperparameter-tuned Random Forest Classifier

## Model Evaluation

The tuned Random Forest model was evaluated at different probability thresholds. The following results were obtained on the test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Original Random Forest | 86.64% | 54.86% | 82.72% | 65.97% |
| Tuned Random Forest (0.50 threshold) | 88.45% | 60.04% | 78.27% | 67.95% |
| Tuned Random Forest (0.55 threshold) | 89.43% | 63.96% | 74.35% | 68.77% |

The tuned Random Forest with a 0.55 probability threshold achieved the highest F1-score among these evaluated configurations.

**Note:** The threshold comparison was performed on the test set during experimentation. For a rigorous final evaluation, the threshold should be selected using validation data or cross-validation, followed by evaluation on an untouched test set.

## Key Findings

- `PageValues` was the most influential feature in the Random Forest feature-importance analysis.
- `ExitRates`, `ProductRelated_Duration`, and `ProductRelated` were also among the important features.
- The dataset is imbalanced, with fewer purchase sessions than non-purchase sessions.
- Precision, recall, and F1-score provide useful insights beyond accuracy when evaluating an imbalanced classification problem.

Feature importance indicates predictive contribution within the trained model; it does not establish causation.

## How to Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/AnsuKurian40/online-shoppers-purchase-prediction.git
   ```

2. Navigate to the project folder:

   ```bash
   cd online-shoppers-purchase-prediction
   ```

3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `online_shoppers_purchase_prediction.ipynb` and run the notebook cells.

## Future Improvements

- Select the classification threshold using validation data.
- Evaluate models using precision-recall curves and ROC-AUC.
- Apply cross-validation for more robust model comparisons.
- Investigate feature availability at prediction time to avoid data leakage.
- Develop a simple interface for predicting purchase intention.

## Author

**Ansu Kurian**

- GitHub: [AnsuKurian40](https://github.com/AnsuKurian40)

## Acknowledgements

Dataset source: [UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
