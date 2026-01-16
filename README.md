# Income Classification Using Machine Learning

## Project Overview
This project applies supervised machine learning techniques to predict whether an individual earns more than \$50,000 per year based on demographic and employment attributes. Multiple models are trained, tuned, and evaluated to compare performance and generalization behavior.

The project emphasizes real-world machine learning practices including preprocessing, model selection, hyperparameter tuning, and validation-based evaluation.

---

## Dataset
- **Source:** UCI Adult Census Income Dataset
- **Target Variable:** Income (`<=50K` or `>50K`)
- **Features:** Age, education, occupation, hours worked per week, marital status, race, sex, and other demographic attributes
- **Key Challenge:** Moderate class imbalance and mixed data types (categorical + numerical)

---

## Data Preprocessing
- Cleaned raw census data and handled missing values
- Encoded categorical variables using one-hot encoding
- Scaled numerical features where appropriate
- Created consistent preprocessing pipelines across models
- Established a **Dummy Classifier baseline** for comparison

---

## Models Evaluated
The following models were implemented and evaluated:

- Dummy Classifier (baseline)
- Logistic Regression
- Decision Tree
- Neural Network (PyTorch)
- Support Vector Machine (SVM)

Hyperparameter tuning was performed using **GridSearchCV** and **RandomizedSearchCV** where applicable.

---

## Model Performance Comparison

| Model     | Train Accuracy | Validation Accuracy |
|-----------|----------------|---------------------|
| Dummy     | 0.76           | 0.76                |
| LogReg    | 0.81           | 0.81                |
| DecTree  | 0.87           | 0.86                |
| NN       | 0.83           | 0.81                |
| SVM      | 0.94           | 0.82                |

---

## Results & Analysis
- The **Dummy Classifier** confirms the baseline accuracy driven by class imbalance.
- **Logistic Regression** provides a strong linear baseline with stable generalization.
- The **Decision Tree** achieved the **highest validation accuracy (0.86)** with a minimal gap between training and validation performance.
- The **Neural Network** performed competitively but did not outperform simpler models on tabular data.
- The **SVM** showed signs of overfitting, achieving high training accuracy but lower validation performance.

**Final Model Selection:**  
The Decision Tree was selected as the best-performing model based on validation accuracy and generalization behavior.

---

## Visualization
Visual comparisons were created to highlight:
- Training vs. validation accuracy across models
- Overfitting and generalization trends

These plots help illustrate the bias–variance tradeoff across different algorithms.

---

## Key Takeaways
- Validation performance is more important than training accuracy when selecting models.
- Simpler models can outperform more complex models on structured datasets.
- Hyperparameter tuning improves reliability but must be carefully controlled.
- Neural networks are not always the best choice for tabular classification tasks.

---

## Tools & Technologies
- **Python**
- **Libraries:** NumPy, Pandas, scikit-learn, PyTorch, Matplotlib
- **Techniques:** Supervised learning, feature engineering, hyperparameter tuning, model evaluation

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

