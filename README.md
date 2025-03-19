# Logistic Regression for Binary Classification - Titanic Dataset

## Overview
This project demonstrates the implementation of a Logistic Regression model to predict passenger survival on the Titanic. The goal is to classify whether a passenger survived (1) or did not survive (0) based on features such as age, gender, ticket class, and more. This project covers the entire data science workflow, including data preprocessing, model training, evaluation, and visualization.

## Dataset
The dataset used in this project is the **Titanic Dataset**, which contains information about passengers aboard the Titanic, including their survival status, age, gender, ticket class, and other relevant features. The dataset is publicly available and is commonly used for binary classification tasks.

### Features
- **Survived**: Target variable (0 = Did not survive, 1 = Survived)
- **Pclass**: Ticket class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- **Sex**: Gender of the passenger (male or female)
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Passenger fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Steps
1. **Data Preprocessing**:
   - Removed unnecessary columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
   - Handled missing values in the `Age` column by filling them with the median age.
   - Filled missing values in the `Embarked` column with the most frequent value.
   - Converted categorical variables (`Sex` and `Embarked`) into numerical values using Label Encoding.

2. **Train-Test Split**:
   - Split the dataset into training (80%) and testing (20%) sets to evaluate the model's performance.

3. **Feature Scaling**:
   - Standardized the features using `StandardScaler` to ensure all features are on the same scale, which is important for logistic regression.

4. **Model Training**:
   - Trained a Logistic Regression model using the training data.

5. **Model Evaluation**:
   - Made predictions on the test set and evaluated the model's performance using:
     - **Accuracy**: The percentage of correctly predicted outcomes.
     - **Confusion Matrix**: A table showing true vs. predicted labels.
     - **Classification Report**: Precision, recall, F1-score, and support for each class.
     - **ROC-AUC Curve**: A plot showing the trade-off between true positive rate and false positive rate.

6. **Visualizations**:
   - Plotted a heatmap of the confusion matrix to visualize the model's performance.
   - Created an ROC curve to evaluate the model's ability to distinguish between the two classes.

7. **Feature Importance**:
   - Analyzed the coefficients of the logistic regression model to understand the impact of each feature on the prediction.

## Results
- **Accuracy**: The model achieved an accuracy of approximately 81% on the test set.
- **Confusion Matrix**: The heatmap shows the number of true positives, true negatives, false positives, and false negatives.
- **ROC-AUC Score**: The area under the ROC curve (AUC) is 0.86, indicating good performance in distinguishing between survivors and non-survivors.
- **Feature Importance**: The coefficients of the logistic regression model show that features like `Sex` and `Pclass` have a significant impact on survival predictions.

## How to Run the Code
1. Open the provided Google Colab notebook.
2. Run each cell sequentially to load the dataset, preprocess the data, train the model, and evaluate its performance.
3. The visualizations and evaluation metrics will be displayed automatically.

## Dependencies
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Files
- `logistic_regression_titanic.ipynb`: The Jupyter notebook containing the code and explanations.
- `README.md`: This file, providing an overview of the project.

## Conclusion
This project demonstrates the application of Logistic Regression for binary classification using the Titanic dataset. It covers essential data science concepts such as data preprocessing, model training, evaluation, and visualization. The results show that the model performs well in predicting passenger survival, making it a useful example for understanding logistic regression and its applications in real-world scenarios.

## References
- Titanic Dataset: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
- Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
- Google Colab file for code: https://colab.research.google.com/drive/1FF0uNfGnytGbPYHOIL3cLx0LD90TOK7V?usp=sharing
