# Telco Customer Churn Prediction

## Overview
This project aims to predict customer churn for a telecom company using machine learning models. The dataset used is the **Telco Customer Churn** dataset from Kaggle. The models implemented include **Random Forest** and **XGBoost**, and performance is evaluated using accuracy, classification reports, and confusion matrices.

## Features and Implementation
- **Data Preprocessing:**
  - Handling missing values
  - Encoding categorical variables
  - Standardizing numerical features
- **Machine Learning Models:**
  - Random Forest Classifier
  - XGBoost Classifier
- **Evaluation Metrics:**
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
  - Feature Importance Analysis

## Installation
To run this project locally, follow these steps:

### Prerequisites
Ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Clone the Repository
```bash
git clone https://github.com/Adithya-5369/CustomerChurnModel.git
cd TelcoChurnPrediction
```

### Running the Project
1. Ensure the **Telco Customer Churn** dataset is placed in the `data/` directory.
2. Run the Python script to preprocess the data, train models, and evaluate results.

```bash
python telco_churn_prediction.py
```

## Dataset
The dataset consists of customer details, contract information, and service usage features. The target variable is **Churn**, indicating whether a customer leaves the service.

## Results
- Model performances are evaluated using confusion matrices and feature importance plots.
- The best-performing model can be fine-tuned further for better results.

## Visualizations
The project provides visualizations such as:
- Confusion matrices for model evaluation
- Feature importance charts for interpretability

## Project Structure
```
TelcoChurnPrediction/
│── data/                         # Directory containing dataset files
│   └── Telco-Customer-Churn.csv   # Dataset file
│── telco_churn_prediction.py     # Main script
│── README.md                     # Project documentation
```

## Future Enhancements
- Implement additional models like Logistic Regression and Neural Networks.
- Hyperparameter tuning for better model accuracy.
- Deploy as a web application using Flask or Streamlit.

## License
This project is open-source and available under the MIT License.

## Author
[Adithya Sai Srinivas](https://github.com/Adithya-5369)

Feel free to fork, contribute, and improve this project!
