# Healthcare Stroke Prediction using Enssemble Learning

This project aims to predict the occurrence of strokes based on various health-related features using ensemble learning techniques. The dataset used contains information about patients' demographics, health conditions, and lifestyle choices.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build and evaluate machine learning models to predict whether a patient is likely to have a stroke, with a particular emphasis on ensemble learning techniques. Ensemble learning methods combine the predictions of multiple machine learning algorithms to improve the performance and robustness of the prediction.

## Dataset
The dataset used in this project is the "Healthcare Dataset Stroke Data," which contains the following columns:
- `id`: Unique identifier for each patient
- `gender`: Gender of the patient
- `age`: Age of the patient
- `hypertension`: Whether the patient has hypertension
- `heart_disease`: Whether the patient has heart disease
- `ever_married`: Marital status of the patient
- `work_type`: Type of work the patient does
- `Residence_type`: Whether the patient lives in an urban or rural area
- `avg_glucose_level`: Average glucose level in the blood
- `bmi`: Body Mass Index
- `smoking_status`: Smoking status of the patient
- `stroke`: Whether the patient had a stroke (target variable)

## Data Preprocessing
- Handled missing values in the `bmi` column by replacing them with the mode of the column.
- Encoded categorical variables such as `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status` to numerical values.
- Split the data into training and testing sets.
- Scaled the features using MinMaxScaler.

## Exploratory Data Analysis
Visualizations were created to understand the distribution of the data and relationships between different features and the target variable (stroke). Some of the visualizations include:
- Count plots of gender, hypertension, heart disease, marital status, work type, and smoking status against the stroke occurrence.
- Histograms and line plots for continuous features like age, BMI, and average glucose level.

## Model Building
The primary focus of this project is on ensemble learning. Various ensemble methods were trained and evaluated, including:
- Bagging Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Voting Classifier

For comparison, individual models were also implemented:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The models were evaluated based on accuracy, precision, recall, and F1-score.

## Results
The results from different models were compared to find the best performing model. Below are the accuracy scores for some models:

- Logistic Regression: 93.93%
- Decision Tree: 90.91%
- Random Forest: 93.84%
- Support Vector Machine (SVM): 93.93%
- K-Nearest Neighbors (KNN): 93.73%
- Bagging Classifier: 93.84%
- Gradient Boosting Classifier: 93.84%
- AdaBoost Classifier: 93.93%

Note: Precision, recall, and F1-score were particularly low for the minority class (stroke cases) due to class imbalance.

## Installation
To run this project locally, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:
```bash
pip install numpy pandas seaborn scikit-learn matplotlib
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/healthcare-stroke-prediction.git
```
2. Navigate to the project directory:
```bash
cd healthcare-stroke-prediction
```
3. Run the Jupyter notebook or Python script to preprocess the data, build and evaluate the models.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.
