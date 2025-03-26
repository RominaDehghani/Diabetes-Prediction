# Diabetes-Prediction

This is a machine learning project that predicts whether a person has diabetes or not using the PIMA Diabetes dataset. The project uses a Support Vector Machine (SVM) model with a linear kernel to classify the input data. The dataset consists of several medical attributes such as age, BMI, blood pressure, insulin levels, and more.

Table of Contents:
Project Overview
Dataset
Dependencies
How to Run
Model Evaluation
Predictive System
License

Project Overview:
This project is focused on building a predictive model that can identify whether a person is diabetic or not based on several medical features. The project follows these main steps:
Data Collection: Import and explore the PIMA Diabetes dataset.
Data Preprocessing: Standardize the data to improve model performance.
Model Training: Train an SVM classifier on the preprocessed data.
Model Evaluation: Evaluate the model’s performance using accuracy scores on training and test datasets.
Save and Load the Model: Save the trained model to a file and demonstrate loading it again for making predictions.
Predictive System: Build a system that allows predictions for new input data.

Dataset:
The PIMA Diabetes dataset is a collection of medical records used to predict the onset of diabetes based on various diagnostic measures. The dataset includes the following columns:
Pregnancies: Number of pregnancies
Glucose: Plasma glucose concentration
BloodPressure: Diastolic blood pressure
SkinThickness: Triceps skinfold thickness
Insulin: 2-hour serum insulin
BMI: Body mass index
DiabetesPedigreeFunction: Diabetes pedigree function (a measure of family history)
Age: Age of the patient
Outcome: Whether the patient has diabetes (1) or not (0)

Accessing the Dataset:
Dataset File link: https://www.dropbox.com/s/uh7o7uyeghq...

Dependencies:
The following libraries are required to run this project:
numpy: For handling arrays and numerical operations.
pandas: For data manipulation and analysis.
scikit-learn: For machine learning models, preprocessing, and evaluation.
joblib: For saving and loading models.

To install the necessary dependencies, you can use the following command:
pip install numpy pandas scikit-learn joblib

How to Run:
To run this project:
Clone the repository to your local machine or work directly in Google Colab.
Make sure the PIMA Diabetes dataset (diabetes.csv) is available in your project directory.
Run the provided code in a Python environment (Jupyter notebook or a Python script).

Model Evaluation:
The trained model is evaluated using accuracy scores:
Training Data Accuracy: This score tells us how well the model performs on the data it was trained on.
Test Data Accuracy: This score tells us how well the model performs on unseen data (i.e., test data).
The accuracy score for both the training and test data is printed in the output.

Predictive System:
The model is capable of making predictions based on new input data. To use the model for a new prediction, you can input the required medical attributes (as a tuple) and the model will output whether the person is diabetic or not.

License:
This project is open-source and available under the MIT License.
