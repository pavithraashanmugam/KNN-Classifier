# KNN Classifier for Diabetes Prediction

This project implements a K-Nearest Neighbors (KNN) classifier to predict whether an individual has diabetes or not based on medical features such as age, blood pressure, BMI, insulin levels, etc.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Implementation](#implementation)
    - [KNN Classifier](#knn-classifier)
    - [Diabetes Prediction](#diabetes-prediction)
4. [Data](#data)
5. [How to Use](#how-to-use)
6. [Results](#results)

---

## Project Overview

The goal of this project is to implement a K-Nearest Neighbors (KNN) classifier using two different distance metrics: **Euclidean** and **Manhattan**. The classifier is then used to predict the likelihood of diabetes based on a set of medical features. The dataset used for this project is the **Pima Indians Diabetes Database**.

---

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `sklearn`
- `statistics`

You can install the required libraries using `pip`:

```bash
pip install numpy pandas scikit-learn
```

---

## Implementation

### KNN Classifier

The KNN classifier is implemented as a class `KNN_Classifier` with the following methods:

#### 1. `__init__(self, distance_metric)`
The constructor initializes the KNN classifier by setting the distance metric (either 'euclidean' or 'manhattan').

#### 2. `get_distance_metrics(self, training_data_point, testing_data_point)`
This function calculates the distance between a training data point and a testing data point based on the chosen distance metric.

- **Euclidean Distance**: The square root of the sum of squared differences between corresponding features of the two points.
- **Manhattan Distance**: The sum of absolute differences between corresponding features of the two points.

#### 3. `nearest_neighbor(self, X_train, test_data, k)`
This function identifies the `k` nearest neighbors of the given test data. It calculates the distance between the test data and all training data points, sorts them by distance, and returns the `k` closest neighbors.

#### 4. `predict(self, X_train, test_data, k)`
This function predicts the class of the test data by finding the most common label among the `k` nearest neighbors using the mode (most frequent value).

---

### Diabetes Prediction

The main script loads the **Pima Indians Diabetes Database** from a CSV file and splits it into training and test datasets. It uses the KNN classifier to predict the diabetes outcome based on the features.

1. **Loading the Dataset**: The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`. It contains 8 features and a target variable (Outcome).
  
2. **Preprocessing**: The features (X) and target (Y) are separated, and the data is split into training and test sets using `train_test_split()` from `sklearn`.

3. **Training the Model**: The KNN classifier is trained using the `X_train` and `Y_train` datasets, and predictions are made on the `X_test` dataset.

4. **Evaluating the Model**: The accuracy of the model is evaluated based on the predictions on the test set.

---

## Data

The dataset used in this project is the **Pima Indians Diabetes Dataset**. The dataset consists of the following features:

1. `Pregnancies`: Number of times pregnant
2. `Glucose`: Plasma glucose concentration
3. `BloodPressure`: Diastolic blood pressure
4. `SkinThickness`: Triceps skin fold thickness
5. `Insulin`: 2-hour serum insulin
6. `BMI`: Body mass index
7. `DiabetesPedigreeFunction`: Diabetes pedigree function
8. `Age`: Age of the person

The target variable is `Outcome`, where `1` indicates that the person has diabetes, and `0` indicates no diabetes.

You can download the dataset from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).

---

## How to Use

1. **Clone the Repository**: First, clone this repository to your local machine.

```bash
git clone https://github.com/yourusername/knn-diabetes-prediction.git
cd knn-diabetes-prediction
```

2. **Run the Script**: You can run the `knn_diabetes.py` script to load the dataset, train the KNN classifier, and make predictions.

```bash
python knn_diabetes.py
```

3. **View the Results**: The model will display the accuracy of the prediction based on the test data. Additionally, you can modify the number of neighbors (`k`) and the distance metric (Euclidean or Manhattan) to see how the model performs under different configurations.

---

## Results

The model uses the KNN algorithm to classify diabetes based on medical features. You can experiment with different values of `k` and distance metrics to observe changes in model accuracy.

For example, using `k=3` and the Euclidean distance metric, the model might give you the following accuracy:

```
Accuracy: 0.773
```

This means the classifier correctly predicted the diabetes status 77.3% of the time on the test data.

---


## Conclusion

This project demonstrates how to implement a K-Nearest Neighbors classifier using Python for predicting diabetes. The code offers a basic introduction to classification models and showcases the practical use of KNN for binary classification tasks. Feel free to experiment with different configurations and improve the model further!

---
