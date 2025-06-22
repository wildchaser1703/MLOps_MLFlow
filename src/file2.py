import pandas as pandas
import numpy as numpy
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='wildchaser1703', repo_name='MLOps_MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/wildchaser1703/MLOps_MLFlow.mlflow")

# Load the dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# Mention your experiment name below
mlflow.set_experiment("MLOps_Exp1")

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# params
max_depth = 10
n_estimators = 15

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, random_state = 42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot = True, cmap = "Blues", xticklabels = wine_data.target_names, yticklabels = wine_data)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author" : "wildchaser1703", "Project" : "Wine_Classification"})

    # log the model
    mlflow.sklearn.log_model(rf, "RandomForest")

    print(accuracy)
