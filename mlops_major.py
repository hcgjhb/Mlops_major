import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
import mlflow
import mlflow.sklearn

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Class to process Iris data
class IrisDataProcessor:
    def __init__(self):
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.X = iris.data
        self.y = iris.target

    def prepare_data(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def get_feature_stats(self):
        return self.df.describe()

iris_data = IrisDataProcessor()
iris_data.prepare_data()

# Function to log model and metrics with MLflow
def log_model_with_mlflow(model, model_name):
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log metrics        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"{model_name} logged with accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1_score: {f1}")
        
    mlflow.end_run()

# Train/test split
X = df
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment and tracking URI
mlflow.set_experiment("MLflow_exp")
mlflow.set_tracking_uri("http://localhost:5000")

# Logistic Regression model
lr = LogisticRegression(max_iter=200)
log_model_with_mlflow(lr, "Logistic Regression")

# Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
log_model_with_mlflow(rf, "Random Forest")
