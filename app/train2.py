import pandas as pd
import numpy as np
import joblib
import mlflow
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import logging

"""from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,ConfusionMatrixDisplay,
)"""

#Load csv
def load_data(url):
    df = pd.read_csv(url, sep= ',', index_col="Unnamed: 0")
    df.drop(columns=['first', 'last','street', 'trans_date_trans_time',
                    'cc_num','dob', 'trans_num'], axis=1, inplace=True )

    return df

# Preprocess
def preprocess(df):
    target_variable = "is_fraud"

    X = df.drop(target_variable, axis = 1)
    y = df.loc[:,target_variable]

    return train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)

#Pipeline
def pipeline():
    numeric_features = ['amt', 'zip', 'lat', 'long',
                    'city_pop', 'unix_time',
                    'merch_lat', 'merch_long']

    categorical_features = ['merchant','category','gender',
                     'city', 'state', 'job']
    

    processor = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])),
        ('classifier', RandomForestClassifier())
    ])

    return processor

#train model
def train_model(pipe, X_train, y_train, param_grid, cv=3, n_jobs=-1, verbose=1):
    
    model = GridSearchCV(pipe, param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv, scoring="f1")
    model.fit(X_train, y_train)
    return model

# Log metrics and model to MLflow
def log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
    
    mlflow.log_metric("Train Score", model.score(X_train, y_train))
    mlflow.log_metric("Test Score", model.score(X_test, y_test))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )

def run_experiment(experiment_name, data_url, param_grid, artifact_path, registered_model_name):

    start_time = time.time()

    logging.info("Chargement des données...")
    df = load_data(data_url)

    logging.info("Données chargées. Début du prétraitement...")
    X_train, X_test, y_train, y_test = preprocess(df)

    # Create pipeline
    logging.info("Prétraitement terminé. Création du pipeline...")
    pipe = pipeline()

    # Set experiment's info 
    mlflow.set_experiment(experiment_name)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model
        model = train_model(pipe, X_train, y_train, param_grid)
        log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)
    # Print timing
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    # Define experiment parameters
    logging.basicConfig(level=logging.INFO)

    experiment_name = "hyperparameter_tuning"
    data_url = r"C:\Users\Malika\Desktop\JEDHA\jedha_formation\Lead\fraud_detection_project\project\df.csv"
    param_grid = {
        "classifier__n_estimators": list(range(90, 101, 10))
}
    artifact_path = "fraud_detection/rf_v1"
    registered_model_name = "rf_model"

    # Run the experiment
    run_experiment(experiment_name, data_url, param_grid, artifact_path, registered_model_name)







