import pytest
from unittest import mock
from app.train import load_data, preprocess, pipeline, train_model
from sklearn.preprocessing import StandardScaler

# Test data loading
def test_load_data():
    url = "https://fraud-proj-s3.s3.us-east-1.amazonaws.com/df.csv"
    df = load_data(url)
    assert not df.empty, "Dataframe is empty"

# Test data preprocessing
def test_preprocess_data():
    df = load_data("https://fraud-proj-s3.s3.us-east-1.amazonaws.com/df.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"

# Test pipeline creation
def test_create_pipeline():
    pipe = pipeline()
    assert any(isinstance(transformer[1], StandardScaler) for transformer in pipe.named_steps['preprocessor'].transformers), "Scaler missing in preprocessor"
    #assert "Random_Forest" in pipe.named_steps, "RandomForest missing in pipeline"

# Test model training (mocking GridSearchCV)
"""@mock.patch('app.train.GridSearchCV.fit', return_value=None)
def test_train_model(mock_fit):
    pipe = pipeline()
    X_train, X_test, y_train, y_test = pipeline(load_data("https://fraud-proj-s3.s3.us-east-1.amazonaws.com/df.csv"))
    param_grid = {"Random_Forest__n_estimators": [90]}
    model = train_model(pipe, X_train, y_train, param_grid)
    assert model is not None, "Model training failed"""
