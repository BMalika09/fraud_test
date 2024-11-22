import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from key import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# Crée un jeu de données factice
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Démarrer une session MLflow
mlflow.set_tracking_uri("https://malika09-test.hf.space")  
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

experiment_name = "test-experiment"  # Exemple de nom d'expérience
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    # Entraîner un modèle simple
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Faire des prédictions et calculer l'accuracy
    predictions = model.predict(X_test) 
    accuracy = accuracy_score(y_test, predictions)

    # Loguer l'accuracy du modèle
    mlflow.log_metric("accuracy", accuracy)

    # Loguer le modèle
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model accuracy: {accuracy:.4f}")
