import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from joblib import load


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os
import argparse


modelo = joblib.load("kmeans2.pkl")           # Ajusta si tu ruta es distinta
scaler = joblib.load("scaler2.pkl")
 


# Configurar el experimento en MLflow
experiment = mlflow.set_experiment("Kmeans_Experiment")



# Iniciar el registro del experimento
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Parámetros del modelo
    nClusters = 3
    random_state = 42

    # Realizar predicciones
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=nClusters, help="Número de clústeres a usar")
    parser.add_argument("--random_state", type=int, default=random_state, help="Semilla aleatoria")
    args = parser.parse_args()

    # ============= 2. CARGAR DATOS =============
    df = pd.read_csv("datos_clientes.csv")  # Tus variables: recency, frequency, monetary, payment_entropy
    X = df[["recency_z", "frequency_z", "monetary_z", "payment_entropy"]]
 
    # ============= 3. ESCALAR =============
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # ============= 4. ENTRENAR MODELO NUEVO =============
    modelo = KMeans(n_clusters=args.n_clusters, random_state=args.random_state)
    labels = modelo.fit_predict(X_scaled)
 
    
 

    # Registrar parámetros del modelo
    mlflow.log_param("numero de clusters", nClusters)
    mlflow.log_param("semilla", random_state)

    # Calcular métricas
    sil_score = silhouette_score(X_scaled, labels)

    # Registrar métricas
    mlflow.log_metric("Sillouette Score", sil_score)
    #mlflow.log_metric("r2_score", r2)

    # Crear ejemplo de entrada y firma del modelo
    #input_example = pd.DataFrame(X_train[:1], columns=db.feature_names)
    #signature = infer_signature(X_train, rf.predict(X_train))

    # Registrar el modelo con ejemplo de entrada y firma
    #mlflow.sklearn.log_model(
     #   rf,
      #  name="random-forest-model",
       # input_example=input_example,
        #signature=signature
    #)

    # Imprimir métricas
    print(df_nuevo)


 

 
