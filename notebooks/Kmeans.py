import pickle
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from joblib import load

objeto = load('kmeans.pkl')


# Configurar el experimento en MLflow
experiment = mlflow.set_experiment("Kmeans_Experiment")

################# por ajustar #################
# Usarlo (por ejemplo, si es un modelo)




# Iniciar el registro del experimento
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Parámetros del modelo
    n_estimators = 200 
    max_depth = 6
    max_features = 3

    # Realizar predicciones
    predictions = objeto.predict(X)  # X puede ser tus datos de entrada
    print(predictions)

    # Registrar parámetros del modelo
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_feat", max_features)

    # Calcular métricas
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Registrar métricas
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

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
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")