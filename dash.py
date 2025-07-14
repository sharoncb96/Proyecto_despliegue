import os
os.chdir('c:/Users/Felip/OneDrive - Universidad de los andes/MAESTRIA/Despliegue de Soluciones/Proyecto Ent 2/202502-DESPLIEGUE-PROYECTO/202502-DESPLIEGUE/modelo_guardado')
import pandas as pd
import joblib
 
# =======================
# CARGAR MODELO Y SCALER
# =======================
modelo = joblib.load("kmeans2.pkl")           # Ajusta si tu ruta es distinta
scaler = joblib.load("scaler2.pkl")
 
# =======================
# INGRESAR NUEVOS DATOS MANUALMENTE
# =======================
nuevos_clientes = [
    {"recency_z": 12, "frequency_z": 5, "monetary_z": 720.5, "payment_entropy": 0.85}
]
 
df_nuevo = pd.DataFrame(nuevos_clientes)
 
# =======================
# NORMALIZAR DATOS
# =======================
X = df_nuevo[["recency_z", "frequency_z", "monetary_z", "payment_entropy"]]
X_scaled = scaler.transform(X)
 
# =======================
# PREDICCIÓN DEL CLÚSTER
# =======================
df_nuevo["cluster_predicho"] = modelo.predict(X_scaled)
 
# =======================
# RESULTADO
# =======================
print(df_nuevo)
 
# OPCIONAL: guardar resultado
df_nuevo.to_csv("nuevos_segmentados.csv", index=False)