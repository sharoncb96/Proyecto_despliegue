import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
 
# ===============================
# CARGA DE DATOS
# ===============================
# Este archivo debe contener las columnas:
# - recency_z, frequency_z, monetary_z, payment_entropy
# - cluster (KMeans), cluster_agg_completo (Agglomerative + tópicos)
 
df = pd.read_csv("resultados_clustering.csv")  # Ajusta con tu archivo real
 
# ===============================
# INICIALIZACIÓN DE DASH
# ===============================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
 
# ===============================
# LAYOUT
# ===============================
app.layout = dbc.Container([
    html.H2("Comparación de Modelos de Clustering"),
    html.Hr(),
 
    dbc.Row([
        dbc.Col([
            html.Label("Selecciona modelo de clustering"),
            dcc.Dropdown(
                id="modelo-selector",
                options=[
                    {"label": "KMeans (sin tópicos)", "value": "cluster"},
                    {"label": "Agglomerative (con tópicos)", "value": "cluster_agg"},
                ],
                value="cluster"
            )
        ], width=6)
    ], className="mb-4"),
 
    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-pca"))
    ]),
 
    dbc.Row([
        dbc.Col(html.Div(id="tabla-resumen"))
    ])
])
 
# ===============================
# CALLBACKS
# ===============================
@app.callback(
    Output("grafico-pca", "figure"),
    Output("tabla-resumen", "children"),
    Input("modelo-selector", "value")
)
def actualizar_dashboard(cluster_col):
    # Asegurar que la columna exista
    if cluster_col not in df.columns:
        return {}, "Columna no encontrada"
 
    # Gráfico PCA (ya asumimos columnas pca_1 y pca_2 precalculadas)
    fig = px.scatter(df, x="pca_1", y="pca_2", color=df[cluster_col].astype(str),
                     title="Visualización PCA de Clústeres",
                     labels={"color": "Clúster"})
 
    # Tabla resumen
    resumen = df.groupby(cluster_col)[["recency_z", "frequency_z", "monetary_z", "payment_entropy"]].mean().round(2).reset_index()
    resumen_table = dbc.Table.from_dataframe(resumen, striped=True, bordered=True, hover=True)
 
    return fig, resumen_table
 
# ===============================
# EJECUCIÓN
# ===============================
if __name__ == '__main__':
    app.run_server(debug=True)
