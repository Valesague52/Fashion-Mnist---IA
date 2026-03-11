import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


st.title("Clasificación de prendas - Fashion MNIST")

st.write("Red neuronal MLP / DNN usando sklearn")

# =========================
# Cargar dataset
# =========================

@st.cache_data
def load_data():
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True)
    X = X / 255.0
    y = y.astype(int)
    return X, y

X, y = load_data()

# =========================
# División de datos
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Sidebar configuración
# =========================

st.sidebar.header("Configuración de la red")

activation = st.sidebar.selectbox(
    "Función de activación",
    ["relu", "tanh", "logistic"]
)

num_layers = st.sidebar.slider(
    "Número de capas ocultas",
    1,
    5,
    2
)

neurons = []

for i in range(num_layers):
    n = st.sidebar.slider(
        f"Neuronas capa {i+1}",
        32,
        512,
        128
    )
    neurons.append(n)

hidden_layers = tuple(neurons)

st.write("Arquitectura:", hidden_layers)

# =========================
# Entrenamiento
# =========================

if st.button("Entrenar red neuronal"):

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        max_iter=20,
        random_state=42
    )

    with st.spinner("Entrenando modelo..."):
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    st.subheader("Resultados")

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)

    # matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    st.pyplot(fig)
