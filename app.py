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
st.write("Redes neuronales MLP / DNN usando sklearn")

# =========================
# Cargar dataset
# =========================

@st.cache_data
def load_data():

    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True)

    # reducir tamaño para evitar caídas
    X = X[:10000]
    y = y[:10000]

    X = X / 255.0
    y = y.astype(int)

    return X, y

X, y = load_data()

# =========================
# Mostrar ejemplos
# =========================

st.subheader("Ejemplos del dataset")

fig, axes = plt.subplots(1,5, figsize=(10,3))

for i in range(5):
    axes[i].imshow(X.iloc[i].values.reshape(28,28), cmap="gray")
    axes[i].axis("off")

st.pyplot(fig)

# =========================
# División datos
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Configuración red
# =========================

st.sidebar.header("Configuración de la red")

activation = st.sidebar.selectbox(
    "Función de activación",
    ["relu","tanh","logistic"]
)

num_layers = st.sidebar.slider(
    "Número de capas ocultas",
    1,
    4,
    2
)

neurons = []

for i in range(num_layers):

    n = st.sidebar.slider(
        f"Neuronas capa {i+1}",
        32,
        256,
        128
    )

    neurons.append(n)

hidden_layers = tuple(neurons)

st.write("Arquitectura seleccionada:", hidden_layers)

# =========================
# Entrenar modelo
# =========================

if st.button("Entrenar red neuronal"):

    with st.spinner("Entrenando modelo..."):

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver="adam",
            max_iter=10,
            early_stopping=True,
            random_state=42
        )

        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # =========================
    # Métricas
    # =========================

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    st.subheader("Resultados del modelo")

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)

    # =========================
    # Matriz de confusión
    # =========================

    st.subheader("Matriz de confusión")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    st.pyplot(fig)
