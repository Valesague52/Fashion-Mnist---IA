import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.datasets import fashion_mnist


st.title("Clasificación de Prendas - Fashion MNIST")
st.write("Redes neuronales MLP / DNN usando sklearn")

# =============================
# Cargar dataset
# =============================

@st.cache_data
def load_data():

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # convertir 28x28 → 784
    X_train = X_train.reshape(-1,784)
    X_test = X_test.reshape(-1,784)

    # normalizar
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # reducir tamaño para que Streamlit no se caiga
    X_train = X_train[:8000]
    y_train = y_train[:8000]

    X_test = X_test[:2000]
    y_test = y_test[:2000]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()

# =============================
# Mostrar ejemplos
# =============================

st.subheader("Ejemplos del dataset")

fig, axes = plt.subplots(1,5, figsize=(10,3))

for i in range(5):
    axes[i].imshow(X_train[i].reshape(28,28), cmap="gray")
    axes[i].axis("off")

st.pyplot(fig)

# =============================
# Escalar datos
# =============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# Configuración de la red
# =============================

st.sidebar.header("Configuración de la Red")

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

# =============================
# Entrenar red
# =============================

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

    # =============================
    # Métricas
    # =============================

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    st.subheader("Resultados")

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)

    # =============================
    # Matriz de confusión
    # =============================

    st.subheader("Matriz de Confusión")

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
