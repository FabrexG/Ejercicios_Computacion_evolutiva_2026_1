# ============================================================
# ⚙️ SCRIPT PARA TU AMIGO: ENTRENAMIENTO DEL SVM CON TU MODELO
# ============================================================
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("📂 Cargando los datos generados por Qwen...")

# 1. Cargar la matriz matemática (.npy) en la variable X
X = np.load("embeddings_qwen_humor_embeddings.npy")

# 2. Cargar las etiquetas reales (.csv) en la variable y
meta_df = pd.read_csv("embeddings_qwen_humor_meta.csv")
y = meta_df["label"].values

print(f"✅ Datos cargados correctamente.")
print(f"-> Matriz de características (X): {X.shape} (Tweets x Dimensiones)")
print(f"-> Vector de respuestas reales (y): {y.shape}\n")

# 3. Hacer el split clásico de entrenamiento y prueba (ej. 80% train, 20% test)
print("✂️ Dividiendo el dataset en Entrenamiento y Prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_split=0.2, random_state=42, stratify=y
)

# 4. Definir y entrenar el SVM
# Usamos un kernel lineal porque en altas dimensiones (como las de un LLM) suele funcionar mejor y más rápido
print("🧠 Entrenando la Máquina de Soporte Vectorial (SVM)...")
svm_model = SVC(kernel="linear", C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# 5. Evaluar los resultados finales
print("📊 Evaluando el rendimiento del modelo en el subset de prueba...")
y_pred = svm_model.predict(X_test)

# Mostrar métricas en pantalla (F1-Score, Precisión, Recall)
print("\n" + "="*50)
print("🏆 REPORTE DE CLASIFICACIÓN DEL SVM HÍBRIDO")
print("="*50)
print(classification_report(y_test, y_pred, target_names=["No Humor", "Humor"]))
print(f"Exactitud General (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("="50)