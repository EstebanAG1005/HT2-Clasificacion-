import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ydata_profiling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Leer los datos del archivo CSV
df = pd.read_csv("breast-cancer-wisconsin.data", header=None)

# Agregar nombres a las columnas
column_names = [
    "Sample_code_number",
    "Clump_Thickness",
    "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape",
    "Marginal_Adhesion",
    "Single_Epithelial_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
    "Class",
]
df.columns = column_names

# Eliminar la columna Sample_code_number
df = df.drop(columns=["Sample_code_number"])

# Reemplazar los valores faltantes en la columna Bare_Nuclei por NaN
df = df.replace("?", np.nan)

# Convertir la variable Class a numérica (0 para muestras benignas y 1 para muestras malignas)
df["Class"] = df["Class"].replace({2: 0, 4: 1})

# Eliminar las filas con valores faltantes
df = df.dropna()

# Convertir todas las columnas excepto Class a tipo numérico
df = df.apply(pd.to_numeric, errors="coerce")

# Eliminar las filas con valores negativos
df = df[df >= 0].dropna()

# # Mostrar las primeras filas del dataframe resultante
# print(df.head())

# # Generar un informe detallado de los datos
# profile = ProfileReport(df, title="Breast Cancer Dataset", explorative=True)

# # Guardar el informe en un archivo HTML
# profile.to_file("breast_cancer_report1.html")

# Guardar el dataframe limpio en un archivo CSV
df.to_csv("preprocessed_data.csv", index=False)

# Parte 2
# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajustar un modelo de Regresión Logística
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluar el modelo de Regresión Logística
print("Regresión Logística:\n")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("accuracy_score Regresion Logistica:", accuracy_score(y_test, y_pred_lr))

# # Ajustar un modelo de K-NN
# knn = KNeighborsClassifier(n_neighbors=1)
# # knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred_knn = knn.predict(X_test)


# Seleccion de un valor de K
tasa_error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    tasa_error.append(np.mean(y_pred_knn != y_test))


plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 40),
    tasa_error,
    color="blue",
    linestyle="dashed",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
plt.title("Tasa de Error vs. Valor de K")
plt.xlabel("K")
plt.ylabel("Tasa de Error")
plt.show()

# PRIMERO UNA RAPIDA COMPARACION CON NUESTRO VALOR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# Evaluar el modelo de K-NN
print("K-NN:\n")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("accuracy_score K-NN:", accuracy_score(y_test, y_pred_knn))