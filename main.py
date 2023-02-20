import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

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

# Mostrar las primeras filas del dataframe resultante
print(df.head())

# Generar un informe detallado de los datos
profile = ProfileReport(df, title="Breast Cancer Dataset", explorative=True)

# Guardar el informe en un archivo HTML
profile.to_file("breast_cancer_report1.html")

# Guardar el dataframe limpio en un archivo CSV
df.to_csv("preprocessed_data.csv", index=False)
