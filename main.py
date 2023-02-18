import pandas as pd
from pandas_profiling import ProfileReport

# Leer los datos del archivo CSV
data = pd.read_csv("breast-cancer-wisconsin.data", header=None)

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
data.columns = column_names

# Generar un informe detallado de los datos
profile = ProfileReport(data, title="Breast Cancer Dataset", explorative=True)

# Guardar el informe en un archivo HTML
profile.to_file("breast_cancer_report.html")
