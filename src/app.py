from utils import db_connect
engine = db_connect()

# your code here

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
# Cargar los datos desde la URL
url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
datos = pd.read_csv(url, sep=";")

# Mostrar las primeras filas del DataFrame
datos.head()

data_limpio = datos.drop_duplicates()


num_cols = data_limpio.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data_limpio.select_dtypes(include=['object']).columns

import matplotlib.pyplot as plt
import seaborn as sns
import math

# Lista de columnas a graficar
columns = cat_cols

# Fijar el número de columnas a 3
num_cols = 3

# Calcular el número de filas necesarias
num_rows = math.ceil(len(columns) / num_cols)

# Crear la figura con subgráficos (axes) y un tamaño más grande
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))  # Ajusta el tamaño de la figura

# Asegurarse de que 'axes' sea una lista bidimensional
axes = axes.reshape(num_rows, num_cols) if num_rows > 1 else [axes]

# Generar los gráficos
col_index = 0
for row in range(num_rows):
    for col in range(num_cols):
        if col_index < len(columns):
            sns.countplot(data=datos, x="y", hue=columns[col_index], palette="coolwarm", ax=axes[row][col])
            axes[row][col].set_title(f"Count Plot of y with hue {columns[col_index]}")
            axes[row][col].tick_params(axis='x', rotation=45)  # Rotar las etiquetas del eje x para que queden dentro
            col_index += 1
        else:
            fig.delaxes(axes[row][col])  # Eliminar gráficos vacíos si hay menos de 3 columnas

# Ajustar el espaciado entre los gráficos
plt.tight_layout()  # Ajuste automático del espacio para evitar que se solapen

# Mostrar la figura
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import math

# Identificar columnas numéricas
num_cols = data_limpio.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Convertir a lista

# Fijar el número de columnas por fila a 3
cols_per_row = 3

# Calcular el número de filas necesarias
num_rows = math.ceil(len(num_cols) / cols_per_row)

# Crear la figura con subgráficos (axes) y un tamaño más grande
fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))

# Asegurarse de que 'axes' sea una lista bidimensional
axes = axes.reshape(num_rows, cols_per_row) if num_rows > 1 else [axes]

# Generar los gráficos
col_index = 0
for row in range(num_rows):
    for col in range(cols_per_row):
        if col_index < len(num_cols):
            # Histograma para cada columna numérica
            sns.histplot(data=data_limpio, x=num_cols[col_index], kde=True, ax=axes[row][col], color='purple')
            axes[row][col].set_title(f"Histogram of {num_cols[col_index]}")
            axes[row][col].set_xlabel(num_cols[col_index])  # Etiqueta el eje x con el nombre de la columna
            col_index += 1
        else:
            fig.delaxes(axes[row][col])  # Eliminar gráficos vacíos si hay menos de 3 columnas

# Ajustar el espaciado entre los gráficos
plt.tight_layout()  # Ajuste automático del espacio para evitar que se solapen

# Mostrar la figura
plt.show()

target = ["y"]
norm_target = "y_n"

# Asegúrate de que no haya valores nulos en "y"
data_limpio.loc[:, "y"] = data_limpio["y"].fillna("Desconocido")

# Codificar la variable "y" y asignar el resultado a la nueva columna "y_n"
data_limpio.loc[:, norm_target] = pd.factorize(data_limpio["y"])[0]

# Crear el diccionario de reglas de codificación
target_rules = {row["y"]: row["y_n"] for _, row in data_limpio[["y", "y_n"]].drop_duplicates().iterrows()}

# Guardar las reglas en un archivo JSON
import json
with open("../models/target_rules.json", "w") as f:
    json.dump(target_rules, f)

# Ver las primeras filas de la columna "y_n"
print(data_limpio[norm_target].head())

def graficar_heatmap_y_dispersion(df, columnas, target_col):
   
    # 1. Heatmap de correlación
    plt.figure(figsize=(8, 6))  # Tamaño de la figura
    sns.heatmap(df[columnas + [target_col]].corr(), annot=True,fmt=".2f", cmap="coolwarm", cbar=True,linewidths=0.5)

    plt.title("Heatmap de Correlación", fontsize=14)
    plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejor legibilidad
    plt.tight_layout()  # Ajustar el layout
    plt.show()

    # 2. Gráfico de dispersión (scatter plot)
    for col in columnas:
        plt.figure(figsize=(6, 4))  # Tamaño de la figura
        sns.scatterplot(data=data_limpio, x=col,y=target_col, alpha=0.5, color="blue")
        
        plt.title(f'Dispersión: {col} vs {target_col}', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel(target_col, fontsize=10)
        plt.tight_layout()  # Ajustar el layout
        plt.show()

graficar_dispersiones_y_heatmaps(data_limpio, num_cols[0:5], norm_target)
graficar_dispersiones_y_heatmaps(data_limpio, num_cols[5:10], norm_target)

def graficar_countplots(df, cat_cols, cols_per_row=3):

    # Calcular el número de filas necesarias
    n = len(cat_cols)
    num_rows = math.ceil(n / cols_per_row)

    # Crear la figura con subgráficos (axes) y un tamaño más grande
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))

    # Aplanar el array de ejes para facilitar la iteración
    axes = axes.flatten() if num_rows > 1 else [axes]

    # Generar los countplots
    for i, col in enumerate(cat_cols):
        if i < len(axes):  # Verificar que el índice esté dentro del rango de ejes
            sns.countplot(data=df, x=col, hue=col, ax=axes[i], palette="vlag", legend=False)
            axes[i].set_title(f'Countplot de {col}', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)  # Rotar etiquetas del eje x
        else:
            break  # Salir del bucle si no hay más ejes disponibles

    # Ocultar ejes vacíos si hay menos columnas que cols_per_row * num_rows
    for j in range(n, cols_per_row * num_rows):
        if j < len(axes):  # Verificar que el índice esté dentro del rango de ejes
            fig.delaxes(axes[j])

    # Ajustar el layout para evitar que los gráficos se sobrepongan
    plt.tight_layout()
    plt.show()

# Graficar countplots
graficar_countplots(datos, cat_cols, cols_per_row=3)

import pandas as pd
import json

def automatizar_factorizacion(df, columnas):
    transformacion_reglas = {}
    col_factorizadas = []
    
    # Iterar sobre las columnas especificadas para la factorización
    for col in columnas:
        # Aplicar la factorización y agregar la columna al DataFrame de forma segura usando .loc
        df.loc[:, f"{col}_n"] = pd.factorize(df[col])[0]
        
        # Agregar el nombre de la columna factorizada a la lista
        col_factorizadas.append(f"{col}_n")
        
        # Guardar las reglas de transformación para la columna actual en el diccionario
        transformacion_reglas[col] = {row[col]: row[f"{col}_n"] for _, row in df[[col, f"{col}_n"]].drop_duplicates().iterrows()}
    
    # Guardar el diccionario de reglas en un archivo JSON
    with open("../models/transformacion_reglas.json", "w") as f:
        json.dump(transformacion_reglas, f)
    
    print("Factorización completada y reglas guardadas en 'transformacion_reglas.json'.")
    
    # Retornar el DataFrame actualizado con las nuevas columnas numéricas y la lista de columnas factorizadas
    return df, col_factorizadas

# Asegúrate de que 'cat_cols' contenga las columnas categóricas que deseas factorizar
cat_cols = data_limpio.select_dtypes(include=['object']).columns

# Llamar a la función para automatizar la factorización
data_limpio, col_factorizadas = automatizar_factorizacion(data_limpio, cat_cols)

# Imprimir las columnas que han sido factorizadas
print("Columnas factorizadas:", col_factorizadas)

# Para revisar el archivo JSON guardado, puedes cargarlo y mostrarlo de forma legible:
with open("../models/transformacion_reglas.json", "r") as f:
    reglas = json.load(f)
    print(json.dumps(reglas, indent=4))  # Imprimir el archivo JSON de forma legible


n_data = data_limpio.drop(columns=cat_cols + target, errors='ignore')
n_data

n_data = data_limpio.select_dtypes(exclude=['object'])

def generar_boxplots_automaticos(df):
   
    # Filtrar las columnas numéricas del DataFrame, como ya esta normalizado, son todas. 
    columnas = df.columns

    n_columnas = 4
    n_filas = (len(columnas) + n_columnas - 1) // n_columnas  # Redondeo hacia arriba

    # Crear la figura y los ejes para los subgráficos
    fig, axis = plt.subplots(n_filas, n_columnas, figsize=(n_columnas * 5, n_filas * 5))

    # Aplanar el array de ejes para facilitar la iteración
    axis = axis.flatten()

    # Graficar un boxplot para cada columna del DataFrame
    for i, col in enumerate(columnas):
        sns.boxplot(ax=axis[i], data=df, y=col)
        axis[i].set_title(f'Boxplot de {col}')

    # Si hay menos gráficos que subgráficos, ocultar los ejes restantes
    for j in range(i + 1, len(axis)):
        axis[j].axis('off')

    # Ajustar el layout para evitar que los gráficos se sobrepongan
    plt.tight_layout()
    plt.show()

generar_boxplots_automaticos(n_data)

import os 

total_data_con_outliers = n_data.copy() 
total_data_sin_outliers = n_data.copy()

col_con_outliers = ["age", "duration", "campaign", "pdays", "job_n", "poutcome_n", "loan_n", "default_n", "previous", "month_n"] # añadir outliers

def replace_outliers_from_column(column, df):
  column_stats = df[column].describe()
  column_iqr = column_stats["75%"] - column_stats["25%"]
  upper_limit = column_stats["75%"] + 1.5 * column_iqr
  lower_limit = column_stats["25%"] - 1.5 * column_iqr
  if lower_limit < 0: lower_limit = min(df[column])
  # Remove upper outliers
  df[column] = df[column].apply(lambda x: x if (x <= upper_limit) else upper_limit)
  # Remove lower outliers
  df[column] = df[column].apply(lambda x: x if (x >= lower_limit) else lower_limit)
  return df.copy(), [lower_limit, upper_limit]

outliers_dict = {}
for column in col_con_outliers:
  total_data_sin_outliers, limits_list = replace_outliers_from_column(column, total_data_sin_outliers)
  outliers_dict[column] = limits_list

with open("../models/outliers_replacement.json", "w") as f:
  json.dump(outliers_dict, f)

outliers_dict

from sklearn.preprocessing import MinMaxScaler

def normalize_and_create(data, columns_to_normalize, new_column_name='econ_index'):

    # Crear un objeto MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizar las columnas especificadas
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    # Crear la nueva variable 'risk' como la media de las columnas normalizadas
    data[new_column_name] = data[columns_to_normalize].mean(axis=1)

    return data

# Definir las columnas que se van a normalizar
columns_to_normalize_econ_index = ['emp.var.rate', 'euribor3m', 'nr.employed']

# Normalizar y crear la variable 'risk' para el conjunto con outliers
total_data_con_outliers = normalize_and_create(total_data_con_outliers, columns_to_normalize_econ_index)

# Normalizar y crear la variable 'risk' para el conjunto sin outliers
total_data_sin_outliers = normalize_and_create(total_data_sin_outliers, columns_to_normalize_econ_index)

total_data_con_outliers.drop(columns_to_normalize_econ_index, axis= 1, inplace= True)
total_data_sin_outliers.drop(columns_to_normalize_econ_index, axis= 1, inplace= True)

total_data_sin_outliers

from sklearn.model_selection import train_test_split



num_variables = total_data_con_outliers.copy().drop("y_n", axis= 1).columns.tolist()

# Dividimos el conjunto de datos en muestras de train y test
X_con_outliers = total_data_con_outliers.drop("y_n", axis = 1)[num_variables]
X_sin_outliers = total_data_sin_outliers.drop("y_n", axis = 1)[num_variables]
y = total_data_con_outliers["y_n"]

X_train_con_outliers, X_test_con_outliers, y_train, y_test = train_test_split(X_con_outliers, y, test_size = 0.2, random_state = 42)
X_train_sin_outliers, X_test_sin_outliers = train_test_split(X_sin_outliers, test_size = 0.2, random_state = 42)

# GUARDAR LOS DATASETS
X_train_con_outliers.to_excel("../data/processed/X_train_con_outliers.xlsx", index = False)
X_train_sin_outliers.to_excel("../data/processed/X_train_sin_outliers.xlsx", index = False)
X_test_con_outliers.to_excel("../data/processed/X_test_con_outliers.xlsx", index = False)
X_test_sin_outliers.to_excel("../data/processed/X_test_sin_outliers.xlsx", index = False)
y_train.to_excel("../data/processed/y_train.xlsx", index = False)
y_test.to_excel("../data/processed/y_test.xlsx", index = False)

X_train_con_outliers.head()