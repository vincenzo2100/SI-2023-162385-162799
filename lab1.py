import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("diabetes.txt", sep=" ", header=None)
info = pd.read_csv("_info-data-discrete.txt", sep=" ", header=None)
attribute_list = ["a1n", "a2n", "a3n", "a4n", "a5n", "a6n", "a7n", "a8n", "a9d"]
print(f"Symbole klas decyzyjnych: {pd.unique(data[8])}")
print(f"Wielkoś klas decyzyjnych (liczby obiektów w klasach): {info[2][9]}")
print("Minimalne wartość atrybutów: ")
for col in data:
    print(f"{attribute_list[col]}: {data[col].min()}")
print("Maksymalna wartość atrybutów: ")
for col in data:
    print(f"{attribute_list[col]}: {data[col].max()}")
print("Liczba różnych dostępnych wartości dla każdego atrybutu:")
for col in data:
    print(f"{attribute_list[col]}: {len(data[col].unique())}")
print("Lista różnych dostępnych wartości dla każdego atrybutu: ")
for col in data:
    print(f"{attribute_list[col]}: {(data[col].unique())}")
print("Odchylenie standardowe dla poszczególnych atrybutów w całym systemie: ")
for col in data:
    print(f"{attribute_list[col]}: {(data[col].std())}")

num_missing = (data[[0, 1, 2, 3, 4, 5, 6, 7]] == 0).sum()
print(num_missing)

data[[0, 1, 2, 3, 4, 5, 6, 7]] = data[[0, 1, 2, 3, 4, 5, 6, 7]].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print(data_normalized)

scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print(data_normalized)

scaler = MinMaxScaler(feature_range=(-10, 10))
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print(data_normalized)

for col in data:
    data[col] = (data[col] - data[col].mean()) / (data[col].var())
print(data)

churn = pd.read_csv("Churn_Modelling.csv", sep=",")
print(churn)
churn_with_dummies = pd.get_dummies(data=churn, columns=["Geography"])
print(churn_with_dummies)
churn_with_dummies = churn_with_dummies.drop("Geography_France", axis=1)
print(churn_with_dummies)
