import pandas as pd
import random


# подключение файла
data= pd.read_csv("./Data/labeled.csv", sep=",")

print("\n=====================================\n")
print( f"Всего комментов,колонок: {data.shape}")
data["toxic"]=data["toxic"].apply(int)
print(data["toxic"].value_counts())
print("\n=====================================\n")
# print("один из комментов")
# print(data[data["toxic"] == 0].iloc[0]['comment'])



#Разделение выбороk
random.seed(42)# Фиксируем seed для воспроизводимости
indices = list(data.index)
random.shuffle(indices)

test_data = data.loc[indices[:500]]
train_data = data.loc[indices[500:]]


print(f"Тренировочная выборка: {len(train_data)} примеров")
print(f"Тестовая выборка: {len(test_data)} примеров")
print("\nРаспределение в тестовой выборке:")
print(test_data["toxic"].value_counts())
print("\nРаспределение в тренировочной выборке:")
print(train_data["toxic"].value_counts())
print("\n=====================================\n")


