# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:42:10 2022

@author: Elena Eremeeva
"""
import pandas as pd
import matplotlib.pyplot as plt
#Импортируем метод классификации ближайшего соседа
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pickle #библиотека загрузки объектов в файл
from joblib import dump, load


#метод соседа
def neighbor (x_train, x_test, y_train, n):
    #classif - объект, который хранит модель
    classif = KNeighborsClassifier(n_neighbors = n)
    classif.fit(x_train, y_train)
    y_pred = classif.predict(x_test)
    #сериализация модели в файл (объект в словарь) и запись в файл с библиотекой pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump(classif, f)
    #сериализация модели в файл (объект в словарь) и запись в файл с библиотекой joblib
    dump(classif, 'neighbor.joblib')
    
    return y_pred 


def Bayes_bern (x_train, x_test, y_train):
    classif = BernoulliNB()
    #Обучение модели Наивного Байеса
    classif.fit(x_train, y_train)
    #Проверка модели на тестовой выборке
    y_pred = classif.predict(x_test)
    return  y_pred
    
 
def Bayes_gaus (x_train, x_test, y_train):
    classif = GaussianNB()
    #Обучение модели Наивного Байеса
    classif.fit(x_train, y_train)
    #Проверка модели на тестовой выборке
    y_pred = classif.predict(x_test)
    
    return  y_pred
    
def LogRegr (x_train, x_test, y_train):
    #Создание модели логистической регрессии
    classif = LogisticRegression(max_iter = 10000)
    #Обучение модели логистической регрессии
    classif.fit(x_train, y_train)
    #Проверка модели на тестовой выборке
    y_pred = classif.predict(x_test)
    
    return y_pred
#функция расчета метрик качества
def verif (y_test, y_pred):
    print(accuracy_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()

def des_tree (x_train, x_test, y_train):
    #построение дерева решений с указанием параметров. (критерий разбиения, минимальное количество примеров для разбиения узлов, образования листьев)
    classif = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 10, min_samples_leaf = 10, max_depth=6)
#обучение дерева решений   
    classif.fit(x_train, y_train)
    #расчет на тестовой выборке
    y_pred = classif.predict(x_test)
    #построение дерева решений визуализация
    fig = plt.figure(dpi = 300, figsize = (20,30))
    plot_tree(classif, filled=True, feature_names = x_train.columns)
    plt.show()
    
   
    return y_pred



def RandomForest (x_train, x_test, y_train):
    #построение случайного леса с использованием параметров: количество деревьев в лесу, минимальное количество примеров для разбиения узлов, образования листьев
    classif = RandomForestClassifier(n_estimators=5, random_state=1, min_samples_split = 6, min_samples_leaf = 6)
    #обучение случайного леса
    classif.fit(x_train, y_train)
    #расчет на тестовой выборке
    y_pred = classif.predict(x_test)

    return y_pred
# создание функции перевода не числовых данных в числовые
def transform (dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype != 'int64' and dataframe[column].dtype != 'float64':
            trans = LabelEncoder()
            trans.fit(dataframe[column])
            dataframe[column] = trans.transform(dataframe[column])
            print(dataframe[column].unique())
    return dataframe
def fill_nan (df, sex):
    df_1_class_m_clear = df[(df.Sex==sex) & (df.Age)>0]
    mean_1_m = df_1_class_m_clear['Age'].mean()
    df_1_class_m = df[(df.Sex==sex)]
    df_1_class_m = df_1_class_m.fillna({'Age':mean_1_m})
    print(df_1_class_m['Age'].isnull().sum())
    return (df_1_class_m)

df = pd.read_csv('train.csv')
print(df.info())

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace = True, axis = 1)
print(df.info())
df_1_class = df[df['Pclass'] == 1]
df_2_class = df[df['Pclass'] == 2]
df_3_class = df[df['Pclass'] == 3]
print(df_1_class['Age'].isnull().sum())
print(df_2_class['Age'].isnull().sum())
print(df_3_class['Age'].isnull().sum())

df_1_class_w = df_1_class[(df_1_class.Sex == 'female') & (pd.isna(df_1_class.Age))==False]
print(df_1_class_w['Age'].isnull().sum())

df_2_class_w = df_2_class[(df_2_class.Sex == 'female') & (pd.isna(df_2_class.Age))==False]
print(df_2_class_w['Age'].isnull().sum())


#Заполнение поля возраст у мужчин  и женщин 
df1 = fill_nan(df_1_class, 'male')
df2 = fill_nan(df_1_class, 'female')
df3 = fill_nan(df_2_class, 'male')
df4 = fill_nan(df_2_class, 'female')
df5 = fill_nan(df_3_class, 'male')
df6 = fill_nan(df_3_class, 'female')

train_data = pd.concat((df1, df2, df3, df4, df5, df6))
train_data.dropna(inplace=True)

print(train_data.info())

print(train_data['Survived'].value_counts())

test_data = pd.read_csv('test.csv')
print(test_data.info())
test_y = pd.read_csv('gender_submission.csv')

test_data = pd.concat((test_data, test_y['Survived']), axis = 1)

test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace = True, axis = 1)
print(test_data.info())
test_data.dropna(inplace=True)

df_1_class_test = test_data[test_data['Pclass'] == 1]
df_2_class_test = test_data[test_data['Pclass'] == 2]
df_3_class_test = test_data[test_data['Pclass'] == 3]

df1 = fill_nan(df_1_class_test, 'male')
df2 = fill_nan(df_1_class_test, 'female')
df3 = fill_nan(df_2_class_test, 'male')
df4 = fill_nan(df_2_class_test, 'female')
df5 = fill_nan(df_3_class_test, 'male')
df6 = fill_nan(df_3_class_test, 'female')

test_data = pd.concat((df1, df2, df3, df4, df5, df6))
test_data.dropna(inplace=True)
print(test_data.info())
train_data = transform(train_data)
test_data = transform(test_data)
print(train_data.info())
print(test_data.info())

x_train = train_data.drop(columns = 'Survived')
y_train = train_data['Survived']

x_test = test_data.drop(columns = 'Survived')
y_test = test_data['Survived']


y_pred_log = LogRegr (x_train, x_test, y_train)
y_pred_neighbor = neighbor (x_train, x_test, y_train, 5)
y_pred_bayes_bern = Bayes_bern (x_train, x_test, y_train)
y_pred_bayes_gaus = Bayes_gaus (x_train, x_test, y_train)
y_pred_dt = des_tree (x_train, x_test, y_train)
y_pred_rf = RandomForest (x_train, x_test, y_train)
print('Логистическая регрессия')
verif (y_test, y_pred_log)
print('Ближайший сосед')
verif (y_test, y_pred_neighbor)
print('Дерево решений')
verif (y_test, y_pred_dt)
print('Случайный лес')
verif (y_test, y_pred_rf)
print('Байес Бернули')
verif (y_test, y_pred_bayes_bern)
print('Байес Гаус')
verif (y_test, y_pred_bayes_gaus)










