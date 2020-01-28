# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('Kyoto_Restaurant_info.csv')
dataset = pd.read_csv('Kyoto_Restaurant_info.csv')

df.drop(df.columns[[0,1,2,12,13]], axis=1,inplace=True)

df.fillna(value={'LunchPrice':0, 'LunchRating':0}, inplace=True)

df.replace({' ～￥999': 1,
            '￥1000～￥1999':2,
            '￥2000～￥2999':3,
            '￥3000～￥3999':4,
            '￥4000～￥4999':5,
            '￥5000～￥5999':6,
            '￥6000～￥6999':7,
            '￥6000～￥7999':8,
            '￥7000～￥7999':8,
            '￥8000～￥8999':9,
            '￥8000～￥9999':10,
            '￥9000～￥9999':10,
            '￥10000～￥14999':15,
            '￥15000～￥19999':20,
            '￥20000～￥29999':30,
            '￥30000～':40},inplace=True)

#
df.loc[df['LunchRating'] < 3.2, 'LunchRating'] = 1
df.loc[df['LunchRating'] >= 3.2, 'LunchRating'] = 0
df['LunchRating'] = df['LunchRating'].astype(int)

df.loc[df['DinnerRating'] < 3.2, 'DinnerRating'] = 1
df.loc[df['DinnerRating'] >= 3.2, 'DinnerRating'] = 0
df['DinnerRating'] = df['DinnerRating'].astype(int)

df.loc[df['TotalRating'] < 3.2, 'TotalRating'] = 1
df.loc[df['TotalRating'] >= 3.2, 'TotalRating'] = 0
df['TotalRating'] = df['TotalRating'].astype(int)

# %%
df_t = df
df_t = pd.get_dummies(df_t, drop_first=True, columns=['Station', 'FirstCategory', 'SecondCategory'])
x_t = df_t.drop(['TotalRating','LunchRating','DinnerRating'], axis=1).values
y_t = df_t.iloc[:, 2].values

# %%
from sklearn.model_selection import train_test_split
x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(x_t, y_t, test_size = 0.2)

# %%
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')

t_classifiers = [dtc, rfc]
# %%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

t_index = ['Decision Tree', 'Random Forest']
score = ['Accuracy','Precision', 'Recall', 'F-score']

t_data_list = []

for classifier in t_classifiers:

    classifier.fit(x_t_train, y_t_train)
    y_pred = classifier.predict(x_t_test)

    cm = confusion_matrix(y_t_test, y_pred)
    print(cm)
    print('')
    accuracy = accuracy_score(y_t_test,y_pred)
    precision = precision_score(y_t_test,y_pred)
    recall = recall_score(y_t_test,y_pred)
    f1 = f1_score(y_t_test,y_pred)

    data = [round(accuracy, 2), round(precision, 2), round(recall,2), round(f1,2)]
    t_data_list.append(data)

df_t_data = pd.DataFrame(data=t_data_list,index=t_index, columns=score)
print(df_t_data)

# %%
df_station = df[df['Station'] == 'Kyoto']
x_station = df_station.loc[:, ['FirstCategory', 'DinnerPrice']]
x_station = pd.get_dummies(x_station, drop_first=True, columns=['FirstCategory'])
x_station = x_station.values
y = df_station.iloc[:, -3].values

# %%
x_station_train, x_station_test, y_train, y_test = train_test_split(x_station, y, test_size = 0.2)

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', gamma='scale')

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

classifiers = [knn, svc, nb, dtc, rfc]

# %%
index = ['K Naighnors', 'kernel SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest']
score = ['Accuracy','Precision', 'Recall', 'F-score']

s_data_list = []

for classifier in classifiers:

    classifier.fit(x_station_train, y_train)
    y_pred = classifier.predict(x_station_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('')
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    data = [round(accuracy, 2), round(precision, 2), round(recall,2), round(f1,2)]
    s_data_list.append(data)

df_s_data = pd.DataFrame(data=s_data_list,index=index, columns=score)
print(df_s_data)

# %%
df_category = df[(df['FirstCategory'] == 'Izakaya (Tavern)') | (df['SecondCategory'] == 'Izakaya (Tavern)')]
x_category = df_category.loc[:, ['Station', 'DinnerPrice']]
print(x_category)
x_category = pd.get_dummies(x_category, drop_first=True, columns=['Station'])
x_category = x_category.values
y_c = df_category.iloc[:, 6]

# %%
x_category_train, x_category_test, y_c_train, y_c_test = train_test_split(x_category, y_c, test_size = 0.2)

# %%
c_data_list = []

for classifier in classifiers:

    classifier.fit(x_category_train, y_c_train)
    y_pred = classifier.predict(x_category_test)

    cm = confusion_matrix(y_c_test, y_pred)
    print(cm)
    print('')
    accuracy = accuracy_score(y_c_test,y_pred)
    precision = precision_score(y_c_test,y_pred)
    recall = recall_score(y_c_test,y_pred)
    f1 = f1_score(y_c_test,y_pred)

    data = [round(accuracy, 2), round(precision, 2), round(recall,2), round(f1,2)]
    c_data_list.append(data)

df_c_data = pd.DataFrame(data=c_data_list,index=index, columns=score)
print(df_c_data)
