# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


df = pd.read_csv('Kyoto_Restaurant_info.csv')
dataset = pd.read_csv('Kyoto_Restaurant_info.csv')

df.drop(df.columns[[0,1,2,5,7,8,10,12,13]], axis=1,inplace=True) #3Station,4Category,6Price,9Rating
df = df[df['ReviewNum'] > 30]
df.drop('ReviewNum', axis=1, inplace=True)

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

# %%
station = df.iloc[:, 0].values
category = df.iloc[:, 1].values
# price = df.iloc[:, 2].values
# rating = df.iloc[:, 3].values
# print(df)
# %%
count_station = Counter(station)
plt.bar(range(len(count_station)), list(count_station.values()))
plt.xticks(range(len(count_station)), list(count_station.keys()), rotation=90,fontsize=50,size=10)
plt.xlabel('Staion')
plt.subplots_adjust(top=0.9, bottom=0.3)
plt.show()

# %%
count_category = Counter(category)
plt.bar(range(len(count_category)), list(count_category.values()))
plt.xticks(range(len(count_category)), list(count_category.keys()), rotation=90,fontsize=50,size=10)
plt.xlabel('category')
plt.subplots_adjust(top=0.9, bottom=0.3)
plt.show()

# %%
df.loc[df['DinnerRating'] < 3.2, 'DinnerRating'] = 1
# df.loc[(3.3 <= df['DinnerRating']) & (df['DinnerRating'] < 3.6), 'DinnerRating'] = 2
df.loc[df['DinnerRating'] >= 3.2, 'DinnerRating'] = 0
df['DinnerRating'] = df['DinnerRating'].astype(int)
df = pd.get_dummies(df, drop_first=True, columns=['Station', 'FirstCategory'])

# %%
# print(df)
# %%
x = df.drop(df.columns[1],axis=1).values
y = df.iloc[:,1].values

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', gamma='scale')

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')

classifiers = [knn, svc, nb, dtc, rfc]

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

index = ['K Naighnors', 'kernel SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest']
score = ['Accuracy','Precision', 'Recall', 'F-score']

ave_list = []
num = 300

print(f"\nN = {num}")
for classifier in classifiers:
    sum_acc = 0
    sum_pre = 0
    sum_rec = 0
    sum_f1 = 0

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    for i in range(num):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        # print('')
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)

        sum_acc += accuracy
        sum_pre += precision
        sum_rec += recall
        sum_f1 += f1

    ave_acc = sum_acc/num
    ave_pre = sum_pre/num
    ave_rec = sum_rec/num
    ave_f1 = sum_f1/num

    data = [round(ave_acc, 2), round(ave_pre, 2), round(ave_rec,2), round(ave_f1,2)]
    ave_list.append(data)

df_ave = pd.DataFrame(data=ave_list,index=index, columns=score)
print(df_ave)
