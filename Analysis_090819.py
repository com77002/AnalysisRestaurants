# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns


df = pd.read_csv('Kyoto_Restaurant_info.csv')
dataset = pd.read_csv('Kyoto_Restaurant_info.csv')
df.drop(df.columns[[0,1,2,5,12,13]], axis=1,inplace=True)

df.fillna(value={'LunchPrice':0, 'LunchRating':0}, inplace=True)

sort = df.loc[:,['FirstCategory', 'TotalRating']]
sort.sort_values(by=['TotalRating'], inplace=True, ascending=False)

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
# data = df
# data.drop(data.columns[[3,4,6,7]], axis=1,inplace=True)
# data.loc[data['DinnerRating'] < 3.3, 'DinnerRating'] = 1
# data.loc[(3.3 <= data['DinnerRating']) & (data['DinnerRating'] < 3.6), 'DinnerRating'] = 2
# data.loc[data['DinnerRating'] >= 3.6, 'DinnerRating'] = 3
# data['DinnerRating'] = data['DinnerRating'].astype(int)
print(df)
# %%
df["ReviewNum"].describe()

# %%
# print('\n---------Dinner Only-----------------------------------\n')
# print(df['Name'][df['LunchPrice'] == 0])
# print('\n---------Lunch and Dinner-----------------------------------\n')
# print(df['Name'][df['LunchPrice'] != 0])
# print('\nTotal Rating Average: ')
# print(df['DinnerRating'].mean())
# print('\n---------Restaurants with rating more than the average----------\n')
# print(sort_totalrating[sort_totalrating['TotalRating'] > df['TotalRating'].mean()])
# print('\n---------Top 10 Restaurants---------------------------\n')
# print(sort_totalrating.head(10))
station = df.iloc[:, 0].values
category = df.iloc[:, 1].values
dinner_price = df.iloc[:, 2].values
# total_rating = df.iloc[:, 4].values
dinner_rating = df.iloc[:, 5].values
station_str = df.iloc[:, 0].values
category_str = df.iloc[:, 1].values
reviewnum = df.iloc[:, -1].values

# %%

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
onehotencoder = OneHotEncoder()
station = labelencoder_x.fit_transform(station)
category = labelencoder_x.fit_transform(category)

# %%
plt.scatter(reviewnum, dinner_rating, color = 'red')
plt.xlabel('Review Num')
plt.ylabel('Dinner Rate')
plt.show()


# %%
# plt.scatter(category, total_rating, color = 'red')
# plt.xlabel('Category')
# plt.ylabel('total Rate')
# plt.show()
# print(sort[sort['TotalRating'] > 3.8])

# %%
# plt.scatter(station, total_rating, color = 'red')
# plt.xlabel('Staion')
# plt.ylabel('total Rate')
# # plt.show()


# %%
from mpl_toolkits.mplot3d.axes3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)
# p = ax.scatter(station, dinner_price, dinner_rating, c='r',cmap='Reds')
# plt.xlabel('Staion')
# plt.ylabel('Dinner Price')
# ax.set_zlabel('Dinner Rating')
# plt.show()

# %%
# plt.scatter(station.flatten(), dinner_price.flatten(), c=dinner_rating)
# plt.colorbar()
# plt.xlabel('Staion')
# plt.ylabel('Dinner Price')
# plt.show()
station_tuple = set(station_str)
# print(station_tuple)
# %%
count_station = Counter(station_str)
plt.bar(range(len(count_station)), list(count_station.values()))
plt.xticks(range(len(count_station)), list(count_station.keys()), rotation=90,fontsize=20,size=5)
plt.xlabel('Staion')
plt.subplots_adjust(top=0.9, bottom=0.3)
plt.show()
# %%
count_category = Counter(category_str)
plt.bar(range(len(count_category)), list(count_category.values()))
plt.xticks(range(len(count_category)), list(count_category.keys()), rotation=90,fontsize=20,size=5)
plt.xlabel('category')
plt.subplots_adjust(top=0.9, bottom=0.3)
plt.show()


# %%
df.loc[df['DinnerRating'] < 3.3, 'DinnerRating'] = 1
df.loc[(3.3 <= df['DinnerRating']) & (df['DinnerRating'] < 3.6), 'DinnerRating'] = 2
df.loc[df['DinnerRating'] >= 3.6, 'DinnerRating'] = 3
df['DinnerRating'] = df['DinnerRating'].astype(int)

# station_cate =pd.get_dummies(df['Station'])
# category_cate = pd.get_dummies(df['FirstCategory'])
# print(category_cate)
df = pd.get_dummies(df)
print(df)

# %%
# sns.set(style='ticks', color_codes=True)
# sns.pairplot(df, hue='DinnerRating')

# # %%
x = df.iloc[:, [0,2]].values


y = df.iloc[:, 5].values
# print(x)
# print(y)
# %%
print(type(df))
plt.scatter(x[:,0].flatten(), x[:,1].flatten(), c=y, alpha=0.7)
# plt.colorbar()
plt.xlabel('Staion')
plt.ylabel('Dinner Price')
plt.title('Staion vs Dinner Price vs Dinner Rate')
# plt.legend()
# plt.show()

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# %%
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

y_pred_svc = classifier.predict(x_test)
# print(y_pred_svc)
# print(y_test)

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_svc)
print(cm)

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier.predict(x_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_knn)
print(cm)

# %%
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_nb = classifier.predict(x_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_nb)
print(cm)

# %%
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_dtc = classifier.predict(x_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dtc)
print(cm)

# %%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_rfc = classifier.predict(x_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rfc)
print(cm)

# %%
print(df)
