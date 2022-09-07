''' ---------- Import Libraries ---------- '''
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler

### ----- bicycleMerge -----
print("Bicycle")
beginTime = datetime.datetime.now()

df = pd.read_csv('bicycleMerge2.csv')
output = df['output']
df = df.drop(['district', 'output'], axis=1)
df['output'] = output
df1 = df.to_csv("testMerge.csv", encoding='utf-8')
x = pd.DataFrame(df[df.columns[1:35]])
y = pd.DataFrame(df['output'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ## accuracy
depth = [0] * 25
for i in range(25):
    depth[i] = i + 1
score_list = []
for i in depth:
    clf = DecisionTreeClassifier(criterion="gini", max_depth=i)
    score = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
    score_list.append(sum(score) / len(score))

plt.plot(depth, score_list)
plt.xlabel("depth")
plt.ylabel('accuracy')
plt.title("10-fold validation")
plt.show()

# 由於 depth = 13 時結果就已經與後面的準確度差不多了，因此訓練上我使用該參數
# gini
clf_cv = DecisionTreeClassifier(criterion="gini", max_depth=13)
clf_cv.fit(x_train, y_train)
# 預測結果
y_pred = clf_cv.predict(x_test)

matric = confusion_matrix(y_test, y_pred)
sns.heatmap(matric, square=True, annot=True, cbar=False)
plt.xlabel("predict value")
plt.ylabel("true value")
plt.title("CART confusion matrix")
plt.show()

print("CART Accuracy score:", clf_cv.score(x_test, y_test))

print("report:\n", classification_report(y_test, y_pred, labels=[0, 1], target_names=['not stole', 'stole']))

endTime = datetime.datetime.now()
interval = endTime - beginTime
total_seconds = interval.total_seconds()
print("Runtime: ", total_seconds, "s", sep="")


### ----- carMerge -----
print("Car")
beginTime = datetime.datetime.now()

df = pd.read_csv('carMerge2.csv')
output = df['output']
df = df.drop(['district', 'output'], axis=1)
df['output'] = output
df1 = df.to_csv("testMerge.csv", encoding='utf-8')
x = pd.DataFrame(df[df.columns[1:35]])
y = pd.DataFrame(df['output'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ## accuracy
depth = [0] * 25
for i in range(25):
    depth[i] = i + 1
score_list = []
for i in depth:
    clf = DecisionTreeClassifier(criterion="gini", max_depth=i)
    score = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
    score_list.append(sum(score) / len(score))

plt.plot(depth, score_list)
plt.xlabel("depth")
plt.ylabel('accuracy')
plt.title("10-fold validation")
plt.show()

# 由於 depth = 11 時結果是最高的
# gini
clf_cv = DecisionTreeClassifier(criterion="gini", max_depth=13)
clf_cv.fit(x_train, y_train)
# 預測結果
y_pred = clf_cv.predict(x_test)

matric = confusion_matrix(y_test, y_pred)
sns.heatmap(matric, square=True, annot=True, cbar=False)
plt.xlabel("predict value")
plt.ylabel("true value")
plt.title("CART confusion matrix")
plt.show()

print("CART Accuracy score:", clf_cv.score(x_test, y_test))

print("report:\n", classification_report(y_test, y_pred, labels=[0, 1], target_names=['not stole', 'stole']))

endTime = datetime.datetime.now()
interval = endTime - beginTime
total_seconds = interval.total_seconds()
print("Runtime: ", total_seconds, "s", sep="")


### ----- motorMerge -----
print("Motor")
beginTime = datetime.datetime.now()

df = pd.read_csv('motorMerge2.csv')
output = df['output']
df = df.drop(['district', 'output'], axis=1)
df['output'] = output
df1 = df.to_csv("testMerge.csv", encoding='utf-8')
x = pd.DataFrame(df[df.columns[1:35]])
y = pd.DataFrame(df['output'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ## accuracy
depth = [0] * 25
for i in range(25):
    depth[i] = i + 1
score_list = []
for i in depth:
    clf = DecisionTreeClassifier(criterion="gini", max_depth=i)
    score = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
    score_list.append(sum(score) / len(score))

plt.plot(depth, score_list)
plt.xlabel("depth")
plt.ylabel('accuracy')
plt.title("10-fold validation")
plt.show()

# 由於 depth = 13 時結果就已經與後面的準確度差不多了，因此訓練上我使用該參數
# gini
clf_cv = DecisionTreeClassifier(criterion="gini", max_depth=13)
clf_cv.fit(x_train, y_train)
# 預測結果
y_pred = clf_cv.predict(x_test)

matric = confusion_matrix(y_test, y_pred)
sns.heatmap(matric, square=True, annot=True, cbar=False)
plt.xlabel("predict value")
plt.ylabel("true value")
plt.title("CART confusion matrix")
plt.show()

print("CART Accuracy score:", clf_cv.score(x_test, y_test))

print("report:\n", classification_report(y_test, y_pred, labels=[0, 1], target_names=['not stole', 'stole']))

endTime = datetime.datetime.now()
interval = endTime - beginTime
total_seconds = interval.total_seconds()
print("Runtime: ", total_seconds, "s", sep="")


### ----- homeMerge -----
print("Home")
beginTime = datetime.datetime.now()

df = pd.read_csv('homeMerge2.csv')
output = df['output']
df = df.drop(['district', 'output'], axis=1)
df['output'] = output
df1 = df.to_csv("testMerge.csv", encoding='utf-8')
x = pd.DataFrame(df[df.columns[1:35]])
y = pd.DataFrame(df['output'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ## accuracy
depth = [0] * 25
for i in range(25):
    depth[i] = i + 1
score_list = []
for i in depth:
    clf = DecisionTreeClassifier(criterion="gini", max_depth=i)
    score = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
    score_list.append(sum(score) / len(score))

plt.plot(depth, score_list)
plt.xlabel("depth")
plt.ylabel('accuracy')
plt.title("10-fold validation")
plt.show()

# 由於 depth = 14 時結果就已經與後面的準確度差不多了，因此訓練上我使用該參數
# gini
clf_cv = DecisionTreeClassifier(criterion="gini", max_depth=14)
clf_cv.fit(x_train, y_train)
# 預測結果
y_pred = clf_cv.predict(x_test)

matric = confusion_matrix(y_test, y_pred)
sns.heatmap(matric, square=True, annot=True, cbar=False)
plt.xlabel("predict value")
plt.ylabel("true value")
plt.title("CART confusion matrix")
plt.show()

print("CART Accuracy score:", clf_cv.score(x_test, y_test))

print("report:\n", classification_report(y_test, y_pred, labels=[0, 1], target_names=['not stole', 'stole']))

endTime = datetime.datetime.now()
interval = endTime - beginTime
total_seconds = interval.total_seconds()
print("Runtime: ", total_seconds, "s", sep="")
