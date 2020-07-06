

import os

import sys
reload(sys)  
sys.setdefaultencoding('utf8')


import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor



url = "SONISLEMDOSYAM.csv"
labels = ['image', 'age_approx', 'anatom_site_general', 'gender','Sinif_Adi']
dataset = pandas.read_csv(url, names = labels)

#shape of the data
print(dataset.shape)

#head of the datasets - labels
print(dataset.head(5))

# summary of the dataset
print(dataset.describe())

# class distribution

print(dataset.groupby('Sinif_Adi').size())

#To numeric metodu
lb_make = LabelEncoder()
dataset["dagilim"] = lb_make.fit_transform(dataset["Sinif_Adi"])
dataset[["Sinif_Adi", "dagilim"]].head(11)

# plot the data - first, plots of each individual variable
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

dataset.hist()

#plt.show()

# muti variate plots
scatter_matrix(dataset)
#plt.show()

#kategorik verileri numerik hale getirme
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# butun kolonlardaki unique degerleri gosterir
dtype_object=dataset.select_dtypes(include=['object'])
print dtype_object.head()
for x in dtype_object.columns:
    dataset[x]=le.fit_transform(dataset[x])

#create a validation dataset
# use 80% for training, 20% for test
array = dataset.values # this will filter the label attributes
X = array[:, 0:5] # use all columns for training
Y = dataset["dagilim"].values # class of the data - the answer that ML wants to predict
validation_size = 0.20
seed = 5
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
                                               test_size = validation_size, random_state = seed)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)


scoring = 'accuracy'

#include different algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('ANN', MLPClassifier()))
models.append(('SVM', SVC())),



results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Make predictions on validation dataset
print("***********KNN***********")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print("*********** LR ***********")
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_validation)
print(logreg.score(X_train, Y_train))
print(accuracy_score(Y_validation,Y_pred))
print(confusion_matrix(Y_validation, Y_pred))
print(classification_report(Y_validation, Y_pred))


print("*********** Support Vector Classifier ***********")
svc = SVC()
svc.fit(X_train, Y_train)
pred_svc = svc.predict(X_validation)
print(svc.score(X_train, Y_train))
print(accuracy_score(Y_validation,Y_pred))
print(confusion_matrix(Y_validation, Y_pred))
print(classification_report(Y_validation, pred_svc))

print("*********** Naive Bayes ***********")
nb=GaussianNB()
nb.fit(X_train,Y_train)
nb_pred=nb.predict(X_validation)
print(logreg.score(X_train, Y_train))
print(accuracy_score(Y_validation,nb_pred))
print(confusion_matrix(Y_validation, nb_pred))
print(classification_report(Y_validation, nb_pred))

print("*********** Decision Tree ***********")
dt=DecisionTreeClassifier(criterion='entropy', max_depth=120)
dt.fit(X_train,Y_train)
dt_pred=dt.predict(X_validation)
print(logreg.score(X_train, Y_train))
print(accuracy_score(Y_validation,dt_pred))
print(confusion_matrix(Y_validation, dt_pred))
print(classification_report(Y_validation, dt_pred))

print("*********** MLP ***********")
Ann= MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 4), random_state=1)
Ann.fit(X_train, Y_train)
Ann_pred = Ann.predict(X_validation)
print(logreg.score(X_train, Y_train))
print(accuracy_score(Y_validation,Ann_pred))
print(confusion_matrix(Y_validation, Ann_pred))
print(classification_report(Y_validation, Ann_pred))


