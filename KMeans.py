from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV as gsc
import pandas
import numpy

SAMPLES = 700

data = numpy.load('BINData.npy').reshape(SAMPLES, -1)
mskData = numpy.load('BINMasks.npy').reshape(SAMPLES, -1)
labels = pandas.read_csv('./info_data.csv').label.astype('int').head(SAMPLES)

data = data @ mskData.T
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.25, random_state=42)

params = {'algorithm': ('ball_tree', 'kd_tree'),
          'weights': ['uniform'],
          'n_neighbors': range(5, 31, 5),
          'metric': ('manhattan', 'euclidean')}

knn = KNC()
clf = gsc(knn, params)
clf.fit(xtrain, ytrain)
print("Best Params:", clf.best_params_)
print("Best score:", clf.best_score_)

knn = KNC(**clf.best_params_)
knn.fit(xtrain, ytrain)
pred = knn.predict(xtest)
print(classification_report(ytest, pred))
