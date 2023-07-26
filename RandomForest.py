from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV as gsc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas
import numpy

SAMPLES = 700
data = numpy.load('BINData.npy').reshape(SAMPLES, -1)
mskData = numpy.load('BINMasks.npy').reshape(SAMPLES, -1)
labels = pandas.read_csv('./info_data.csv').label.astype('int').head(SAMPLES)

data = data @ mskData.T
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.25, random_state=42)

params = {'n_estimators': range(50, 301, 50),
          'criterion': ['gini', 'entropy', 'log_loss'],
          'max_features': ['sqrt', 'log2']}

rfc = RFC()
clf = gsc(rfc, params)
clf.fit(xtrain, ytrain)
print("Best Params:", clf.best_params_)
print("Best score:", clf.best_score_)

rfc = RFC(**clf.best_params_)
rfc.fit(xtrain, ytrain)

pred = rfc.predict(xtest)
print(classification_report(ytest, pred))
