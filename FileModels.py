import joblib
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
import pandas
import numpy

SAMPLES = 700
xtrain = []
imgs = numpy.load('BINData.npy')
mskData = numpy.load('BINMasks.npy')
ytrain = pandas.read_csv('./info_data.csv').label.astype('int').head(SAMPLES)
for img, mask in zip(imgs, mskData):
    xtrain.append(img @ mask.T)
xtrain = numpy.asarray(xtrain).reshape(SAMPLES, -1)

scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)

mnb = MNB()
mnb.fit(xtrain, ytrain)
filename = 'MNB.joblib'
joblib.dump(mnb, filename)
del mnb
print(1)

svc = SVC(kernel='poly', degree=5, decision_function_shape='ovo')
svc.fit(xtrain, ytrain)
filename = 'SVC.joblib'
joblib.dump(svc, filename)
del svc
print(2)

# votingClf = VotingClassifier([('clf1', MNB()),
#                              ('clf3', SVC(kernel='poly', degree=5, decision_function_shape='ovo', probability=True))],
#                              voting='soft')
# abc = AdaBoostClassifier(base_estimator=votingClf)
# abc.fit(xtrain[:400, :], ytrain[:400])
# filename = 'ABC.joblib'
# joblib.dump(abc, filename)
# del(abc, votingClf, scaler, filename)
# print(3)

rfc = RFC(**{'criterion': 'entropy', 'max_features': 'log2', 'n_estimators': 200})
rfc.fit(xtrain, ytrain)
filename = 'RFC.joblib'
joblib.dump(rfc, filename)
del(rfc, filename)
print(4)

knn = KNC(** {'algorithm': 'ball_tree', 'metric': 'manhattan', 'n_neighbors': 25, 'weights': 'uniform'})
knn.fit(xtrain, ytrain)
filename = 'KNN.joblib'
joblib.dump(knn, filename)
del(knn, filename)
print('Done!!')
