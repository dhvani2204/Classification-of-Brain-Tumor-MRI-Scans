from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn.metrics import classification_report

SAMPLES = 700
labels = pandas.read_csv('./info_data.csv').label.astype('int').head(SAMPLES)
data = []
imgs = numpy.load('BINData.npy')
mskData = numpy.load('BINMasks.npy')
ytrain = pandas.read_csv('./info_data.csv').label.astype('int').head(SAMPLES)
for img, mask in zip(imgs, mskData):
    data.append(img @ mask.T)
data = numpy.asarray(data).reshape(SAMPLES, -1)
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.25, random_state=42)
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

votingClf = VotingClassifier([('clf1', MNB()),
                             ('clf3', SVC(decision_function_shape='ovo', degree=5, kernel='poly', probability=True))],
                             voting='soft')
abc = AdaBoostClassifier(base_estimator=votingClf)
abc.fit(xtrain, ytrain)
pred = abc.predict(xtest)
hm = sns.heatmap(confusion_matrix(ytest, pred))
hm.get_figure().savefig('AdaBoost.png', dpi=400)
plt.clf()

rfc = RFC(**{'criterion': 'entropy', 'max_features': 'log2', 'n_estimators': 200})
rfc.fit(xtrain, ytrain)
pred = rfc.predict(xtest)
print(classification_report(ytest, pred))
del (rfc, pred)
# hm = sns.heatmap(confusion_matrix(ytest, pred))
# hm.get_figure().savefig('Random Forest.png', dpi=400)
# plt.clf()

knn = KNC(** {'algorithm': 'ball_tree', 'metric': 'manhattan', 'n_neighbors': 25, 'weights': 'uniform'})
knn.fit(xtrain, ytrain)
pred = knn.predict(xtest)
print(classification_report(ytest, pred))
del (knn, pred)
# hm = sns.heatmap(confusion_matrix(ytest, pred))
# hm.get_figure().savefig('KNN.png', dpi=400)
# plt.clf()

mnb = MultinomialNB()
mnb.fit(xtrain, ytrain)
pred = mnb.predict(xtest)
print(classification_report(ytest, pred))
del (mnb, pred)
# hm = sns.heatmap(confusion_matrix(ytest, pred))
# hm.get_figure().savefig('Naive Bayes.png', dpi=400)
# plt.clf()

# svc = SVC(kernel='poly', degree=5, decision_function_shape='ovo')
# svc.fit(xtrain, ytrain)
# pred = svc.predict(xtest)
# hm = sns.heatmap(confusion_matrix(ytest, pred))
# hm.get_figure().savefig('SVC.png', dpi=400)
# plt.clf()
