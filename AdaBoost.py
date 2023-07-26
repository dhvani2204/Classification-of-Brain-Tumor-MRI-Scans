from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas
import numpy

SAMPLES = 300
labels = pandas.read_csv('./info_data.csv').label.astype('int').head(SAMPLES)
data = []
imgs = numpy.load('PCAData.npy')
mskData = numpy.load('PCAMasks.npy')
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

abc = AdaBoostClassifier(estimator=votingClf)

abc.fit(xtrain, ytrain)

pred = abc.predict(xtest)
print("Best Score:", accuracy_score(ytest, pred))
print(classification_report(ytest, pred))
