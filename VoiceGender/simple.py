import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn import metrics

df = pd.read_csv('input/voice.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

svc = SVC()  # Default hyperparameters
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
