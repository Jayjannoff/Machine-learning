import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from sklearn import preprocessing

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(data['buying'])
maint = le.fit_transform(data['maint'])
door = le.fit_transform(data['door'])
persons = le.fit_transform(data['persons'])
lugboot = le.fit_transform(data['lug_boot'])
safety = le.fit_transform(data['safety'])
cls = le.fit_transform(data['class'])
predict = 'class'
x = list(zip(buying,maint,door,persons,lugboot,safety))
y = list(cls)
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

predictions = model.predict(x_test)

names = ['unacc','acc','good','vgood']
for x in range(len(predictions)):
    print('prediction:',predictions[x],'data:',x_test[x],'actual value',y_test[x])