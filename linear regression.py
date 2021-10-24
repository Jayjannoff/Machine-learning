import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('student-mat.csv',sep=';')
data = data[['G1','G2','G3','failures','studytime','absences']]
predict = 'G3'
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
print(data)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

best = 0
for _ in range(30):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    model = LinearRegression()
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(acc)
    if acc > best:
        acc=best
        with open('studytime.pickle','wb') as f:
            pickle.dump(model,f)

pik = open('studytime.pickle','rb')
model = pickle.load(pik)

prediction = model.predict(x_test)
for i in range(len(prediction)):
    print(prediction[i], x_test[i], y_test[i])

plt.style.use('ggplot')
p = 'G1'
plt.scatter(data['G3'],data[p])
plt.xlabel('final grade')
plt.ylabel(p)
plt.show()