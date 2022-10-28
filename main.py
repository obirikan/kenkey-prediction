import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model,preprocessing
import sklearn
import numpy as np
from sklearn import linear_model,preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pickle

data = pd.read_csv('kenkey.csv',encoding = "ISO-8859-1")

#changing non numerical to numerical
le=preprocessing.LabelEncoder()

day=le.fit_transform(list(data['Day']))
month=le.fit_transform(list(data['Month']))
amt=le.fit_transform(list(data['Total_Amt']))
target_amount=np.array(data['Total_Amt'])


x=list(zip(day,month,target_amount))
y=list(data['Qty_Sales'])

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

best=0


for _ in range(100):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.09)

    #use the KNN module
    model=linear_model.LinearRegression()

    #minimizing error(gradient decent)
    model.fit(x_train,y_train)

    #test the accuracy of the error
    acc=model.score(x_test,y_test)

    if acc>best:
        best=acc
        with open('kenkey.pickle','wb') as f:
           pickle.dump(model,f)


savedmodel=open('kenkey.pickle', 'rb')
newmodel=pickle.load(savedmodel)
nd=1
nm=11
nta=2546
myvalues=[[nd,nm,nta]]

predicted=newmodel.predict(myvalues)



# loop through prediction to see if your data is corresponding well
for x in range(len(predicted)):
    print('predicted:',predicted[x])
    print(f'therefore our weekly sales will be {nta*7} ghana cedis and our monthly sales will be {nta*30} ghana cedis')



