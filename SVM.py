from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

#split it in features and labels
X = iris.data
y = iris.target

print(X, y)
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
print(X.shape)
print(y.shape)

#hours of study vs good/bad grades
#10 different students
#train with 8 student 
#predict with the remaining 2
#allows for determining the model accuracy
#level of accuracy 

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2)

model = svm.SVC()
model.fit(X_train, y_train)


print(model)


prediction = model.predict(X_test)
acc = accuracy_score(y_test, prediction)

print(prediction)
print("actual:  ", y_test)
print("Accuracy : ", acc)

for i in range(len(prediction)):
    print(classes[prediction[i]])







