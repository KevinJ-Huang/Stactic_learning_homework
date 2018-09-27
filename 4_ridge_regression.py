import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def generate():
    y = []
    x = []
    base= 0.041
    e = np.random.normal(0, 0.3, 25)
    for i in range(25):
        x.append(i*base)
        y.append(math.sin(2*math.pi*x[i]))
    Y = y
    y = y+e
    return x,y,Y

def coffecient(lamda):
    x,y,Y = generate()
    X = np.zeros(shape=[25,8])
    for i in range(25):
        for j in range(8):
           X[i,j] = x[i]**j
    XT = X.T
    I = np.eye(8)
    coef = np.dot(np.dot((np.matrix(np.dot(XT,X)+lamda*I)).I,XT),y)
    # clf = Ridge(lamda,fit_intercept=False)
    # clf.fit(X,y)
    # coef = clf.coef_
    return coef,X,x

# plt.figure()
# for i in range(100):
#     x,y,Y = generate()
#     plt.plot(x,y)
# plt.show()

coef,X,x = coffecient(0.01)
print(coef)

def select(lamda):
    # plt.figure()
    y_sum = np.zeros(shape=[25])
    for i in range(100):
        y_data = []
        coef,X,x = coffecient(lamda)
        for i in range(25):
           y = 0
           for j in range(8):
               y = y+coef[0,j]*(X[i,j])
           y_data.append(y)
        y_sum = y_sum+y_data
        y_mean = y_sum / 100
        # plt.plot(x,y_data,color = 'r')
    return x,y_mean,y_data


plt.figure()
# lamda = 0.0001
# x, y_mean = select(lamda)
# plt.plot(x, y_mean, label=lamda)
for lamda in 0.001,0.01,0.1,1:
    x,y_mean,y_data = select(lamda)
    plt.plot(x,y_mean,label = lamda)
plt.xlabel('x')
plt.ylabel('y')
x,y,Y = generate()
plt.plot(x,Y,label = 'groundtruth',color='black')
plt.legend()
plt.show()
