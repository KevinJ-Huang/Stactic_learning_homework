from numpy import *
from sklearn import svm
import matplotlib.pyplot as plt

def produce():
    data = array([[1,2,1],
        [2,3,1],
        [3,3,1],
        [2,1,-1],
        [3,2,-1]])
    x=data[:,0:2]
    y=data[:,2]
    return x,y,data

def compute_svm(x,y):
    clf=svm.SVC(kernel='linear',C=1000)
    clf.fit(x,y)
    return clf

def main():
    x,y,data = produce()
    clf = compute_svm(x,y)
    w = clf.coef_[0]
    print('weight for svm:',w)
    x0 = data[:3,:]
    x1 = data[3:,:]
    plt.scatter(x0[:, 0], x0[:, 1], c='r', label='+')
    plt.scatter(x1[:, 0], x1[:, 1], c='b', label='-')
    a = -w[0] / w[1]
    xx = arange(0,4.0,0.1)
    yy = a * xx - clf.intercept_[0] / w[1]
    plt.plot(xx, yy)
    plt.title('svm for linear')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()
