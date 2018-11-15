import numpy as np
import matplotlib.pyplot as plt


def GetDataset():
    x1 = np.array([1,1,1,1,1,2,2,2,2,2,3])
    x2 = np.array([1,2,2,1,1,1,2,2,3,3,3])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1])
    return x1,x2,y

x1,x2,y = GetDataset()
plt.figure()
for i in range(11):
    if y[i]<0:
        plt.scatter(x1[i],x2[i],c='b')
    else:
        plt.scatter(x1[i],x2[i],c='r')

plt.show()

def BaseLearn(dim,threshnum,low_or_high):
    x1,x2,y = GetDataset()
    restresh = np.ones(shape=[11])
    if dim == 0:
        x = x1
    else:
        x = x2
    if low_or_high == 'low':
        restresh[x[:] <= threshnum]=-1
    else:
        restresh[x[:] > threshnum] = -1
    return restresh


def GetStump(w):
    minError = np.inf
    best_val = np.ones(shape=[11])
    x1,x2,y = GetDataset()
    for i in range(2):
        if i==0:
           x = x1
        else:
           x = x2
        rangeMin = x[:].min()
        rangeMax = x[:].max()
        stepSize = (rangeMax - rangeMin)/4.0
        for j in range(4):
            threshnum = rangeMin + stepSize*j
            for item in ['low','high']:
                resVal = BaseLearn(dim=i,threshnum=threshnum,low_or_high=item)
                err = np.ones(shape=[11])
                err[resVal == y] = 0
                err_w = np.dot(w,err.T)
                if err_w < minError:
                    minError = err_w
                    best_tresh = threshnum
                    best_item = item
                    best_dim = i
                    best_val = resVal.copy()
    return best_val,minError,best_tresh,best_dim


def AdaBoost(iters):
    w = np.ones(shape=[11])/11.0
    x1,x2,y = GetDataset()
    val_sum = np.zeros(shape=[11])


    tresh_total_1 = []
    tresh_total_2 = []
    for i in range(iters):
        val,error,tresh,dim = GetStump(w)
        alpha = float(0.5*np.log((1-error)/max(error,1e-16)))
        expon = -1*alpha*y*val
        w_list = w*np.exp(expon)
        w = w_list/w_list.sum()
        val_sum += alpha*val
        error_end = (np.sign(val_sum) != y).sum()/11
        if error_end<0.2:
            break
        if dim == 0:
            tresh_total_1.append(tresh)
        if dim == 1:
            tresh_total_2.append(tresh)

    print(error_end)

    return val_sum

val = AdaBoost(4)

for i in range(11):
    if val[i] < 0:
        plt.scatter(x1[i], x2[i], c='b')
    else:
        plt.scatter(x1[i], x2[i], c='r')

plt.show()

print(val[1],val[2])

