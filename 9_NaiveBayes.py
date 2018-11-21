import numpy as np

def produce_sample():
    x1 = np.array([1,1,1,1,1,2,2,2,2,2,3])
    x2 = np.array([1,2,2,1,1,1,2,2,3,3,3])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1])
    return x1,x2,y

def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = []
    count = 0
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result.append([k,v])
        count+=1
    return np.array(result),count

def cal_xy(x,y):
    arr = np.array(x)
    key_x = np.unique(x)
    key_y = np.unique(y)
    result = []
    for k_y in key_y:
        for k_x in key_x:
            mask_x = (x == k_x)
            mask_y = (y == k_y)
            mask = np.logical_and(mask_x,mask_y)
            arr_new = arr[mask]
            v = arr_new.size
            result.append([k_x,k_y,v])
    return np.array(result)


def NaiveBaiyes(alpha):
    x1,x2,y = produce_sample()
    y_list,count_y = all_np(y)
    x1_list,count_x1 = all_np(x1)
    x2_list,count_x2 = all_np(x2)
    y_prob = (y_list[:,1]+alpha*1)/(y.size+alpha*count_y)

    probx1_y = cal_xy(x1,y)[:,-1]
    probx1_y0 = (probx1_y[:3]+alpha*1)/(y_list[:,1][0]+alpha*count_x1)
    probx1_y1 = (probx1_y[3:] + alpha * 1) / (y_list[:, 1][1] + alpha * count_x1)

    probx2_y = cal_xy(x2,y)[:,-1]
    probx2_y0 = (probx2_y[:3]+alpha*1)/(y_list[:,1][0]+alpha*count_x2)
    probx2_y1 = (probx2_y[3:] + alpha * 1) / (y_list[:, 1][1] + alpha * count_x2)

    return y_prob,probx1_y0,probx1_y1,probx2_y0,probx2_y1


def check(x,y,alpha):
    y_prob, probx1_y0, probx1_y1, probx2_y0, probx2_y1 = NaiveBaiyes(alpha)
    x1_label = x-1
    x2_label = y-1
    y_pre0 = y_prob[0]*probx1_y0[x1_label]*probx2_y0[x2_label]
    y_pre1 = y_prob[1]*probx1_y1[x1_label]*probx2_y1[x2_label]
    if y_pre0>y_pre1:
        print("The label is -1")
    else:
        print("The label is 1")
    print("prediction of -1 is %.4f,prediction 1 is %.4f"%(y_pre0,y_pre1))



check(1,2,alpha=0)


