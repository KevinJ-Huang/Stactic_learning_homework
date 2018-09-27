import numpy as np
import matplotlib.pyplot as plt

def StructNumber(mean,sigma,num):
    number = np.random.normal(mean,sigma,num)
    return number

i = 1
x = []
y = []
t = 1
while i<100001:
    num = i*10
    print(num)
    s = StructNumber(100,10,num)
    i = i*10
    plt.subplot(3,2,t)
    plt.hist(s,30)
    plt.title('num='+str(num))
    mean = s.mean()
    variance = (s*s).sum()/num - mean**2
    print("mean:%.8f,variance:%.8f"%(mean,variance))
    x.append(mean)
    y.append(variance)
    t+=1
plt.show()

plt.plot(x, y,  color='r',markerfacecolor='blue',marker='o')
num = 1
for a, b in zip(x, y):
    num = num*10
    plt.text(a, b, (num), ha='center', va='bottom', fontsize=5)

plt.legend()
plt.show()
