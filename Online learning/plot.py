import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

a, b = np.loadtxt('config.txt', delimiter=',', unpack=True)
colors = ["b", "g", "r", "c", "m", "y"]
idx = 0
for i in range(len(a)):
    plt.plot("", label='a='+str(int(a[i]))+', b='+str(int(b[i])))
    for j in range(11):
        file = "output/a"+str(int(a[i]))+"_b"+str(int(b[i]))+"_"+str(j+1);
        x,y = np.loadtxt(file+'.txt', delimiter=',', unpack=True)
        if j==10:
            plt.plot(x,y, color=colors[idx+1], label='a='+str(int(a[i]))+', b='+str(int(b[i]))+', case=11')
        else:
            plt.plot(x,y, color=colors[idx])
    idx=idx+2
    if idx==6:
        idx = 0
plt.legend()
plt.show()
plt.savefig("output/show.png");