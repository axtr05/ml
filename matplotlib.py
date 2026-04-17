import matplotlib.pyplot as plt

K1=[8,4,6,3,5,10,13,16,12,21] 
R1=[11,6,13,15,17,5,3,2,8,19]
K2=[6,9,18,14,16,15,11,16,12,20]
R2=[16,4,10,13,18,20,6,2,17,15]

plt.scatter(K1,R1)
plt.scatter(K2,R2)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()