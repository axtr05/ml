import matplotlib.pyplot as plot

K_1=[8,4,6,3,5,10,13,16,12,21]
R_1=[11,6,13,15,17,5,3,2,8,19]
K_2=[6,9,18,14,16,15,11,16,12,20]
R_2=[16,4,10,13,18,20,6,2,17,15]

plot.scatter(K_1, R_1, c="Black", linewidths=2, marker="s", edgecolor="Brown", s=100)
plot.scatter(K_2, R_2, c="Purple", linewidths=2, marker="^", edgecolor="Grey", s=100)

plot.xlabel("X-axis")
plot.ylabel("Y-axis")

print("Scatter Plot")
plot.show()