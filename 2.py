import math
import numpy as np
import scipy.stats as st
import statistics as stats
square_root = math.sqrt(16)
print("Square root of 16:", square_root)
data = [5, 10, 15, 20, 25]
mean = stats.mean(data)
print("Mean using statistics:", mean)
arr = np.array([1, 2, 3, 4, 5])
mean_numpy = np.mean(arr)
print("Mean using numpy:", mean_numpy)
z_score = st.zscore(data)
print("Z-score using scipy:", z_score)