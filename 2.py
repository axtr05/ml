import math
import numpy as np
import scipy.stats as st
import statistics as stats
print("Square root of 16:", math.sqrt(16))
data = [5, 10, 15, 20, 25]
print("Mean using statistics:", stats.mean(data))
arr = np.array([1, 2, 3, 4, 5])
print("Mean using numpy:", np.mean(arr))
print("Z-score using scipy:", st.zscore(data))