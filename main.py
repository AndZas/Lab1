import numpy as np
from scipy.stats import lognorm

data = np.array([
    814, 491, 346, 310, 211, 151, 125, 89, 73, 71, 65, 58, 57, 54, 48, 46, 45, 37, 33, 28, 27, 26, 24, 23, 22, 22, 22, 19, 18
])

shape, loc, scale = lognorm.fit(data)

print("Параметры логнормального распределения:")
print("shape =", shape)
print("loc =", loc)
print("scale =", scale)
