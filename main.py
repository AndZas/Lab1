import numpy as np
from scipy import stats

data = np.array([814, 491, 346, 310, 211, 151, 125, 89, 73, 71, 65, 58, 57, 54, 48, 46, 45, 37, 33, 28, 27, 26, 24, 23, 22, 22, 22, 19, 18])

# --- ЛОГНОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ---
shape, loc, scale = stats.lognorm.fit(data)

print("Lognormal params:")
print("shape =", shape)
print("loc =", loc)
print("scale =", scale)

# --- KS тест ---
ks_stat, p_value = stats.kstest(data, 'lognorm', args=(shape, loc, scale))

print("KS statistic:", ks_stat)
print("p-value:", p_value)

# --- ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ---
mean = np.mean(data)
std = np.std(data, ddof=1)

print("Mean:", mean)
print("Std:", std)