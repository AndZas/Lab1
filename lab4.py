import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# ЗАГРУЗКА ДАННЫХ
file_path = 'FAOSTAT_data_en_5-24-2026 (4).xls'
raw_df = pd.read_excel(file_path)
df = raw_df[['Year', 'Value']].copy()
df.columns = ['Год', 'Площадь лесов']
df = df.sort_values('Год')
df.set_index('Год', inplace=True)

print(df.head(n=31))
print('\nКоличество наблюдений:', len(df))


# ОСНОВНОЙ ГРАФИК
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Площадь лесов'], marker='o')
plt.title('Временные ряды площади лесов')
plt.xlabel('Год')
plt.ylabel('Площадь лесов (1000 га)')
plt.grid(True)
plt.show()

# БАР-ГРАФИК
plt.figure(figsize=(14, 6))
plt.bar(df.index, df['Площадь лесов'])
plt.title('Площадь лесов по годам')
plt.xlabel('Год')
plt.ylabel('Площадь лесов (1000 га)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ОПИСАТЕЛЬНАЯ СТАТИСТИКА
print('\nОписательная статистика:\n')
print(df.describe())

# ПРОВЕРКА НА СТАЦИОНАРНОСТЬ
result = adfuller(df['Площадь лесов'])

print('\nРезультаты теста Дики-Фуллера:\n')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print(f'Critical Values:')

for key, value in result[4].items():
    print(f'   {key}: {value}')

# Автокорреляция
fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(df['Площадь лесов'], lags=15, ax=ax)
plt.title('Автокорреляция')
plt.grid(True)
plt.show()

# Частичная автокорреляция
fig, ax = plt.subplots(figsize=(12, 6))
plot_pacf(df['Площадь лесов'], lags=15, ax=ax)
plt.title('Частичная автокорреляция')
plt.grid(True)
plt.show()


# СЕЗОННОСТЬ И ДЕКОМПОЗИЦИЯ
try:
    decomposition = seasonal_decompose(
        df['Площадь лесов'],
        model='additive',
        period=3
    )

    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.show()

except Exception as e:
    print('\nОшибка decomposition:', e)


# АНОМАЛЬНЫЕ ЗНАЧЕНИЯ
mean = df['Площадь лесов'].mean()
std = df['Площадь лесов'].std()

threshold_upper = mean + 2 * std
threshold_lower = mean - 2 * std

anomalies = df[
    (df['Площадь лесов'] > threshold_upper) |
    (df['Площадь лесов'] < threshold_lower)
]

print('\nАномальные значения:\n')
print(anomalies)


# График аномалий
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Площадь лесов'], label='Исходные данные')

if len(anomalies) > 0:
    plt.scatter(
        anomalies.index,
        anomalies['Площадь лесов'],
        s=100,
        marker='o',
        label='Аномалии'
    )

plt.title('Анализ аномальных значений')
plt.xlabel('Год')
plt.ylabel('Площадь лесов')
plt.legend()
plt.grid(True)
plt.show()


# СКОЛЬЗЯЩЕЕ СРЕДНЕЕ
df['MA_3'] = df['Площадь лесов'].rolling(window=3).mean()
df['MA_5'] = df['Площадь лесов'].rolling(window=5).mean()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Площадь лесов'], label='Исходные данные')
plt.plot(df.index, df['MA_3'], label='Сдвиг среднего (3)')
plt.plot(df.index, df['MA_5'], label='Сдвиг среднего (5)')

plt.title('Сглаживание временного ряда')
plt.xlabel('Год')
plt.ylabel('Площадь лесов')
plt.legend()
plt.grid(True)
plt.show()


# EXPONENTIAL SMOOTHING
model_exp = ExponentialSmoothing(
    df['Площадь лесов'],
    trend='add',
    seasonal=None
)

fit_exp = model_exp.fit()

df['Exp_Smoothing'] = fit_exp.fittedvalues

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Площадь лесов'], label='Исходные данные')
plt.plot(df.index, df['Exp_Smoothing'], label='Экспоненциальное сглаживание')

plt.title('Экспоненциальное сглаживание')
plt.xlabel('Год ')
plt.ylabel('Площадь лесов')
plt.legend()
plt.grid(True)
plt.show()


# ЛИНЕЙНЫЙ ТРЕНД
X = np.array(df.index).reshape(-1, 1)
y = df['Площадь лесов'].values

model = LinearRegression()
model.fit(X, y)

trend = model.predict(X)

# Коэффициенты
print('\nПараметры линейного тренда:\n')
print('a (наклон):', model.coef_[0])
print('b (пересечение):', model.intercept_)


# ОЦЕНКА АДЕКВАТНОСТИ МОДЕЛИ
r2 = r2_score(y, trend)
rmse = np.sqrt(mean_squared_error(y, trend))

print('\nОценка модели:\n')
print('R^2:', r2)
print('RMSE:', rmse)


# ГРАФИК ТРЕНДА
plt.figure(figsize=(12, 6))
plt.plot(df.index, y, label='Исходные данные')
plt.plot(df.index, trend, linewidth=3, label='Линейный тренд')

plt.title('Модель линейного тренда')
plt.xlabel('Год')
plt.ylabel('Площадь лесов')
plt.legend()
plt.grid(True)
plt.show()

# Полиномиальный тренд 2 степени
coeffs_poly = np.polyfit(df.index, df['Площадь лесов'], 2)
poly_model = np.poly1d(coeffs_poly)

df['Poly_Trend'] = poly_model(df.index)

# Оценка полиномиальной модели
r2_poly = r2_score(df['Площадь лесов'], df['Poly_Trend'])
rmse_poly = np.sqrt(mean_squared_error(df['Площадь лесов'], df['Poly_Trend']))

print("\nПараметры полиномиального тренда:")
print(coeffs_poly)

print("\nОценка полиномиальной модели:")
print("R^2:", r2_poly)
print("RMSE:", rmse_poly)

# График сравнения трендов
plt.figure(figsize=(12, 6))

plt.plot(df.index, df['Площадь лесов'],
         label='Исходные данные')

plt.plot(df.index, trend,
         label='Линейный тренд')

plt.plot(df.index, df['Poly_Trend'],
         label='Полиномиальный тренд')

plt.xlabel('Год')
plt.ylabel('Площадь лесов, тыс. га')
plt.title('Сравнение моделей тренда')
plt.legend()
plt.grid(True)

plt.show()


# ОСТАТОЧНАЯ КОМПОНЕНТА
residuals = y - trend

plt.figure(figsize=(12, 6))
plt.plot(df.index, residuals)
plt.axhline(y=0, linestyle='--')

plt.title('Остаточная компонента')
plt.xlabel('Год')
plt.ylabel('Остаток')
plt.grid(True)
plt.show()


# ГИСТОГРАММА ОСТАТКОВ
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=10)

plt.title('Распределение остатков')
plt.xlabel('Значения остатков')
plt.ylabel('Частота')
plt.grid(True)
plt.show()


# ФИНАЛЬНАЯ ТАБЛИЦА
print('\nИтоговая таблица:\n')
print(df.head(31))
