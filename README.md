
# Ex.No: 04  FIT ARMA MODEL FOR TIME SERIES

### Date: 22-09-2025


### AIM:

To implement ARMA model in Python.

### ALGORITHM:

1. Import necessary libraries like `pandas`, `numpy`, `matplotlib`, and `statsmodels`.
2. Load and preprocess the dataset, converting date columns to datetime and handling missing values.
3. Plot the original time series to visualize the data.
4. Display autocorrelation (ACF) and partial autocorrelation (PACF) plots for the series.
5. Fit ARMA(1,1) model and generate a simulated time series using the estimated AR and MA coefficients. Plot the series and its ACF and PACF.
6. Fit ARMA(2,2) model and generate a simulated time series using the estimated AR and MA coefficients. Plot the series and its ACF and PACF.


### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
df = pd.read_csv("weatherHistory.csv")
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True, errors='coerce')
df['YearFrac'] = df['Formatted Date'].dt.year + (df['Formatted Date'].dt.dayofyear / 365)
df = df.dropna(subset=['Temperature (C)', 'YearFrac'])

X = df['Temperature (C)'].values

# Original time series plot
plt.figure(figsize=(12, 6))
plt.plot(X)
plt.title('Temperature Time Series')
plt.show()

# ACF and PACF of original series
plt.subplot(2,1,1)
plot_acf(X, lags=50, ax=plt.gca())
plt.title("ACF of Temperature Data")

plt.subplot(2,1,2)
plot_pacf(X, lags=50, ax=plt.gca())
plt.title("PACF of Temperature Data")

plt.tight_layout()
plt.show()

# Fit ARMA(1,1)
arma11_model = ARIMA(X, order=(1, 0, 1), trend='n').fit()
phi1 = arma11_model.params[0]
theta1 = arma11_model.params[1]

ar = np.array([1, -phi1])
ma = np.array([1, theta1])
arma11_process = ArmaProcess(ar, ma).generate_sample(nsample=500)

plt.figure(figsize=(12, 6))
plt.plot(arma11_process)
plt.title('Simulated ARMA(1,1) Process for Temperature')
plt.show()

plt.figure(figsize=(12, 4))
plot_acf(arma11_process, lags=50, ax=plt.gca())
plt.title("ACF of Simulated ARMA(1,1)")
plt.show()

plt.figure(figsize=(12, 4))
plot_pacf(arma11_process, lags=50, ax=plt.gca())
plt.title("PACF of Simulated ARMA(1,1)")
plt.show()

# Fit ARMA(2,2)
arma22_model = ARIMA(X, order=(2,0,2), trend='n').fit()
phi1, phi2, theta1, theta2 = arma22_model.params[:4]

ar = np.array([1, -phi1, -phi2])
ma = np.array([1, theta1, theta2])
arma22_process = ArmaProcess(ar, ma).generate_sample(nsample=500)

plt.figure(figsize=(12, 6))
plt.plot(arma22_process)
plt.title('Simulated ARMA(2,2) Process for Temperature')
plt.show()

plt.figure(figsize=(12, 4))
plot_acf(arma22_process, lags=50, ax=plt.gca())
plt.title("ACF of Simulated ARMA(2,2)")
plt.show()

plt.figure(figsize=(12, 4))
plot_pacf(arma22_process, lags=50, ax=plt.gca())
plt.title("PACF of Simulated ARMA(2,2)")
plt.show()
```

### OUTPUT:

![alt text](/Images/image.png)

**SIMULATED ARMA(1,1) PROCESS:**

* Plot of simulated ARMA(1,1) time series
* Partial autocorrelation (PACF) plot
* Autocorrelation (ACF) plot



![alt text](/Images/image-1.png)

![alt text](/Images/image-4.png)

![alt text](/Images/image-5.png)



**SIMULATED ARMA(2,2) PROCESS:**

* Plot of simulated ARMA(2,2) time series
* Partial autocorrelation (PACF) plot
* Autocorrelation (ACF) plot

![alt text](/Images/image-6.png)

![alt text](/Images/image-2.png)

![alt text](/Images/image-3.png)


### RESULT:

Thus, a Python program is created to fit ARMA(1,1) and ARMA(2,2) models successfully, simulate time series data using the estimated parameters, and visualize the autocorrelation and partial autocorrelation plots for analysis.
