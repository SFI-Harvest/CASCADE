# This is a test using the curve_fit function from scipy.optimize

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm
import os


class Function:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.n_calls = 0


    def func(self, x):
        return self.a * np.exp(-self.b * (x - self.c)**2) + self.c + np.sin(x / 3) * self.d

    def __call__(self, x):
        return self.func(x, self.a, self.b, self.c)
    

    def set_and_call(self, x, a, b, c, d):
        self.set_a(a)
        self.set_b(b)
        self.set_c(c)
        self.set_d(d)

        self.n_calls += 1

        return self.func(x)
    

    def set_a(self, a):
        self.a = a

    def set_b(self, b):
        self.b = b

    def set_c(self, c):
        self.c = c

    def set_d(self, d):
        self.d = d
    



f = Function(2.5, 0.03, 0.5, 0.1)

x = np.random.rand(100) * 10    
a_true = 2.5
b_true = 0.03
c_true = 0.5
d_true = 0.1
true_params = [a_true, b_true, c_true, d_true]

y = f.func(x) + np.random.normal(size=x.size, scale=0.2)




xdata = x
ydata = y
p0 = [1, 1, 1, 1]  # Initial guess for the parameters
popt, pcov = curve_fit(f.set_and_call, xdata, ydata, p0=p0)
perr = np.sqrt(np.diag(pcov))
print("Fitted parameters:", popt)
print("Parameter errors:", perr)
print("True parameters:", a_true, b_true, c_true)
print("True parameters errors:", 0.1, 0.1, 0.1)
print("Covariance matrix:", pcov)
print("Covariance matrix errors:", np.sqrt(np.diag(pcov)))
print("Correlation matrix:", np.corrcoef(pcov))
print("Correlation matrix errors:", np.sqrt(np.diag(pcov)))
print("Number of function calls:", f.n_calls)


# Plotting the results
xdata_sort = np.sort(xdata)

plt.figure(figsize=(10, 6))
plt.scatter(xdata, ydata, label='Data', color='red')
plt.plot(xdata_sort, f.set_and_call(xdata_sort, *popt), label='Fitted function', color='blue')
plt.plot(xdata_sort, f.set_and_call(xdata_sort, *p0), label='Initial guess', color='green', linestyle='--')
plt.plot(xdata_sort, f.set_and_call(xdata_sort, *true_params), label='True function', color='orange', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting Example')
plt.legend()
plt.show()


# Try running the script multiple times to see if the results are consistent

n_tries = 20

f = Function(2.5, 0.03, 0.5, 0.1)

x = np.random.rand(100) * 10    
xdata = x

params_estimates = []
for i in range(n_tries):
    
    y = f.func(x) + np.random.normal(size=x.size, scale=0.2)
    
    ydata = y
    p0 = [1, 1, 1, 1]  # Initial guess for the parameters
    popt, pcov = curve_fit(f.set_and_call, xdata, ydata, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    print("Fitted parameters:", popt)
    print("Parameter errors:", perr)
    print("n calls:", f.n_calls)

    params_estimates.append(popt)

xdata_sort = np.sort(xdata)
for i, params in enumerate(params_estimates):
    plt.plot(xdata_sort, f.set_and_call(xdata_sort, *params), label=f'Fitted function {i+1}', color='blue', alpha=0.5)
plt.plot(xdata_sort, f.set_and_call(xdata_sort, *p0), color='green', linestyle='--')
plt.plot(xdata_sort, f.set_and_call(xdata_sort, *true_params), label='True function', color='orange', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting Example - Multiple Runs')
plt.legend()
plt.show()