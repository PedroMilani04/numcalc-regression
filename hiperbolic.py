import numpy as np
import matplotlib.pyplot as plt

# data
anos = np.array([1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2022])
x = anos - 1940
y = np.array([12637, 24515, 54055, 71188, 129733, 167966, 182023, 207610, 225668])
n = len(x)

# hyperbolic transformation
Y = 1 / y

# base vectors
v0 = np.ones(n)
v1 = x

# scalar product
def prod_escalar(a, b):
    return np.dot(a, b)

# system
A = np.array([
    [prod_escalar(v0, v0), prod_escalar(v0, v1)],
    [prod_escalar(v1, v0), prod_escalar(v1, v1)]
])

b_vec = np.array([
    prod_escalar(v0, Y),
    prod_escalar(v1, Y)
])

# solve system
coef = np.linalg.solve(A, b_vec)
a0, a1 = coef

# fitted function
def f_hiper(x):
    return 1 / (a0 + a1 * x)

# estimate population in 2030 (x=90)
print("Estimated population for 2030:", f_hiper(90))

# plot
x_plot = np.linspace(0, 90, 300)
y_plot = f_hiper(x_plot)

plt.scatter(x, y, label="Real data", color='black')
plt.plot(x_plot, y_plot, label="Hyperbolic fit", color='green')
plt.xlabel("Years since 1940")
plt.ylabel("Population")
plt.title("Least Squares Fit (Hyperbolic) - Limited")
plt.ylim(0, 300000)  
plt.grid()
plt.legend()
plt.show()


plt.scatter(x, y, label="Real data", color='black')
plt.plot(x_plot, y_plot, label="Hyperbolic fit", color='green')
plt.xlabel("Years since 1940")
plt.ylabel("Population")
plt.title("Least Squares Fit (Hyperbolic) - Expanded")
plt.grid()
plt.legend()
plt.show()
