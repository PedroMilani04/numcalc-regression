import numpy as np
import matplotlib.pyplot as plt

# data (years transformed to avoid large numbers)
years = np.array([1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2022])
x_original = years - 1940
y = np.array([12637, 24515, 54055, 71188, 129733, 167966, 182023, 207610, 225668])

# avoid log of zero or negative
x = x_original[x_original > 0]
y = y[x_original > 0]

# linearization
X = np.log(x)
Y = np.log(y)

# dot product
def dot_product(a, b):
    return np.dot(a, b)

# build normal system for least squares (line: Y = A + B·X)
n = len(X)
v0 = np.ones(n)
v1 = X

A_mat = np.array([
    [dot_product(v0, v0), dot_product(v0, v1)],
    [dot_product(v1, v0), dot_product(v1, v1)]
])

b_vec = np.array([
    dot_product(v0, Y),
    dot_product(v1, Y)
])

# solve linear system
coef = np.linalg.solve(A_mat, b_vec)
A, B = coef  # A = ln(a), B = b

a = np.exp(A)
b = B

print(f"Coefficients: a = {a:.3f}, b = {b:.3f}")

# model function: f(x) = a * x^b
def f_geom(x):
    return a * x**b

# plot
x_plot = np.linspace(0, 90, 300)
y_plot = f_geom(x_plot)

plt.scatter(x, y, label="Real data", color='black')
plt.plot(x_plot, y_plot, label="Geometric fit (a·x^b)", color='orange')
plt.xlabel("Years since 1940")
plt.ylabel("Population")
plt.title("Least Squares Fit (Geometric)")
plt.grid()
plt.legend()
plt.ylim(0, 300000)
plt.show()
