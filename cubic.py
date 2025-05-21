import numpy as np
import matplotlib.pyplot as plt

# data
years = np.array([1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2022])
x = years - 1940
y = np.array([12637, 24515, 54055, 71188, 129733, 167966, 182023, 207610, 225668])
n = len(x)

# dot product (escalar)
def dot_product(a, b):
    return np.dot(a, b)

# basis vectors
v0 = np.ones(n)     # 1
v1 = x              # x
v2 = x**2           # x²
v3 = x**3           # x³

# build matrix A explicitly
A = np.array([
    [dot_product(v0, v0), dot_product(v0, v1), dot_product(v0, v2), dot_product(v0, v3)],
    [dot_product(v1, v0), dot_product(v1, v1), dot_product(v1, v2), dot_product(v1, v3)],
    [dot_product(v2, v0), dot_product(v2, v1), dot_product(v2, v2), dot_product(v2, v3)],
    [dot_product(v3, v0), dot_product(v3, v1), dot_product(v3, v2), dot_product(v3, v3)],
])

# build vector b 
b_vec = np.array([
    dot_product(v0, y),
    dot_product(v1, y),
    dot_product(v2, y),
    dot_product(v3, y)
])

# solve system
coef = np.linalg.solve(A, b_vec)
print("Coefficients found:")
print("a0 = {:.3f}, a1 = {:.3f}, a2 = {:.3f}, a3 = {:.6f}".format(*coef))

# fitted function
def f_approx(x):
    return coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3

# prediction for 2030 (x = 2030 - 1940 = 90)
x_pred = 90
print("Estimated population for 2030:", f_approx(x_pred))

# plot
x_plot = np.linspace(0, 90, 300)
y_plot = f_approx(x_plot)

plt.scatter(x, y, label="Actual data", color='black')
plt.plot(x_plot, y_plot, label="Cubic polynomial fit", color='green')
plt.xlabel("Years since 1940")
plt.ylabel("Population")
plt.title("Least Squares Fit (degree 3)")
plt.grid()
plt.legend()
plt.show()
