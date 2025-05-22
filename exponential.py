import numpy as np
import matplotlib.pyplot as plt

# data
anos = np.array([1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2022])
x = anos - 1940
y = np.array([12637, 24515, 54055, 71188, 129733, 167966, 182023, 207610, 225668])
n = len(x)

# transform y to ln(y)
Y = np.log(y)

# base vectors
v0 = np.ones(n)
v1 = x

# scalar product
def prod_escalar(a, b):
    return np.dot(a, b)

# build matrix A (2x2)
A = np.array([
    [prod_escalar(v0, v0), prod_escalar(v0, v1)],
    [prod_escalar(v1, v0), prod_escalar(v1, v1)]
])

# vector b
b_vec = np.array([
    prod_escalar(v0, Y),
    prod_escalar(v1, Y)
])

# solve system
coef_ln = np.linalg.solve(A, b_vec)
a0_ln, a1_ln = coef_ln

# recover real a and b
a = np.exp(a0_ln)
b = np.exp(a1_ln)

print("Found coefficients:")
print(f"a0 = ln(a) = {a0_ln:.4f}, a1 = ln(b) = {a1_ln:.4f}")
print(f"a = e^a0 = {a:.4f}, b = e^a1 = {b:.4f}")

# fitted function: f(x) = a * b^x
def f_exp(x):
    return a * (b ** x)

# prediction for 2030 (x = 90)
x_pred = 90
print("Estimated population for 2030:", f_exp(x_pred))

# plot
x_plot = np.linspace(0, 90, 300)
y_plot = f_exp(x_plot)

plt.scatter(x, y, label="Real data", color='black')
plt.plot(x_plot, y_plot, label="Exponential fit", color='red')
plt.xlabel("Years since 1940")
plt.ylabel("Population")
plt.title("Least Squares Fit (Exponential)")
plt.grid()
plt.legend()
plt.show()
