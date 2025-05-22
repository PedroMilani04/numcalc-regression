import numpy as np
import matplotlib.pyplot as plt

# --- data ---
years = np.array([1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2022])
x_original = years - 1940
y = np.array([12637, 24515, 54055, 71188, 129733, 167966, 182023, 207610, 225668])
n = len(x_original)

# dot product
def prod_escalar(a, b):
    return np.dot(a, b)

# base vectors
v0 = np.ones(n)
v1 = x_original
v2 = v1**2
v3 = v1**3

# -------------------- linear model --------------------
A_lin = np.array([
    [prod_escalar(v0, v0), prod_escalar(v0, v1)],
    [prod_escalar(v1, v0), prod_escalar(v1, v1)]
])
b_lin = np.array([
    prod_escalar(v0, y),
    prod_escalar(v1, y)
])
a0_lin, a1_lin = np.linalg.solve(A_lin, b_lin)
f_lin = lambda x: a0_lin + a1_lin * x

# -------------------- quadratic model --------------------
A_quad = np.array([
    [prod_escalar(v0, v0), prod_escalar(v0, v1), prod_escalar(v0, v2)],
    [prod_escalar(v1, v0), prod_escalar(v1, v1), prod_escalar(v1, v2)],
    [prod_escalar(v2, v0), prod_escalar(v2, v1), prod_escalar(v2, v2)]
])
b_quad = np.array([
    prod_escalar(v0, y),
    prod_escalar(v1, y),
    prod_escalar(v2, y)
])
a0_q, a1_q, a2_q = np.linalg.solve(A_quad, b_quad)
f_quad = lambda x: a0_q + a1_q*x + a2_q*x**2

# -------------------- cubic model --------------------
A_cub = np.array([
    [prod_escalar(v0, v0), prod_escalar(v0, v1), prod_escalar(v0, v2), prod_escalar(v0, v3)],
    [prod_escalar(v1, v0), prod_escalar(v1, v1), prod_escalar(v1, v2), prod_escalar(v1, v3)],
    [prod_escalar(v2, v0), prod_escalar(v2, v1), prod_escalar(v2, v2), prod_escalar(v2, v3)],
    [prod_escalar(v3, v0), prod_escalar(v3, v1), prod_escalar(v3, v2), prod_escalar(v3, v3)]
])
b_cub = np.array([
    prod_escalar(v0, y),
    prod_escalar(v1, y),
    prod_escalar(v2, y),
    prod_escalar(v3, y)
])
a0_c, a1_c, a2_c, a3_c = np.linalg.solve(A_cub, b_cub)
f_cub = lambda x: a0_c + a1_c*x + a2_c*x**2 + a3_c*x**3

# -------------------- exponential model --------------------
y_exp = np.log(y)
A_exp = np.array([
    [prod_escalar(v0, v0), prod_escalar(v0, v1)],
    [prod_escalar(v1, v0), prod_escalar(v1, v1)]
])
b_exp = np.array([
    prod_escalar(v0, y_exp),
    prod_escalar(v1, y_exp)
])
a0_e, a1_e = np.linalg.solve(A_exp, b_exp)
a_exp = np.exp(a0_e)
b_exp = np.exp(a1_e)
f_exp = lambda x: a_exp * (b_exp**x)

# -------------------- hyperbolic model --------------------
y_hip = 1 / y
A_hip = np.array([
    [prod_escalar(v0, v0), prod_escalar(v0, v1)],
    [prod_escalar(v1, v0), prod_escalar(v1, v1)]
])
b_hip = np.array([
    prod_escalar(v0, y_hip),
    prod_escalar(v1, y_hip)
])
a0_h, a1_h = np.linalg.solve(A_hip, b_hip)
f_hip = lambda x: 1 / (a0_h + a1_h * x)

# -------------------- geometric model (power) --------------------
# avoid log(0)
x_geo = x_original[x_original > 0]
y_geo = y[x_original > 0]
X_geo = np.log(x_geo)
Y_geo = np.log(y_geo)
v0g = np.ones(len(X_geo))
v1g = X_geo
A_geo = np.array([
    [prod_escalar(v0g, v0g), prod_escalar(v0g, v1g)],
    [prod_escalar(v1g, v0g), prod_escalar(v1g, v1g)]
])
b_geo = np.array([
    prod_escalar(v0g, Y_geo),
    prod_escalar(v1g, Y_geo)
])
A_g, B_g = np.linalg.solve(A_geo, b_geo)
a_geom = np.exp(A_g)
b_geom = B_g
f_geom = lambda x: a_geom * x**b_geom

# calculate predictions for 2030 (90 years since 1940)
x_2030 = 90
predictions = {
    "Linear": f_lin(x_2030),
    "Quadratic": f_quad(x_2030),
    "Cubic": f_cub(x_2030),
    "Exponential": f_exp(x_2030),
    "Hyperbolic": f_hip(x_2030),
    "Geometric": f_geom(x_2030)
}

# calculate sse for each model
sse = {
    "Linear": np.sum((y - f_lin(x_original))**2),
    "Quadratic": np.sum((y - f_quad(x_original))**2),
    "Cubic": np.sum((y - f_cub(x_original))**2),
    "Exponential": np.sum((y - f_exp(x_original))**2),
    "Hyperbolic": np.sum((y - f_hip(x_original))**2),
    "Geometric": np.sum((y - f_geom(x_original))**2)
}

# find best model (lowest sse)
best_model = min(sse.items(), key=lambda x: x[1])[0]

print("\npredictions for 2030 (90 years since 1940):")
print("-" * 50)
for model, pred in predictions.items():
    print(f"{model:12} → {pred:,.0f} inhabitants")

print("\nsum of squared errors (sse) for each model:")
print("-" * 50)
for model, error in sse.items():
    print(f"{model:12} → {error:,.2e}")

print(f"\nbest model: {best_model} (lowest sse)")

# -------------------- plot --------------------
x_plot = np.linspace(0, 90, 300)

plt.figure(figsize=(12, 8))
plt.scatter(x_original, y, color='black', label="Real data")
plt.plot(x_plot, f_lin(x_plot), label="Linear", color='blue')
plt.plot(x_plot, f_quad(x_plot), label="Quadratic", color='green')
plt.plot(x_plot, f_cub(x_plot), label="Cubic", color='purple')
plt.plot(x_plot, f_exp(x_plot), label="Exponential", color='orange')
plt.plot(x_plot, f_hip(x_plot), label="Hyperbolic", color='red')
plt.plot(x_plot, f_geom(x_plot), label="Geometric (a·x^b)", color='brown')

plt.xlabel("Years since 1940")
plt.ylabel("Population")
plt.title("Comparison of Least Squares Fits")
plt.ylim(0, 300000)
plt.grid(True)
plt.legend()
plt.show()