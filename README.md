# Population Forecast with Least Squares Regression

This project applies multiple regression models using the **Least Squares Method** to fit population data and forecast future values. Various models—Linear, Quadratic, Cubic, Exponential, Hyperbolic, and Geometric—are compared based on their error and accuracy (R²) to determine the best fit.

---

## Project Overview

This project focuses on forecasting population values using historical census data. The pipeline includes:

1. **Model Fitting**  
   Implements least squares regression to fit different models to the population data across time (from 1940 to 2022).

2. **Model Evaluation**  
   Calculates **Sum of Squared Errors (SSE)**, **Total Sum of Squares (TSS)**, and **R²** for each model.  
   The model with the highest R² is selected as the best fit.

3. **Forecasting**  
   The fitted models are used to predict population for the year **2030**, which corresponds to 90 years since 1940.

4. **Visualization**  
   A comparative plot shows how each model fits the historical data and extrapolates to 2030.

---

## Key Features

1. **Multiple Regression Models**  
   The project includes:
   - **Linear Model**: \( f(x) = a_0 + a_1x \)
   - **Quadratic Model**: \( f(x) = a_0 + a_1x + a_2x^2 \)
   - **Cubic Model**: \( f(x) = a_0 + a_1x + a_2x^2 + a_3x^3 \)
   - **Exponential Model**: \( f(x) = ae^{bx} \)
   - **Hyperbolic Model**: \( f(x) = \frac{1}{a + bx} \)
   - **Geometric (Power) Model**: \( f(x) = ax^b \)

2. **Numerical Stability**  
   All models are fitted using custom-built normal equations via NumPy’s linear algebra solver, ensuring precision and independence from scikit-learn.

3. **Model Comparison with R²**  
   Evaluates models using R² (coefficient of determination), allowing a robust comparison of how well each model explains the variance in the data.

4. **Visual Analysis**  
   Generates a comparative plot to help interpret model performance and reliability of future projections.

---

## Technical Steps

- **Preprocessing**
  - Normalize years as "years since 1940"
  - Remove zero values for logarithmic models

- **Model Construction**
  - Assemble design matrices for each model
  - Solve normal equations using `np.linalg.solve`
  - Define prediction functions

- **Forecasting**
  - Predict population in 2030 (i.e., 90 years from 1940) for each model

- **Evaluation**
  - Compute SSE, TSS, and R² for each model
  - Identify and print the best-fitting model

- **Visualization**
  - Scatter plot of real data
  - Fitted curves for all models
  - Clearly labeled axes and legend

---

## Conclusion

This project demonstrates how different regression models behave on the same dataset and how statistical metrics like **R²** and **SSE** can guide model selection.  
The modular approach allows straightforward expansion to new data, additional models, or adaptations for other forecasting problems.

---
