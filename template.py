import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import curve_fit
import math
import textwrap

# Load the data
file_path = 'aol_data.xlsx'  
data = pd.read_excel(file_path)

# Extract month numbers and production values
months = np.arange(1, 145)
production_values = data.iloc[0].values

def plotData(ax, production_values, fitted_values, fitted_label_text="Fitted model", plot_title="Bag Production Trend", equation_text="", r_squared=None, mse=None, mae=None, rmse=None, display_axs=None):

    if display_axs is None:
        display_axs = ax
    ax.plot(months, production_values, label='Actual Production', color='blue')
    ax.plot(months, fitted_values, label=fitted_label_text, color='red')
    ax.set_xlabel('Month')
    ax.set_ylabel('Production')
    ax.set_title(plot_title)
    ax.legend()
    text_str = equation_text  + (f"\n$R^2$ = {r_squared:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}" if r_squared is not None else "")
    wrapped_text = '\n'.join([textwrap.fill(line, width=70) for line in text_str.split('\n')])
    display_axs.annotate(wrapped_text, xy=(1, 0), xycoords='axes fraction', fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

def calculate_statistics(y_actual, y_pred):
    residuals = y_actual - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
    r_squared = 1 - (ss_res / ss_tot)
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(mse)
    return r_squared, mse, mae, rmse

results = []

# Polynomial fit function
def polynomial_fit(ax,display_axs=None):
    # Define the polynomial degree
    degree = 4
    # Create the design matrix for polynomial regression
    X = np.vander(months, N=degree+1, increasing=True)
    def poly_reg(X, production_values):
        # Calculate the transpose of X
        X_transpose = np.transpose(X)
        # Calculate the product of X_transpose and X
        XtX = np.dot(X_transpose, X)
        # Calculate the inverse of XtX
        XtX_inv = np.linalg.inv(XtX)
        # Calculate the product of X_transpose and production_values
        Xty = np.dot(X_transpose, production_values)
        # Calculate the coefficients by multiplying XtX_inv and Xty
        coefficients = np.dot(XtX_inv, Xty)
        # Calculate the residuals
        residuals = production_values - np.dot(X, coefficients)
        return coefficients, residuals


    coefficients, residuals = poly_reg(X, production_values)
    # Calculate the fitted values
    fitted_values = np.dot(X, coefficients)
    # Calculate statistics
    r_squared, mse, mae, rmse = calculate_statistics(production_values, fitted_values)
    # Generate the equation text
    equation_text = "y = " + " + ".join([f"{coefficients[i]:.3f}x^{i}" for i in range(degree, -1, -1)])
    equation_text = equation_text.replace("x^0", "").replace(" + -", " - ").replace(" 1x", " x")
    plotData(ax, production_values, fitted_values, f'Fitted Polynomial of Degree {degree}', 'Polynomial Regression', equation_text, r_squared, mse, mae, rmse, display_axs)
    def poly_func(month):
        return sum(c * month**i for i, c in enumerate(coefficients))
    results.append(("Polynomial", r_squared, mse, mae, rmse, poly_func, polynomial_fit, coefficients))

# Exponential fit function
def exponential_fit(ax,display_axs=None):
    def linearize_exponential(X, y):
        y_transformed = np.log(y)
        return X, y_transformed
    
    def exp_reg(X, y_transformed):
        # Perform polynomial regression on the linearized data
        X_transpose = np.transpose(X)
        XtX = np.dot(X_transpose, X)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X_transpose, y_transformed)
        coefficients = np.dot(XtX_inv, Xty)
        residuals = y_transformed - np.dot(X, coefficients)
        return coefficients, residuals

    X = np.vander(months, N=2, increasing=True)
    X, y_transformed = linearize_exponential(X, production_values)
    
    # Fit the exponential model
    coefficients, residuals = exp_reg(X, y_transformed)
    a = np.exp(coefficients[0])
    b = coefficients[1]
    c = 0  # This model assumes no constant offset
    
    def exp_func(month):
        return a * np.exp(b * month) + c
    
    fitted_values = exp_func(months)
    r_squared, mse, mae, rmse = calculate_statistics(production_values, fitted_values)
    equation_text = f"y = {a:.3f} * exp({b:.3f} * x)"
    plotData(ax, production_values, fitted_values, 'Fitted Exponential Model', 'Exponential Model', equation_text, r_squared, mse, mae, rmse, display_axs)
    results.append(("Exponential", r_squared, mse, mae, rmse, exp_func, exponential_fit))

# Logarithmic fit function
def logarithmic_fit(ax,display_axs=None):
    def linearize_logarithmic(X, y):
        X_transformed = np.log(X[:, 1])
        return X_transformed.reshape(-1, 1), y
    
    def log_reg(X_transformed, y):
        X_transpose = np.transpose(X_transformed)
        XtX = np.dot(X_transpose, X_transformed)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X_transpose, y)
        coefficients = np.dot(XtX_inv, Xty)
        residuals = y - np.dot(X_transformed, coefficients)
        return coefficients, residuals

    X = np.vander(months, N=2, increasing=True)
    X_transformed, y = linearize_logarithmic(X, production_values)
    X_transformed = np.hstack([np.ones_like(X_transformed), X_transformed])
    
    # Fit the logarithmic model
    coefficients, residuals = log_reg(X_transformed, y)
    a = coefficients[0]
    b = coefficients[1]
    c = 0  # This model assumes no constant offset
    
    def log_func(month):
        return a + b * np.log(month)
    
    fitted_values = log_func(months)
    r_squared, mse, mae, rmse = calculate_statistics(production_values, fitted_values)
    equation_text = f"y = {a:.3f} + {b:.3f} * log(x)"
    plotData(ax, production_values, fitted_values, 'Fitted Logarithmic Model', 'Logarithmic Model', equation_text, r_squared, mse, mae, rmse, display_axs)
    results.append(("Logarithmic", r_squared, mse, mae, rmse, log_func, logarithmic_fit))

# Power fit function
def power_fit(ax,display_axs=None):
    def linearize_power(X, y):
        X_transformed = np.log(X[:, 1])
        y_transformed = np.log(y)
        return X_transformed.reshape(-1, 1), y_transformed
    
    def power_reg(X_transformed, y_transformed):
        X_transpose = np.transpose(X_transformed)
        XtX = np.dot(X_transpose, X_transformed)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X_transpose, y_transformed)
        coefficients = np.dot(XtX_inv, Xty)
        residuals = y_transformed - np.dot(X_transformed, coefficients)
        return coefficients, residuals
    
    X = np.vander(months, N=2, increasing=True)
    X_transformed, y_transformed = linearize_power(X, production_values)
    X_transformed = np.hstack([np.ones_like(X_transformed), X_transformed])
    
    # Fit the power model
    coefficients, residuals = power_reg(X_transformed, y_transformed)
    a = np.exp(coefficients[0])
    b = coefficients[1]
    
    def power_func(month):
        return a * np.power(month, b)
    
    fitted_values = power_func(months)
    r_squared, mse, mae, rmse = calculate_statistics(production_values, fitted_values)
    equation_text = f"y = {a:.3f} * x^{b:.3f}"
    plotData(ax, production_values, fitted_values, 'Fitted Power Model', 'Power Model', equation_text, r_squared, mse, mae, rmse, display_axs)
    results.append(("Power", r_squared, mse, mae, rmse, power_func, power_fit))

# Logistic fit function (retaining nonlinear optimization)
def logistic_fit(ax, display_axs=None):
    def logistic_func(month, L, k, x0):
        return L / (1 + np.exp(-k * (month - x0)))

    def fit_logistic_model(months, production_values):
        # Initial guess for the parameters
        initial_guess = [max(production_values), 0.1, np.median(months)]
        # Fit the logistic model
        popt, pcov = curve_fit(logistic_func, months, production_values, p0=initial_guess)
        return popt, pcov

    # Fit the logistic model
    popt, pcov = fit_logistic_model(months, production_values)

    # Calculate the fitted values
    fitted_values = logistic_func(months, *popt)
    
    # Calculate statistics
    r_squared, mse, mae, rmse = calculate_statistics(production_values, fitted_values)

    # Generate the equation text
    equation_text = f"y = {popt[0]:.3f} / (1 + exp(-{popt[1]:.3f} * (x - {popt[2]:.3f})))"
    
    plotData(ax, production_values, fitted_values, 'Fitted Logistic Model', 'Logistic Model', equation_text, r_squared, mse, mae, rmse, display_axs)

    def logistic_func_final(month):
        return popt[0] / (1 + np.exp(-popt[1] * (month - popt[2])))

    results.append(("Logistic", r_squared, mse, mae, rmse, logistic_func_final, logistic_fit))


def plot_conclusion(ax):
    best_model = results[0]
    text_str = f"Best Model: {best_model[0]}\n$R^2$ = {best_model[1]:.3f}\nMSE = {best_model[2]:.3f}\nMAE = {best_model[3]:.3f}\nRMSE = {best_model[4]:.3f}"
    ax.text(0.5, 0.5, text_str, horizontalalignment='center', verticalalignment='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    ax.axis('off')
    if best_model[0] == "Polynomial":
        print(f"Coefficients: {best_model[7]}")
    print("Best Model:", best_model)

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(20, 12))

# Fit models and plot in subplots
polynomial_fit(axs[0, 0])
exponential_fit(axs[0, 1])
logarithmic_fit(axs[1, 0])
power_fit(axs[1, 1])
logistic_fit(axs[2, 0])

results.sort(key=lambda x: (x[2], x[3], x[4], -x[1]))  # Sort by MSE, MAE, RMSE, then -R^2
# Plot the conclusion in the last subplot
plot_conclusion(axs[2, 1])

# Adjust layout
plt.tight_layout()
plt.show()

# Create subplots for Taylor Series comparison
plt.close()
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
best_model = results[0]
best_model_func = best_model[5]
best_model_fit = best_model[6]
results = []
def calculate_derivative(func, order):
        x = sp.symbols('x')
        derivative = func(x)
        for _ in range(order):
            derivative = sp.diff(derivative, x)
        return sp.lambdify(x, derivative, 'numpy')
def taylor_series(ax, func_to_model, a=0, taylor_order=5, display_axs=None):

    x_values = np.array(months)
    y_values = func_to_model(x_values)
    # Calculate the Taylor series expansion manually
    taylor_series_expansion = np.zeros_like(x_values, dtype=np.float64)

    for n in range(taylor_order + 1):
        nth_derivative_func = calculate_derivative(func_to_model, n)
        nth_derivative_at_a = nth_derivative_func(a)
        taylor_series_expansion += (nth_derivative_at_a / math.factorial(n)) * (x_values - a)**(n)

    y_taylor_values = taylor_series_expansion

    r_squared, mse, mae, rmse = calculate_statistics(production_values, y_taylor_values)

    # Generate the equation text for display
    equation_text = " + ".join([f"{calculate_derivative(func_to_model, n)(a) / math.factorial(n):.3f}(x - {a})^{n}" for n in range(taylor_order + 1)])
    plotData(ax, production_values, y_taylor_values, f'Taylor series (order {taylor_order})', 'Taylor Series Approximation', equation_text, r_squared, mse, mae, rmse, display_axs)

taylor_series(ax[0,0], best_model_func, display_axs=ax[0,1])
best_model_fit(ax[1,0], display_axs=ax[1,1])
explanation_text_taylor1 = f"The numerical approximation done by the taylor series converges to the original model, taking the taylor series of a polynomial function will eventually converge back to the function itself"
explanation_text_taylor1 = '\n'.join([textwrap.fill(line, width=70) for line in explanation_text_taylor1.split('\n')])
explanation_text_taylor2 = f"Since the most effective model is the polynomial function, it is also in the most efficient form and will not be required to be converted to another form with taylor series"
explanation_text_taylor2 = '\n'.join([textwrap.fill(line, width=70) for line in explanation_text_taylor2.split('\n')])
ax[0,1].annotate(explanation_text_taylor1, xy=(0.5, 0.7), xycoords='axes fraction', fontsize=14, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
ax[0,1].axis('off')
ax[1,1].annotate(explanation_text_taylor2, xy=(0.5, 0.7), xycoords='axes fraction', fontsize=14, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
ax[1,1].axis('off')

plt.show()



#QUESTION 3
plt.close()


INITIAL_GUESS = 150
PRODUCTION_LIMIT = 25000
LEAD_TIME = 13
root_func = lambda x: best_model_func(x) - PRODUCTION_LIMIT

def newtonRhapson(fx, f_prime, x0, tol=1e-6, max_iter=10000):
    x = x0
    for i in range(max_iter):
        fx_val = fx(x)
        print(f"Iteration {i+1} Month: {x:.3f}, Production: {(fx_val+PRODUCTION_LIMIT):.3f}")
        if abs(fx_val) < tol:
            return x
        x = x - fx_val / f_prime(x)
    print("Failed to converge")
    return x

month_x = newtonRhapson(root_func, calculate_derivative(root_func,1), INITIAL_GUESS)
build_x = month_x - LEAD_TIME
print(f"EGIER needs to start building the new warehouse in month: {(month_x-LEAD_TIME):.3f}")
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
future_months = np.arange(1, 200)
predicted_production = best_model_func(future_months)
axs[0].plot(months, production_values, label='Actual Production', color='blue')
axs[0].plot(future_months, predicted_production, label='Predicted Production', color='red')
axs[0].axhline(y=PRODUCTION_LIMIT, color='green', linestyle='--', label='Production Limit (25,000 bags)')
axs[0].axvline(x=month_x, color='orange', linestyle='--', label='Exceed Month')
axs[0].axvline(x=build_x, color='purple', linestyle='--', label='Start Building Month')
axs[0].set_xlabel('Month')
axs[0].set_ylabel('Production')
axs[0].set_title('Bag Production Prediction and Warehouse Planning')
axs[0].legend()

axs[1].axis('off')
disp_text = f"The data span consists of 144 months, the model predicts that EGIER will exceed the production limit of {PRODUCTION_LIMIT} bags in month {month_x:.3f}. To ensure that the new warehouse is ready before the production exceeds the limit, EGIER should start building the new warehouse in month {build_x:.3f}."
wrapped_text = '\n'.join([textwrap.fill(line, width=70) for line in disp_text.split('\n')])
axs[1].annotate(wrapped_text, xy=(0.5, 0.7), xycoords='axes fraction', fontsize=14, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

def month_to_year_month(add_month,start_year,start_month):
    year = start_year + (start_month + add_month - 1) // 12
    month = (start_month + add_month - 1) % 12 + 1
    return year, month

def month_to_text(month):
    if month == 1:
        return "January"
    elif month == 2:
        return "February"
    elif month == 3:
        return "March"
    elif month == 4:
        return "April"
    elif month == 5:
        return "May"
    elif month == 6:
        return "June"
    elif month == 7:
        return "July"
    elif month == 8:
        return "August"
    elif month == 9:
        return "September"
    elif month == 10:
        return "October"
    elif month == 11:
        return "November"
    elif month == 12:
        return "December"
    

predicted_month_year = month_to_year_month(int(month_x), 2018, 1)
build_month_year=  month_to_year_month(int(build_x), 2018, 1)
predicted_month_name = month_to_text(predicted_month_year[1])
build_month_name = month_to_text(build_month_year[1])
disp_text_2 = f"Since the data starts from January 2018, the production is predicted to exceed the limit in {predicted_month_name} of {predicted_month_year[0]}. To ensure the new warehouse is ready before the production exceeds the limit, EGIER should start building the new warehouse in {build_month_name} of {build_month_year[0]}."
wrapped_text_2 = '\n'.join([textwrap.fill(line, width=70) for line in disp_text_2.split('\n')])
axs[1].annotate(wrapped_text_2, xy=(0.5, 0.3), xycoords='axes fraction', fontsize=14, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
plt.show()


