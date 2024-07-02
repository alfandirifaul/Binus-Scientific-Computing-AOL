import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from scipy.optimize import curve_fit

# Read the data from the file
data = pd.read_excel('aol_data.xlsx')

# Extract month number and production data
time_periods = np.arange(1, len(data) + 1)
production_data = data.iloc[:, 1].values

# Define the variable to store the result
results_list = []

# Define the function to plot the data
def plot_data(plot_axis, actual_values, fitted_values, fitted_label="Fitted Model", graph_title="Production Trend", equation="", r2_value=None, mse=None, mae=None, rmse=None, display_axes=None):
    if display_axes is None:
        display_axes = plot_axis
    plot_axis.plot(time_periods, actual_values, label='Actual Production', color='blue')
    plot_axis.plot(time_periods, fitted_values, label=fitted_label, color='red')
    plot_axis.set_xlabel('Time Period')
    plot_axis.set_ylabel('Production')
    plot_axis.set_title(graph_title)
    plot_axis.legend()
    text = equation  + (f"\n$R^2$ = {r2_value:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}" if r2_value is not None else "")
    wrapped_text = '\n'.join([textwrap.fill(line, width=70) for line in text.split('\n')])
    display_axes.annotate(wrapped_text, xy=(1, 0), xycoords='axes fraction', fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

# Define the function to calculate the statistics
def compute_statistics(actual_values, fitted_values):
    r2_value = 1 - np.sum((actual_values - fitted_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
    mse = np.mean((actual_values - fitted_values) ** 2)
    mae = np.mean(np.abs(actual_values - fitted_values))
    rmse = np.sqrt(mse)
    return r2_value, mse, mae, rmse

# Define the function to create the polynomial fit
def fit_polynomial(plot_axis, display_axes=None):
    poly_degree = 4

    # Fit the polynomial model
    coeffs = np.polyfit(time_periods, production_data, poly_degree)
    polynomial_function = np.poly1d(coeffs)

    fitted_values = polynomial_function(time_periods)
    r2_score, mse, mae, rmse = compute_statistics(production_data, fitted_values)

    equation = "y = " + " + ".join([f"{coeffs[i]:.3f}x^{poly_degree-i}" for i in range(poly_degree+1)])
    equation = equation.replace("x^0", "").replace(" + -", " - ").replace(" 1x", " x")
    
    plot_data(plot_axis, production_data, fitted_values, f'Fitted Polynomial of Degree {poly_degree}', 'Polynomial Regression', equation, r2_score, mse, mae, rmse, display_axes)
    
    results_list.append(("Polynomial", r2_score, mse, mae, rmse, polynomial_function, fit_polynomial, coeffs))

# Define the function to create the exponential fit
def fit_exponential(plot_axis, display_axes=None):
    # Define the exponential function
    def exponential_function(x, a, b):
        return a * np.exp(b * x)

    # Initial guess for the parameters
    initial_guess = [1.0, 0.01]

    # Perform the fit
    popt, _ = curve_fit(exponential_function, time_periods, production_data, p0=initial_guess)
    a, b = popt

    fitted_values = exponential_function(time_periods, *popt)
    r2_score, mse, mae, rmse = compute_statistics(production_data, fitted_values)
    equation = f"y = {a:.3f} * exp({b:.3f} * x)"
    
    plot_data(plot_axis, production_data, fitted_values, 'Fitted Exponential Model', 'Exponential Model', equation, r2_score, mse, mae, rmse, display_axes)
    
    results_list.append(("Exponential", r2_score, mse, mae, rmse, lambda x: exponential_function(x, *popt), fit_exponential))

def fit_power(plot_axis, display_axes=None):
    def linearize_power(X, y):
        X_transformed = np.log(X)
        y_transformed = np.log(y)
        return X_transformed, y_transformed
    
    def power_reg(X_transformed, y_transformed):
        X_transpose = np.transpose(X_transformed)
        XtX = np.dot(X_transpose, X_transformed)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X_transpose, y_transformed)
        coefficients = np.dot(XtX_inv, Xty)
        residuals = y_transformed - np.dot(X_transformed, coefficients)
        return coefficients, residuals
    
    X = np.vstack([np.ones_like(time_periods), np.log(time_periods)]).T
    X_transformed, y_transformed = linearize_power(time_periods, production_data)
    
    # Fit the power model
    coefficients, _ = power_reg(X_transformed[:, np.newaxis], y_transformed)
    a = np.exp(coefficients[0])
    b = coefficients[1]
    
    def power_func(time_period):
        return a * np.power(time_period, b)
    
    fitted_values = power_func(time_periods)
    r2_score, mse, mae, rmse = compute_statistics(production_data, fitted_values)
    equation = f"y = {a:.3f} * x^{b:.3f}"
    plot_data(plot_axis, production_data, fitted_values, 'Fitted Power Model', 'Power Model', equation, r2_score, mse, mae, rmse, display_axes)
    results_list.append(("Power", r2_score, mse, mae, rmse, power_func, fit_power))

def draw_conclusion(plot_axis):
    # Select the model with the lowest MSE
    optimal_model = min(results_list, key=lambda x: x[2])
    text_content = f"Best Model: {optimal_model[0]}\n$R^2$ = {optimal_model[1]:.3f}\nMSE = {optimal_model[2]:.3f}\nMAE = {optimal_model[3]:.3f}\nRMSE = {optimal_model[4]:.3f}"
    plot_axis.text(0.5, 0.5, text_content, horizontalalignment='center', verticalalignment='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    plot_axis.axis('off')
    if optimal_model[0] == "Polynomial":
        print(f"Coefficients: {optimal_model[7]}")
    elif optimal_model[0] == "Exponential":
        print(f"Parameters: {optimal_model[5](np.array([0]))}")
    elif optimal_model[0] == "Power":
        print(f"Function: {optimal_model[5]}")
    print("Best Model:", optimal_model)

# Plotting all the models and conclusion
fig, ax = plt.subplots(4, 1, figsize=(10, 20))

fit_polynomial(ax[0])
fit_exponential(ax[1])
fit_power(ax[2])
draw_conclusion(ax[3])

# Display the plot
plt.tight_layout()
plt.show()
