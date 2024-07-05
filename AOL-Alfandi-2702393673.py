import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.optimize import fsolve
from sympy import symbols, diff, lambdify
import json
import math

def load_data(data_production):
    data = pd.DataFrame(data_production)
    return data.values.flatten()

def generate_time_points(data_length):
    return np.arange(1, data_length + 1)

def fit_polynomial(months, production_data, degree):
    p = Polynomial.fit(months, production_data, degree)
    poly_coefficients = p.convert().coef
    poly_func = np.poly1d(poly_coefficients[::-1])
    return poly_func

def calculate_taylor_series(poly_func, months, degree):
    x = symbols('x')
    a = np.median(months)
    taylor_approx = poly_func(a)
    for i in range(1, degree + 1):
        derivative = diff(poly_func(x), x, i).subs(x, a)
        taylor_approx += (derivative / math.factorial(i)) * (x - a) ** i
    taylor_func = lambdify(x, taylor_approx, 'numpy')

    ply_equation = f"{poly_func[0]:.3f} + {poly_func[1]:.3f}*x + {poly_func[2]:.3f}*x**2 + {poly_func[3]:.3f}*x**3 + {poly_func[4]:.3f}*x**4 + {poly_func[5]:.3f}*x**5"
    taylorSeries = sp.series(ply_equation, x, 0, 6)
    print(f"Taylor series expansion: {taylorSeries}")
    return taylor_func

def find_breach_month(poly_func, months, threshold):
    def production_threshold(month):
        return poly_func(month) - threshold
    breach_month = fsolve(production_threshold, months[-1])[0]

    return breach_month

def plot_data(months, production_data, poly_func, taylor_func, breach_month, start_building_month, degree, threshold):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Plot the polynomial fit
    axs[0].plot(months, production_data, label='Actual Production')
    axs[0].plot(months, poly_func(months), label=f'Polynomial Fit (degree {degree})', linestyle='--')
    axs[0].set_xlabel('Month')
    axs[0].set_ylabel('Production')
    axs[0].legend()
    axs[0].set_title('Monthly Bag Production with Polynomial Trend')

    # Plot the Taylor series approximation
    axs[1].plot(months, production_data, label='Actual Production')
    axs[1].plot(months, poly_func(months), label=f'Polynomial Fit (degree {degree})', linestyle='--')
    axs[1].plot(months, taylor_func(months), label='Taylor Series Approximation', linestyle='-.')
    axs[1].set_xlabel('Month')
    axs[1].set_ylabel('Production')
    axs[1].legend()
    axs[1].set_title('Polynomial Fit vs Taylor Series Approximation')

    # Plot the prediction of when production will exceed 25,000 bags
    axs[2].plot(months, production_data, label='Actual Production')
    axs[2].axhline(y=threshold, color='r', linestyle='--', label='Threshold (25,000 bags)')
    if breach_month:
        axs[2].axvline(x=breach_month, color='g', linestyle='--', label=f'Exceeds 25,000 bags at month {breach_month:.1f}')
        axs[2].axvline(x=start_building_month, color='b', linestyle='--', label=f'Start building new warehouse at month {start_building_month:.1f}')
    axs[2].set_xlabel('Month')
    axs[2].set_ylabel('Production')
    axs[2].legend()
    axs[2].set_title('Production Exceeding 25,000 Bags and Warehouse Planning')

    plt.tight_layout()
    plt.show()

def month_to_text(month):
    month = int(month % 12)
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
    

def main():
    # Load data from JSON
    json_data = {
        "months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144],
        "production_values": [1863, 1614, 2570, 1685, 2101, 1811, 2457, 2171, 2134, 2502, 2358, 2399, 2048, 2523, 2086, 2391, 2150, 2340, 3129, 2277, 2964, 2997, 2747, 2862, 3405, 2677, 2749, 2755, 2963, 3161, 3623, 2768, 3141, 3439, 3601, 3531, 3477, 3376, 4027, 3175, 3274, 3334, 3964, 3649, 3502, 3688, 3657, 4422, 4197, 4441, 4736, 4521, 4485, 4644, 5036, 4876, 4789, 4544, 4975, 5211, 4880, 4933, 5079, 5339, 5232, 5520, 5714, 5260, 6110, 5334, 5988, 6235, 6365, 6266, 6345, 6118, 6497, 6278, 6638, 6590, 6271, 7246, 6584, 6594, 7092, 7326, 7409, 7976, 7959, 8012, 8195, 8008, 8313, 7791, 8368, 8933, 8756, 8613, 8705, 9098, 8769, 9544, 9050, 9186, 10012, 9685, 9966, 10048, 10244, 10740, 10318, 10393, 10986, 10635, 10731, 11749, 11849, 12123, 12274, 11666, 11960, 12629, 12915, 13051, 13387, 13309, 13732, 13162, 13644, 13808, 14101, 13992, 15191, 15018, 14917, 15046, 15556, 15893, 16388, 16782, 16716, 17033, 16896, 17689 ]
    }

    data_load = json_data
    data_read = pd.DataFrame(data_load['production_values'], columns=['production_values'])

    months = np.arange(1, 145)
    production_values = data_read['production_values'].values

    degree = 6
    threshold = 25000

    production_data = load_data(production_values)
    months = generate_time_points(len(production_data))
    poly_func = fit_polynomial(months, production_data, degree)
    taylor_func = calculate_taylor_series(poly_func, months, degree)
    breach_month = find_breach_month(poly_func, months, threshold)
    
    start_building_month = int(breach_month - 13)
    print("Month when production will exceed 25,000 bags:", breach_month)
    print("Month to start building a new warehouse:", month_to_text(start_building_month))
        
    plot_data(months, production_data, poly_func, taylor_func, breach_month, start_building_month, degree, threshold)


if __name__ == "__main__":
    main()
