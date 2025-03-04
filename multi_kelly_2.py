import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize

csv_filename = "data/test1.csv"

def objective(f, p, b):
    # objective: maximize expected log wealth
    # expected log return: p_i * ln(1 + b_i * f_i) + (1-p_i)*ln(1 - f_i)
    eps = 1e-9  # small constant
    term_win = np.log(np.clip(1 + b * f, eps, None)) # if win: wealth * (1 + b_i * f_i)
    term_lose = np.log(np.clip(1 - f, eps, None)) # if lose: wealth * wealth multiplies by (1 - f_i)
    total_log = np.sum(p * term_win + (1 - p) * term_lose)
    return -total_log # return the negative

def constraint_total(f):
    return 1 - np.sum(f) # sum of fractions <= 1

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)
    
    df = pd.read_csv(csv_path)
    p = df['Assessed Probability'].values 
    q = df['Bookmaker Probability'].values
    
    # b: decimal odds minus 1
    b = (1 - q) / q
    n = len(p)

    # f_i* = (p_i - q_i) / (1 - q_i), but only if p_i > q_i.
    f_unconstrained = np.maximum((p - q) / (1 - q), 0)
    
    # use scaled version as initial guess if the sum > 1.
    sum_unconstrained = np.sum(f_unconstrained)
    if sum_unconstrained <= 1:
        initial_guess = f_unconstrained
    else:
        initial_guess = f_unconstrained / sum_unconstrained

    bounds = [(0, 0.9999) for _ in range(n)]
    
    # constraint dictionary: sum(f) <= 1.
    constraints = [{'type': 'ineq', 'fun': constraint_total}]
    
    # SOLVE w/ SLSQP
    result = minimize(
        objective,
        initial_guess,
        args=(p, b),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )
    
    if result.success:
        optimal_f = result.x
        print("Optimal fractions for each bet:")
        for i, f_val in enumerate(optimal_f):
            print(f"  Bet {i+1}: {f_val*100:.2f}% of wealth")
        total_fraction = np.sum(optimal_f)
        print("Total fraction bet: {:.2f}%".format(total_fraction*100))
        
        expected_log_growth = -objective(optimal_f, p, b)
        print(f"Expected log growth: {expected_log_growth:.4f} ln(multiplier)")
        
        # arithmetic expected multiplier: E[multiplier]_i = p_i*(1+b_i*f_i) + (1-p_i)*(1-f_i)
        individual_expectations = p * (1 + b * optimal_f) + (1 - p) * (1 - optimal_f)
        portfolio_multiplier = np.prod(individual_expectations)
        portfolio_expected_return = portfolio_multiplier - 1
        print(f"Expected portfolio return (arithmetic): {portfolio_expected_return*100:.2f}%")
        
        log_returns = p * np.log(1 + b * optimal_f) + (1 - p) * np.log(1 - optimal_f)
        portfolio_expected_log_return = np.sum(log_returns)
        log_returns_sq = p * (np.log(1 + b * optimal_f))**2 + (1 - p) * (np.log(1 - optimal_f))**2
        portfolio_log_variance = np.sum(log_returns_sq) - portfolio_expected_log_return**2
        portfolio_log_std = np.sqrt(portfolio_log_variance)
        print(f"Portfolio expected log return: {portfolio_expected_log_return:.4f} ln(multiplier)")
        print(f"Portfolio log-return standard deviation: {portfolio_log_std:.4f}")
        
    else:
        print("Optimization failed:", result.message)

if __name__ == '__main__':
    main()
