# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:00:00 2026

@author: lfval
"""

import functions
import numpy as np
import pandas as pd
from config import OUTPUTS
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution


if __name__ == '__main__':

    ### --- Question 1 --- ###
    
    print("=" * 60)
    print("Question 1")
    print("=" * 60)

    tol = 1e-4

    # Nelder-Mead
    res_nm = minimize(functions.f_q1, x0=0.0, method='Nelder-Mead',options={'xatol': tol, 'fatol': tol})

    # L-BFGS-B
    res_lbfgsb = minimize(functions.f_q1, x0=0.0, method='L-BFGS-B',bounds=[(0, 10)],options={'ftol': tol, 'gtol': tol})

    # Differential Evolution (global optimizer)
    res_de = differential_evolution(functions.f_q1, bounds=[(0, 10)],tol=tol, atol=tol, seed=13051905)

    print(f"Nelder-Mead:            x* = {res_nm.x[0]:.6f},  f(x*) = {res_nm.fun:.6f}")
    print(f"L-BFGS-B:               x* = {res_lbfgsb.x[0]:.6f},  f(x*) = {res_lbfgsb.fun:.6f}")
    print(f"Differential Evolution: x* = {res_de.x[0]:.6f},  f(x*) = {res_de.fun:.6f}")

    # Item (b)
    starting_points = np.linspace(0, 10, 50)
    results_nm     = []
    results_lbfgsb = []

    for x0 in starting_points:
        r_nm = minimize(functions.f_q1, x0=x0, method='Nelder-Mead',
                        options={'xatol': tol, 'fatol': tol})
        r_lb = minimize(functions.f_q1, x0=x0, method='L-BFGS-B',
                        bounds=[(0, 10)],
                        options={'ftol': tol, 'gtol': tol})
        results_nm.append(r_nm.fun)
        results_lbfgsb.append(r_lb.fun)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(linestyle='--', zorder=0)
    ax.plot(starting_points, results_nm, label='Nelder-Mead', color='blue', linewidth=1.2, zorder=2)
    ax.plot(starting_points, results_lbfgsb, label='L-BFGS-B', color='red', linewidth=1.2, zorder=2)
    ax.axhline(y=res_de.fun, color='green', linestyle='--', label='Diff. Evolution (global)', zorder=2)
    ax.set_xlabel('Starting Point $x_0$')
    ax.set_ylabel('$f(x^*)$')
    ax.set_title('Sensitivity of Optimum to Starting Point')
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(OUTPUTS / 'chart_q1_starting_points.pdf')
    plt.close()

    # Plot the function
    x_grid = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(linestyle='--', zorder=0)
    ax.plot(x_grid, functions.f_q1(x_grid), color='black', linewidth=1.2, zorder=2)
    ax.scatter([res_de.x[0]], [res_de.fun], color='red', s=60, zorder=5, label=f'Global min: x*={res_de.x[0]:.4f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x) = x \\sin(5x)$')
    ax.set_title('$f(x) = x \\sin(5x)$ on $[0, 10]$')
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(OUTPUTS / 'chart_q1_function.pdf')
    plt.close()



    ### --- Question 2 --- ###
    
    print("\n" + "=" * 60)
    print("Question 2")
    print("=" * 60)

    # Item (b)
    theta_1 = np.array([0.0, 0.0, 0.0, 0.0])
    
    g_theta_1 = functions.g_q2(theta_1)
    print(f"\nItem b) g(theta_1) = g((0,0,0,0)) = {g_theta_1:.4f}")

    # Items (c-e)
    recorder = functions.CallbackRecorder()
    res_q2 = minimize(functions.g_q2, 
                      x0       = np.array([1.0, 1.0, 1.0, 1.0]),
                      method   = 'Nelder-Mead',
                      callback = recorder,
                      options  = {'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})

    theta_hat = res_q2.x
    n_iter = len(recorder.g_vals)

    print(f"\nItem c/e) Estimate theta_hat = ({theta_hat[0]:.6f}, {theta_hat[1]:.6f}, {theta_hat[2]:.6f}, {theta_hat[3]:.6f})")
    print(f"          g(theta_hat) = {res_q2.fun:.10f}")
    print(f"          Iterations: {n_iter}")

    functions.plot_q2_convergence(recorder)
    print("          Convergence plots saved to outputs/")


    ### --- Question 3 --- ###
    
    print("\n" + "=" * 60)
    print("Question 3")
    print("=" * 60)

    analytical = 1 / np.sqrt(np.pi) 

    print(f"\nAnalytical solution: E[max(X,Y)] = 1/sqrt(pi) = {analytical:.6f}")

    # Item (a)
    print("\n  Gauss-Hermite quadrature:")
    for n_nodes in [5, 10, 20, 50, 100]:
        gh_result = functions.gauss_hermite_expected_max(n_nodes=n_nodes)
        gh_error  = abs(gh_result - analytical)
        print(f"    n={n_nodes:3d} nodes: E[max(X,Y)] = {gh_result:.6f},  |error| = {gh_error:.2e}")

    # Item (b)
    print("\n  Monte Carlo integration:")
    for n_draws in [1_000, 10_000, 100_000, 1_000_000]:
        mc_result = functions.monte_carlo_expected_max(n_draws=n_draws, seed=42)
        mc_error  = abs(mc_result - analytical)
        print(f"    n={n_draws:>9,} draws: E[max(X,Y)] = {mc_result:.6f},  |error| = {mc_error:.2e}")



    ### --- Question 4 --- ###
    
    print("\n" + "=" * 60)
    print("Question 4")
    print("=" * 60)

    nodes_list = [3, 5, 10, 15, 20]

    # Analytical solutions
    analytical_a = 0.5                                     
    analytical_b = np.sin(1) - 1 * np.cos(1) 
    analytical_c = np.pi / 4                 

    funcs     = [functions.f_q4a, functions.f_q4b, functions.f_q4c]
    labels    = ['int x dx', 'int x*sin(x) dx', 'int sqrt(1-x^2) dx']
    analytics = [analytical_a, analytical_b, analytical_c]
    names     = ['a', 'b', 'c']

    q4_results = []
    for func, label, true_val, name in zip(funcs, labels, analytics, names):
        print(f"\n  {label}  (analytical = {true_val:.6f})")
        row = {}
        for n in nodes_list:
            approx = functions.trapezoid(func, 0, 1, n)
            error  = abs(approx - true_val)
            row[f'n={n}'] = approx
            print(f"    n={n:2d}: approx = {approx:.6f},  |error| = {error:.2e}")
        q4_results.append(pd.Series(row, name=label))

    q4_table = pd.DataFrame(q4_results)
    q4_table.to_excel(OUTPUTS / 'table_q4_trapezoid.xlsx')


    ### --- Question 5 --- ###
    
    print("\n" + "=" * 60)
    print("Question 5")
    print("=" * 60)

    h_steps = [0.001, 0.005, 0.01, 0.05]

    q5_funcs   = [functions.f_q5a, functions.f_q5b, functions.f_q5c]
    q5_primes  = [functions.f_q5a_prime, functions.f_q5b_prime, functions.f_q5c_prime]
    q5_points  = [5, 10, 1]
    q5_labels  = ['f(x)=x^2 at x=5', 'f(x)=log(x) at x=10', 'f(x)=x*sin(x) at x=1']

    for func, fprime, x0, label in zip(q5_funcs, q5_primes, q5_points, q5_labels):
        analytical_val = fprime(x0)
        print(f"\n  {label}  (analytical f'(x) = {analytical_val:.6f})")

        rows = []
        for h in h_steps:
            cd2 = functions.centered_diff_2pt(func, x0, h)
            cd4 = functions.centered_diff_4pt(func, x0, h)
            err2 = abs(cd2 - analytical_val)
            err4 = abs(cd4 - analytical_val)
            rows.append({'h': h,
                         '2-pt centered': cd2,
                         '|error| 2-pt': err2,
                         '4-pt centered': cd4,
                         '|error| 4-pt': err4})
            print(f"    h={h:.3f}: 2pt = {cd2:.8f} (err={err2:.2e}),  4pt = {cd4:.8f} (err={err4:.2e})")

        df = pd.DataFrame(rows)
        safe_name = label.replace("^", "").replace("*", "").replace("(", "").replace(")", "").replace(" ", "_")
        df.to_excel(OUTPUTS / f'table_q5_{safe_name}.xlsx', index=False)