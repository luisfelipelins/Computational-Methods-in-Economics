# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

import os
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import *
from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel

# We'll consider the following exercise to decompose the impacts of each mechanism we add to the model.
# 1. Solve the analytical baby model (representative household)
# 2. Solve the numerical model without shocks to z but with shocks to l
# 3. Solve the numerical model with shocks to z but without shocks to l
# 4. Solve the numerical model with shocks to both z and l
# In all scenarios we'll estimate a prior and a posterior model. Assuming
# we were in an equilibirum with β=2 and move to β=3, how does the economy respond?


def build_calib(rh_N):
    return TypeCalibParameters(rh_N             = rh_N ,
                               rh_r             = 2    ,
                               rh_c             = 0    ,
                               sup_side_eq_eps  = 1e-5 ,
                               vfi_lb           = 0    ,
                               vfi_ubmul        = 15   ,
                               vfi_N            = 500  ,
                               vfi_eps          = 1e-5 ,
                               vfi_howard_steps = 20   ,
                               gmc_eps          = 1e-3 ,
                               kmc_eps          = 1e-3 ,
                               p_init_guess     = 1    ,
                               r_init_guess     = 0.01 ,
                               inner_loop_eps   = 1e-5 ,
                               inner_loop_p_lb  = 0.1  ,
                               inner_loop_p_ub  = 1.2  ,
                               inner_loop_marg  = 0.3  ,
                               outer_loop_eps   = 1e-5 ,
                               outer_loop_r_lb  = 0.001)


def build_model_par(β, π_LL, π_HH):
    return TypeModelParameters(α_x    = 0.600,
                               α_y    = 0.450,
                               γ      = 0.200,
                               β      = β    ,
                               w_star = 0.650,
                               θ      = 0.500,
                               η      = 0.700,
                               σ      = 2.000,
                               δ      = 0.960,
                               ρ      = 0.900,
                               σ_ϵ    = 0.150,
                               π_LL   = π_LL ,
                               π_HH   = π_HH ,
                               M      = 1.000)


model_specs = {
    'M1': {'rh_N': 1, 'π_LL': 0.700, 'π_HH': 0.800, 'solver': 'rep_hh'    },
    'M2': {'rh_N': 1, 'π_LL': 0.700, 'π_HH': 0.800, 'solver': 'outer_loop'},
    'M3': {'rh_N': 7, 'π_LL': 1.000, 'π_HH': 1.000, 'solver': 'outer_loop'},
    'M4': {'rh_N': 7, 'π_LL': 0.700, 'π_HH': 0.800, 'solver': 'outer_loop'},
}

scenarios = [('T0', 2.000), ('T1', 3.000)]


def solve_single(task):
    name     = task['name']
    ModelPar = task['ModelPar']
    CalibPar = task['CalibPar']
    solver   = task['solver']

    t0 = time.time()
    model = GeneralEquilibriumModel(ModelPar=ModelPar, CalibPar=CalibPar)
    if solver == 'rep_hh':
        model.solve_representative_household()
    else:
        model.outer_loop_solver()
    elapsed = (time.time() - t0) / 60
    return name, model, elapsed


if __name__ == '__main__':

    # Build task list: 4 models × 2 scenarios = 8 independent solves
    tasks = []
    for m_name, spec in model_specs.items():
        calib = build_calib(rh_N=spec['rh_N'])
        for t_name, β in scenarios:
            mp = build_model_par(β=β, π_LL=spec['π_LL'], π_HH=spec['π_HH'])
            tasks.append({'name'    : f'{m_name}_{t_name}',
                          'ModelPar': mp,
                          'CalibPar': calib,
                          'solver'  : spec['solver']})

    max_workers = min(len(tasks), os.cpu_count() or 4)
    print(f"Solving {len(tasks)} models in parallel with max_workers={max_workers}.")
    print("")

    start = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_single, t): t['name'] for t in tasks}
        for future in as_completed(futures):
            name, model, elapsed = future.result()
            results[name] = model
            print(f"[DONE] {name} ({elapsed:.2f}m)")

    print("")
    print(f"All models solved in {(time.time() - start)/60:.2f}m.")

    # Saving individual model files (preserves original output filenames)
    for m_name in model_specs.keys():
        save_dict = {f'{m_name}_T0': results[f'{m_name}_T0'],
                     f'{m_name}_T1': results[f'{m_name}_T1']}
        with open(OUTPUTS / f'prem_res_2_{m_name}.p', 'wb') as file:
            pickle.dump(save_dict, file)

    # Saving combined results
    with open(OUTPUTS / 'model_results.p', 'wb') as file:
        pickle.dump(results, file)
