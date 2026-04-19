# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

import pickle
from config import *
from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel

if __name__ == '__main__':
    
    # We'll consider the following exercise to decompose the impacts of each mechanism we add to the model.
    # 1. Solve the analytical baby model (representative household)
    # 2. Solve the numerical model without shocks to z but with shocks to l
    # 3. Solve the numerical model with shocks to z but without shocks to l
    # 4. Solve the numerical model with shocks to both z and l
    # In all scenarios we'll estimate a prior and a posterior model. Assuming 
    # we were in an equilibirum with β=2 and move to β=3, how does the economy respond?
    
    #######################
    ### --- Model 1 --- ###
    #######################

    CalibPar = TypeCalibParameters(rh_N             = 1    ,
                                   rh_r             = 2    , 
                                   rh_c             = 0    ,
                                   sup_side_eq_eps  = 1e-5 ,
                                   vfi_lb           = 0    ,
                                   vfi_ubmul        = 15    ,
                                   vfi_N            = 500  ,
                                   vfi_eps          = 1e-5 ,
                                   vfi_howard_steps = 20   ,
                                   gmc_eps          = 1e-3 ,
                                   kmc_eps          = 1e-3 ,
                                   p_init_guess     = 1,
                                   r_init_guess     = 0.01,
                                   inner_loop_eps   = 1e-5,
                                   inner_loop_p_lb  = 0.1,
                                   inner_loop_p_ub  = 1.2,
                                   inner_loop_marg  = 0.3,
                                   outer_loop_eps   = 1e-5,
                                   outer_loop_r_lb  = 0.001)
    
    ModelPar_T0 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 2.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 0.700,
                                      π_HH   = 0.800,
                                      M      = 1.000)
    
    ModelPar_T1 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 3.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 0.700,
                                      π_HH   = 0.800,
                                      M      = 1.000)
    
    M1_T0 = GeneralEquilibriumModel(ModelPar = ModelPar_T0,
                                    CalibPar = CalibPar)
    M1_T1 = GeneralEquilibriumModel(ModelPar = ModelPar_T1,
                                    CalibPar = CalibPar)
    
    
    M1_T0.solve_representative_household()
    M1_T1.solve_representative_household()
    
    # Saving results
    save_dict = {'M1_T0':M1_T0,
                 'M1_T1':M1_T1}
    with open(OUTPUTS / 'prem_res_2_M1.p', 'wb') as file:
        pickle.dump(save_dict, file)
    
    #######################
    ### --- Model 2 --- ###
    #######################
    
    CalibPar = TypeCalibParameters(rh_N             = 1    ,
                                   rh_r             = 2    , 
                                   rh_c             = 0    ,
                                   sup_side_eq_eps  = 1e-5 ,
                                   vfi_lb           = 0    ,
                                   vfi_ubmul        = 15    ,
                                   vfi_N            = 500  ,
                                   vfi_eps          = 1e-5 ,
                                   vfi_howard_steps = 20   ,
                                   gmc_eps          = 1e-3 ,
                                   kmc_eps          = 1e-3 ,
                                   p_init_guess     = 1,
                                   r_init_guess     = 0.01,
                                   inner_loop_eps   = 1e-5,
                                   inner_loop_p_lb  = 0.1,
                                   inner_loop_p_ub  = 1.2,
                                   inner_loop_marg  = 0.3,
                                   outer_loop_eps   = 1e-5,
                                   outer_loop_r_lb  = 0.001)
    
    ModelPar_T0 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 2.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 0.700,
                                      π_HH   = 0.800,
                                      M      = 1.000)
    
    ModelPar_T1 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 3.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 0.700,
                                      π_HH   = 0.800,
                                      M      = 1.000)
    
    M2_T0 = GeneralEquilibriumModel(ModelPar = ModelPar_T0,
                                    CalibPar = CalibPar)
    M2_T1 = GeneralEquilibriumModel(ModelPar = ModelPar_T1,
                                    CalibPar = CalibPar)
    
    
    
    M2_T0.outer_loop_solver()
    M2_T1.outer_loop_solver()
    
    # Saving results
    save_dict = {'M2_T0':M2_T0,
                 'M2_T1':M2_T1}
    with open(OUTPUTS / 'prem_res_2_M2.p', 'wb') as file:
        pickle.dump(save_dict, file)
    
    #######################
    ### --- Model 3 --- ###
    #######################
    
    CalibPar = TypeCalibParameters(rh_N             = 7    ,
                                   rh_r             = 2    , 
                                   rh_c             = 0    ,
                                   sup_side_eq_eps  = 1e-5 ,
                                   vfi_lb           = 0    ,
                                   vfi_ubmul        = 15    ,
                                   vfi_N            = 500  ,
                                   vfi_eps          = 1e-5 ,
                                   vfi_howard_steps = 20   ,
                                   gmc_eps          = 1e-3 ,
                                   kmc_eps          = 1e-3 ,
                                   p_init_guess     = 1,
                                   r_init_guess     = 0.01,
                                   inner_loop_eps   = 1e-5,
                                   inner_loop_p_lb  = 0.1,
                                   inner_loop_p_ub  = 1.2,
                                   inner_loop_marg  = 0.3,
                                   outer_loop_eps   = 1e-5,
                                   outer_loop_r_lb  = 0.001)
    
    ModelPar_T0 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 2.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 1.000,
                                      π_HH   = 1.000,
                                      M      = 1.000)
    
    ModelPar_T1 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 3.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 1.000,
                                      π_HH   = 1.000,
                                      M      = 1.000)
    
    M3_T0 = GeneralEquilibriumModel(ModelPar = ModelPar_T0,
                                    CalibPar = CalibPar)
    M3_T1 = GeneralEquilibriumModel(ModelPar = ModelPar_T1,
                                    CalibPar = CalibPar)
    
    
    
    M3_T0.outer_loop_solver()
    M3_T1.outer_loop_solver()
    
    # Saving results
    save_dict = {'M3_T0':M3_T0,
                 'M3_T1':M3_T1}
    with open(OUTPUTS / 'prem_res_2_M3.p', 'wb') as file:
        pickle.dump(save_dict, file)
    
    #######################
    ### --- Model 4 --- ###
    #######################
    
    CalibPar = TypeCalibParameters(rh_N             = 7    ,
                                   rh_r             = 2    , 
                                   rh_c             = 0    ,
                                   sup_side_eq_eps  = 1e-5 ,
                                   vfi_lb           = 0    ,
                                   vfi_ubmul        = 15    ,
                                   vfi_N            = 500  ,
                                   vfi_eps          = 1e-5 ,
                                   vfi_howard_steps = 20   ,
                                   gmc_eps          = 1e-3 ,
                                   kmc_eps          = 1e-3 ,
                                   p_init_guess     = 1,
                                   r_init_guess     = 0.01,
                                   inner_loop_eps   = 1e-5,
                                   inner_loop_p_lb  = 0.1,
                                   inner_loop_p_ub  = 1.2,
                                   inner_loop_marg  = 0.3,
                                   outer_loop_eps   = 1e-5,
                                   outer_loop_r_lb  = 0.001)
    
    ModelPar_T0 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 2.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 0.700,
                                      π_HH   = 0.800,
                                      M      = 1.000)
    
    ModelPar_T1 = TypeModelParameters(α_x    = 0.600,
                                      α_y    = 0.450,
                                      γ      = 0.200,
                                      β      = 3.000,
                                      w_star = 0.650,
                                      θ      = 0.500,
                                      η      = 0.700,
                                      σ      = 2.000,
                                      δ      = 0.960,
                                      ρ      = 0.900,
                                      σ_ϵ    = 0.150,
                                      π_LL   = 0.700,
                                      π_HH   = 0.800,
                                      M      = 1.000)
    
    M4_T0 = GeneralEquilibriumModel(ModelPar = ModelPar_T0,
                                    CalibPar = CalibPar)
    M4_T1 = GeneralEquilibriumModel(ModelPar = ModelPar_T1,
                                    CalibPar = CalibPar)
    
    
    
    M4_T0.outer_loop_solver()
    M4_T1.outer_loop_solver()
    
    # Saving results
    save_dict = {'M4_T0':M4_T0,
                 'M4_T1':M4_T1}
    with open(OUTPUTS / 'prem_res_2_M4.p', 'wb') as file:
        pickle.dump(save_dict, file)
    
    # Saving results
    save_dict = {'M1_T0':M1_T0,
                 'M1_T1':M1_T1,
                 'M2_T0':M2_T0,
                 'M2_T1':M2_T1,
                 'M3_T0':M3_T0,
                 'M3_T1':M3_T1,
                 'M4_T0':M4_T0,
                 'M4_T1':M4_T1}
    with open(OUTPUTS / 'model_results.p', 'wb') as file:
        pickle.dump(save_dict, file)
    