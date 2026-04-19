# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

import pickle
import pandas as pd
from config import *
from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel
import numpy as np
import matplotlib.pyplot as plt

# Import the GeneralEquilibriumModel objects and calculates economy statistics
with open(OUTPUTS / 'model_results.p', 'rb') as file:
    res = pickle.load(file)

M1_T0 = res['M1_T0']
M1_T1 = res['M1_T1']
M2_T0 = res['M2_T0']
M2_T1 = res['M2_T1']
M3_T0 = res['M3_T0']
M3_T1 = res['M3_T1']
M4_T0 = res['M4_T0']
M4_T1 = res['M4_T1']

# Compute economy statistics for numerical models (M2, M3, M4)
for model in [M2_T0, M2_T1, M3_T0, M3_T1, M4_T0, M4_T1]:
    model.economy_statistics()

def extract_vars(model, is_rep_hh=False):
    """
    Extracts the key equilibrium variables from a model object.

    Parameters
    ----------
    model : GeneralEquilibriumModel
        A solved model object containing equilibrium results.
    is_rep_hh : bool, optional
        If True, reads variables from the representative-household solution
        instead of the heterogeneous-agent economy statistics. Default is False.

    Returns
    -------
    dict
        Dictionary with keys: 'I', 'w', 's', 'r', 'K', 'p', 'U_L', 'U_H', 'Gini'.

    """
    if is_rep_hh:
        rh = model.rep_hh_res
        return {'I': rh['I'], 'w': rh['w'], 's': rh['s'], 'r': rh['r'],
                'K': rh['K_s'], 'p': rh['p'], 'U_L': rh['U_L'],
                'U_H': rh['U_H'], 'Gini': rh['income_gini']}
    es = model.economy_stats
    return {'I': model.I, 'w': model.w, 's': model.s, 'r': model.r,
            'K': es['K'], 'p': model.p, 'U_L': es['mean_V_L'],
            'U_H': es['mean_V_H'], 'Gini': es['income_gini']}

T0 = {'M1': extract_vars(M1_T0, True), 'M2': extract_vars(M2_T0),
      'M3': extract_vars(M3_T0),        'M4': extract_vars(M4_T0)}
T1 = {'M1': extract_vars(M1_T1, True), 'M2': extract_vars(M2_T1),
      'M3': extract_vars(M3_T1),        'M4': extract_vars(M4_T1)}

rate_vars = {'I', 'r'}
models    = ['M1', 'M2', 'M3', 'M4']
all_vars  = ['I', 'w', 's', 'r', 'K', 'p', 'U_L', 'U_H', 'Gini']

def delta_pct(t0, t1, var):
    """
    Computes the percentage change between two equilibrium values.

    For rate variables (e.g. I, r) returns the change in percentage points;
    for all other variables returns the percentage change relative to t0.

    Parameters
    ----------
    t0 : float
        Value in the baseline (pre-reform) equilibrium.
    t1 : float
        Value in the counterfactual (post-reform) equilibrium.
    var : str
        Variable name, used to decide the change formula.

    Returns
    -------
    float
        Percentage change (or percentage-point change) from t0 to t1.

    """
    return (t1 - t0) * 100 if var in rate_vars else (t1 - t0) / abs(t0) * 100

delta = {m: {v: delta_pct(T0[m][v], T1[m][v], v) for v in all_vars} for m in models}

def fmt_level(val, var):
    """
    Formats an equilibrium level value as a LaTeX string.

    Parameters
    ----------
    val : float
        The equilibrium value to format.
    var : str
        Variable name, used to select the appropriate format and unit.

    Returns
    -------
    str
        LaTeX-ready string representation of val.

    """
    if var == 'I':            return f'{val*100:.1f}\\%'
    if var == 'r':            return f'{val*100:.2f}\\%'
    if var == 'K':            return f'{val:.2f}'
    if var in ('U_L', 'U_H'): return f'{val:.1f}'
    return f'{val:.3f}'        # w, s, p, Gini

def fmt_delta(val, var):
    """
    Formats a percentage change value as a signed string for LaTeX tables.

    Positive values receive an explicit '+' prefix; zero values are left
    unsigned; negative values keep their '-' sign.

    Parameters
    ----------
    val : float
        The percentage (or percentage-point) change to format.
    var : str
        Variable name, used to select decimal precision.

    Returns
    -------
    str
        Signed string representation of val.

    """
    dec = 2 if var == 'r' else 1
    s   = f'{val:.{dec}f}'
    if float(s) == 0.0: return s
    return f'+{s}' if val > 0 else s

table_groups = [
    ('Labour Market',        ['I', 'w', 's']),
    ('Capital & Prices',     ['r', 'K', 'p']),
    ('Welfare & Inequality', ['U_L', 'U_H', 'Gini']),
]

var_labels = {
    'I': 'I', 'w': 'w', 's': 's', 'r': 'r', 'K': 'K', 'p': 'p',
    'U_L': 'U^L', 'U_H': 'U^H', 'Gini': r'\text{Gini}',
}

def build_latex_table(variables):
    """
    Builds a LaTeX tabular string comparing equilibrium levels and changes
    across all models (M1–M4) for the given set of variables.

    Each variable gets four columns (one per model) showing the baseline
    level, the counterfactual level, and the percentage change.

    Parameters
    ----------
    variables : list of str
        Subset of all_vars to include as column groups in the table.

    Returns
    -------
    str
        A multi-line LaTeX tabular environment as a single string.

    """
    n        = len(variables)
    col_spec = 'l|' + '|'.join(['cccc'] * n)
    lines    = []

    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    var_hdrs = []
    for i, v in enumerate(variables):
        sep = 'c|' if i < n - 1 else 'c'
        var_hdrs.append(r'\multicolumn{4}{' + sep + r'}{\textcolor{blue}{$' + var_labels[v] + r'$}}')
    lines.append('& ' + ' \n& '.join(var_hdrs) + r' \\')

    lines.append('& ' + ' & '.join(['M1 & M2 & M3 & M4'] * n) + r' \\')
    lines.append(r'\midrule')

    row_defs = [
        (r'$\beta=2$',  lambda v, m: fmt_level(T0[m][v], v)),
        (r'$\beta=3$',  lambda v, m: fmt_level(T1[m][v], v)),
        (r'$\Delta\%$', lambda v, m: f'${fmt_delta(delta[m][v], v)}$'),
    ]

    for label, cell_fn in row_defs:
        cells = [cell_fn(v, m) for v in variables for m in models]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)

table_comments = ['% Table 1: Labour Market',
                  '% Table 2: Capital and Prices',
                  '% Table 3: Welfare and Inequality']

latex_sections = [f'{comment}\n{build_latex_table(vars_)}'
                  for (_, vars_), comment in zip(table_groups, table_comments)]
latex_output   = ('\n\n' + r'\vspace{0.3cm}' + '\n\n').join(latex_sections)

txt_path = OUTPUTS / 'results_tables.txt'
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(latex_output)
print(f'LaTeX saved to {txt_path}')

# Analysing the policy functions
a_dens   = M2_T0.mod_res.groupby('a_0')['dens'].sum().reset_index().sort_values('a_0')
cum_w    = a_dens['dens'].cumsum() / a_dens['dens'].sum()
a_cutoff = float(a_dens.loc[cum_w >= 0.99, 'a_0'].iloc[0])

fig, axes = plt.subplots(2, 2, figsize=(12, 5),
                         gridspec_kw={'height_ratios': [3, 1]},
                         sharex='col')

for col, (skill, title) in enumerate([('L', 'Low Skill'), ('H', 'High Skill')]):

    ax_diff = axes[0, col]
    ax_dens = axes[1, col]

    df_T0 = M2_T0.mod_res[M2_T0.mod_res['skill_type'] == skill].sort_values('a_0')
    df_T1 = M2_T1.mod_res[M2_T1.mod_res['skill_type'] == skill].sort_values('a_0')

    a0   = df_T0['a_0'].values
    diff = df_T1['a_1'].values - df_T0['a_1'].values

    ax_diff.plot(a0, diff, color='steelblue', linewidth=2)
    ax_diff.fill_between(a0, diff, 0, where=(diff >= 0), color='steelblue', alpha=0.15, label=r'Higher $a^\prime$')
    ax_diff.fill_between(a0, diff, 0, where=(diff <  0), color='tomato',    alpha=0.15, label=r'Lower $a^\prime$')
    ax_diff.axhline(0, color='black', linestyle=':', linewidth=1)

    ax_diff.set_xlim(0, a_cutoff)
    ax_diff.set_title(title, fontsize=13)
    ax_diff.set_ylabel(r"$a'(\beta=3)\, -\, a'(\beta=2)$", fontsize=11)
    ax_diff.legend(fontsize=10, loc='upper right')
    ax_diff.tick_params(labelbottom=False)

    dens_skill_b2 = (M2_T0.mod_res[M2_T0.mod_res['skill_type'] == skill].groupby('a_0')['dens'].sum().reset_index())
    mean_a_b2     = (dens_skill_b2['a_0'] * dens_skill_b2['dens']).sum() / dens_skill_b2['dens'].sum()

    dens_skill_b3 = (M2_T1.mod_res[M2_T1.mod_res['skill_type'] == skill].groupby('a_0')['dens'].sum().reset_index())
    mean_a_b3     = (dens_skill_b3['a_0'] * dens_skill_b3['dens']).sum() / dens_skill_b3['dens'].sum()

    ax_dens.fill_between(dens_skill_b2['a_0'], dens_skill_b2['dens'], color='gray', alpha=0.4)
    ax_dens.axvline(mean_a_b2, color='gray', linestyle='--', linewidth=1.3, label=r'Mean ($\beta=2$) = '+f'{mean_a_b2:.2f}')
    ax_diff.axvline(mean_a_b2, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    ax_dens.fill_between(dens_skill_b3['a_0'], dens_skill_b3['dens'], color='steelblue', alpha=0.4)
    ax_dens.axvline(mean_a_b3, color='steelblue', linestyle='--', linewidth=1.3, label=r'Mean ($\beta=3$) = '+f'{mean_a_b3:.2f}')
    ax_diff.axvline(mean_a_b3, color='steelblue', linestyle='--', linewidth=1, alpha=0.6)

    ax_dens.set_xlabel(r'Current assets $a_0$', fontsize=11)
    ax_dens.set_ylabel('Density', fontsize=10)
    ax_dens.set_yticks([])
    ax_dens.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUTS / 'policy_diff_M2.pdf', dpi=150, bbox_inches='tight')
plt.show()

# Running the simulations to confirm hypothesis
hh_count           = 50000
n_periods          = 500
burned_periods     = 200
sim_saved_periods  = n_periods - burned_periods
pre_shock_periods  = 5
post_shock_periods = 30
min_periods_L      = 5
recover_level      = 0.90
assets_lb_thresh   = [0.1, 0.5, 1.0]

def build_policy(mod_res):
    """
    Builds an array with the policy function used in the simulations.

    Parameters
    ----------
    mod_res : pandas.DataFrame
        GeneralEQuilibriumModel mod res attribute

    Returns
    -------
    a_grid : numpy.ndarray
        Grid with the discretized possible values of assets
    
    pol : numpy.ndarray
        Array with the policy function (if in state x go to state y)

    """    
    
    a_grid = np.sort(mod_res['a_0'].unique())
    pol    = np.empty((2, len(a_grid)), dtype=int)
    for s_idx, skill in enumerate(['L', 'H']):
        df         = mod_res[mod_res['skill_type'] == skill].sort_values('a_0')
        pol[s_idx] = np.searchsorted(a_grid, df['a_1'].values)
    return a_grid, pol

def sample_init(mod_res, a_grid, hh_count, rng):
    """
    Draws initial states for the simulation from the stationary distribution.

    Each household is assigned an asset grid index and a skill type index
    by sampling with replacement from the stationary distribution, using
    the stationary density as sampling weights.

    Parameters
    ----------
    mod_res : pandas.DataFrame
        GeneralEquilibriumModel mod_res attribute, containing columns
        'a_0', 'skill_type', and 'dens'.
    a_grid : numpy.ndarray
        Sorted asset grid returned by build_policy.
    hh_count : int
        Number of households to initialise.
    rng : numpy.random.Generator
        Random number generator to use for sampling.

    Returns
    -------
    a_idx : numpy.ndarray of int, shape (hh_count,)
        Asset grid indices for each household's initial state.
    s_idx : numpy.ndarray of int, shape (hh_count,)
        Skill indices (0 = Low, 1 = High) for each household's initial state.

    """

    df          = mod_res[['a_0', 'skill_type', 'dens']].copy()
    df['a_idx'] = np.searchsorted(a_grid, df['a_0'].values)
    df['s_idx'] = (df['skill_type'] == 'H').astype(int)
    p           = df['dens'].values / df['dens'].sum()
    chosen      = rng.choice(len(df), size=hh_count, replace=True, p=p)
    return df['a_idx'].values[chosen].astype(np.int32), df['s_idx'].values[chosen].astype(np.int8)

def simulate(mod_res, π_LL, π_HH, hh_count, n_periods, burned_periods, rng):
    """
    Simulates the asset and skill paths of a cross-section of households.

    Households are initialised from the stationary distribution, then
    iterated forward using the optimal policy function and a Markov skill
    transition. The first burned_periods periods are discarded to remove
    dependence on initial conditions.

    Parameters
    ----------
    mod_res : pandas.DataFrame
        GeneralEquilibriumModel mod_res attribute.
    π_LL : float
        Probability of remaining Low-skill in the next period.
    π_HH : float
        Probability of remaining High-skill in the next period.
    hh_count : int
        Number of households to simulate.
    n_periods : int
        Total number of periods to run (including burn-in).
    burned_periods : int
        Number of initial periods to discard as burn-in.
    rng : numpy.random.Generator
        Random number generator used for both initialisation and skill transitions.

    Returns
    -------
    a_grid : numpy.ndarray
        Sorted asset grid used in the simulation.
    a_sim : numpy.ndarray of int, shape (sim_saved_periods, hh_count)
        Asset grid indices for each household in each saved period.
    s_sim : numpy.ndarray of int, shape (sim_saved_periods, hh_count)
        Skill indices (0 = Low, 1 = High) for each household in each saved period.

    """

    a_grid, pol       = build_policy(mod_res)
    sim_saved_periods = n_periods - burned_periods
    a_idx, skill      = sample_init(mod_res, a_grid, hh_count, rng)
    a_sim             = np.empty((sim_saved_periods, hh_count), dtype=np.int32)
    s_sim             = np.empty((sim_saved_periods, hh_count), dtype=np.int8)
    for t in range(n_periods):
        if t >= burned_periods:
            a_sim[t - burned_periods] = a_idx
            s_sim[t - burned_periods] = skill
        a_idx             = pol[skill, a_idx]
        rnd               = rng.random(hh_count)
        new_s             = skill.copy()
        new_s[skill == 0] = np.where(rnd[skill == 0] < π_LL, 0, 1)
        new_s[skill == 1] = np.where(rnd[skill == 1] < π_HH, 1, 0)
        skill             = new_s
    return a_grid, a_sim, s_sim

print('Simulating β=2...')
a_grid2, a_sim2, s_sim2 = simulate(M2_T0.mod_res, M2_T0.ModelPar.π_LL, M2_T0.ModelPar.π_HH,
                                    hh_count, n_periods, burned_periods, np.random.default_rng(42))
print('Simulating β=3...')
a_grid3, a_sim3, s_sim3 = simulate(M2_T1.mod_res, M2_T1.ModelPar.π_LL, M2_T1.ModelPar.π_HH,
                                    hh_count, n_periods, burned_periods, np.random.default_rng(42))

def event_study(a_sim, s_sim, a_grid, τ_pre, τ_post, k_min, from_s=1, to_s=0):
    """
    Extracts asset trajectories around skill-transition events.

    An event is defined as a period in which a household's skill changes from
    from_s to to_s. Only events where the household remains in to_s for at least
    k_min consecutive periods are kept, to focus on persistent transitions.

    Parameters
    ----------
    a_sim : numpy.ndarray of int, shape (T, N)
        Asset grid indices from simulate.
    s_sim : numpy.ndarray of int, shape (T, N)
        Skill indices from simulate.
    a_grid : numpy.ndarray
        Sorted asset grid used to convert indices back to asset values.
    τ_pre : int
        Number of periods to include before the transition (event at index τ_pre).
    τ_post : int
        Number of periods to include after the transition.
    k_min : int
        Minimum number of consecutive periods in to_s required to keep an event.
    from_s : int, optional
        Skill state before the transition (default 1 = High, for H→L).
    to_s : int, optional
        Skill state after the transition (default 0 = Low, for H→L).

    Returns
    -------
    numpy.ndarray of float, shape (n_events, τ_pre + τ_post + 1)
        Asset levels (in value, not index) for each qualifying event,
        centred so that column τ_pre corresponds to the period of the transition.

    """

    t_sim, N    = a_sim.shape
    t_min       = τ_pre
    t_max       = t_sim - τ_post - 1
    a_val       = a_grid[a_sim]
    ev_t, ev_n  = np.where((s_sim[t_min:t_max+1] == to_s) & (s_sim[t_min-1:t_max] == from_s))

    ev_t      += t_min

    k_idx      = ev_t[:, None] + np.arange(k_min)[None, :]
    stay       = (s_sim[k_idx, ev_n[:, None]] == to_s).all(axis=1)
    ev_t, ev_n = ev_t[stay], ev_n[stay]
    w_idx      = ev_t[:, None] + np.arange(-τ_pre, τ_post + 1)[None, :]

    return a_val[w_idx, ev_n[:, None]]

print('Extracting event windows...')
trajs2    = event_study(a_sim2, s_sim2, a_grid2, pre_shock_periods, post_shock_periods, min_periods_L)
trajs3    = event_study(a_sim3, s_sim3, a_grid3, pre_shock_periods, post_shock_periods, min_periods_L)
trajs2_LH = event_study(a_sim2, s_sim2, a_grid2, pre_shock_periods, post_shock_periods, min_periods_L, from_s=0, to_s=1)
trajs3_LH = event_study(a_sim3, s_sim3, a_grid3, pre_shock_periods, post_shock_periods, min_periods_L, from_s=0, to_s=1)
print(f'  β=2: {len(trajs2):,} H→L  |  {len(trajs2_LH):,} L→H')
print(f'  β=3: {len(trajs3):,} H→L  |  {len(trajs3_LH):,} L→H')

pre2  = trajs2[:, pre_shock_periods - 1][:, None]
pre3  = trajs3[:, pre_shock_periods - 1][:, None]
norm2 = np.where(pre2 > 1e-6, trajs2 / pre2, np.nan)
norm3 = np.where(pre3 > 1e-6, trajs3 / pre3, np.nan)

pre2_LH  = trajs2_LH[:, pre_shock_periods - 1][:, None]
pre3_LH  = trajs3_LH[:, pre_shock_periods - 1][:, None]
norm2_LH = np.where(pre2_LH > 1e-6, trajs2_LH / pre2_LH, np.nan)
norm3_LH = np.where(pre3_LH > 1e-6, trajs3_LH / pre3_LH, np.nan)

def recovery_periods(norm, τ_pre, p_recover):
    """
    Computes, for each event, the first post-shock period in which assets
    recover to a given fraction of their pre-shock level.

    Parameters
    ----------
    norm : numpy.ndarray of float, shape (n_events, τ_pre + τ_post + 1)
        Asset paths normalised by the pre-shock level (output of event_study
        divided by the period-τ_pre value).
    τ_pre : int
        Column index of the shock period; post-shock columns start at τ_pre.
    p_recover : float
        Recovery threshold as a fraction of the pre-shock level (e.g. 0.90).

    Returns
    -------
    numpy.ndarray of float, shape (n_events,)
        Number of periods after the shock until first recovery. NaN for
        events that never recover within the observation window.

    """

    post      = norm[:, τ_pre:]
    hit       = post >= p_recover
    first_hit = np.where(hit.any(axis=1), np.argmax(hit, axis=1), np.nan).astype(float)
    return first_hit

rec2 = recovery_periods(norm2, pre_shock_periods, recover_level)
rec3 = recovery_periods(norm3, pre_shock_periods, recover_level)

def left_tail(mod_res, thresholds):
    """
    Computes the stationary mass of households below each asset threshold.

    Parameters
    ----------
    mod_res : pandas.DataFrame
        GeneralEquilibriumModel mod_res attribute, containing columns
        'a_0' and 'dens'.
    thresholds : list of float
        Asset cutoff values below which the density mass is summed.

    Returns
    -------
    dict
        Mapping from each threshold in thresholds to the corresponding
        cumulative stationary mass (a number in [0, 1]).

    """
    
    return {ā: float(mod_res.loc[mod_res['a_0'] < ā, 'dens'].sum()) for ā in thresholds}

tail2 = left_tail(M2_T0.mod_res, assets_lb_thresh)
tail3 = left_tail(M2_T1.mod_res, assets_lb_thresh)

rows = [
    {'Metric': f'Mean recovery time (periods, {int(recover_level*100)}% threshold)',
     'β=2': f'{np.nanmean(rec2):.1f}', 'β=3': f'{np.nanmean(rec3):.1f}'},
    {'Metric': f'% recovering within {post_shock_periods} periods',
     'β=2': f'{np.mean(~np.isnan(rec2)):.1%}', 'β=3': f'{np.mean(~np.isnan(rec3)):.1%}'},
    *[{'Metric': f'Left tail mass  (a < {ā})',
       'β=2': f'{tail2[ā]:.4f}', 'β=3': f'{tail3[ā]:.4f}'} for ā in assets_lb_thresh],
]
summary_df = pd.DataFrame(rows).set_index('Metric')
print('\n', summary_df.to_string())

with open(OUTPUTS / 'simulation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_df.to_latex())

τ_axis = np.arange(-pre_shock_periods, post_shock_periods + 1)
mean2  = np.nanmedian(norm2, axis=0)
mean3  = np.nanmedian(norm3, axis=0)
p25_2  = np.nanpercentile(norm2, 25, axis=0)
p75_2  = np.nanpercentile(norm2, 75, axis=0)
p25_3  = np.nanpercentile(norm3, 25, axis=0)
p75_3  = np.nanpercentile(norm3, 75, axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(τ_axis, mean2, color='steelblue', linewidth=2, label=r'$\beta=2$')
ax.fill_between(τ_axis, p25_2, p75_2, color='steelblue', alpha=0.15)
ax.plot(τ_axis, mean3, color='tomato', linewidth=2, label=r'$\beta=3$')
ax.fill_between(τ_axis, p25_3, p75_3, color='tomato', alpha=0.15)
ax.axvline(0, color='black', linestyle=':', linewidth=1)
ax.axhline(1, color='black', linestyle=':', linewidth=1)
ax.axhline(recover_level, color='gray', linestyle='--', linewidth=1,
           label=f'{int(recover_level*100)}% pre-shock level')
ax.set_xlabel(r'Periods relative to H→L transition ($\tau$)', fontsize=11)
ax.set_ylabel(r'Assets / pre-shock level', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(τ_axis[0], τ_axis[-1])
plt.tight_layout()
plt.savefig(OUTPUTS / 'event_study_HL_M2.pdf', dpi=150, bbox_inches='tight')
plt.show()


