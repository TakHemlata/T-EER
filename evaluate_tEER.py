import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import special
import random
import sys
import os
import eval_metrics as em
import loading_function as lf
import simulated_function as sim

args = sys.argv

#SASV = True # pass the boolan argument  
#ASVspoof = False # pass the boolan argument 

SASV = args[1].lower() == 'true'
ASVspoof = args[2].lower() == 'true'

if SASV:
    # Replace CM scores with your own scores.
    cm_score_file = 'scores\SASV\SASV_CM_AASIST_scores.txt'
    # Replace ASV with your own scores.
    asv_score_file = 'scores\SASV\SASV_ASV_ECAPA_scores.txt'

elif ASVspoof:
    # Replace CM scores with your own scores.
    cm_score_file = 'scores\ASVspoof\CM_RawNet2_2021_scores_file.txt'
    # Replace ASV with your own scores.
    asv_score_file = 'scores\ASVspoof\ASV_X_vector_2021_baseline_scores.txt'

else:
    print('Using synthetic scores')

# load scores file
if SASV:
    X_tar, X_non, X_spf = lf.process_sasv_score_files(cm_score_file, asv_score_file)
    
elif ASVspoof:
    cond = 'eval'
    X_tar, X_non, X_spf = lf.process_ASVspoof2021_score_files(cm_score_file, asv_score_file,cond)

else:
    print('Using synthetic scores')

    np.random.seed(seed=233423)

    ## SIMULATOR PARAMETERS
    sim_param = {}
    sim_param['asv_eer_tar_non']    = 0.08
    sim_param['asv_eer_tar_spoof']  = 0.35
    sim_param['cm_eer_tar_spoof']   = 0.10
    sim_param['cm_eer_non_spoof']   = sim_param['cm_eer_tar_spoof']
    sim_param['asv_cm_xcorr_tar']   = 0.00
    sim_param['asv_cm_xcorr_non']   = 0.00
    sim_param['asv_cm_xcorr_spf']   = 0.00

    ## SAMPLE SOME SCORES FROM THE THREE CLASSES
    n_trials_per_class = 10000
    tar_mu, non_mu, spf_mu, tar_cov, non_cov, spf_cov = sim.all_bivariate_gauss_params_from_sim_params(sim_param)
    X_tar = np.random.multivariate_normal(tar_mu, tar_cov, size=n_trials_per_class, check_valid='warn', tol=1e-8)
    X_non = np.random.multivariate_normal(non_mu, non_cov, size=n_trials_per_class, check_valid='warn', tol=1e-8)
    X_spf = np.random.multivariate_normal(spf_mu, spf_cov, size=n_trials_per_class, check_valid='warn', tol=1e-8)


# Obtain t-EER path and t-EER values

plt.close('all')
fig, ax = plt.subplots(1,2, figsize=(16, 8))

if SASV:
    xmin = min([min(X_tar[:,0]), min(X_non[:,0]), min(X_spf[:,0])])  # ASV scores
    xmax = max([max(X_tar[:,0]), max(X_non[:,0]), max(X_spf[:,0])])
    ymin = min([min(X_tar[:,1]), min(X_non[:,1]), min(X_spf[:,1])])  # CM scores
    ymax = max([max(X_tar[:,1]), max(X_non[:,1]), max(X_spf[:,1])])

    ax[0].scatter(X_tar[:,0], X_tar[:,1], c='green', label='target', marker='x', zorder=1, alpha=.5, s=50)
    ax[0].scatter(X_non[:,0], X_non[:,1], c='red', label='non-target', marker='o', zorder=1, alpha=.5, s=50)
    ax[0].scatter(X_spf[:,0], X_spf[:,1], c='black', label='spoof', marker='d', zorder=1, alpha=.5, s=50)

    ax[0].set_xlabel(r'$\mathcal{S}_\mathrm{asv}$',fontsize=34)
    ax[0].set_ylabel(r'$\mathcal{S}_\mathrm{cm}$',fontsize=34)
    ax[0].legend(loc=3,edgecolor='black',fontsize=16);

    # Obtain ASV error curves and ASV thresholds
    Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV = em.compute_Pmiss_Pfa_Pspoof_curves(X_tar[:,0], X_non[:,0], X_spf[:,0])

    # Obtain CM error curves and CM thresholds. Here we use both target and nontarget scores on the bonafide side.
    Pmiss_CM, Pfa_CM, tau_CM = em.compute_det_curve(np.concatenate([X_tar[:,1], X_non[:,1]]), X_spf[:,1])

elif ASVspoof:
    xmin = min([min(X_tar[0]), min(X_non[0]), min(X_spf[0])])  # ASV scores
    xmax = max([max(X_tar[0]), max(X_non[0]), max(X_spf[0])])
    ymin = min([min(X_tar[1]),  min(X_spf[1])])                # CM scores
    ymax = max([max(X_tar[1]),  max(X_spf[1])])

    # Obtain ASV error curves and ASV thresholds
    Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV = em.compute_Pmiss_Pfa_Pspoof_curves(X_tar[0], X_non[0], X_spf[0])

    # Obtain CM error curves and CM thresholds.
    Pmiss_CM, Pfa_CM, tau_CM = em.compute_det_curve(X_tar[1], X_spf[1])

else:
    xmin = min([min(X_tar[:,0]), min(X_non[:,0]), min(X_spf[:,0])])  # ASV scores
    xmax = max([max(X_tar[:,0]), max(X_non[:,0]), max(X_spf[:,0])])
    ymin = min([min(X_tar[:,1]), min(X_non[:,1]), min(X_spf[:,1])])  # CM scores
    ymax = max([max(X_tar[:,1]), max(X_non[:,1]), max(X_spf[:,1])])

    ax[0].scatter(X_tar[:,0], X_tar[:,1], c='green', label='target', marker='x', zorder=1, alpha=.5, s=50)
    ax[0].scatter(X_non[:,0], X_non[:,1], c='red', label='non-target', marker='o', zorder=1, alpha=.5, s=50)
    ax[0].scatter(X_spf[:,0], X_spf[:,1], c='black', label='spoof', marker='d', zorder=1, alpha=.5, s=50)

    ax[0].set_xlabel(r'$\mathcal{S}_\mathrm{asv}$',fontsize=34)
    ax[0].set_ylabel(r'$\mathcal{S}_\mathrm{cm}$',fontsize=34)
    ax[0].legend(loc=3,edgecolor='black',fontsize=16);

    # Obtain ASV error curves and ASV thresholds for synthetic_scores
    Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV = em.compute_Pmiss_Pfa_Pspoof_curves(X_tar[:,0], X_non[:,0], X_spf[:,0])

    # Obtain CM error curves and CM thresholds.
    Pmiss_CM, Pfa_CM, tau_CM = em.compute_det_curve(np.concatenate([X_tar[:,1], X_non[:,1]]), X_spf[:,1])


# Different spoofing prevalence priors (rho) parameters values
rho_vals            = [0,0.2,0.5,0.8,1]
rho_lw              = [0.5,1.3, 1.8,2.2, 2.7]
rho_min_pt_size     = [9,9,9,9,9]
rho_marker          = ['o','o','o','o','o']
ymins               = np.empty(len(rho_vals))

tEER_val    = np.empty([len(rho_vals),len(tau_ASV)], dtype=float)

for rho_idx, rho_spf in enumerate(rho_vals):

    # Table to store the CM threshold index, per each of the ASV operating points
    tEER_idx_CM = np.empty(len(tau_ASV), dtype=int)

    tEER_path   = np.empty([len(rho_vals),len(tau_ASV),2], dtype=float)

    # Tables to store the t-EER, total Pfa and total miss valuees along the t-EER path
    Pmiss_total = np.empty(len(tau_ASV), dtype=float)
    Pfa_total   = np.empty(len(tau_ASV), dtype=float)
    min_tEER    = np.inf
    argmin_tEER = np.empty(2)

    # best intersection point
    xpoint_crit_best = np.inf
    xpoint = np.empty(2)

    # Loop over all possible ASV thresholds
    for tau_ASV_idx, tau_ASV_val in enumerate(tau_ASV):

        # Tandem miss and fa rates as defined in the manuscript
        Pmiss_tdm = Pmiss_CM + (1 - Pmiss_CM) * Pmiss_ASV[tau_ASV_idx]
        Pfa_tdm   = (1 - rho_spf) * (1 - Pmiss_CM) * Pfa_non_ASV[tau_ASV_idx] + rho_spf * Pfa_CM * Pfa_spf_ASV[tau_ASV_idx]

        # Store only the INDEX of the CM threshold (for the current ASV threshold)
        h = Pmiss_tdm - Pfa_tdm
        tmp = np.argmin(abs(h))
        tEER_idx_CM[tau_ASV_idx] = tmp

        if Pmiss_ASV[tau_ASV_idx] < (1 - rho_spf) * Pfa_non_ASV[tau_ASV_idx] + rho_spf * Pfa_spf_ASV[tau_ASV_idx]:
            Pmiss_total[tau_ASV_idx] = Pmiss_tdm[tmp]
            Pfa_total[tau_ASV_idx] = Pfa_tdm[tmp]

            tEER_val[rho_idx,tau_ASV_idx] = np.mean([Pfa_total[tau_ASV_idx], Pmiss_total[tau_ASV_idx]])

            tEER_path[rho_idx,tau_ASV_idx, 0] = tau_ASV_val
            tEER_path[rho_idx,tau_ASV_idx, 1] = tau_CM[tmp]

            if tEER_val[rho_idx,tau_ASV_idx] < min_tEER:
                min_tEER = tEER_val[rho_idx,tau_ASV_idx]
                argmin_tEER[0] = tau_ASV_val
                argmin_tEER[1] = tau_CM[tmp]

            # Check how close we are to the INTERSECTION POINT for different prior (rho) values:
            LHS = Pfa_non_ASV[tau_ASV_idx]/Pfa_spf_ASV[tau_ASV_idx]
            RHS = Pfa_CM[tmp]/(1 - Pmiss_CM[tmp])
            crit = abs(LHS - RHS)

            if crit < xpoint_crit_best:
                xpoint_crit_best = crit
                xpoint[0] = tau_ASV_val
                xpoint[1] = tau_CM[tmp]
                xpoint_tEER = Pfa_spf_ASV[tau_ASV_idx]*Pfa_CM[tmp]
        else:
            # Not in allowed region
            tEER_path[rho_idx,tau_ASV_idx, 0] = np.nan
            tEER_path[rho_idx,tau_ASV_idx, 1] = np.nan
            Pmiss_total[tau_ASV_idx] = np.nan
            Pfa_total[tau_ASV_idx] = np.nan
            tEER_val[rho_idx,tau_ASV_idx] = np.nan


    print("concurrent-teer for [rho :{}] = {:.2f} ".format(rho_spf,xpoint_tEER*100))

    ##======================= t-EER plots =========================##
    ax[0].plot(tEER_path[rho_idx,:,0], tEER_path[rho_idx,:,1], c='blue', lw=rho_lw[rho_idx])
    ax[0].plot(xpoint[0], xpoint[1], marker=rho_marker[rho_idx], markersize=rho_min_pt_size[rho_idx], color='magenta', fillstyle='full', markeredgecolor='yellow')

    ax[0].tick_params(axis='x', labelsize=24)
    ax[0].tick_params(axis='y', labelsize=24)

    ax[1].plot(tau_ASV, tEER_val[rho_idx,:], c='blue', lw=rho_lw[rho_idx],label=r'$\rho={}$'.format(rho_spf), fillstyle='full')
    ax[1].plot(xpoint[0], xpoint_tEER, marker=rho_marker[rho_idx], markersize=rho_min_pt_size[rho_idx], color='magenta', fillstyle='full', markeredgecolor='yellow')

    ax[1].set_xlabel(r'$\tau_\mathrm{asv}$',fontsize=34)
    ax[1].set_ylabel(r't-EER',fontsize=24)
    ax[1].yaxis.set_ticks(np.arange(0.0, 0.7, 0.1))
    ax[1].tick_params(axis='x', labelsize=24)
    ax[1].tick_params(axis='y', labelsize=24)


ax[1].legend(loc=1,edgecolor='black',fontsize=20);
fig.savefig("Teer_plot.pdf", bbox_inches='tight', dpi=600)
