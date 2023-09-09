import numpy as np
import sys

def gauss_params_from_eer(desired_eer):

    m = 2*(stats.norm.ppf(desired_eer)**2)
    v = 2*m

    m_pos = m
    m_neg = -m
    v_pos = v
    v_neg = v

    return m_pos, m_neg, v_pos, v_neg

# -----------------------------------------------------------------------------
def set_bivariate_gauss_params(m1, m2, v1, v2, r):
# m1,m2 - means, v1,v2 - vars, r = correlation

    m       = np.array([m1, m2])
    x_term  = r * np.sqrt(v1) * np.sqrt(v2)
    C       = np.matrix([[v1, x_term], [x_term, v2]])
    return m, C

# -----------------------------------------------------------------------------
def all_bivariate_gauss_params_from_sim_params(sim_param):

    # First, "lock" the target-nontarget gaussians (ASV) and target-spoof Gaussians (CM)
    asv_tar_m, asv_non_m,  asv_tar_v, asv_non_v = gauss_params_from_eer(sim_param['asv_eer_tar_non'])
    cm_tar_m,  cm_spf_m,   cm_tar_v,  cm_spf_v  = gauss_params_from_eer(sim_param['cm_eer_tar_spoof'])

    # ASV - insert spoof distribution with the same shared variance, and with desired EER between (target, spoof)
    F = np.sqrt(2) * special.erfinv(1 - 2*sim_param['asv_eer_tar_spoof'])
    asv_spf_m = asv_tar_m - 2*np.sqrt(asv_tar_v)*F
    asv_spf_v = asv_tar_v

    # CM - similarly, insert nontarget distribution. Note here we've defined the discrimation from spoof (rather than target)
    F = np.sqrt(2) * special.erfinv(1 - 2*sim_param['cm_eer_non_spoof'])
    cm_non_m = cm_spf_m + 2*np.sqrt(cm_tar_v)*F
    cm_non_v = cm_tar_v

    # Now we have all the building blocks!
    tar_mu, tar_cov = set_bivariate_gauss_params(asv_tar_m, cm_tar_m, asv_tar_v, cm_tar_v, sim_param['asv_cm_xcorr_tar'])
    non_mu, non_cov = set_bivariate_gauss_params(asv_non_m, cm_non_m, asv_non_v, cm_non_v, sim_param['asv_cm_xcorr_non'])
    spf_mu, spf_cov = set_bivariate_gauss_params(asv_spf_m, cm_spf_m, asv_spf_v, cm_spf_v, sim_param['asv_cm_xcorr_spf'])

    return tar_mu, non_mu, spf_mu, tar_cov, non_cov, spf_cov