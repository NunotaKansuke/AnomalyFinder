
import numpy as np
import scipy.optimize as op
import warnings
warnings.filterwarnings("ignore", category=np.RankWarning)

#--------------------------------single lens ---------------------------------------#

def single_magnification(t0,tE,u0,t):
    u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
    A = (u**2 + 2)/(u*np.sqrt(u**2 +4))
    return A

def get_flux_single(t0,tE,u0,data_time,data_flux,data_ferr):
    A = single_magnification(t0,tE,u0,data_time)
    fs, fb, chi2 = linear_fit(A, data_flux,(1/data_ferr)**2)
    return fs, fb
    
def get_chi2_single(t0,tE,u0,data_time,data_flux,data_ferr):
    A = single_magnification(t0,tE,u0,data_time)
    fs, fb = get_flux_single(t0,tE,u0,data_time,data_flux,data_ferr)
    model_flux = A*fs + fb
    chi2 = np.sum(((data_flux-model_flux)/data_ferr)**2)
    return chi2
    
def chi2_fun_single(theta,data_time,data_flux,data_ferr):
    if theta[1] <= 0 or theta[2] <= 0 or theta[2]>1.5:
        return np.inf
    else:
        return get_chi2_single(theta[0],theta[1],theta[2],data_time,data_flux,data_ferr)

def single_fit(t0_init,tE_init,u0_init,data_time,data_flux,data_ferr):
    initial_guess = [t0_init, tE_init,u0_init]
    
    result = op.minimize(chi2_fun_single, x0=initial_guess,args=(data_time,data_flux,data_ferr),method='Nelder-Mead')
    
    (fit_t0, fit_tE,fit_u0) = result.x
    return fit_t0, fit_tE, fit_u0

def get_lc_PSPL(t0,tE,u0,fs,fb,t_range):
    t_ref = np.linspace(t_range[0],t_range[1],10000)
    f_model = fs* single_magnification(t0,tE,u0,t_ref) + fb
    lc_model = np.empty(len(t_ref), dtype=[("time", float), ("flux", float)])
    lc_model["time"] = t_ref
    lc_model["flux"] = f_model
    return lc_model

def linear_fit(x, y, w):
    w_sum = np.sum(w)
    wxy_sum = np.sum(w * x * y)
    wx_sum = np.sum(w * x)
    wy_sum = np.sum(w * y)
    wxx_sum = np.sum(w * x * x)
    bunbo = w_sum * wxx_sum - wx_sum**2
    a = (w_sum * wxy_sum - wx_sum * wy_sum) / bunbo
    b = (wxx_sum * wy_sum - wx_sum * wxy_sum) / bunbo
    residuals = y - (a * x + b)
    chi2 = np.sum((residuals**2) / w)
    return a, b, chi2

#------------------------------------------------------------------------------------#

#--------------------------------discrete single lens---------------------------------------#

def calc_A_j_0(t0,teff,t):
    Q = 1+((t-t0)/teff)**2
    A = 1/np.sqrt(Q)  
    return A

def calc_A_j_1(t0,teff,t):
    Q = 1+((t-t0)/teff)**2
    A = (Q+2)/np.sqrt(Q*(Q+4))
    return A

def calc_A_comb(t0,teff,t):
    A_j_0 = calc_A_j_0(t0,teff,t)
    A_j_1 = calc_A_j_1(t0,teff,t)
    A = A_j_0 + A_j_1
    return A
    
def get_chi2_comb(t0,teff,data_time,data_flux,data_ferr):
    A_comb = calc_A_comb(t0,teff,data_time)
    fs, fb = get_flux_comb(t0,teff,data_time,data_flux,data_ferr)
    model_flux = A_comb*fs + fb
    chi2s = ((data_flux-model_flux)/data_ferr)**2
    return np.sum(chi2s), chi2s

def get_flux_comb(t0,teff,data_time,data_flux,data_ferr):
    A_comb = calc_A_comb(t0,teff,data_time)
    f1, f0, chi2 = linear_fit(A_comb, data_flux,(1/data_ferr)**2)
    return f1,f0

def chi2_fun_comb(theta,data_time,data_flux,data_ferr):
    return get_chi2_comb(theta[0],theta[1],data_time,data_flux,data_ferr)

def single_fit_2D(t0_init,teff_init,data_time,data_flux,data_ferr):
    initial_guess = [t0_init, teff_init]
    result = op.minimize(chi2_fun_comb, x0=initial_guess,args=(data_time,data_flux,data_ferr),method='Nelder-Mead')
    (fit_t0, fit_teff) = result.x
    return fit_t0, fit_teff

def get_chi2_flat(data_time,data_flux,data_ferr):
    bunsi = np.sum(data_flux/(data_ferr**2))
    bunbo= np.sum(1/(data_ferr**2))
    model_flux = bunsi/bunbo
    return np.sum(((data_flux-model_flux)/data_ferr)**2), ((data_flux-model_flux)/data_ferr)**2

def get_chi2_0fit(data_time,data_flux,data_ferr):
    model_flux = np.zeros(data_time.shape[0])
    return np.sum(((data_flux-model_flux)/data_ferr)**2), ((data_flux-model_flux)/data_ferr)**2

def get_lc_comb(t0,teff,f1,f0):
    t_range = [t0-3*teff ,t0+3*teff]
    t_ref = np.linspace(t_range[0],t_range[1],10000)
    f_model = f1* calc_A_comb(t0,teff,t_ref) + f0
    lc_model = np.empty(len(t_ref), dtype=[("time", float), ("flux", float)])
    lc_model["time"] = t_ref
    lc_model["flux"] = f_model
    return lc_model

#------------------------------------------------------------------------------------#
