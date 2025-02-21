from utils import *
import numpy as np
import time

def search_init_params(data,t_range=False,teff_init=3,common_ratio=2,dt0_coeff=1/6,teff_coeff=3,teff_grid=10,sigma=5,nout=10):
    teff_k = teff_init * np.power(common_ratio, np.arange(teff_grid))

    if t_range:
        t0_start, t0_end = t_range[0],t_range[1]
    else:
        t0_start, t0_end = np.min(data["time"]), np.max(data["time"])

    chi2_list, t0_ref_list, teff_ref_list, chi2_flat_list, chi2_0fit_list, d_chi2_flat_list, chi2_0fit_list, d_chi2_0fit_list,n_outs_flat,n_outs_zero =[],[],[],[],[],[],[],[],[],[]
    
    t_before = time.time()
    for teff_ref in teff_k:
        dt0 = dt0_coeff * teff_ref

        t0_j = np.arange(t0_start, t0_end+dt0,dt0)

        for t0_ref in t0_j:
            cond1, cond2 = t0_ref-teff_coeff*teff_ref <data["time"], data["time"] < t0_ref+teff_coeff*teff_ref #調節パラメータ
            ind=np.where(cond1&cond2)[0]
            if ind.shape[0] < 4:
                continue
            else:
                chi2, res = get_chi2_comb(t0_ref,teff_ref,data["time"][ind],data["flux"][ind],data["ferr"][ind])
                chi2_flat, res_flat  = get_chi2_flat(data["time"][ind],data["flux"][ind],data["ferr"][ind])
                chi2_0fit, res_0fit = get_chi2_0fit(data["time"][ind],data["flux"][ind],data["ferr"][ind])
                n_out_flat = int(np.shape(np.where((res-res_flat)>(sigma**2))[0])[0])
                n_out_zero = int(np.shape(np.where((res-res_0fit)>(sigma**2))[0])[0])
            dchi2_flat = chi2_flat-chi2
            dchi2_0= chi2_0fit-chi2
            chi2_list.append(chi2/ind.shape[0])
            t0_ref_list.append(t0_ref)
            teff_ref_list.append(teff_ref)
            chi2_flat_list.append(chi2_flat)
            chi2_0fit_list.append(chi2_0fit)
            d_chi2_flat_list.append(dchi2_flat)
            d_chi2_0fit_list.append(dchi2_0)
            n_outs_flat.append(n_out_flat)
            n_outs_zero.append(n_out_zero)
    t_after = time.time()
    
    print(f"Total time for the PSPL initial params search: {t_after-t_before} seconds")
            
    chi2_init_array = np.empty(len(t0_ref_list), dtype=[("t0", float), ("teff", float), ("chi2_flat", float), ("chi2_zero", float), ("nout_flat",float), ("nout_zero",float),("chi2",float)])
    
    chi2_init_array["t0"], chi2_init_array["teff"]= np.array(t0_ref_list), np.array(teff_ref_list)
    chi2_init_array["chi2_flat"], chi2_init_array["chi2_zero"]= np.array(d_chi2_flat_list), np.array(d_chi2_0fit_list)
    chi2_init_array["chi2"]= np.array(chi2_list)
    chi2_init_array["nout_flat"] = np.array(n_outs_flat)
    chi2_init_array["nout_zero"] = np.array(n_outs_zero)   
    cand_ind = np.where(chi2_init_array["nout_flat"]>=nout)[0]
    tmp = chi2_init_array[cand_ind]
    best_grid_ind = np.nanargmax(tmp["chi2_flat"]) 
    t0_init, teff_init,chi2_init = tmp["t0"][best_grid_ind], tmp["teff"][best_grid_ind], tmp["chi2"][best_grid_ind]

    return t0_init, teff_init, chi2_init
    
def auto_single_fit(data,nout):
    t0_init, teff_init, chi2_init = search_init_params(data,nout=nout)
    u0_inits = np.array([0.5,0.4,0.6,0.3,0.7,0.2,0.8,0.1,0.9,0.05,0.01])
    tE_inits = teff_init/u0_inits

    t0_list, tE_list, u0_list, chi2_list = [],[],[],[]
    for tE_init, u0_init in zip(tE_inits, u0_inits):
        t0, tE, u0 = single_fit(t0_init, tE_init, u0_init,data["time"],data["flux"],data["ferr"])
        chi2 = get_chi2_single(t0,tE,u0,data["time"],data["flux"],data["ferr"])
        t0_list.append(t0)
        tE_list.append(tE)
        u0_list.append(u0)
        chi2_list.append(chi2)
    t0_list, tE_list, u0_list, chi2_list = np.array(t0_list), np.array(tE_list), np.array(u0_list), np.array(chi2_list)
    best_ind = np.argmin(chi2_list)
    return t0_list[best_ind],tE_list[best_ind],u0_list[best_ind], chi2_init