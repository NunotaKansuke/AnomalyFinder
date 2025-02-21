import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
from matplotlib import rcParams
from utils import *
from SingleFit import *
from tqdm import tqdm
rcParams["font.size"] = 10
rcParams["axes.linewidth"] = 3
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 4
rcParams['xtick.minor.width'] = 1.5
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 4
rcParams['ytick.minor.width'] = 1.5
warnings.filterwarnings("ignore", category=np.RankWarning)

class AnomalyFinder():
    
    def set_data(self,path,subtract_2450000=False):
        self.data = np.genfromtxt(path,usecols=[0,1,2],names=["time","flux","ferr"])
        if subtract_2450000:
            self.data["time"]-=2450000
            
        del_ind = np.where((self.data["time"] < 1000) | (self.data["ferr"] <= 0)| np.isnan(self.data["flux"])|(np.abs(self.data["flux"])>10**7))
        self.data = np.delete(self.data, del_ind)
        
    def cut_data(self,t_range):
        cond1, cond2 = t_range[0] <self.data["time"], self.data["time"] < t_range[1]
        ind=np.where(cond1& cond2)[0]
        self.data = self.data[ind]
        
    def plot_data(self,t_range=False,**kwargs):
        if not t_range:
            t_range = [np.min(self.data["time"]),np.max(self.data["time"])]
        cond1, cond2 = t_range[0] <self.data["time"], self.data["time"] < t_range[1]
        ind=np.where(cond1&cond2)[0]
        plt.errorbar(self.data["time"][ind],self.data["flux"][ind],yerr=self.data["ferr"][ind],fmt="o",**kwargs)
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.minorticks_on()
        
    def PSPL_fit(self,nout=10,index=False):
        if index:
            selected_data = self.data[index]
        else:
            selected_data = self.data

        self.t0_PSPL, self.tE_PSPL, self.u0_PSPL, self.chi2_init = auto_single_fit(selected_data,nout=nout)

        self.fs_PSPL, self.fb_PSPL = get_flux_single(self.t0_PSPL, self.tE_PSPL, self.u0_PSPL, selected_data["time"], selected_data["flux"], selected_data["ferr"])
        self.teff_PSPL = self.tE_PSPL * self.u0_PSPL
        self.f_PSPL = self.fs_PSPL*single_magnification(self.t0_PSPL,self.tE_PSPL,self.u0_PSPL,self.data["time"]) + self.fb_PSPL
        self.f_residual = self.data["flux"]-self.f_PSPL

    def normalize_error(self,toeff=100):
        base_ind = np.where((self.data["time"] < self.t0_PSPL-self.teff_PSPL*toeff) | (self.data["time"] > self.t0_PSPL+self.teff_PSPL*toeff))[0]
        base_data = self.data[base_ind]
        base_A = single_magnification(self.t0_PSPL,self.tE_PSPL,self.u0_PSPL,base_data["time"])
        base_model_flux = base_A*self.fs_PSPL + self.fb_PSPL
        nout_ind = np.where(((base_data["flux"]-base_model_flux)/base_data["ferr"])**2 < 5**2)[0]
        base_chi2 = np.sum(((base_data["flux"][nout_ind]-base_model_flux[nout_ind])/base_data["ferr"][nout_ind])**2)
        coeff = np.sqrt(base_chi2/(base_data[nout_ind].shape[0]-5))
        self.data["ferr"] *= coeff
        print(f"coefficient for error normalization is {coeff}")
        
    def plot_residual_from_PSPL(self,t_range=False,ax0_ylims=False,save=False, **kwargs):
        if not t_range:
            t_range = [self.t0_PSPL-np.max([10*self.teff_PSPL,10]), self.t0_PSPL+np.max([10*self.teff_PSPL,10])]
        cond1, cond2 = t_range[0] <self.data["time"], self.data["time"] < t_range[1]
        ind=np.where(cond1&cond2)[0]
        self.lc_PSPL = get_lc_PSPL(self.t0_PSPL,self.tE_PSPL,self.u0_PSPL,self.fs_PSPL,self.fb_PSPL,t_range)
        fig, ax = plt.subplots(2, 1, figsize=(7.5, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 3])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.set_title(label=f"t0= {round(self.t0_PSPL,2)}     tE= {round(self.tE_PSPL,2)}     u0= {round(self.u0_PSPL,5)}     chi2init= {int(self.chi2_init)}")

        ax0.plot(self.lc_PSPL["time"],self.lc_PSPL["flux"],c="orange")
        ax0.errorbar(self.data["time"],self.data["flux"],yerr=self.data["ferr"],fmt="o",**kwargs)
        ax0.set_ylabel("Flux")
        ax0.minorticks_on()
        ax0.set_xlim(t_range[0],t_range[1])

        ax1.errorbar(self.data["time"],self.f_residual,yerr=self.data["ferr"],fmt="o",**kwargs)
        ax1.plot([np.min(self.data["time"]), np.max(self.data["time"])], [0, 0], color='orange', linestyle='-')
        ax1.set_ylabel("Residual")
        ax1.set_xlabel("Time")
        ax1.minorticks_on()
        ax1.set_xlim(t_range[0],t_range[1])

        f_residual_t_range = self.f_residual[ind]
        ymax_abs = np.max(np.abs(f_residual_t_range))
        margin = ymax_abs * 0.1
        ax1.set_ylim(- (ymax_abs + margin), ymax_abs + margin)

        if ax0_ylims:
            ax0.set_ylim(ax0_ylims[0], ax0_ylims[1])
        else:
            y_max = np.max(self.lc_PSPL["flux"])
            y_margin = 0.2
            y_max +=  y_margin * y_max
            med = (np.max(self.lc_PSPL["flux"]) + 1) / 2
            y_min = -(y_max - med)
            ax0.set_ylim(y_min,y_max)

        if save:
            plt.savefig(save)

    def run_grid_search(self,t_range=False,teff_init=0.01,common_ratio=4/3,dt0_coeff=1/6,teff_coeff=3,teff_grid=25,sigma=3):
        #grid serchのじかん幅、グリッドの下限、teffのgridの幅、coefisincy係数のいみ、toのgrid幅決まる際の係数、fittingに使うデータ転の幅を決める係数、線の数
        
        self.teff_coeff=teff_coeff #show anomaly signalで使うから
        
        teff_k = teff_init * np.power(common_ratio, np.arange(teff_grid))
        #np.powerで縦軸振る、common_ratioが等比、np.arangeは等間隔の値を持つ配列を作る、gridの値決めれた。
        chi2_list, t0_ref_list, teff_ref_list, chi2_flat_list, chi2_0fit_list, d_chi2_flat_list, chi2_0fit_list, d_chi2_0fit_list,n_outs_flat,n_outs_zero =[],[],[],[],[],[],[],[],[],[]

        if t_range:
            t0_start, t0_end = t_range[0],t_range[1]
        else:
            t0_start, t0_end = np.min(self.data["time"]), np.max(self.data["time"])
            #t_rangeがデフォルトやと、elseが実行されて、toのrangeが決まる。データの最小値と最大値が入る。

        t_before = time.time()
        for teff_ref in teff_k:
            dt0 = dt0_coeff * teff_ref

            t0_j = np.arange(t0_start, t0_end+dt0,dt0)
            
            for t0_ref in t0_j:
                cond1, cond2 = t0_ref-teff_coeff*teff_ref <self.data["time"], self.data["time"] < t0_ref+teff_coeff*teff_ref #調節パラメータ
                ind=np.where(cond1&cond2)[0]
                if ind.shape[0] < 4:
                    continue
                else:
                    chi2, res = get_chi2_comb(t0_ref,teff_ref,self.data["time"][ind],self.f_residual[ind],self.data["ferr"][ind])
                    chi2_flat, res_flat = get_chi2_flat(self.data["time"][ind],self.f_residual[ind],self.data["ferr"][ind])
                    chi2_0fit, res_0fit = get_chi2_0fit(self.data["time"][ind],self.f_residual[ind],self.data["ferr"][ind])
                    n_out_flat = int(np.shape(np.where((res_flat-res)>(sigma**2))[0])[0])
                    n_out_zero = int(np.shape(np.where((res_0fit-res)>(sigma**2))[0])[0])
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
        
        print(f"Total time for the search: {t_after-t_before} seconds")
                
        self.chi2_array = np.empty(len(t0_ref_list), dtype=[("t0",float),("teff",float),("chi2_flat",float),("chi2_zero",float),("nout_flat",float),("nout_zero",float),("chi2",float)])
        
        self.chi2_array["t0"], self.chi2_array["teff"]= np.array(t0_ref_list), np.array(teff_ref_list)
        self.chi2_array["chi2_flat"], self.chi2_array["chi2_zero"]= np.array(d_chi2_flat_list), np.array(d_chi2_0fit_list)
        self.chi2_array["chi2"] = np.array(chi2_list)
        self.chi2_array["nout_flat"] = np.array(n_outs_flat)
        self.chi2_array["nout_zero"] = np.array(n_outs_zero)
        
    def show_grid_search_result(self, nout=3, save=False, **kwargs):
        cand_ind = np.where(self.chi2_array["nout_flat"] >= nout)[0]
        notcand_ind = np.where(self.chi2_array["nout_flat"] < nout)[0]
        rcParams["font.size"] = 12
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        plt.subplots_adjust(left=0.082, right=0.88, top=0.95, bottom=0.08)

        sort_ind_flat = np.argsort(self.chi2_array["chi2_flat"])

        ax[0].scatter(self.chi2_array["t0"][notcand_ind], self.chi2_array["chi2_flat"][notcand_ind], c="C0", **kwargs)
        ax[0].scatter(self.chi2_array["t0"][cand_ind], self.chi2_array["chi2_flat"][cand_ind], c="C1", **kwargs)

        ymin, ymax = ax[0].get_ylim()
        ax[0].scatter(self.t0_PSPL, ymax, s=200, marker="*", c="red",zorder=10)

        im1 = ax[1].scatter(self.chi2_array["t0"][sort_ind_flat], self.chi2_array["teff"][sort_ind_flat],
                            c=self.chi2_array["chi2_flat"][sort_ind_flat], s=10, cmap="jet", marker='o')

        ax[1].set_yscale("log")

        ax[0].minorticks_on()
        ax[1].minorticks_on()

        ax[0].set_ylabel(r"$\Delta\chi^{2}_{flat}$",fontsize=15)
        ax[1].set_xlabel(r"$\rm t_{\rm 0}$",fontsize=15)
        ax[1].set_ylabel(r"$\rm t_{\rm eff}$",fontsize=15)


        cbar_ax = fig.add_axes([0.89, 0.08, 0.03, 0.395])
        cb = fig.colorbar(im1, cax=cbar_ax)
        cbar_ax.minorticks_on()

        if save:
            plt.savefig(save)
        
    def show_anomaly_signal(self,nout=3,which="flat",t_range=False,save=False,**kwargs):
        chi2_array_cand = self.chi2_array[np.where((self.chi2_array["nout_flat"]>=nout)&(self.chi2_array["nout_zero"]>=nout))[0]]
        while chi2_array_cand.shape[0]==0:
            nout-=1
            chi2_array_cand = self.chi2_array[np.where((self.chi2_array["nout_flat"]>=nout)&(self.chi2_array["nout_zero"]>=nout))[0]]
            
        if which == "zero": 
            best_grid_ind = np.argmax(chi2_array_cand["chi2_zero"])
            best_chi2= chi2_array_cand["chi2_zero"][best_grid_ind]
            best_nout = int(chi2_array_cand["nout_zero"][best_grid_ind])
            best_chi2raw = chi2_array_cand["chi2"][best_grid_ind]
        else:
            best_grid_ind = np.argmax(chi2_array_cand["chi2_flat"])
            best_nout = int(chi2_array_cand["nout_flat"][best_grid_ind])
            best_chi2= chi2_array_cand["chi2_flat"][best_grid_ind]
            best_chi2raw = chi2_array_cand["chi2"][best_grid_ind]

        t0_best, teff_best = chi2_array_cand["t0"][best_grid_ind], chi2_array_cand["teff"][best_grid_ind]
        cond1, cond2 = t0_best-self.teff_coeff*teff_best <self.data["time"], self.data["time"] < t0_best+self.teff_coeff*teff_best
        ind=np.where(cond1&cond2)[0]
        
        if not t_range:
            t_range = [t0_best-5*teff_best,t0_best+5*teff_best]
        
        cond3, cond4 = t_range[0]<self.data["time"], self.data["time"] < t_range[1] 
        ind2 = np.where(cond3&cond4)[0]

        f1_best, f0_best = get_flux_comb(t0_best,teff_best,self.data["time"][ind],self.f_residual[ind],self.data["ferr"][ind])

        lc = get_lc_comb(t0_best,teff_best,f1_best,f0_best)
        
        plt.figure(figsize=(6,2))

        plt.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.21)
        
        plt.plot(lc["time"],lc["flux"],c="red")

        plt.errorbar(self.data["time"][ind2],self.f_residual[ind2],yerr=self.data["ferr"][ind2],fmt="o",label=f"Nout={best_nout} chi2={int(best_chi2)}",**kwargs)
        plt.plot([np.min(self.data["time"]), np.max(self.data["time"])], [0, 0], color='orange', linestyle='-')
        plt.title(f"Nout={best_nout} dchi2={int(best_chi2)} chi2={int(best_chi2raw)}")
        
        if not which == "zero": 
            plt.plot([np.min(self.data["time"]), np.max(self.data["time"])], 
                     [np.mean(self.f_residual[ind]), np.mean(self.f_residual[ind])], color='blue', linestyle='-',label="flat")
#             plt.legend(loc="best")

        plt.xlim(t_range[0],t_range[1])
        ylims=[np.min(lc["flux"])-0.6*(np.median(lc["flux"])-np.min(lc["flux"])),np.max(lc["flux"])+0.3*(np.max(lc["flux"])-np.median(lc["flux"]))]
        plt.ylim(ylims[0],ylims[1])
        plt.minorticks_on()
        #plt.legend(loc="best")
        plt.xlabel("Time")
        plt.ylabel("Residual")

        if save:
            plt.savefig(save)
        
    def save(self,file):
        np.save(file,self.chi2_array)
        
#---------------------------------------------------#
if __name__ == "__main__":
    list_path = "suzuki16/list"
    names = np.genfromtxt(list_path,dtype=str)
    i = 17
    tmp= AnomalyFinder()
    
    file_name = f"suzuki16/data/MOAR_{names[i]}.dat"
    print(file_name)
    tmp.set_data(file_name,subtract_2450000=True)
    
    tmp.PSPL_fit()
    
    tmp.normalize_error()
    print(tmp.t0_PSPL,tmp.tE_PSPL,tmp.u0_PSPL)
    tmp.plot_residual_from_PSPL()
    plt.show()
    tmp.run_grid_search(teff_init=1,teff_grid=10)
    #tmp.show_grid_search_result(s=1)
    #plt.show()
    tmp.show_anomaly_signal()
    plt.show()
