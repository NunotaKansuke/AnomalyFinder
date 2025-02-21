import numpy as np
import matplotlib.pyplot as plt
import MulensModel as mm

class GenMockCurve():

    def get_base_line(self,path):
        self.raw_data = np.genfromtxt(path,usecols=[0,1,2],names=["time","flux","ferr"])
        ind = np.argmin(self.raw_data["flux"])
        offset = np.abs(self.raw_data["flux"][ind])+self.raw_data["ferr"][ind]+1 #MulensModelはマイナスのフラックスを扱えないため.
        self.raw_data["flux"] = self.raw_data["flux"]+np.abs(self.raw_data["flux"][ind])+offset
        
    def set_params(self,param_dict,coeff=5): #coeff: finite sourceを考慮する時間を指定する.
        self.param_dict = param_dict
        self.model=mm.Model(param_dict)
        self.t_range = [param_dict["t_0"]-coeff*param_dict["t_E"],param_dict["t_0"]+coeff*param_dict["t_E"]]
        self.model.set_magnification_methods([self.t_range[0], 'VBBL', self.t_range[1]])
        self.model.set_default_magnification_method("point_source_point_lens")
        self.model_single_test=mm.Model({'t_0': param_dict["t_0"], 'u_0': param_dict["u_0"], 't_E': param_dict["t_E"]})
        
    def plot_model(self,caustics=True,t_range=False):
        self.model.plot_magnification(label="Binary Model")
        self.model_single_test.plot_magnification(label="PSPL Model")
        
    def plot_caustic(self):
        self.model.plot_trajectory(caustics=True)
            
    def make_mock_flux(self, fs_in=30000):
        self.fs_in = fs_in
        model_magnification = self.model.get_magnification(self.raw_data["time"])
        self.mock_flux = fs_in*(model_magnification-1)+self.raw_data["flux"]
        self.mock_data = mm.MulensData([self.raw_data["time"],self.mock_flux,self.raw_data["ferr"]],phot_fmt="flux")
        
    def output(self,path,subtract_median=True):
        if subtract_median:
            subtract = np.median(self.mock_data.flux)
        else:
            subtract = 0
        with open(path,"a") as f:
            f.write(f"#{self.param_dict} fs= {self.fs_in}\n")
            for i in range(self.mock_data.time.shape[0]):
                f.write(f"{self.mock_data.time[i]}         {self.mock_data.flux[i]-subtract}         {self.mock_data.err_flux[i]} \n")