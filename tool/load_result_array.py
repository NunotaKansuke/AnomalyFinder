import numpy as np
import matplotlib.pyplot as plt

data_path = "/Users/nunotahiroshisuke/Desktop/iral/work/AnomalyFinder/tool/CheckResult/MB21431.npy"
data = np.load(data_path)

nout = 3
cand_ind = np.where(data["nout_flat"]>=nout)
not_cand_ind = np.where(data["nout_flat"]<nout)

#plt.scatter(data["t0"][not_cand_ind],data["chi2_flat"][not_cand_ind])
plt.scatter(data["t0"][cand_ind],data["chi2_flat"][cand_ind])
plt.show()
