import sys
sys.path.append("../source/") 
import numpy as np
import matplotlib.pyplot as plt
import AnomalyFinder as af

data_path = sys.argv[1]
output_dir = sys.argv[2]

finder = af.AnomalyFinder()
finder.set_data(data_path,subtract_2450000=True)

finder.PSPL_fit()
finder.normalize_error(toeff=1)

finder.plot_residual_from_PSPL()
plt.savefig(output_dir+"residual.png")


finder.run_grid_search(teff_init=0.3,teff_grid=10,common_ratio=4/3)

finder.show_grid_search_result(s=1)
plt.savefig(output_dir+"gird.png")

finder.show_anomaly_signal(which="zero",markersize=3,capsize=2)
plt.savefig(output_dir+"flat_chi2.png")

finder.show_anomaly_signal(which="flat",markersize=3,capsize=2)
plt.savefig(output_dir+"zero_chi2.png")