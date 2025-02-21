import sys
sys.path.append("./source/") #AnomalyFinder.pyが入っているpathを指定
import numpy as np
import matplotlib.pyplot as plt
import AnomalyFinder as af

data_path = sys.argv[1]

finder = af.AnomalyFinder()
finder.set_data(data_path,subtract_2450000=True)

finder.PSPL_fit()

print("t0= ",finder.t0_PSPL)
print("tE= ",finder.tE_PSPL)
print("u0= " ,finder.u0_PSPL)

finder.plot_residual_from_PSPL()
plt.show()
finder.normalize_error(toeff=1)

finder.run_grid_search(teff_init=0.3,teff_grid=10,common_ratio=4/3)

finder.show_grid_search_result(s=1)
plt.show()

finder.show_anomaly_signal(which="zero",markersize=3,capsize=2)
finder.show_anomaly_signal(which="flat",markersize=3,capsize=2)

plt.show()


