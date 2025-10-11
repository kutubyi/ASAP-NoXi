import numpy as np
import math
import scipy.stats as sst
from dtw import dtw


class EvalUtils:
	def rmse_fct(y_real, y_pred, decimals=3):
		'RMSE metric'
		rmse_all_feat = []
		for f in range(y_pred.shape[1]):
			mse_f = np.square(np.subtract(y_real[:,f],y_pred[:,f])).mean() 
			rmse_f = math.sqrt(mse_f)
			rmse_f = np.around(rmse_f, decimals=decimals)
			rmse_all_feat.append(rmse_f)
		rmse_mean = np.mean(rmse_all_feat)
		rmse_mean = np.around(rmse_mean, decimals=decimals)

		return [rmse_mean, rmse_all_feat]

	def ks_test(X, y_pred, feat_idx):
		'KS test'
		X_f = np.copy(X[:,feat_idx])
		y_pred_f = np.copy(y_pred[:,feat_idx])

		rng = 42
		mean_x = np.mean(X_f)
		std_x = np.std(X_f) 
		mean_y = np.mean(y_pred_f)
		std_y = np.std(y_pred_f)

		# Normal continuous random variable for each distribution 
		rvs_x = sst.norm.rvs(size=len(X_f), loc=mean_x, scale=std_x, random_state=rng)
		rvs_y = sst.norm.rvs(size=len(y_pred_f), loc=mean_y, scale=std_y, random_state=rng)

		ks_f = sst.ks_2samp(rvs_x, rvs_y)

		return ks_f

	def dtw_fct(y_real, y_pred, t_len = 25*60, t_stride = 25*30):
		'Dynamic Time Warping (DTW)'
		ts = [t for t in range(0,len(y_real),t_stride)]
		dtw_f_tot = []
		for t in ts:
			y_r = np.float64(y_real[t:t+t_len]).copy(order='C')
			y_p = np.float64(y_pred[t:t+t_len]).copy(order='C')
			alignment = dtw(y_r, y_p, keep_internals=True)
			c_mat = alignment.costMatrix
			dtw_f = alignment.distance
			align_r = alignment.index1
			align_p = alignment.index2
			if (not np.isnan(dtw_f)) and (np.abs(dtw_f) != np.inf):
				dtw_f_tot.append(dtw_f)
		dtw_f = np.mean(dtw_f_tot) / t_len

		return [dtw_f]

