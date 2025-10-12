import numpy as np
import pandas as pd
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from utils.eval_utils import EvalUtils


# Input features (Visual and audio features = total of 28 features with 12 visual & 16 audio features)
IND_xij_U1V_rot = [i for i in range(0,3)]		
IND_xij_U1V_au = [i for i in range(3,3+7)]		
IND_xij_U1V_gaze = [i for i in range(3+7,3+7+2)]
IND_xij_U1V = np.concatenate((IND_xij_U1V_rot, IND_xij_U1V_au, IND_xij_U1V_gaze), axis=0)	# Head Rotation(x,y,z) + Upper AUs (1,2,4,5,6,7) + Smile(AU12) + Gaze(x,y) of U1
IND_xij_U1A = [i for i in range(12, 28)]
IND_xij_U1 = np.concatenate((IND_xij_U1V, IND_xij_U1A), axis=0)

IND_xij_U2V_rot = [i for i in range(28,28+3)]
IND_xij_U2V_au = [i for i in range(28+3,28+3+7)]
IND_xij_U2V_gaze = [i for i in range(28+3+7,28+3+7+2)]
IND_xij_U2V = np.concatenate((IND_xij_U2V_rot, IND_xij_U2V_au, IND_xij_U2V_gaze), axis=0)	# Head Rotation(x,y,z) + Upper AUs (1,2,4,5,6,7) + Smile(AU12) + Gaze(x,y) of U2
IND_xij_U2A = [i for i in range(28+12, 28+28)]
IND_xij_U2 = np.concatenate((IND_xij_U2V, IND_xij_U2A), axis=0)

IND_xij = np.concatenate((IND_xij_U1, IND_xij_U2), axis=0)
IND_xij_inv = np.concatenate((IND_xij_U2, IND_xij_U1), axis=0)

# Output features
IND_yij_U1V_rot = IND_xij_U1V_rot	
IND_yij_U1V_au = IND_xij_U1V_au
IND_yij_U1V_gaze = IND_xij_U1V_gaze
IND_yij_U2V_rot = [i for i in range(12,12+3)]
IND_yij_U2V_au = [i for i in range(12+3,12+3+7)]
IND_yij_U2V_gaze = [i for i in range(12+3+7,12+3+7+2)]

IND_yij_U1V = np.concatenate((IND_yij_U1V_rot, IND_yij_U1V_au, IND_yij_U1V_gaze), axis=0)
IND_yij_U2V = np.concatenate((IND_yij_U2V_rot, IND_yij_U2V_au, IND_yij_U2V_gaze), axis=0)


class DataGenerator(keras.utils.Sequence):
	'''Data Augmentation'''
	def __init__(self, Xij, yij, batch_size=256, shuffle=True):
		self.Xij = np.copy(Xij)
		self.Xij_perm = np.copy(self.Xij)
		self.yij = np.copy(yij)
		self.list_IDs = np.arange(len(self.Xij))
		self.indexes = np.arange(len(self.Xij))
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
				np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp, do_perm):
		'Generates data containing batch_size samples with the option of randomly permutating the sample input feature order (U1&U2 or U2&U1)'
		batch_X = self.Xij[list_IDs_temp,:,:]
		batch_y = []
		# Generate permutated data (U2&U1)
		if do_perm:
				batch_X[:,:,IND_xij] = batch_X[:,:,IND_xij_inv]
				for feat in range(int(self.yij.shape[2]/2)):
						batch_y.append(self.yij[list_IDs_temp,:,feat])
		# Generate non-permutated data (U1&U2)
		else:
				for feat in range(int(self.yij.shape[2]/2)):
						batch_y.append(self.yij[list_IDs_temp,:,feat+int(self.yij.shape[2]/2)])
		return batch_X, batch_y
					
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		# Generate randomly if the batch will be permuted
		# Permutation of user feature indexes (U1 & U2 or U2 & U1; when batch_X:U1&U2 => batch_Y:U2, else batch_X:U2&U1 => batch_Y:U1)
		do_perm = np.random.choice(a = [False,True], p = [0.5,0.5])
		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		# Generate data
		batch_X, batch_y = self.__data_generation(list_IDs_temp, do_perm)
		return batch_X, batch_y


class MultiHeadAttention(tf.keras.layers.Layer):
	'''Self-Attention Pruning'''
	def __init__(self, d_model, num_heads, causal=False, dropout=0.0):
		super(MultiHeadAttention, self).__init__()
		assert d_model % num_heads == 0
		depth = d_model // num_heads
		self.num_heads = num_heads
		self.w_query = tf.keras.layers.Dense(d_model)
		self.split_reshape_query = tf.keras.layers.Reshape((-1,num_heads,depth))  
		self.split_permute_query = tf.keras.layers.Permute((2,1,3))      
		self.w_value = tf.keras.layers.Dense(d_model)
		self.split_reshape_value = tf.keras.layers.Reshape((-1,num_heads,depth))
		self.split_permute_value = tf.keras.layers.Permute((2,1,3))
		self.w_key = tf.keras.layers.Dense(d_model)
		self.split_reshape_key = tf.keras.layers.Reshape((-1,num_heads,depth))
		self.split_permute_key = tf.keras.layers.Permute((2,1,3))
		self.attention = tf.keras.layers.Attention(causal=causal, dropout=dropout)
		self.join_permute_attention = tf.keras.layers.Permute((2,1,3))
		self.join_reshape_attention = tf.keras.layers.Reshape((-1,d_model))
		self.pruning = Dense(num_heads, activation='hard_sigmoid')
		self.split_permute_pruning = tf.keras.layers.Permute((3,2,1))
		self.dense = tf.keras.layers.Dense(d_model)

	def call(self, inputs, mask=None, pruning=False):
		q = inputs[0]
		v = inputs[1]
		k = inputs[2] if len(inputs) > 2 else v

		query = self.w_query(q)
		query = self.split_reshape_query(query)    
		query = self.split_permute_query(query)
		value = self.w_value(v)
		value = self.split_reshape_value(value)
		value = self.split_permute_value(value)
		key = self.w_key(k)
		key = self.split_reshape_key(key)
		key = self.split_permute_key(key)

		if mask is not None:
			if mask[0] is not None:
				mask[0] = tf.keras.layers.Reshape((-1,1))(mask[0])
				mask[0] = tf.keras.layers.Permute((2,1))(mask[0])
			if mask[1] is not None:
				mask[1] = tf.keras.layers.Reshape((-1,1))(mask[1])
				mask[1] = tf.keras.layers.Permute((2,1))(mask[1])

		attention = self.attention([query, value, key], mask=mask)
		
		if pruning:
			pruning_mask = self.pruning(self.split_permute_pruning(attention))
			pruning_mask = tf.round(pruning_mask)
			pruning_mask = self.split_permute_pruning(pruning_mask)
			attention = attention*pruning_mask

		attention = self.join_permute_attention(attention)
		attention = self.join_reshape_attention(attention)

		x = self.dense(attention)

		return x


class NumpyEncoder(json.JSONEncoder):
	""" Special json encoder for numpy types """
	def default(self, obj):
			if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
													np.int16, np.int32, np.int64, np.uint8,
													np.uint16, np.uint32, np.uint64)):
					return int(obj)
			elif isinstance(obj, (np.float_, np.float16, np.float32,
														np.float64)):
					return float(obj)
			elif isinstance(obj, (np.ndarray,)):
					return obj.tolist()
			return json.JSONEncoder.default(self, obj)


def build_model(params_config, params_model):
	'Build ASAP model'
	cell_lstm = params_model["cell_lstm"]
	cell_dense = params_model["cell_dense"]
	cell_att = params_model["cell_multiheadatt"]
	num_head_att = params_model["num_head_multiheadatt"]
	pruning_stat = params_model["pruning_stat"]
	nb_inputs = params_config["nb_inputs"]
	in_seq_len = params_config["in_seq_len"]

	inputs = keras.layers.Input(shape=(in_seq_len, nb_inputs))
	mha = MultiHeadAttention(d_model=cell_att,num_heads=num_head_att)
	x = mha([inputs,inputs,inputs], pruning=pruning_stat)
	x = LSTM(cell_lstm, input_shape=(in_seq_len, nb_inputs))(x)
	x = Dense(cell_dense, activation='relu')(x)

	output_RotXYZ = Dense(3, activation='linear', name='RotXYZ')(x)
	output_au1_intensity = Dense(1, activation='relu', name='AU1_int')(x)
	output_au2_intensity = Dense(1, activation='relu', name='AU2_int')(x)
	output_au4_intensity = Dense(1, activation='relu', name='AU4_int')(x)
	output_au5_intensity = Dense(1, activation='relu', name='AU5_int')(x)
	output_au6_intensity = Dense(1, activation='relu', name='AU6_int')(x)
	output_au7_intensity = Dense(1, activation='relu', name='AU7_int')(x)
	output_au12_intensity = Dense(1, activation='relu', name='AU12_int')(x)
	output_GazeXY = Dense(2, activation='linear', name='GazeXY')(x)
	output_RotX = tf.identity(output_RotXYZ[:,0], name="RotX")
	output_RotY = tf.identity(output_RotXYZ[:,1], name="RotY")
	output_RotZ = tf.identity(output_RotXYZ[:,2], name="RotZ")
	output_GazeX = tf.identity(output_GazeXY[:,0], name="GazeX")
	output_GazeY = tf.identity(output_GazeXY[:,1], name="GazeY")

	output_list = [output_RotX, output_RotY, output_RotZ,\
									output_au1_intensity, output_au2_intensity, output_au4_intensity, output_au5_intensity, output_au6_intensity, output_au7_intensity, output_au12_intensity,\
									output_GazeX, output_GazeY]

	model = Model(inputs=inputs, outputs=output_list)

	loss_list = ['mse','mse','mse',\
								'mse','mse','mse','mse','mse','mse','mse',\
								'mse','mse']
									
	model.compile(loss=loss_list, optimizer='adam')

	return model

def train_model(model_tr, xij_tr, yij_tr, xij_val, yij_val, params_train, params_generator, dir_path_batchtr="../trainedASAP"):
	'Train ASAP model'
	nb_epoch = params_train["nb_epoch"]
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=1e-7, cooldown=1)
	early_stop = EarlyStopping(monitor='val_loss', patience=50)
	os.makedirs(dir_path_batchtr, exist_ok=True)
	# Checkpoint
	dir_weight_path_batchtr = dir_path_batchtr + '/weights'
	os.makedirs(dir_weight_path_batchtr, exist_ok=True)
	checkpoint_name = dir_weight_path_batchtr + '/best_weights-{epoch}.hdf5'
	checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
	# Logger for history
	dir_hist_path_batchtr = dir_path_batchtr + '/histories'
	os.makedirs(dir_hist_path_batchtr, exist_ok=True)
	logger_name = dir_hist_path_batchtr + '/history_log.csv'
	logger = CSVLogger(logger_name, append=True, separator=',')
	# Tensorboard log
	dir_tensorboard_log = dir_path_batchtr + "/tensorboard_logs"
	os.makedirs(dir_tensorboard_log, exist_ok=True)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_tensorboard_log)
	# Generators
	training_generator = DataGenerator(xij_tr, yij_tr, **params_generator)
	validation_generator = DataGenerator(xij_val, yij_val, **params_generator)
	# Fit
	history_callback = model_tr.fit(training_generator,validation_data=validation_generator,epochs=nb_epoch,use_multiprocessing=False, callbacks=[reduce_lr,early_stop,checkpoint,logger,tensorboard_callback], verbose=2)

def save_prediction(pred_dict, dir_predBase_path = '../predictions'):
	'Save prediction made by the ASAP model'
	os.makedirs(dir_predBase_path, exist_ok=True)
	pred_name = dir_predBase_path + '/prediction'
	dumped = json.dumps(pred_dict, cls=NumpyEncoder)
	with open(pred_name + '.json', 'w') as f:
			f.write(dumped)
	print('Saved Prediction')

def predict_model(model, xij_test, params_config, save=False):
	'Prediction via the ASAP model'
	in_seq_len = params_config["in_seq_len"]
	sample_range_end = len(xij_test)
	pred_dict = {}
	for s in range(0,sample_range_end):
			pred_t_U1_list_s = []
			timestep_range_end = xij_test[s].shape[0]-in_seq_len
			for t in range(timestep_range_end):
					if (t==0):
							#Flip U1&U2 to predict U1
							xij_test_t_U1 = np.copy(xij_test[s][t:t+in_seq_len,IND_xij_U1])
							xij_test_t_U2 = np.copy(xij_test[s][t:t+in_seq_len,IND_xij_U2])
							xij_test_t = np.concatenate((xij_test_t_U2, xij_test_t_U1),axis=1)
					# Autoregression
					else:
							xij_test_t = np.delete(xij_test_t, [0], axis=0)
							# Use previous prediction as input (rot and au of U1) & Rest as GT (Audio of U1 and All Audio & Vis of U1)
							xij_test_new_t_U1V = np.copy(pred_t_U1)
							xij_test_new_t_U1A = np.copy(xij_test[s][t+in_seq_len-1,IND_xij_U1A])
							xij_test_new_t_U1 = np.concatenate((xij_test_new_t_U1V, xij_test_new_t_U1A))
							xij_test_new_t_U2 = np.copy(xij_test[s][t+in_seq_len-1,IND_xij_U2])
							#Flip U1&U2 to predict U1
							xij_test_new_t = np.concatenate((xij_test_new_t_U2, xij_test_new_t_U1))
							xij_test_new_t = np.reshape(xij_test_new_t,(1,len(xij_test_new_t)))
							xij_test_t = np.vstack((xij_test_t,xij_test_new_t))
					xij_test_t_in = np.reshape(xij_test_t,(1,xij_test_t.shape[0],xij_test_t.shape[1]))
					pred_t = model.predict_on_batch(xij_test_t_in)
					pred_t_U1 = pred_t
					pred_t_U1[:len(IND_yij_U1V_rot)] = [pred[0] for pred in pred_t[:len(IND_yij_U1V_rot)]]
					pred_t_U1[len(IND_yij_U1V_rot):len(IND_yij_U1V_rot)+len(IND_yij_U1V_au)] = [pred[0,0] for pred in pred_t[len(IND_yij_U1V_rot):len(IND_yij_U1V_rot)+len(IND_yij_U1V_au)]]
					pred_t_U1[len(IND_yij_U1V_rot)+len(IND_yij_U1V_au):] = [pred[0] for pred in pred_t[len(IND_yij_U1V_rot)+len(IND_yij_U1V_au):]]
					pred_t_U1_list_s.append(pred_t_U1)
			# Reset states for new sample
			model.reset_states()
			# Save pred of each sample into dictionary
			key_s = "Sample" + str(s)
			pred_dict[key_s] = pred_t_U1_list_s
			print('Sample'+str(s)+' complete')

	if save:
			save_prediction(pred_dict)

	return pred_dict

def evaluate_model(model, xij_test, params_config, data_path, saved_prediction=False):
	'Evaluate the model'
	if not save_prediction:
			pred_dict = predict_model(model, xij_test, params_config, save=True)
	else:
			pred_dict = load_prediction()

	in_seq_len = params_config["in_seq_len"]
	sample_range_end = len(xij_test)

	# RMSE
	rmse_mean = 0
	for s in range(0,sample_range_end):
			yij_test_U1_s = np.copy(xij_test[s][in_seq_len:,IND_xij_U1V])
			yij_pred_U1_s = np.array(pred_dict["Sample"+str(s)])
			[rmse_mean_s, rmse_all_feat_s] = EvalUtils.rmse_fct(yij_test_U1_s,yij_pred_U1_s)
			rmse_mean += rmse_mean_s
	rmse_mean /= 4

	# KS Test
	yij_tr_U1 = load_train_data_labels(data_path)
	yij_tr_U1 = np.reshape(yij_tr_U1, (yij_tr_U1.shape[0]*yij_tr_U1.shape[1],yij_tr_U1.shape[2]))
	pred_U1 = np.array(pred_dict["Sample0"])
	for s in range(1,len(xij_test)):
			pred_U1_s = np.array(pred_dict["Sample"+str(s)])
			pred_U1 = np.vstack((pred_U1,pred_U1_s))

	ks_mean = 0
	for f in range(pred_U1.shape[1]):
			ks_f = EvalUtils.ks_test(yij_tr_U1, pred_U1, f)
			ks_mean += ks_f[0]
	ks_mean /= pred_U1.shape[1]

	# DTW between ¬PA&PB and Ground Truth(PA&PB) for smile(AU12)
	au12_idx = 9

	feat_list = ['RotX', 'RotY', 'RotZ', 'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU12', 'GazeX', 'GazeY']
	dtw_df_real = pd.DataFrame(columns=[feat_list[au12_idx]], dtype=object)
	dtw_df_pred = pd.DataFrame(columns=[feat_list[au12_idx]], dtype=object)
	for s in range(0,sample_range_end):
			au12_real_U1 = np.copy(xij_test[s][in_seq_len:,au12_idx])
			au12_real_U2 = np.copy(xij_test[s][in_seq_len:,len(IND_yij_U1V)+au12_idx])
			au12_pred_U2 = np.array(pred_dict["Sample"+str(s)])[:,au12_idx]

			dtw_au12_real_s = EvalUtils.dtw_fct(au12_real_U1,au12_real_U2)
			dtw_au12_pred_s = EvalUtils.dtw_fct(au12_pred_U2,au12_real_U2)

			dtw_series_real = pd.Series(dtw_au12_real_s, index = dtw_df_real.columns)
			dtw_series_pred = pd.Series(dtw_au12_pred_s, index = dtw_df_pred.columns)
			dtw_df_real = dtw_df_real.append(dtw_series_real, ignore_index = True)
			dtw_df_pred = dtw_df_pred.append(dtw_series_pred, ignore_index = True)

	# # Mean of samples 
	# DTW of ASAP predictions(¬PA&PB)
	dtw_mean_pred = pd.Series(dtw_df_pred[~dtw_df_pred.isin([np.nan,np.inf,-np.inf]).any(1)].mean(axis=0), index = dtw_df_pred.columns).values[0]
	# DTW of Ground Truth(PA&PB)
	dtw_mean_gt = pd.Series(dtw_df_real[~dtw_df_real.isin([np.nan,np.inf,-np.inf]).any(1)].mean(axis=0), index = dtw_df_real.columns).values[0]

	return [rmse_mean, ks_mean, dtw_mean_pred, dtw_mean_gt]


def load_training_data(data_path):
	'Load preprocessed train and validation datasets'
	xij_tr = np.load(data_path+"/Xij_train.npy", allow_pickle=True).astype(np.float32)
	yij_tr = np.load(data_path+"/Yij_train.npy", allow_pickle=True).astype(np.float32)
	xij_val = np.load(data_path+"/Xij_val.npy", allow_pickle=True).astype(np.float32)
	yij_val = np.load(data_path+"/Yij_val.npy", allow_pickle=True).astype(np.float32)

	# Randomize training dataset (shuffle samples)
	np.random.seed(42)
	np.random.shuffle(xij_tr)
	np.random.seed(42)
	np.random.shuffle(yij_tr)

	return [xij_tr, yij_tr, xij_val, yij_val]

def load_test_data(data_path):
	'Load preprocessed test dataset'
	xij_test = np.load(data_path+"/Xij_test.npy", allow_pickle=True)

	return xij_test

def load_pretrained_model(params_config, params_model, best_weight_path='../trainedASAP/weights'):
	'Load pretrained ASAP model'
	model_tr = build_model(params_config, params_model)
	checkpoint_filepath = best_weight_path + '/best_weights.hdf5'
	model_tr.load_weights(checkpoint_filepath)

	return model_tr

def load_train_data_labels(data_path):
	'Load preprocessed train dataset labels'
	yij_tr = np.load(data_path+"/YijTrain_inseq100_outseq1_stride1.npy",allow_pickle=True, mmap_mode='r')[0].astype(np.float32)
	yij_tr = yij_tr[:,:,IND_yij_U1V]
	return yij_tr
				
def load_prediction():
	'Load prediction made by the ASAP model'
	dir_predBase_path = '../predictions'
	pred_name = dir_predBase_path + '/prediction'
	with open(pred_name + '.json', 'r') as f:
			dumped_pred_dict = f.read()
	pred_dict = json.loads(dumped_pred_dict)
	print('Loaded Prediction')
	return pred_dict




