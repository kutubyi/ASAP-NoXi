import tensorflow as tf
from utils.ASAP_utils import load_training_data, build_model, train_model


# Use GPU
physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')
if (len(physical_device) >= 1):
    if (tf.config.experimental.get_memory_growth(physical_device[0]) == 1):
        tf.config.experimental.set_memory_growth(physical_device[0],True)


# Params for Config
params_config = {'nb_inputs' : 56,
                'in_seq_len' : 100,
                }
# Params for Training
params_train = {'nb_epoch' : 1000,
                }
# Params for Model
params_model = {'cell_multiheadatt' : 16*4,
                'num_head_multiheadatt' : 4,
                'pruning_stat' : True,
                'cell_lstm' : 20,
                'cell_dense' : 20,
                }
# Params for Generators
params_generator = {'batch_size' : 32,
                      'shuffle' : True}


if __name__ == "__main__":
  '''
  Training of ASAP model
  '''
  data_path = 'data/processed'

  [xij_tr, yij_tr, xij_val, yij_val] = load_training_data(data_path)

  model_tr = build_model(params_config, params_model)
  train_model(model_tr, xij_tr, yij_tr, xij_val, yij_val, params_train, params_generator)

