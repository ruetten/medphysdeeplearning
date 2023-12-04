import pickle as pickle
from keras import regularizers 
from utils.models import Parameters, CNN_Net
#from keras.optimizers import Adam
from keras.optimizers.legacy import Adam

import sys
# TODO
sys.path.append('//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN/utils')
print(sys.version)

target_rows = 91
target_cols = 109
depth = 91
axis = 1
num_clinical = 2
CNN_drop_rate = 0.3
RNN_drop_rate = 0.1
CNN_w_regularizer = regularizers.l2(2e-2)
RNN_w_regularizer = regularizers.l2(1e-6)
CNN_batch_size = 10
RNN_batch_size = 5
val_split = 0.2
optimizer = Adam(lr=1e-5)
final_layer_size = 5

# TODO
model_filepath = './dir'
# model_filepath = '//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN'

params_dict = { 'CNN_w_regularizer': CNN_w_regularizer, 'RNN_w_regularizer': RNN_w_regularizer,
               'CNN_batch_size': CNN_batch_size, 'RNN_batch_size': RNN_batch_size,
               'CNN_drop_rate': CNN_drop_rate, 'epochs': 2,
          'gpu': "/gpu:0", 'model_filepath': model_filepath, 
          'image_shape': (target_rows, target_cols, depth, axis),
          'num_clinical': num_clinical,
          'final_layer_size': final_layer_size,
          'optimizer': optimizer, 'RNN_drop_rate': RNN_drop_rate,}

params = Parameters(params_dict)
netCNN = CNN_Net(params)
netCNN.load_the_weights("dir/SavedCNNWeights")
print(netCNN)
# pickle_in = open(model_filepath+'/'+picklename+'.pickle', 'rb') 
# pickle0=pickle.load(pickle_in)
# pickle_in.close()
# test_data = pickle0[5][0]
# pickle0 = 0  #to save memory
# test_lossCNN, test_accCNN  = netCNN.evaluate(test_data)
# test_predsCNN = netCNN.predict(test_data)
# print('check_lossCNN, check_accCNN: '+ str(test_lossCNN)+', '+ str(test_accCNN))
