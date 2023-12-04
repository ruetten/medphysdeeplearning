import tensorflow as tf
from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Reshape, Dense, ELU, concatenate, add, Lambda, MaxPooling2D, GRU, Masking, advanced_activations
from keras.models import Model, save_model 
from keras.optimizers import Adam
from keras import backend as K
from tensorflow import reshape, transpose
from keras.callbacks import LearningRateScheduler
from keras.metrics import binary_crossentropy
from keras import activations
import numpy as np
import math
from keras import activations
from keras.utils import CustomObjectScope
from skimage.transform import resize
from tensorflow.python.framework import ops
import innvestigate
import innvestigate.utils
import os

import sys
sys.path.append('//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN/utils')
from sepconv3D import SeparableConv3D
from augmentation import CustomIterator

model_filepath = '//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN'

####for 2 class model + RNN###

class Parameters():
    def __init__ (self, param_dict):
        self.CNN_w_regularizer = param_dict['CNN_w_regularizer']
        #self.RNN_w_regularizer = param_dict['RNN_w_regularizer']
        self.CNN_batch_size = param_dict['CNN_batch_size']
        #self.RNN_batch_size = param_dict['RNN_batch_size']
        self.CNN_drop_rate = param_dict['CNN_drop_rate']
        #self.RNN_drop_rate = param_dict['RNN_drop_rate']
        self.epochs = param_dict['epochs']
        self.gpu = param_dict['gpu']
        self.model_filepath = param_dict['model_filepath'] + '/net.h5'
        self.num_clinical = param_dict['num_clinical']
        self.image_shape = param_dict['image_shape']
        self.final_layer_size = param_dict['final_layer_size']
        self.optimizer = param_dict['optimizer']
        
class CNN_Net ():
    def __init__ (self, params):
        self.params = params

        self.xls = Input (shape = (self.params.num_clinical,),name='input_xls')
        self.mri = Input (shape = (self.params.image_shape),name='input_mri')
        self.jac = Input (shape = (self.params.image_shape),name='input_jac')
        
        xalex3D = XAlex3D(w_regularizer = self.params.CNN_w_regularizer, drop_rate = self.params.CNN_drop_rate, final_layer_size=self.params.final_layer_size)
    
        with tf.device(self.params.gpu):
            self.fc_CNN = xalex3D (self.mri, self.jac, self.xls) 
            self.CNNoutput_class = Dense(units = 2, activation = 'softmax', name = 'CNNclass_output') (self.fc_CNN)  #use either 1, sigmoid, binarycrossent OR 2, softmax, sparsecategoricalcrossent
        
    def train (self, data):
        train_data, val_data = data

        train_samples = train_data[0].shape[0]  
        val_samples = len(val_data[0])
               
        data_flow_train = CustomIterator (train_data, batch_size = self.params.CNN_batch_size,
                                          shuffle = True)  
        data_flow_val = CustomIterator (val_data, batch_size = self.params.CNN_batch_size,
                                          shuffle = True)
        self.model = Model(inputs = [self.mri,self.jac,self.xls], outputs = [self.CNNoutput_class])
        
        lrate = LearningRateScheduler(step_decay_CNN)    
        callback = [lrate]    
        #optimizer = Adam(lr=1e-5) 
        self.optimizer = self.params.optimizer
        self.model.compile(optimizer = self.optimizer, loss = 'sparse_categorical_crossentropy', metrics =['acc']) 
        self.model.summary()
        
        history = self.model.fit_generator (data_flow_train,
                   steps_per_epoch = train_samples/self.params.CNN_batch_size,
                   epochs = self.params.epochs,
                   callbacks = callback,
                   shuffle = True,  #might be being ignored if the input data is a generator??
                   validation_data = data_flow_val,
                   validation_steps =  val_samples/self.params.CNN_batch_size)
        
        #Save the model
        save_model(self.model,'SavedCNNModel')
        self.model.save_weights('SavedCNNWeights')
        
        #get features from last layer
        featuresModel = Model(inputs = self.model.input, outputs = self.model.layers[-2].output) 
        featuresModel.compile(optimizer = self.params.optimizer, loss = 'sparse_categorical_crossentropy', metrics =['acc']) 
        
        return history.history, featuresModel

    def predict (self, data_test):
        test_mri, test_jac, test_xls, test_labels, test_ptid, test_imageID, test_confid, test_csf = data_test
#        with open('//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class/figchecks/test_data.txt', 'w') as testdata:
#            testdata.write('{}\n'.format(test_data))
#            testdata.write('{}\n{}\n{}\n{}\n{}\n'.format(len(test_data),len(test_data[0]),len(test_data[1]),len(test_data[2]),len(test_data[3])))
        preds = self.model.predict ([test_mri, test_jac, test_xls])
        return preds

    def evaluate (self, data_test):
        test_mri, test_jac, test_xls, test_labels, test_ptid, test_imageID, test_confid, test_csf = data_test
        metrics = self.model.evaluate (x = [test_mri, test_jac, test_xls], y = test_labels, batch_size = self.params.CNN_batch_size)
        return metrics
    
    def load_the_weights (self, SavedWeights):
        self.model = Model(inputs = [self.mri,self.jac,self.xls], outputs = [self.CNNoutput_class])
        self.model.compile(optimizer = self.params.optimizer, loss = 'sparse_categorical_crossentropy', metrics =['acc']) 
        loaded = self.model.load_weights(SavedWeights)
        return loaded
    
    def LRP_heatmap(self, img_data, img_number):    #https://github.com/albermax/innvestigate
        test_mri, test_jac, test_xls, test_labels, test_ptid, test_imageID, test_confid, test_csf = img_data
        #clear some memory: (these are just pointers anyway?)
        test_labels=0
        test_ptid=0
        test_imageID=0
        test_confid=0
        test_csf=0
        print('kill check models130')
        #create the model without the final softmax layer
        nosoftmax_model = innvestigate.utils.model_wo_softmax(self.model)
        print('kill check models135')
        #create the analyzer
        analyzer = innvestigate.create_analyzer("lrp.z",nosoftmax_model,disable_model_checks=True)  
        print('kill check models138')
        ##analyzer = innvestigate.analyzer.LRPZ(nosoftmax_model,disable_model_checks=True)
        #analyze
        analysis = analyzer.analyze([[test_mri[img_number]],[test_jac[img_number]],[test_xls[img_number]]])
        print('shape of initial LRP heatmap: ', analysis.shape)
        analysis /= np.max(np.abs(analysis))
        analysis = np.squeeze(analysis,0)
        print('shape of squeezed LRP heatmap: ', analysis.shape)
        #analysis = np.moveaxis(analysis,0,3)
        #analysis = resize(analysis,(test_mri[img_number].shape))   #maybe this is actually turning the 1 into 91 instead of moving it to the end. maybe try a channels last adjustment or something??
        #analysis = analysis[:,:,:,0,:]
        print('shape of resized LRP heatmap: ', analysis.shape)
        
        return analysis
    
    def make_gradcam_heatmap2(self,img_data,img_number):   #https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-gradcam-554a85dd4e48 and https://keras.io/examples/vision/grad_cam/
        test_mri, test_jac, test_xls, test_labels, test_ptid, test_imageID, test_confid, test_csf = img_data
        last_conv = self.model.layers[-15]    #.get_layer('name')
        
        grads = K.gradients(self.model.output[:,1],last_conv.output)[0]   #[:,x] corresponds to the class I want I think? So AD = 0
        #print('grads: ', grads)
        print('shape of grads: ', grads.shape)
        pooled_grads = K.mean(grads,axis=(0,1,2,3)) 
        print('shape of pooled_grads: ', pooled_grads.shape)
        iterate = K.function([self.model.input[0],self.model.input[1],self.model.input[2]],[pooled_grads,last_conv.output[0]])
        print('shape of test_mri[j]: ', test_mri[img_number].shape)
        pooled_grads_value,conv_layer_output = iterate([[test_mri[img_number]],[test_jac[img_number]],[test_xls[img_number]]])
        
        for i in range(48):  #range = size of conv layer units? aka number of filters/channels
            conv_layer_output[:,:,:,i] *= pooled_grads_value[i]   #conv_layer_output[:,:,i]  #multiplies feature maps with pooled grads
        heatmap = np.mean(conv_layer_output,axis=-1)    #takes the mean over all the filters/channels to get just one map
        #for x in range(heatmap.shape[0]):    #this chunk applies a relu to keep only the features that have a positive influence on the output map
        #    for y in range(heatmap.shape[1]):
        #        heatmap[x,y] = np.max(heatmap[x,y],0)
        print('shape of initial heatmap: ', heatmap.shape)        
        heatmap = np.maximum(heatmap,0)    #keeps only the positive values (only keep the features that have a positive influence on the output map)
        heatmap /= np.max(heatmap)   #normalizes the heatmap to 0-1   #do I actually want to normalize the same way I normed my images? (x-mean)/std. or min-max?
        heatmap = resize(heatmap,(test_mri[img_number].shape))
        print('shape of resized heatmap: ', heatmap.shape)
        
        return  heatmap        
        
    def guided_backprop(self, img_data, img_number):
        """Guided Backpropagation method for visualizing input saliency."""
        #define new model which changes gradient fn for all relu activations acording to Guided Backpropagation
        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * \
                       tf.cast(op.inputs[0] > 0., dtype)

        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
            new_model = self.model
        
        test_mri, test_jac, test_xls, test_labels, test_ptid, test_imageID, test_confid, test_csf = img_data
        layer_output = new_model.layers[-15].output
        grads = K.gradients(layer_output, [new_model.input[0],new_model.input[1],new_model.input[2]])[0]
        backprop_fn = K.function([new_model.input[0],new_model.input[1],new_model.input[2], K.learning_phase()], [grads])
        grads_val = backprop_fn([[test_mri[img_number]],[test_jac[img_number]],[test_xls[img_number]], 0])[0]
        print('shape of initial gb: ', grads_val.shape)
        #grads_val = resize(grads_val,(test_mri[img_number].shape))
        grads_val = grads_val[0]
        print('shape of resized gb: ', grads_val.shape)
        return grads_val

class RNN_Net ():
    def __init__ (self, params):
        self.params = params

        self.fc_CNNt1 = Input (shape = (self.params.final_layer_size,)) #Value corresponds to size of final layer in CNN
        self.fc_CNNt2 = Input (shape = (self.params.final_layer_size,))
        self.fc_CNNt3 = Input (shape = (self.params.final_layer_size,))
        #self.fc_CNN = Input (shape = (3,self.params.final_layer_size,))  #for rnn_bgrus_multiTP
        
        rnn = rnn_bgrus(drop_rate=self.params.RNN_drop_rate, final_layer_size = self.params.final_layer_size, kernel_regularizer=self.params.RNN_w_regularizer)
    
        with tf.device(self.params.gpu):           
            self.fc_RNN = Lambda(rnn, name='rnn')([self.fc_CNNt1,self.fc_CNNt2,self.fc_CNNt3]) #original call without prior masking
            #self.fc_RNN = Lambda(rnn, name='rnn')(self.fc_CNN) #for multi-TP gru
            print('Shape of self.fc_RNN: ', self.fc_RNN.shape)

            self.RNNoutput_class= Dense(units = 2, activation = 'softmax', name = 'RNNclass_output') (self.fc_RNN) #switch to sigmoid from softmax? (for 2 class? for non-sparse categorical?) #back to softmax for multi-class

    def train (self, data):
        train_data, train_labels, val_data, val_labels = data
        #data is now loaded in keeping all scans from same patient aligned across timepoints. 
        #That way train_labels can be just one array which applies to all timepoints (necessary because I only ask for 1 output when I define the model!)

        print('train data shape: ', train_data[0].shape)
        print('train labels shape: ', train_labels.shape)
             
#        self.fc_CNNt1,self.fc_CNNt2,self.fc_CNNt3, train_dataT1,train_dataT2,train_dataT3 = train_data

        train_samples = train_data[0].shape[0]  
#        val_samples = len(val_data[0])
        
        self.model = Model(inputs = [self.fc_CNNt1,self.fc_CNNt2,self.fc_CNNt3], outputs = [self.RNNoutput_class])
        #self.model = Model(inputs = [self.fc_CNN], outputs = [self.RNNoutput_class])    #for multi-TP gru
                           
        lrate = LearningRateScheduler(step_decay_RNN)    
        callback = [lrate]    
        #optimizer = Adam(lr=1e-5)
        self.optimizer = self.params.optimizer
        self.model.compile(optimizer = self.optimizer, loss = 'sparse_categorical_crossentropy', metrics =['acc']) 
        self.model.summary()
        
        history = self.model.fit(x = train_data, 
                    y = train_labels,
                    batch_size = self.params.RNN_batch_size, #can change this now  ...previously seemed to have to be 1...? Otherwise got a mismatch in my last layer being [1,2] instead of [None,2]
                    epochs = self.params.epochs,
                    callbacks = callback,
                    shuffle = True,
                    verbose = 1,
                    #steps_per_epoch = int(train_samples/self.params.batch_size))#,
                    validation_data = (val_data,val_labels)) 
        
        #Save the model
        save_model(self.model,'SavedRNNModel')
        self.model.save_weights('SavedRNNWeights')
        
        return history.history

    def predict (self, data_test):
        test_data, test_labels = data_test
#        with open('//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class/figchecks/test_data.txt', 'w') as testdata:
#            print(test_mri)
        print('shape of test_predsT1 inside predict: ' , test_data[0].shape)
#        print('test_predsT1 inside predict: ' , test_data[0])
        preds = self.model.predict (test_data, batch_size=self.params.RNN_batch_size)
        return preds

    def evaluate (self, data_test):
#        test_mri, test_jac, test_xls, test_labels = data_test
        test_data, test_labels = data_test
        metrics = self.model.evaluate (x = test_data, y = test_labels, batch_size = self.params.RNN_batch_size)  #self.params.batch_size? 
        return metrics
        
    def load_the_weights (self, SavedWeights):
        self.model = Model(inputs = [self.fc_CNNt1,self.fc_CNNt2,self.fc_CNNt3], outputs = [self.RNNoutput_class])
        self.model.compile(optimizer = self.params.optimizer, loss = 'sparse_categorical_crossentropy', metrics =['acc'])  
        loaded = self.model.load_weights(SavedWeights)
        return loaded


def XAlex3D(w_regularizer = None, drop_rate = 0., final_layer_size = 50) : 
  
    #3D Multi-modal deep learning neural network (refer to fig. 4 for chain graph of architecture)
    ###Create the CNN architecture
    def f(mri_volume, mri_volume_jacobian, clinical_inputs):
    
        #First conv layers
        conv1_left = _conv_bn_relu_pool_drop(192, 11, 13, 11, strides = (4, 4, 4), w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (mri_volume)
        #conv1_right = _conv_bn_relu_pool_drop(48, 15, 18, 15, strides = (4, 4, 4), w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (mri_volume_jacobian)
    
        #Second layer
        conv2_left =_conv_bn_relu_pool_drop(384, 5, 6, 5, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv1_left)
        #conv2_right =_conv_bn_relu_pool_drop(96, 5, 6, 5, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv1_right)
    
        #conv2_concat = concatenate([conv2_left, conv2_right], axis = -1)
        
        #Third layer
        #conv3_left =_conv_bn_relu_pool_drop(96, 3, 4, 3, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv2_left)
    
        #conv3_right =_conv_bn_relu_pool_drop(96, 3, 4, 3, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv2_right)
    
        #conv3_concat = concatenate([conv2_left, conv2_right], axis = -1)
    
        #Introduce Middle Flow (separable convolutions with a residual connection)
        print('residual shape '+str(conv2_left.shape))
        conv_mid_1 = mid_flow (conv2_left, drop_rate, w_regularizer, filters = 384) #changed input to conv2_left from conv2_concat
#        conv_mid_2 = mid_flow (conv_mid_1, drop_rate, w_regularizer, filters = 192) 
    
        #Split channels for grouped-style convolution
        conv_mid_1_1 = Lambda (lambda x:x[:,:,:,:,:192]) (conv_mid_1 )
        conv_mid_1_2 = Lambda (lambda x:x[:,:,:,:,192:]) (conv_mid_1 )
        
        conv5_left = _conv_bn_relu_pool_drop (96, 3, 4, 3, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv_mid_1_1)
    
        conv5_right = _conv_bn_relu_pool_drop (96, 3, 4, 3, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv_mid_1_2)
    
        conv6_left = _conv_bn_relu_pool_drop (48, 3, 4, 3, w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (conv5_left)

        conv6_right = _conv_bn_relu_pool_drop (48, 3, 4, 3, w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (conv5_right)
        
        #conv7_left = _conv_bn_relu_pool_drop (16, 3, 4, 3, w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (conv6_left)

        #conv7_right = _conv_bn_relu_pool_drop (16, 3, 4, 3, w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (conv6_right)
        
        conv6_concat = concatenate([conv6_left, conv6_right], axis = -1)
        
        #convExtra = Conv3D(48, (20,30,20),
        #                     strides = (1,1,1), kernel_initializer="he_normal",
        #                     padding="same", kernel_regularizer = w_regularizer)(conv6_concat)
    
        #Flatten 3D conv network representations
        flat_conv_6 = Reshape((np.prod(K.int_shape(conv6_concat)[1:]),))(conv6_concat)    

        #2-layer Dense network for clinical features
        vol_fc1 = _fc_bn_relu_drop(64,  w_regularizer = w_regularizer,
                               drop_rate = drop_rate)(clinical_inputs)

        flat_volume = _fc_bn_relu_drop(20, w_regularizer = w_regularizer,
                                   drop_rate = drop_rate)(vol_fc1)   
    
        #Combine image and clinical features embeddings
    
        fc1 = _fc_bn_relu_drop (20, w_regularizer, drop_rate = drop_rate, name='final_conv') (flat_conv_6)
        #fc2 = _fc_bn_relu_drop (40, w_regularizer, drop_rate = drop_rate) (fc1)
        flat = concatenate([fc1, flat_volume])
    
        #Final 4D embedding

        fc2 = Dense(units = final_layer_size, activation = 'linear', kernel_regularizer=w_regularizer, name='features') (flat) #was linear activation

        #fc2 = _fc_bn_relu_drop (final_layer_size, w_regularizer, drop_rate = drop_rate) (flat)  #this was the orginal final layer

        return fc2
    return f

###Define pieces of CNN
def _fc_bn_relu_drop (units, w_regularizer = None, drop_rate = 0., name = None):
    #Defines Fully connected block (see fig. 3 in paper)
    def f(input):  
        fc = Dense(units = units, activation = 'linear', kernel_regularizer=w_regularizer, name = name) (input)  #was linear activation
        fc = BatchNormalization()(fc)
        fc = ELU()(fc)
        fc = Dropout (drop_rate) (fc)
        return fc
    return f

def _conv_bn_relu_pool_drop(filters, height, width, depth, strides=(1, 1, 1), padding = 'same', w_regularizer = None, 
                            drop_rate = None, name = None, pool = False):
   #Defines convolutional block (see fig. 3 in paper)
   def f(input):
       conv = Conv3D(filters, (height, width, depth),
                             strides = strides, kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(input)
       norm = BatchNormalization()(conv)
       elu = ELU()(norm)
       if pool == True:       
           elu = MaxPooling3D(pool_size=3, strides=2, padding = 'same') (elu)
       return Dropout(drop_rate) (elu)
   return f

def _sepconv_bn_relu_pool_drop (filters, height, width, depth, strides = (1, 1, 1), padding = 'same', depth_multiplier = 1, w_regularizer = None, 
                            drop_rate = None, name = None, pool = False):
    #Defines separable convolutional block (see fig. 3 in paper)
    def f (input):
        sep_conv = SeparableConv3D(filters, (height, width, depth),
                             strides = strides, depth_multiplier = depth_multiplier,kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(input)
        sep_conv = BatchNormalization()(sep_conv)
        elu = ELU()(sep_conv)
        if pool == True:       
           elu = MaxPooling2D(pool_size=3, strides=2, padding = 'same') (elu)
        return Dropout(drop_rate) (elu)
    return f

def mid_flow (x, drop_rate, w_regularizer, filters):
    #3 consecutive separable blocks with a residual connection (refer to fig. 4)
    residual = x   
    x = _sepconv_bn_relu_pool_drop (filters, 3, 3, 3, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer)(x)
    x = _sepconv_bn_relu_pool_drop (filters, 3, 3, 3, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer)(x)
    x = _sepconv_bn_relu_pool_drop (filters, 3, 3, 3, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer)(x)
#    print('x shape '+str(x.shape))
    x = add([x, residual])
    x = ELU()(x)
    return x

def step_decay_CNN (epoch):
    #Decaying learning rate function
    initial_lrate = 4e-4                                                                              
    drop = 0.3                                                                                        
    epochs_drop = 10.0                                                                                
    lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))                                    
    return lrate  

def step_decay_RNN (epoch):
    #Decaying learning rate function
    initial_lrate = 2e-3                                                                        
    drop = 0.3                                                                                      
    epochs_drop = 10.0                                                                                
    lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))                                    
    return lrate      

###Create the RNN
def rnn_bgrus (drop_rate,final_layer_size, kernel_regularizer=None, mask=None):
    def f(inputs):

        fc2T1 = inputs[0]
        print ('fc2T1_ogShape: ', fc2T1.shape)
        print ('fc2T1_ogShape[0]: ', fc2T1.shape[0])
        fc2T2 = inputs[1]
        fc2T3 = inputs[2]

        batch = K.shape(fc2T1)[0] #just the number of samples in T1 (which is same as T2 and T3)
        unitsA = 100
        unitsB = 100
        unitsC = 100
    #reshape
        fc2T1 = tf.reshape(fc2T1,(batch,1,final_layer_size)) #GRU needs input shape: [batch(aka num_samples), timesteps, feature] should first 1 be params.batch_size?
        fc2T2 = tf.reshape(fc2T2,(batch,1,final_layer_size))  
        fc2T3 = tf.reshape(fc2T3,(batch,1,final_layer_size))
    #Add masking layer to handle missing data
        fc2T1_mask = Masking(mask_value=-1, input_shape=(1,final_layer_size))(fc2T1) #needs input shape of (samples, timesteps, features)
        fc2T2_mask = Masking(mask_value=-1, input_shape=(1,final_layer_size))(fc2T2)  #should give output shape of (samples,timesteps)
        fc2T3_mask = Masking(mask_value=-1, input_shape=(1,final_layer_size))(fc2T3)
        print('fc2T1_masked: ', fc2T1_mask)

    # first BGRU (a)
        a_forwardT1 = GRU(unitsA,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(fc2T1_mask) #output shape = (batch_size, timesteps, units)
        a_backwardT1 = GRU(unitsA, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(fc2T1_mask)
        a_forwardT2 = GRU(unitsA,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(fc2T2_mask)
        a_backwardT2 = GRU(unitsA, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(fc2T2_mask) 
        a_forwardT3 = GRU(unitsA,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(fc2T3_mask)
        a_backwardT3 = GRU(unitsA, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(fc2T3_mask)
        a_gruT1 = concatenate([a_forwardT1, a_backwardT1], axis=-1)
        a_gruT2 = concatenate([a_forwardT2, a_backwardT2], axis=-1)
        a_gruT3 = concatenate([a_forwardT3, a_backwardT3], axis=-1)    
    #reshape
        a_gruT1 = tf.reshape(a_gruT1,(batch,1,unitsA*2)) #had 1,200,1...why??
        a_gruT2 = tf.reshape(a_gruT2,(batch,1,unitsA*2))
        a_gruT3 = tf.reshape(a_gruT3,(batch,1,unitsA*2))
        
        # second BGRU (b)
        b_forwardT1 = GRU(unitsB,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(a_gruT1)  #does this propagate the mask??? Does it need to anymore??
        b_backwardT1 = GRU(unitsB, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(a_gruT1)
        b_forwardT2 = GRU(unitsB,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(a_gruT2)
        b_backwardT2 = GRU(unitsB, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(a_gruT2) 
        b_forwardT3 = GRU(unitsB,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(a_gruT3)
        b_backwardT3 = GRU(unitsB, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(a_gruT3)
        b_gruT1 = concatenate([b_forwardT1, b_backwardT1], axis=-1)
        b_gruT2 = concatenate([b_forwardT2, b_backwardT2], axis=-1)
        b_gruT3 = concatenate([b_forwardT3, b_backwardT3], axis=-1)
    #reshape
        b_gruT1 = tf.reshape(b_gruT1,(batch,1,unitsB*2))
        b_gruT2 = tf.reshape(b_gruT2,(batch,1,unitsB*2))
        b_gruT3 = tf.reshape(b_gruT3,(batch,1,unitsB*2))

        ##ADD a dropout layer or two; or add dropout to GRU (see documentation)
      
        # third BGRU (c)
        c_forwardT1 = GRU(unitsC,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(b_gruT1)
        c_backwardT1 = GRU(unitsC, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(b_gruT1)
        c_forwardT2 = GRU(unitsC,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(b_gruT2)
        c_backwardT2 = GRU(unitsC, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(b_gruT2) 
        c_forwardT3 = GRU(unitsC,activation='tanh',dropout=drop_rate,kernel_regularizer=kernel_regularizer)(b_gruT3)
        c_backwardT3 = GRU(unitsC, activation='tanh', go_backwards=True,dropout=drop_rate,kernel_regularizer=kernel_regularizer)(b_gruT3)
        c_gruT1 = concatenate([c_forwardT1, c_backwardT1], axis=-1)
        c_gruT2 = concatenate([c_forwardT2, c_backwardT2], axis=-1)
        c_gruT3 = concatenate([c_forwardT3, c_backwardT3], axis=-1)   

    #reshape
        #c_gruT1 = tf.reshape(c_gruT1,(batch,1,unitsC*2))
        #c_gruT2 = tf.reshape(c_gruT2,(batch,1,unitsC*2))
        #c_gruT3 = tf.reshape(c_gruT3,(batch,1,unitsC*2))
        
    # fourth BGRU (d)
        #d_forwardT1 = GRU(unitsC,activation='tanh')(c_gruT1)
        #d_backwardT1 = GRU(unitsC, activation='tanh', go_backwards=True)(c_gruT1)
        #d_forwardT2 = GRU(unitsC,activation='tanh')(c_gruT2)
        #d_backwardT2 = GRU(unitsC, activation='tanh', go_backwards=True)(c_gruT2) 
        #d_forwardT3 = GRU(unitsC,activation='tanh')(c_gruT3)
        #d_backwardT3 = GRU(unitsC, activation='tanh', go_backwards=True)(c_gruT3)
        #d_gruT1 = concatenate([d_forwardT1, d_backwardT1], axis=-1)
        #d_gruT2 = concatenate([d_forwardT2, d_backwardT2], axis=-1)
        #d_gruT3 = concatenate([d_forwardT3, d_backwardT3], axis=-1) 

      
        #concatenate final BGRU output
        bgru_total = concatenate([c_gruT1, c_gruT2,  c_gruT3], axis=-1)
               
        #Fully connected layer
        rnn_fc1 = Dense(units = 20, activation = 'linear', name = 'RNNfcFinal') (bgru_total)
        rnn_fc1 = Dropout(drop_rate) (rnn_fc1)
        
        return rnn_fc1
    return f
