import sys,os
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import tensorflow as tf
import keras
from utils.console import Progbar
from nn import patchlayer

sys.path.append(os.path.dirname(os.getcwd()))

# =============================================================================
# Base structure for Neural structure
# =============================================================================

def pseudo_rbf(x):
    return (1+tf.cos(2*tf.atan(x)))/2

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows 
          are taken as centers)
    """
    def __init__(self, X):
        self.X = X 

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx,:]

        
class RBFLayer(Layer):
    """ Layer of Gaussian RBF units. 
    # Example
 
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X), 
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas 
    """
    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas 
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
            #self.initializer = Orthogonal()
        else:
            self.initializer = initializer 
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.centers = self.add_weight(name='centers', 
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(input_shape[1],self.output_dim),
                                     initializer=Constant(value=self.init_betas),
                                     #initializer='ones',
                                     trainable=True)
            
        super(RBFLayer, self).build(input_shape)
        
        self.trainable_weights=[self.centers, self.betas]

    def call(self, inputs):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(inputs))
        return pseudo_rbf(-K.sum(self.betas*H**2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ReductionLayer(keras.models.Layer):
    
    def __init__(self, patch_size, dictionary_size,alpha,**kwargs):
        
        super(ReductionLayer, self).__init__(**kwargs)
        
        self.dict_size = dictionary_size
    
        self.patch_layer = patchlayer.PatchLayer(patch_size)
                
        self.alpha = alpha
        
        self.progbar = Progbar(100,stateful_metrics=["loss"])
        
        self.old_D = 0.0
    def build(self, input_shape):
        
        self.patch_layer.build(input_shape)
        
        self.input_shape_t = self.patch_layer.compute_output_shape(input_shape)
        
        self.dim = self.input_shape_t[-1]
        
        self.filters = self.dict_size
        
        self.strides = (1,self.dim)
        

        self.kernel_shape = (1,self.dim,self.dict_size)
        
        self.D0 = K.random_normal_variable((self.dim,self.dict_size),mean=0,scale = 1)

        self.D = tf.matmul(tf.diag(1/tf.norm(self.D0,axis=1)),self.D0)

        self.D_ols = tf.matmul(tf.linalg.inv(tf.matmul(self.D,self.D,transpose_a=True)+self.alpha*tf.eye(self.dict_size)),
                               self.D,transpose_b=True)
        self.kernel = K.reshape(self.D_ols, self.kernel_shape)
        #self.add_weight(shape=self.kernel_shape,
        #                              initializer='glorot_uniform',
        #                              name='kernel')
        self.D_kernel = K.reshape(tf.matmul(self.D,self.D_ols),(1,self.dim,self.dim))
    
        self.trainable_weights = [self.D0]
    def call(self,inputs):
        
        
        beta = K.conv1d(self.patch_layer(inputs),
                self.kernel,
                strides=1,
                padding='valid',
                data_format='channels_last',
                dilation_rate=1)
        
        return beta
        
    def fit(self,X,Y,batch_size=64):
        print("Fitting the reduction")
        
        n = len(X)
        self.progbar.__init__(n)
        for i in range(0,n,batch_size):
            weights = np.ones(min(n,i+batch_size)-i)
            inputs = X[i:min(i+batch_size,n)]
            targets = Y[i:min(n,i+batch_size)]
            self.fit_op([inputs,targets,weights])
            self.progbar.add(min(batch_size,n-batch_size),
                             values=[('loss',self.loss([inputs,targets,weights])[0])])
    
    def display_update(self):
        res = np.linalg.norm(K.eval(self.D)-self.old_D)
        self.old_D = K.eval(self.D)
        return res
    def set_D(self,D):
        K.set_value(self.D_ridge,D)
        
    def compile(self,model):
        
        self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.opt = self.optimizer.minimize(model.total_loss,var_list=[self.D0])
        self.fit_op = K.Function([model.input,model.targets[0],model.sample_weights[0]],[self.opt])
        self.loss = K.Function([model.input,model.targets[0],model.sample_weights[0]],[model.total_loss])
        print("Reduction Layer Compiled, batch %d"%self.patch_layer.patch_size,"\n",
              "Output shape:",self.compute_output_shape(self.input_shape_t)) 
        
    def compute_output_shape(self, input_shape):
        return self.patch_layer.compute_output_shape(input_shape)[:2] + (self.dict_size,)

    def get_config(self):
        config = {
            'rank': 1,
            'filters': self.dict_size,
            'kernel_size': self.kernel_shape,
            'strides': self.strides,
            'padding': 'valid',
            'data_format': 'channels_last',
            'activation': 'linear',
            'kernel_initializer': 'personal'
        }
        base_config = super(ReductionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def outer(a,b):
    return tf.einsum('ijk,ijm->ijkm',a,b)

