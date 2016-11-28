from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from numpy import linalg as LA

from .. import activations, initializations, regularizers, constraints
from ..layers.core import Layer, MaskedLayer
from ..utils.theano_utils import sharedX, alloc_zeros_matrix

from ..constraints import unitnorm
#This is f1 .. changed

class Embedding(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    input_ndim = 2

    def __init__(self, input_dim, output_dim, init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None, W_constraint=None,
                 mask_zero=False, weights=None, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_dim,)
        super(Embedding, self).__init__(**kwargs)

    def build(self):
        self.input = T.imatrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.params = [self.W]
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None
        else:
            return T.ones_like(X) * (1 - T.eq(X, 0))

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_length, self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        out = self.W[X]
        return out

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "input_length": self.input_length,
                  "mask_zero": self.mask_zero,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class WordTagSupervised_dotp(Layer):
    '''
        This layer turns a tuple of word and its surrounding words (i.e. context)
        identified by their index in a vocabulary, into two dense reprensentations
        (word representation and context representation).

        Then it returns activation(dot(pivot_embedding, context_embedding)),
        which can be trained to encode the probability
        of finding the context word in the context of the pivot word
        (or reciprocally depending on your training procedure).

        The layer ingests integer tensors of shape:
        (nb_samples, 2)
        and outputs a float tensor of shape
        (nb_samples, 1)

        The 2nd dimension encodes (pivot, context).
        input_dim is the size of the vocabulary.

        For more context, see Mikolov et al.:
            Efficient Estimation of Word reprensentations in Vector Space
            http://arxiv.org/pdf/1301.3781v3.pdf
    '''
    input_ndim = 2

    def __init__(self, input_dim, proj_dim=128, neg_samples = 4,
                 init='uniform', activation='sigmoid', weights=None, W_regularizer = None, activity_regularizer = None, **kwargs):

        super(WordTagContextProduct, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.samples = neg_samples + 1
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input = [T.itensor3(), T.itensor3()]
        self.W_w = self.init((input_dim, proj_dim))
        self.params = [self.W_w]
        if weights is not None:
            self.set_weights(weights)
            
        

    @property
    def output_shape(self):
        return (self.input_shape[0], 1)

    def init_tags(self, word_index = None, tag_index = None, W_constraint = 'maxnorm'):
        if tag_index == None:
            return     
        W_t = np.random.uniform(low=-0.05, high=0.05, size=((len(tag_index)+1, self.proj_dim)))
        self.W_t = sharedX(W_t)
        self.params = [self.W_w, self.W_t]
  
    def get_output(self, train=False):
        [X_w, X_t] = self.get_input(train)
        t_w = self.W_t[X_w[:,:, 0]] # doc_l, n_tags*n_samples, n_dim
        w_w = self.W_w[X_w[:,:, 1]]
        dot_tw = T.sum(w_w * t_w, axis=2)
        dot_sum_w = T.sum(dot_tw* dot_tw, axis = 0)/(X_w.shape[0])
        return self.activation(theano.tensor.reshape(dot_sum_w, (X_w.shape[1], 1)))
        '''
        ## uncomment this code if learning from tag tag co-occurrence is also desired.
        t_t = self.W_t[X_t[:,:, 0]] # doc_l, n_tags*n_samples, n_dim
        w_t = self.W_t[X_t[:,:, 1]]
        dot_tt = T.sum(w_t * t_t, axis=2)
        #dot_sum = T.sum(dot_tw, axis = 0)#/(X.shape[0])
        dot_sum_t = T.sum(dot_tt , axis = 0)#/(X_t.shape[0])
        #dot_sum_t = T.sum(dot_tt * dot_tt, axis = 0)/(X_t.shape[0])
        dot_twc_t = dot_sum_t#*dot_sum#_p
        dot_t = theano.tensor.reshape(dot_twc_t, (X_t.shape[1], 1))
        
        return 0.5 * self.activation(dot_w) + 0.5 * self.activation(dot_t)
        '''
    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "proj_dim": self.proj_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(WordContextProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WordTagSupervised_tesnor(Layer):

    input_ndim = 2

    def __init__(self, input_dim, proj_dim=128, neg_samples = 4,tensor_slices = 4, slice_dim = 16,
                 init='uniform', activation='tanh', weights=None,W_regularizer = None, activity_regularizer=None,   **kwargs):

        super(WordTagContextProduct_tensor, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.samples = neg_samples + 1
        #np.random.seed(0)
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input = [T.itensor3(), T.itensor3()]
        self.W_w = self.init((input_dim, proj_dim))
        self.tensor_slices = tensor_slices
        self.slice_dim = slice_dim
        self.params = [self.W_w]

        if weights is not None:
            self.set_weights(weights)
            
        

    @property
    def output_shape(self):
        return (self.input_shape[0], 1)

    def init_tags(self, word_index = None, tag_index = None, W_constraint = 'maxnorm'):
        if tag_index == None:
            return
        
        self.W_t = self.init((len(tag_index)+1, self.proj_dim))
        self.S = self.init((self.tensor_slices,self.slice_dim,self.proj_dim))
        self.P = self.init((self.tensor_slices,self.slice_dim,self.proj_dim))
        self.U = self.init((self.tensor_slices,1))
        self.B = theano.shared(np.asarray(np.random.uniform(low=-0.05, high=0.05, size=((self.tensor_slices))), dtype=theano.config.floatX))
        self.U_t = self.init((self.tensor_slices,1))
        self.params = [self.W_w, self.W_t, self.S, self.P, self.U, self.B, self.U_t]

   
    def get_output(self, train=False):
        [X_w, X_t] = self.get_input(train)
        t_w = self.W_t[X_w[:,:, 0]] # doc_l, n_tags*n_samples, n_dim
        w_w = self.W_w[X_w[:,:, 1]]
        dot_tw = T.sum(w_w * t_w, axis=2)
        inter_1 = T.tensordot(w_w, self.S, axes = [[2],[2]])
        inter_2 = T.tensordot(t_w, self.P, axes = [[2],[2]]) # doc_l, n_tags*n_samples, 2,5
        inter = T.sum(inter_1 * inter_2, axis = 3)
        sim_tw = T.tensordot(inter + T.shape_padleft(self.B, 2), self.U, axes=[[2],[0]]) 
        sim_tw = T.reshape(sim_tw, (X_w.shape[0], X_w.shape[1]))
        dot_sum_w = T.sum(dot_tw * T.nnet.sigmoid(sim_tw), axis = 0)/(X_w.shape[0])
        dot_w = theano.tensor.reshape(dot_sum_w, (X_w.shape[1], 1))
        return self.activation(dot_w)
        '''
        t_t = self.W_t[X_t[:,:, 0]] # doc_l, n_tags*n_samples, n_dim
        w_t = self.W_t[X_t[:,:, 1]]
        dot_tt = T.sum(w_t * t_t, axis=2)
        #dot_sum = T.sum(dot_tw, axis = 0)#/(X.shape[0])
        #dot_sum_t = T.sum(dot_tt , axis = 0)#/(X_t.shape[0])
        inter_t_1 = T.tensordot(t_t, self.P, axes = [[2],[2]])
        inter_t_2 = T.tensordot(w_t, self.P, axes = [[2],[2]]) # doc_l, n_tags*n_samples, 2,5
        inter_t = T.sum(inter_t_1 * inter_t_2, axis = 3)
        sim_tt = T.tensordot(inter_t, self.U_t, axes=[[2],[0]]) 
        sim_tt = T.reshape(sim_tt, (X_t.shape[0], X_t.shape[1]))
        
        dot_sum_t = T.sum(dot_tt * sim_tt, axis = 0)/(X_t.shape[0])
        dot_twc_t = dot_sum_t#*dot_sum#_p
        dot_t = theano.tensor.reshape(dot_twc_t, (X_t.shape[1], 1))
        
        return 0.5 * self.activation(dot_w) + 0.5 * self.activation(dot_t)
        '''
    
    '''def get_output_0(self, train=False):
        X = self.get_input(train)
        t = self.W_t[X[:,:, 0]] # doc_l, n_tags*n_samples, n_dim
        w = self.W_w[X[:,:, 1]]
        dot_tw = T.sum(w * t, axis=2)
        #dot_sum = T.sum(dot_tw, axis = 0)#/(X.shape[0])
        dot_sum = T.sum(dot_tw * dot_tw, axis = 0)/(X.shape[0])
        dot_twc = dot_sum
        dot = theano.tensor.reshape(dot_twc, (X.shape[1], 1))
        return self.activation(dot)'''

        
    '''def get_output(self, train=False):
        X = self.get_input(train)
        ## tag, word, context words
        dot = []
        len_X = X.shape[0]
        for i in range(self.negative_samples):
            t = self.W_t[X[i,:, 0]]
            w = self.W_w[X[i,:, 1]]
            c = T.sum(self.W_c[X[i,:, 2:]], axis=2)/(X.shape[2]-2)
            dot_tw = T.sum(w * t, axis=1)
            dot_tc = T.sum(w * c, axis=1)
            dot.append(T.scalar(T.sum(dot_tw*dot_tc, axis = 0)))
        
        # sum dot_t
        dot = theano.tensor.reshape(sharedX(dot), (len_X, 1))
        return self.activation(dot)'''

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "proj_dim": self.proj_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(WordContextProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



