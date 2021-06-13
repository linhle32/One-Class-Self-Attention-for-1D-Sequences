import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.keras.layers import Dense, Attention,Flatten, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, UpSampling1D, \
                                    AveragePooling1D, Conv2DTranspose, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import *
from tensorflow.compat.v1 import placeholder, Session, global_variables_initializer
from tensorflow import math
from matplotlib import pyplot as plt
import numpy as np


############
#####Auto-Encoder with 1DCNN blocks for 1D sequential data    
#####
class CNN1D_AE(object):
    #initialize model
    #   input_shape: input dimensionality
    #   Encoder: encoder architecture (list of [filter number, filter size]
    #   Decoder: decoder architecture (list of [filter number, filter size]
    def __init__(self, input_shape, Encoder, Decoder):
        
        self.input_ = tf.placeholder(shape=[None, input_shape], dtype=tf.float32, name="inputs")
        self.input_reshaped = Reshape((input_shape,1))(self.input_)
        
        #Encoder
        self.EL = []
        x = self.input_reshaped
        for elayer in Encoder:
            CNNL = Conv1D(elayer[0], elayer[1], activation='relu', padding='valid')
            self.EL.append(CNNL)
            x = CNNL(x)
            max_ = MaxPooling1D(2, padding='valid')(x)
            min_ = - MaxPooling1D(2, padding='valid')(-x)
            avg_ = AveragePooling1D(2, padding='valid')(x)
            x = tf.concat([max_, min_, avg_], axis=-1)
        
        self.encoded = x
        
        #Decoder
        self.DL = []
        x = tf.expand_dims(x, axis=-2)
        for dlayer in Decoder:
            DCNNL = Conv2DTranspose(dlayer[0], [1,dlayer[1]], activation='relu', padding='same')
            self.DL.append(DCNNL)
            x = DCNNL(x)
            x = UpSampling2D((2,1))(x)
        
        x = GlobalAveragePooling2D()(x)
        self.outlayer = Dense(input_shape,activation = 'linear')
        z = self.outlayer(x)
        self.decoded = Reshape((input_shape,1))(z)
        
        #loss
        self.loss_ = tf.reduce_mean(tf.math.abs(self.input_reshaped - self.decoded))
                
        #training, validation loss
        self.train_loss = []
        self.valid_loss = []
        
        #learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        #session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    
    ####training function
    #learning_rate decay when loss fluctuate
    #   n_epochs: number of iterations
    #   init_learning_rate: initial learning rate
    #   train: training data
    #   valid: validation data, optional
    #   min_epochs: minimum iterations to train even without min_improvement reach
    #   min_improvemet: minimum improvement in training cost to keep training
    def train(self,n_epochs,init_learning_rate,train,valid=None, min_epochs=200, min_improvement=0.001): 
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        objective = optimizer.minimize(self.loss_)
        
        if len(self.train_loss) == 0:
            self.train_loss.append(self.sess.run([self.loss_],feed_dict={self.input_: train})[0])
        if valid is not None and len(self.valid_loss) == 0:
            self.valid_loss.append(self.sess.run([self.loss_],feed_dict={self.input_: valid})[0])

        learning_rate = init_learning_rate
        last_update = 0
        
        for i in range(n_epochs):
            tcc = self.sess.run([objective, self.loss_],
                                feed_dict={self.input_: train,
                                           self.learning_rate: learning_rate
                                          })[1]
            self.train_loss.append(tcc)
            out = 'epoch ' + str(i) + ', learning rate ' + str(learning_rate) + ', training loss: ' + str(tcc)
            
            if valid is not None:
                vcc = self.sess.run([self.loss_],feed_dict={self.input_: valid})[0]
                out += ', valid loss: ' + str(vcc)
                self.valid_loss.append(vcc)
                    
            print(out)
        
            #decaying learning rate if loss fluctuates
            if (i - last_update > 10):
                if (np.diff(self.train_loss[-10:]) < 0).sum() < 5:
                    learning_rate = learning_rate / 10
                    last_update = i
                    
            #early stopping
            if (i > min_epochs):
                if (np.diff(self.valid_loss[-101:]) / self.valid_loss[-100:]).mean() > -min_improvement:
                    print("early terminated")
                    return
        
    def plot_train_history(self):
        plt.plot(self.train_loss)
        plt.plot(self.valid_loss)
        plt.legend(['training loss','valid loss'])
        plt.show()
        
    def encode(self,x):
        return self.sess.run([self.encoded],feed_dict={self.input_: x})[0]
    
    def predict(self,x,ano_rate):
        xpred = self.sess.run([self.decoded],feed_dict={self.input_ : x})[0]
        e = np.abs(xpred.reshape(x.shape) - x).mean(axis=1)
        predY = np.ones(x.shape[0])
        pred_ano = np.argsort(-e)[:int(x.shape[0]*ano_rate)]
        predY[pred_ano] = -1
        return predY
    
    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    
############
#####Deep One-Class Classifier with 1DCNN inputs for 1D sequential data
#####        
class DOCC(object):
    
    #initialize model
    #   input_shape: input dimensionality
    #   cnnls: list of 1D CNN layer architectures [filter number, filter size]
    #   emb_shape: output dimensionality
    def __init__(self, input_shape, cnnls, emb_shape):
      
        #input
        self.input_ = tf.placeholder(shape=[None, input_shape], dtype=tf.float32, name="inputs")
        self.center_ = tf.placeholder(shape=[emb_shape], dtype=tf.float32, name="center")
        self.CENTER = np.zeros(emb_shape)
        
        #reshape for cnn layers
        self.input_reshaped = Reshape((input_shape,1))(self.input_)
        
        #cnn layers
        self.layers = []
        x = self.input_reshaped
        for elayer in cnnls:
            CNNL = Conv1D(elayer[0], elayer[1], 
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                          kernel_regularizer = l2(0.01),
                          use_bias=False,
                          padding='valid')
            self.layers.append(CNNL)
            x = CNNL(x)
            max_ = MaxPooling1D(2, padding='valid')(x)
            min_ = - MaxPooling1D(2, padding='valid')(-x)
            avg_ = AveragePooling1D(2, padding='valid')(x)
            x = tf.concat([max_, min_, avg_], axis=-1)
        
        x = Flatten()(x)
        
        #embedding
        self.out_layer = Dense(emb_shape,
                               activation = tf.keras.layers.LeakyReLU(alpha=0.1), 
                               kernel_regularizer = l2(0.01),
                               use_bias = False
                              )
        self.layers.append(self.out_layer)
        self.emb_ = self.out_layer(x)
        
        #loss
        self.loss_ = tf.reduce_mean((self.emb_ - self.center_)**2) + sum(l.losses[0] for l in self.layers)
        self.loss_wo_reg = tf.reduce_mean((self.emb_ - self.center_)**2)
        
        #training, validation loss
        self.train_loss = []
        self.valid_loss = []
        
        #learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        #session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    ####initialize function
    # generate data center
    def initialize(self,x):
        e = self.sess.run([self.emb_], feed_dict={self.input_: x})[0]
        self.CENTER = e.mean(axis=0)
    
    ####training function
    #learning_rate decay when loss fluctuate
    #   n_epochs: number of iterations
    #   init_learning_rate: initial learning rate
    #   train: training data
    #   valid: validation data, optional
    #   min_epochs: minimum iterations to train even without min_improvement reach
    #   min_improvemet: minimum improvement in training cost to keep training
    def train(self,n_epochs,init_learning_rate,train,valid=None, min_epochs=200, min_improvement=0.001): 
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        objective = optimizer.minimize(self.loss_)
        
        if len(self.train_loss) == 0:
            self.train_loss.append(self.sess.run([self.loss_wo_reg],
                                                 feed_dict={self.input_: train, self.center_ : self.CENTER})[0])
        if valid is not None and len(self.valid_loss) == 0:
            self.valid_loss.append(self.sess.run([self.loss_wo_reg],
                                                 feed_dict={self.input_: valid, self.center_ : self.CENTER})[0])

        learning_rate = init_learning_rate
        last_update = 0
        
        for i in range(n_epochs):
            self.sess.run([objective],
                          feed_dict={self.input_: train,
                                     self.center_ : self.CENTER,
                                     self.learning_rate: learning_rate
                                    })
            rcc, tcc = self.sess.run([self.loss_, self.loss_wo_reg],
                                     feed_dict={self.input_: train,
                                                self.center_ : self.CENTER,
                                                self.learning_rate: learning_rate
                                               })
            self.train_loss.append(tcc)
            out = 'epoch ' + str(i) + ', learning rate ' + str(learning_rate) + ', regularized loss: ' \
                  + str(rcc) + ', training loss: ' + str(tcc)
            
            if valid is not None:
                vcc = self.sess.run([self.loss_wo_reg],feed_dict={self.input_: valid, self.center_ : self.CENTER})[0]
                out += ', valid loss: ' + str(vcc)
                self.valid_loss.append(vcc)
                    
            print(out)
        
            #decaying learning rate if loss fluctuates
            if (i - last_update > 10):
                if (np.diff(self.train_loss[-10:]) < 0).sum() < 5:
                    learning_rate = learning_rate / 10
                    last_update = i
                    
            #early stopping
            if (i > min_epochs):
                if (np.diff(self.valid_loss[-min_epochs-1:]) / self.valid_loss[-min_epochs:]).mean() > -min_improvement:
                    print("early terminated")
                    return

    def getE(self,x):
        return self.sess.run([self.emb_], feed_dict={self.input_: x})[0]
            
    def getS(self,x):
        e = self.getE(x)
        s = ((e - self.CENTER)**2).sum(axis=1)
        return s
    
    def predict(self,x,ano_rate):
        s = self.getS(x)
        predY = np.ones(x.shape[0])
        pred_ano = np.argsort(-s)[:int(x.shape[0]*ano_rate)]
        predY[pred_ano] = -1
        return predY
    
    def plot_train_history(self):
        plt.plot(self.train_loss)
        plt.plot(self.valid_loss)
        plt.legend(['training loss','valid loss'])
        plt.show()
        
    def close(self):
        tf.reset_default_graph()
        self.sess.close()