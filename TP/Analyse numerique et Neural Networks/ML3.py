import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import datetime as dt

import numpy as np
import tensorflow
import tensorflow.keras as keras
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Reshape,Flatten,BatchNormalization, Activation,ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# from keras import metrics
from keras import initializers
from keras import utils
from keras.utils.generic_utils import get_custom_objects
# from keras import losses 
# from keras.callbacks import TensorBoard
# from keras.callbacks import LambdaCallback
#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import LearningRateScheduler
#from keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
#from tensorflow.keras.models import Model
#from tensorflow.keras.callbacks import  ModelCheckpoint

#import cv2
#import glob


#import time 
#from sympy import *



import sys
import matplotlib.pyplot as plt



print("\n------------------------------------------\n")

def g1(x):
    return 2*x*(x<0.5) + (2-2*x)*(x>=0.5)
def g2(x):
    return g1(g1(x))
def g3(x):
    return g1(g2(x))
def g4(x):
    return g1(g3(x))
def g5(x):
    return g1(g4(x))
def g6(x):
    return g1(g5(x))
def g7(x):
    return g1(g6(x))
def g8(x):
    return g1(g7(x))
def g9(x):
    return g1(g8(x))

a=2
def Tak(x):
    b=2*( g1(x)/a+g2(x)/a**2+g3(x)/a**3
       +g4(x)/a**4+g5(x)/a**5+g6(x)/a**6 
       +g7(x)/a**7+g8(x)/a**8+g9(x)/a**9 ) #+20*x**5*(1-x)
    return b

n=400
x_p=np.linspace(0,1,n)
y_p=g2(x_p)
z_p=Tak(x_p)

plt.plot(x_p,z_p)

print("Fin partie 1")
#sys.exit(0)




print("\n------------------------------------------\n")

start = dt.datetime.now()

SEED = 2020; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)

# Loading datasets
DATA_DIR = "./DATA/"
train_data = np.loadtxt(DATA_DIR + "TRAIN_tak.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(train_data)
x_train = train_data[:, 2:3]
y_train = train_data[:, 1:2]

print ("Taille x_train=",x_train.shape)
print ("Taille y_train=",y_train.shape)

test_data = np.loadtxt(DATA_DIR + "TEST_tak.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(test_data)
x_test = test_data[:, 2:3]
y_test = test_data[:, 1:2]
print ("Taille x_test=",x_test.shape)
print ("Taille y_test=",y_test.shape)


#------ initialiseur couches ReLU ---#
#------ initialiseur couches ReLU ---#

fac=3
depth=5


def init_W0(shape, dtype=None):
    W=np.array([[1,1,0 ]])
    return K.constant(W)
def init_b0(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W1(shape, dtype=None):
    W=np.array([[2,2,2],[-4,-4,-4],[0,0,0]])
    return K.constant(W)   
def init_b1(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W2(shape, dtype=None):
    W=np.array([[2,2,2/fac],[-4,-4,-4/fac],[0,0,1]])
    return K.constant(W)   
def init_b2(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W3(shape, dtype=None):
    W=np.array([[2,2,2/fac**2],[-4,-4,-4/fac**2],[0,0,1]])
    return K.constant(W)   
def init_b3(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W4(shape, dtype=None):
    W=np.array([[2,2,2/fac**3],[-4,-4,-4/fac**3],[0,0,1]])
    return K.constant(W)   
def init_b4(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W5(shape, dtype=None):
    W=np.array([[2,2,2/fac**4],[-4,-4,-4/fac**4],[0,0,1]])
    return K.constant(W)   
def init_b5(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W_sortie(shape, dtype=None):
    W=np.array([[2/fac**depth],[-4/fac**depth],[1]])
    return K.constant(W)
def init_b_sortie(shape, dtype=None):
    b=np.array([0])
    return K.constant(b)

#------ initialiseur couches TReLU ---#
#------ initialiseur couches TReLU ---#

fac=2.
depth=2


def init_W0_T(shape, dtype=None):
    W=np.array([[2,-2,0 ]])
    return K.constant(W)
def init_b0_T(shape, dtype=None):
    b=np.array([0,2,0 ])
    return K.constant(b)

def init_W1_T(shape, dtype=None):
    W=np.array([[2,-2,1],[2,-2,1],[0,0,0]])
    return K.constant(W)   
def init_b1_T(shape, dtype=None):
    b=np.array([-2,4,-1 ])
    return K.constant(b)

def init_W2_T(shape, dtype=None):
    W=np.array([[2,-2,1/fac**1],[2,-2,1/fac**1],[0,0,1]])
    return K.constant(W)   
def init_b2_T(shape, dtype=None):
    b=np.array([-2,4,-1/fac**1 ])
    return K.constant(b)

def init_W3_T(shape, dtype=None):
    W=np.array([[2,2,2/fac**2],[-4,-4,-4/fac**2],[0,0,1]])
    return K.constant(W)   
def init_b3_T(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W4_T(shape, dtype=None):
    W=np.array([[2,2,2/fac**3],[-4,-4,-4/fac**3],[0,0,1]])
    return K.constant(W)   
def init_b4_T(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W5_T(shape, dtype=None):
    W=np.array([[2,2,2/fac**4],[-4,-4,-4/fac**4],[0,0,1]])
    return K.constant(W)   
def init_b5_T(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W_sortie_T(shape, dtype=None):
    W=np.array([[1/fac**depth],[1/fac**depth],[1]])
    return K.constant(W)
def init_b_sortie_T(shape, dtype=None):
    b=np.array([-1/fac**depth])
    return K.constant(b)


#--- Fonction activatin T-ReLU -------#


def T_relu(x):
    return K.relu(x, max_value=1)


#------ Defining model -------#
model = Sequential()
model.add(Dense(3, input_dim=1,name="couche_entree",
                kernel_initializer=init_W0_T,
                use_bias=True, bias_initializer=init_b0_T, 
                activation=T_relu))

model.add(Dense(3,name="couche_hidden_1",
                kernel_initializer=init_W1_T,
                use_bias=True, bias_initializer=init_b1_T, 
                activation=T_relu))
model.add(Dense(3,name="couche_hidden_2",
                kernel_initializer=init_W2_T,
                use_bias=True, bias_initializer=init_b2_T, 
                activation=T_relu))
# model.add(Dense(3,name="couche_hidden_3",
#                 kernel_initializer=init_W3,
#                 use_bias=True, bias_initializer=init_b3, 
#                 activation='relu'))
# model.add(Dense(3,name="couche_hidden_4",
#                 kernel_initializer=init_W4,
#                 use_bias=True, bias_initializer=init_b4, 
#                 activation='relu'))
# model.add(Dense(3,name="couche_hidden_5",
#                 kernel_initializer=init_W5,
#                 use_bias=True, bias_initializer=init_b5, 
#                 activation='relu'))

model.add(Dense(1,name="couche_sortie",
                kernel_initializer=init_W_sortie_T,
                use_bias=True, bias_initializer=init_b_sortie_T, 
                activation='linear'))

model.compile(loss='mse', optimizer=Adam())

y_predict_ini=model.predict(x_p)    
plt.plot(x_p,y_predict_ini)

print("Fin initialisation")
#sys.exit(0)



# Learning
# model.fit(x_train, y_train,
# 		  batch_size=100,
# 		  epochs=100,
# 		  verbose=1, validation_data=(x_test, y_test)
# 		  ) 

rmse = np.sqrt(model.evaluate(x_test, y_test, verbose=False))
print('Test l2-error: {:.6f}'.format(rmse))


#print (' ')
#y_essai=model.predict(x_test,verbose=1)    
#for i5 in range(0,20): 
#    print ('1  : y_predict',y_essai[[i5]],': y_vrai   ',y_test[[i5]]) #,': x  ',x_test[[i5]])
    
   
y_predict=model.predict(x_p)    
plt.plot(x_p,y_predict)


#------------  Sortie  -----------#
string_file='%5.8f %5.8f %5.8f %5.8f'
string_file=string_file+' \n'

strstr ='./sortie.plot'
fichier = open(strstr,'w')

for i in range(0,n):
    fichier.write(string_file %  (x_p[i],z_p[i],y_predict_ini[i],y_predict[i]) )

fichier.close()
#------------  Sortie  -----------#
    
    

end = dt.datetime.now()
print("Training duration: {} seconds".format(end - start))

