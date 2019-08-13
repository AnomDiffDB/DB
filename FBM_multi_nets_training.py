import numpy as np
from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv2D
from keras.layers import Input,GlobalMaxPooling2D,concatenate
from keras.optimizers import Adam
from utils import fbm_ensemble_regression
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint


batchsize = 64
T = np.arange(2,4,0.1)  # this provides another layer of stochasticity to make the network more robust
steps = 100 # number of steps to generate
initializer = 'he_normal'
f = 16 # number of convolution filters in a single network layer
numTraj = 10 # number of short trajectories the network recieves as input
sigma = 0.1

inputs = Input((numTraj,9,1))


x1 = Conv2D(f,4,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
x1 = BatchNormalization()(x1)
x1 = Conv2D(f,4,dilation_rate=(1,2),padding='same',activation='relu',kernel_initializer=initializer)(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(f,4,dilation_rate=(1,4),padding='same',activation='relu',kernel_initializer=initializer)(x1)
x1 = BatchNormalization()(x1)
x1 = GlobalMaxPooling2D()(x1)


x2 = Conv2D(f,2,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
x2 = BatchNormalization()(x2)
x2 = Conv2D(f,2,dilation_rate=(1,2),padding='same',activation='relu',kernel_initializer=initializer)(x2)
x2 = BatchNormalization()(x2)
x2 = Conv2D(f,2,dilation_rate=(1,4),padding='same',activation='relu',kernel_initializer=initializer)(x2)
x2 = BatchNormalization()(x2)
x2 = GlobalMaxPooling2D()(x2)


x3 = Conv2D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
x3 = BatchNormalization()(x3)
x3 = Conv2D(f,3,dilation_rate=(1,2),padding='same',activation='relu',kernel_initializer=initializer)(x3)
x3 = BatchNormalization()(x3)
x3 = Conv2D(f,3,dilation_rate=(1,4),padding='same',activation='relu',kernel_initializer=initializer)(x3)
x3 = BatchNormalization()(x3)
x3 = Conv2D(f,3,dilation_rate=(1,4),padding='same',activation='relu',kernel_initializer=initializer)(x3)
x3 = BatchNormalization()(x3)
x3 = Conv2D(f,3,dilation_rate=(1,8),padding='same',activation='relu',kernel_initializer=initializer)(x3)
x3 = BatchNormalization()(x3)
x3 = GlobalMaxPooling2D()(x3)


x4 = Conv2D(f,10,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
x4 = BatchNormalization()(x4)
x4 = Conv2D(f,10,dilation_rate=(1,5),padding='same',activation='relu',kernel_initializer=initializer)(x4)
x4 = BatchNormalization()(x4)
x4 = Conv2D(f,10,dilation_rate=(1,10),padding='same',activation='relu',kernel_initializer=initializer)(x4)
x4 = BatchNormalization()(x4)
x4 = GlobalMaxPooling2D()(x4)


con = concatenate([x1,x2,x3,x4])
dense = Dense(512,activation='relu')(con)
dense = Dense(256,activation='relu')(dense)
dense2 = Dense(1,activation='sigmoid')(dense)
 
model = Model(inputs=inputs, outputs=dense2)


optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer,loss='mse',metrics=['mse'])
model.summary()

callbacks = [EarlyStopping(monitor='val_loss',
                       patience=20,
                       verbose=1,
                       min_delta=1e-4),
         ReduceLROnPlateau(monitor='val_loss',
                           factor=0.1,
                           patience=4,
                           verbose=1,
                           min_lr=1e-12),
         ModelCheckpoint(filepath='new_model.h5',
                         monitor='val_loss',
                         save_best_only=True,
                         mode='min',
                         save_weights_only=False)]


gen = fbm_ensemble_regression(batchsize=batchsize,steps=steps,T=T,numTraj=numTraj,sigma=sigma)
history = model.fit_generator(generator=gen,
        steps_per_epoch=50,
        epochs=100,
        callbacks=callbacks,
        validation_data=fbm_ensemble_regression(steps=steps,T=T,numTraj=numTraj,sigma=sigma),
        validation_steps=10)