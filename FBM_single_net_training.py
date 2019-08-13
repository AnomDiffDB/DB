
import numpy as np
from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D,concatenate
from keras.optimizers import Adam
from utils import fbm_regression
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint


batchsize = 64
T = np.arange(19,21,0.1) # this provides another layer of stochasticity to make the network more robust
steps = 500 # number of steps to generate
initializer = 'he_normal'
f = 32 # number of convolution filters in a single network layer
sigma = 0.1

inputs = Input((steps-1,1))

x1 = Conv1D(f,4,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
x1 = BatchNormalization()(x1)
x1 = Conv1D(f,4,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x1)
x1 = BatchNormalization()(x1)
x1 = Conv1D(f,4,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x1)
x1 = BatchNormalization()(x1)
x1 = GlobalMaxPooling1D()(x1)


x2 = Conv1D(f,2,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
x2 = BatchNormalization()(x2)
x2 = Conv1D(f,2,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x2)
x2 = BatchNormalization()(x2)
x2 = Conv1D(f,2,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x2)
x2 = BatchNormalization()(x2)
x2 = GlobalMaxPooling1D()(x2)


x3 = Conv1D(f,3,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
x3 = BatchNormalization()(x3)
x3 = Conv1D(f,3,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x3)
x3 = BatchNormalization()(x3)
x3 = Conv1D(f,3,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x3)
x3 = BatchNormalization()(x3)
x3 = GlobalMaxPooling1D()(x3)


x4 = Conv1D(f,10,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
x4 = BatchNormalization()(x4)
x4 = Conv1D(f,10,dilation_rate=5,padding='causal',activation='relu',kernel_initializer=initializer)(x4)
x4 = BatchNormalization()(x4)
x4 = Conv1D(f,10,dilation_rate=10,padding='causal',activation='relu',kernel_initializer=initializer)(x4)
x4 = BatchNormalization()(x4)
x4 = GlobalMaxPooling1D()(x4)

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


gen = fbm_regression(batchsize=batchsize,steps=steps,T=T,sigma=sigma)
history = model.fit_generator(generator=gen,
        steps_per_epoch=50,
        epochs=100,
        callbacks=callbacks,
        validation_data=fbm_regression(steps=steps,T=T,sigma=sigma),
        validation_steps=10)