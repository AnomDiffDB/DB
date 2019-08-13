
from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D
from keras.optimizers import Adam
from utils import brownian_scalar_regression
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint


batchsize = 64
T = [50,51] # this provides another layer of stochasticity to make the network more robust
steps = 1000 # number of steps to generate 
initializer = 'he_normal'
f = 32
sigma = 0.1
inputs = Input((2,1))

#
x2 = Conv1D(f,2,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
x2 = BatchNormalization()(x2)
x2 = GlobalMaxPooling1D()(x2)

dense = Dense(512,activation='relu')(x2)
dense = Dense(256,activation='relu')(dense)
dense2 = Dense(1,activation='sigmoid')(dense)
model = Model(inputs=inputs, outputs=dense2)

optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss= 'mse')
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


gen = brownian_scalar_regression(batchsize=batchsize,steps=steps,T=T,sigma=sigma)
history = model.fit_generator(generator=gen,
        steps_per_epoch=50,
        epochs=100,
        callbacks=callbacks,
        validation_data=brownian_scalar_regression(steps=steps,T=T,sigma=sigma),
        validation_steps=10)