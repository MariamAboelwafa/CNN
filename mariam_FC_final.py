import time
from keras.models import Model, Sequential
from keras.layers import Input, Dense,Flatten, Dropout,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from MyImageDataGenerator import MyImageDataGenerator
from keras import metrics

Classes = 257

Width = 300
Height = 300
Depth = 3

ImagesPerBatch = 32
Epochs = 100

TotalSet = 30607
ValidationDataSet = 15 * Classes
ValidationBatches = ValidationDataSet / ImagesPerBatch
TrainingDataSet = TotalSet - ValidationDataSet   #26752
TrainingBatches = TrainingDataSet / ImagesPerBatch


#trainingDirectory = 'D:/Mariam/final/256_ObjectCategories'
trainingDirectory = '/home/mariamaboelwafa/256_ObjectCategories'
validationDirectory = 'validation_data'

'''MariamModelFC = Sequential()

digit_input = Input(shape=(250, 250, 3))
x = Flatten()(digit_input)
xx = Dense(250*250*3, activation='relu')(x)
xxx = Dropout(0.5)(xx)
out = Dense(Classes,activation='softmax')(xxx)'''
 

#MariamModelFC = Model(inputs=[digit_input], outputs=[out])


MariamModelFC = Sequential()
#MariamModelFC.add(Flatten(input_shape=(300,300,3)))
MariamModelFC.add(GlobalAveragePooling2D(input_shape=(300,300,3)))
MariamModelFC.add(Dense(50, activation='relu'))
MariamModelFC.add(Dense(50, activation='relu'))
MariamModelFC.add(Dense(50, activation='relu'))
MariamModelFC.add(Dropout(0.5))
MariamModelFC.add(Dense(257, activation='softmax'))

                  
myOptimizer = Adam(lr=0.001)    

MariamModelFC.compile(optimizer=myOptimizer, loss='categorical_crossentropy', metrics=['accuracy',metrics.top_k_categorical_accuracy])   # categorical_crossentropy iss for negative log likelihood

MariamModelFC.summary()     

MonitoringCallback = ModelCheckpoint(filepath="./weights.hdf5", monitor="val_acc", verbose=1, save_best_only=True)

StoppingCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

PlottingCallback = TensorBoard(log_dir="./logs/training_" + time.strftime("%c"), histogram_freq=0, write_graph=True,write_images=False)

reduceOnPlateauCallback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',epsilon=0.0001, cooldown=0, min_lr=0.000001)

terminateOnNaNCallback = TerminateOnNaN()

callbacks = [MonitoringCallback, StoppingCallback, reduceOnPlateauCallback, terminateOnNaNCallback, PlottingCallback]


trainingDataGenerator = MyImageDataGenerator(featurewise_center=True,samplewise_center=False,featurewise_std_normalization=True, samplewise_std_normalization=False,zca_whitening=False,rotation_range=30,horizontal_flip=True,vertical_flip=False,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)

validationDataGenerator = MyImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,zca_whitening=False)

trainingDataGenerator = trainingDataGenerator.flow_from_directory(trainingDirectory,target_size=(Width, Height),batch_size=ImagesPerBatch,shuffle=True)

validationDataGenerator = validationDataGenerator.flow_from_directory(validationDirectory,target_size=(Width,Height),batch_size=ImagesPerBatch,shuffle=False)

print("Training...")


MariamModelFC.fit_generator(trainingDataGenerator, steps_per_epoch=TrainingBatches, epochs=Epochs,validation_data=validationDataGenerator, validation_steps=ValidationBatches, verbose=1,callbacks=callbacks)


MariamModelFc.save('MariamModelFc.h5')

