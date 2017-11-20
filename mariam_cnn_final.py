import time
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from MyImageDataGenerator import MyImageDataGenerator
from keras import metrics

Classes = 257

Width =300 
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

inputFormat = Input(shape=(Width, Height, Depth))
preTrainedModel = Xception(include_top=False, weights='imagenet', input_tensor=inputFormat, pooling='avg')

I = preTrainedModel.output

for layer in preTrainedModel.layers:
    layer.trainable = False

outputLayer = Dense(Classes, activation='softmax')(I)

MariamOptimizer = Adam(lr=0.0001)      

MariamModel = Model(inputs=[preTrainedModel.input], outputs=[outputLayer])

MariamModel.compile(optimizer=MariamOptimizer, loss='categorical_crossentropy', metrics=['accuracy',metrics.top_k_categorical_accuracy])

MariamModel.summary()

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


MariamModel.fit_generator(trainingDataGenerator, steps_per_epoch=TrainingBatches, epochs=Epochs,validation_data=validationDataGenerator, validation_steps=ValidationBatches, verbose=1,callbacks=callbacks)


MariamModel.save('MariamModel.h5')

