from keras.layers import Input, Dense,Flatten, Dropout
from keras.applications import Xception
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from MyImageDataGenerator import MyImageDataGenerator
import time
from keras import metrics

Width = 250
Height = 250
Depth = 3
Classes = 257
ImagesPerBatch = 32
Epochs = 10
TrainingaDataSet = 26752
TrainingBatches = 26752 / 32
ValidationDataSet = 15 * 257
ValidationBatches = (15*257) / 32

#trainingDirectory = 'D:/Mariam/final/256_ObjectCategories'
trainingDirectory = '/home/mariamaboelwafa/256_ObjectCategories'
validationDirectory = 'validation_data'

inputFormat = Input(shape=(Width, Height, Depth))


digit_input = Input(shape=(250, 250, 3))
x = Flatten()(digit_input)
xx = Dense(250*250*3, activation='relu')(x)
xxx = Dropout(0.5)(xx)
out = Dense(Classes,activation='softmax')(xxx)
 

MariamModelFC = Model(digit_input, out)


'''
MariamModelFC = Sequential()
    MariamModelFC.add(Flatten(input_shape=train_data.shape[1:]))
    MariamModelFC.add(Dense(256, activation='relu'))
    MariamModelFC.add(Dropout(0.5))
    MariamModelFC.add(Dense(2, activation='sigmoid'))'''

                  
myOptimizer = Adam(lr=0.001)    

# compile the model
MariamModelFC.compile(optimizer=myOptimizer, loss='categorical_crossentropy', metrics=['accuracy',metrics.top_k_categorical_accuracy])   # categorical_crossentropy iss for negative log likelihood

MariamModelFC.summary()      # print out summary of my model


checkpointCallback = ModelCheckpoint(filepath="./weights.hdf5", monitor="val_acc", verbose=1, save_best_only=True)

earlyStoppingCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

tensorBoardCallback = TensorBoard(log_dir="./logs/training_" + time.strftime("%c"), histogram_freq=0, write_graph=True,write_images=False)
reduceOnPlateauCallback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',epsilon=0.0001, cooldown=0, min_lr=0.000001)
terminateOnNaNCallback = TerminateOnNaN()

callbacks = [checkpointCallback, earlyStoppingCallback, reduceOnPlateauCallback, terminateOnNaNCallback, tensorBoardCallback]


trainingDataGenerator = MyImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # don't set each sample mean to 0
    featurewise_std_normalization=True,  # divide all inputs by std of the dataset
    samplewise_std_normalization=False,  # don't divide each input by its std
    zca_whitening=False,  # don't apply ZCA whitening.
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180).
    horizontal_flip=True,  # randomly flip horizontal images.
    vertical_flip=False,  # don't randomly flip vertical images.
    zoom_range=0.1,  # slightly zoom in.
    width_shift_range=0.1,
    height_shift_range=0.1
)

validationDataGenerator = MyImageDataGenerator(
    featurewise_center=True,  # test images should have input mean set to 0 over the images.
    featurewise_std_normalization=True,  # test images should have all divided by std of the images.
    zca_whitening=False
)


trainingDataGenerator = trainingDataGenerator.flow_from_directory(trainingDirectory,target_size=(Width, Height),batch_size=ImagesPerBatch,shuffle=True)


validationDataGenerator = validationDataGenerator.flow_from_directory(validationDirectory,target_size=(Width,Height),batch_size=ImagesPerBatch,shuffle=False)

print("Data Set loaded!" + '\n')
print("Training...")


MariamModelFC.fit_generator(trainingDataGenerator, steps_per_epoch=TrainingBatches, epochs=Epochs,validation_data=validationDataGenerator, validation_steps=ValidationBatches, verbose=1,callbacks=callbacks)


