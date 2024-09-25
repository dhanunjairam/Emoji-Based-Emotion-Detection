import numpy as np # type: ignore
import cv2 # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from keras.layers import Dropout , Dense,Flatten,Conv2D,MaxPooling2D # type: ignore
from keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

train_dir = 'data/train'
valid_dir = 'data/valid'

test_dir = 'data/test'  # Your actual test dataset directory

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale, no augmentation
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(train_dir,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')
valid_gen = valid_datagen.flow_from_directory(valid_dir,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001,decay=1e-6),metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)


model_info = model.fit(
    train_gen,
    steps_per_epoch=28709//64,
    epochs=100,
    validation_data=valid_gen,
    validation_steps=7178//64,
    callbacks=[early_stopping, model_checkpoint]
)



model.save('model2.keras')
model.save_weights('model2.weights.h5')
