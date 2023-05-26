"""
Testing model using painting patches instead of full paintings.
Used RGB colorway - most accurate for full painting.
Tried on two models - overfitting and underfitting since couldn't optimize to be perfectly fitting.

Acc: 0.5520, loss: 0.6878
Val_acc: 0.8200, val_loss: 0.6264 using underfitting model

Acc: 0.9885, loss: 0.0776
Val_acc: 0.8320, val_loss: 5.3757 using overfitting model.
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, datasets, models
from keras import preprocessing
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import os
from keras import optimizers
from keras import regularizers
from keras.layers import Conv2D
from keras.regularizers import l2
from keras import Sequential

import patched

# set base directory for images - right now only looking at paintings
base_dir = 'patched'
try:
   if not os.path.exists(os.path.dirname(base_dir)):
       os.makedirs(os.path.dirname(base_dir))
except OSError as err:
   print(err)


# Directories for folders with training, validation, test sets
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# make train directory
try:
   if not os.path.exists(os.path.dirname(train_dir)):
       os.makedirs(os.path.dirname(train_dir))
except OSError as err:
   print(err)

# make validation directory
try:
   if not os.path.exists(os.path.dirname(validation_dir)):
       os.makedirs(os.path.dirname(validation_dir))
except OSError as err:
   print(err)

# make test directory
try:
   if not os.path.exists(os.path.dirname(test_dir)):
       os.makedirs(os.path.dirname(test_dir))
except OSError as err:
   print(err)

# Directory with our training corot pictures (confirmed real)
train_corot_dir = os.path.join(train_dir, 'corot')
try:
   if not os.path.exists(os.path.dirname(train_corot_dir)):
       os.makedirs(os.path.dirname(train_corot_dir))
except OSError as err:
   print(err)

# Directory with our training forgery pictures
train_forgery_dir = os.path.join(train_dir, 'forgery')
try:
   if not os.path.exists(os.path.dirname(train_forgery_dir)):
       os.makedirs(os.path.dirname(train_forgery_dir))
except OSError as err:
   print(err)

# Directory with our validation corot pictures
validation_corot_dir = os.path.join(validation_dir, 'corot')
try:
   if not os.path.exists(os.path.dirname(validation_corot_dir)):
       os.makedirs(os.path.dirname(validation_corot_dir))
except OSError as err:
   print(err)

# Directory with our validation forgery pictures
validation_forgery_dir = os.path.join(validation_dir, 'forgery')
try:
   if not os.path.exists(os.path.dirname(validation_forgery_dir)):
       os.makedirs(os.path.dirname(validation_forgery_dir))
except OSError as err:
   print(err)

# Directory with our test corot pictures
test_corot_dir = os.path.join(test_dir, 'corot')
try:
   if not os.path.exists(os.path.dirname(test_corot_dir)):
       os.makedirs(os.path.dirname(test_corot_dir))
except OSError as err:
   print(err)

# Directory with our test forgery pictures
test_forgery_dir = os.path.join(test_dir, 'forgery')
try:
   if not os.path.exists(os.path.dirname(test_forgery_dir)):
       os.makedirs(os.path.dirname(test_forgery_dir))
except OSError as err:
   print(err)

# Overfitting model: 2 million params

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Underfitting model: 1 million params

# model = Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer ='he_normal',
#                         input_shape=(128, 128, 3)))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.25))
# model.add(layers.Convolution2D(64, (3, 3), padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.25))
# model.add(layers.Convolution2D(128, (3, 3), padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.25))
# model.add(layers.Convolution2D(32, (3, 3), padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(layers.Flatten())
# model.add(layers.Dense(32, 
#               kernel_regularizer=l2(0.01),      
#               bias_regularizer=l2(0.01)))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(32, 
#               kernel_regularizer=l2(0.01),      
#               bias_regularizer=l2(0.01)))
# model.add(layers.Activation('relu'))
# model.add(layers.Dense(1, activation='sigmoid'))


model.summary()

# compile model
# model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(learning_rate=1e-4), metrics = ['acc'])
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr=0.001, decay=1e-6), metrics = ['acc'])
# generate data
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        class_mode='binary',
        color_mode="rgb",
        batch_size=20, classes=['corot', 'forgery'])

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=20,
        color_mode="rgb",
        class_mode='binary', classes=['corot', 'forgery'])

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# fit the model
history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      shuffle=True,
      validation_steps=50)


# save model in directory for data
models_dir = "saved models"
model.save(os.path.join(models_dir, 'brushstrokes_model'))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Brushstroke Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Brushstroke Training and validation loss')
plt.legend()

plt.show()

