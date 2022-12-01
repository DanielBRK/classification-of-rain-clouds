import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,save_model,load_model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import os

def get_params():
    train_dir = os.environ.get("train_dir")
    test_dir = os.environ.get("test_dir")
    val_dir = os.environ.get("val_dir")
    print("Parameters loaded")
    return (train_dir,test_dir,val_dir)


def get_data(train_dir,test_dir,val_dir,batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen=ImageDataGenerator(rescale=1./255)

    #train_datagen_2 = ImageDataGenerator(
    #    rotation_range=40,  # Rotations
    #    width_shift_range=0.2,  # Horinzontal shift
    #    height_shift_range=0.2,  # Vertical shift
    #    rescale=1./255,
    #    shear_range=0.2,  # Transvection (shear mapping)
    #    zoom_range=0.2,  # Zoom
    #    horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')

    print("Train/test/validation data loaded")
    return (train_generator,validation_generator,test_generator) # val_dir optional

def model_init():

    base_model = VGG16(weights = "imagenet", include_top = False, input_shape = (32, 32, 3))

    x = base_model.output

    x = layers.Flatten()(x)

    x = layers.Dense(128, activation = "relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation = "relu")(x)
    x = layers.Dropout(0.3)(x)
    pred = layers.Dense(1, activation = "sigmoid")(x)
    base_model.trainable = False
    model = Model(inputs = base_model.input , outputs = pred)

    # We use the keras Functional API to create our keras model

    print("Model initialised")
    return model

def model_compile(model,lr=0.001):
    adam = optimizers.Adam(lr = lr)
    model.compile(loss='binary_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
    print("Model compiled")
    return model

def model_fit(model,train_generator,validation_generator,epochs=1):

    MODEL = "model"

    modelCheckpooint = callbacks.ModelCheckpoint("{}.h5".format(MODEL), monitor="val_loss", verbose=0, save_best_only=True)

    LRreducer = callbacks.ReduceLROnPlateau(monitor="val_loss", factor = 0.1, patience=3, verbose=1, min_lr=0)

    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks = [modelCheckpooint, LRreducer, EarlyStopper])

    print("Model fitted")
    return (model,history)

def sv_model(model,filename):
    save_model(model,filename)
    print("model saved")

def ld_model(filename):
    model=load_model(filename)
    print("model loaded")
    return model

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
