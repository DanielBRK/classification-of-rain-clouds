import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,save_model,load_model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import matplotlib.pyplot as plt
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_data(train_dir,test_dir,val_dir,batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen=ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    print("Train/test/validation data loaded")
    return (train_generator,validation_generator,test_generator)

def model_init():

    base_model = ResNet50V2(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))

    x = base_model.output

    x = layers.Flatten()(x)

    x = layers.Dense(128, activation = "relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation = "relu")(x)
    x = layers.Dropout(0.3)(x)
    pred = layers.Dense(4, activation = "softmax")(x)
    base_model.trainable = False
    model = Model(inputs = base_model.input , outputs = pred)

    # We use the keras Functional API to create our keras model

    print("Model initialised")
    return model

def model_compile(model,lr=0.001):
    adam = optimizers.Adam(learning_rate = lr)
    model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['accuracy']) #,f1_m,precision_m, recall_m])
    print("Model compiled")
    return model

def model_fit(model,train_generator,validation_generator,epochs=1):

    MODEL = "model"

    modelCheckpooint = callbacks.ModelCheckpoint("model_c_checkpoint.h5".format(MODEL), monitor="val_loss", verbose=0, save_best_only=True)

    LRreducer = callbacks.ReduceLROnPlateau(monitor="val_loss", factor = 0.1, patience=3, verbose=1, min_lr=0)

    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks = [modelCheckpooint, LRreducer, EarlyStopper])

    print("Model fitted")
    return (model,history)

def sv_model(model,history,filename):
    np.save('model_c_history.npy',history.history)
    save_model(model,filename)
    # history=np.load('my_history.npy',allow_pickle='TRUE').item()
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
