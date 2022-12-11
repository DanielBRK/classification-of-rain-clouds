# python runpredict.py

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from cnnbinary.main import get_data,ld_model

filename="./data/h5/model_c.h5"
batch_size=64
train_dir="./data/categorical/train/"
test_dir="./data/categorical/test/"
val_dir="./data/categorical/val/"

(train_generator,validation_generator,test_generator)=get_data(train_dir,test_dir,val_dir,batch_size)
model=ld_model(filename)

img = load_img("../Projekt/SkyCam/2018_07_01_0545.jpg")
img = img.resize((224, 224))
img = img_to_array(img)/255
print(type(img))
img = img.reshape((-1, 224, 224, 3))
res=model.predict(img)
print(res)
#model.predict(test_generator)
