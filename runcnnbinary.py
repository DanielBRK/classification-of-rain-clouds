# python runcnnbinary.py

from cnnbinary.main import get_data,model_init,model_compile,model_fit \
    ,model_compile,sv_model,plot_history

lr=0.001
batch_size=64
epochs=1
train_dir="./data/binary/train/"
test_dir="./data/binary/test/"
val_dir="./data/binary/val/"


(train_generator,validation_generator,test_generator)=get_data(train_dir,test_dir,val_dir,batch_size)
model=model_init()
model=model_compile(model,lr)
(model,history)=model_fit(model,train_generator,validation_generator,epochs)
model.evaluate(test_generator)
plot_history(history)
#sv_model(model,history,filename)
model.summary()
