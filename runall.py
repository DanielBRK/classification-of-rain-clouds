
from cnnmodel.main import get_params, \
get_data,model_init,model_compile,model_fit,model_compile,sv_model,ld_model

lr=0.001
filename="exportedmodel.h5"
batch_size=64
epochs=10


(tr,te,v)=get_params()
(train_generator,validation_generator,test_generator)=get_data(tr,te,v,batch_size)
model=model_init()
model=model_compile(model,lr)
(model,history)=model_fit(model,train_generator,validation_generator,epochs)
sv_model(model,history,filename)
model=ld_model(filename)
model.summary()
