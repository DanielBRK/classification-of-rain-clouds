
import numpy as np
from cnnmodel.main import get_params, \
get_data,model_fit,sv_model,ld_model,plot_history

lr=0.001
filename="exportedmodel.h5"
batch_size=64
epochs=30


(tr,te,v)=get_params()
(train_generator,validation_generator,test_generator)=get_data(tr,te,v,batch_size)
model=ld_model("model.h5")
history=np.load("my_history26epochs.npy",allow_pickle='TRUE').item()
#plot_history(history)
model.evaluate(test_generator)
