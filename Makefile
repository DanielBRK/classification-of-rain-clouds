run_params:
	python -c 'from cnnmodel.main import get_params; (tr,te,v)=get_params()'

run_getdata:
	python -c 'from cnnmodel.main import get_data; get_data(tr,te,v,64)'

run_model_init:
	python -c 'from cnnmodel.main import model_init; model=model_init()'

run_model_compile:
	python -c 'from cnnmodel.main import model_compile; model=model_compile(model,0.001)'

run_model_fit:
	python -c 'from cnnmodel.main import model_compile; model=model_fit(model,tr,v,1)'

run_save_model:
	python -c 'from cnnmodel.main import model_compile; model=sv_model(model,"exportedmodel")'

run_save_model:
	python -c 'from cnnmodel.main import model_compile; ld_model("exportedmodel")'
