run_params:
	python -c 'from cnnmodel.main import get_params; (a,b,c)=get_params()'

run_getdata:
	python -c 'from cnnmodel.main import get_data; get_data(a,b,c,64)'
