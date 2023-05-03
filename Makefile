init:
	mkdir -p data
	mkdir -p models
	mkdir -p samples
	pip install -r req.txt
	pip install -r req-dev.txt

data: init
	gsutil cp gs://muse-model/museC_data/test_data.json data/train.json
	
params: init
	echo "getting params"