init:
	mkdir -p data
	mkdir -p models
	mkdir -p samples
	pip install -r req.txt
	pip install -r req-dev.txt

data:
	sudo apt-get install gsutil
	gsutil cp gs://muse-model/museC_data/test_data.json data/train.json
	
params:
	echo "getting params"