conda:
	conda create -n musec python=3.10.8
	
folders:
	mkdir -p data
	mkdir -p models
	mkdir -p samples

init: folders
	pip install -r req.txt
	pip install -r req-dev.txt

data: folders
	gsutil cp gs://muse-model/musec/train.json data/train.json
	gsutil cp gs://muse-model/musec/prompts.json data/prompts.json
	
params: init
	@echo "getting params"