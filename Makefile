init:
	mkdir -p data
	mkdir -p models
	mkdir -p samples
	pip install -r req.txt
	pip install -r req-dev.txt

data:
	echo "getting data"
	
params:
	echo "getting params"