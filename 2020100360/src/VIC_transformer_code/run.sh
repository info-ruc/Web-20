mkdir /data6/wwy/cjt/NeuralNLP_transformer/VIC
mkdir /data6/wwy/cjt/NeuralNLP_transformer/results
python3 train.py conf/train.json 2>&1 | tee results/log_transformer.txt