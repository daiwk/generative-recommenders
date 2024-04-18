pip3 install gin-config absl-py scikit-learn scipy matplotlib numpy apex hypothesis pandas fvcore iopath tensorboard

pip3 install fbgemm-gpu==0.6.0
pip3 install torch==2.2.0


mkdir -p tmp/ && python3 preprocess_public_data.py


nohup tensorboard --logdir `pwd`/exps/ml-1m-l200/ --port 24001 --bind_all &

CUDA_VISIBLE_DEVICES=0 python3 train.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
