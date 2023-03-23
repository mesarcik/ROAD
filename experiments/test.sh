#!/bin/bash
echo "Logging for hyperparams.sh at time: $(date)." >> log.log
limit=None
epochs=100
seed=42 #$(openssl rand -hex 3)
model=ssl
latent_dim=64
patch_size=64
[ "$model" = supervised ] &&
   batch_size=256 ||
   batch_size=64
backbone=resnet50

for amount in 1; do 
		python -u main.py -model $model\
						  -limit $limit \
						  -epochs $epochs \
						  -latent_dim $latent_dim \
						  -amount $amount\
						  -backbone $backbone\
						  -patch_size $patch_size\
						  -data_path /data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_07-03-23.h5\
						  -output_path outputs/LOFAR_refactor.csv\
						  -batch_size $batch_size\
						  -neighbours 5\
						  -seed $seed| tee -a log.log 
done 
