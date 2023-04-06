#!/bin/bash
echo "Logging for hyperparams.sh at time: $(date)." >> log.log
limit=None
epochs=100
model=all
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
						  -data_path /data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_06-04-23.h5\
						  -output_path outputs/LOFAR_ouputs_new_dataset.csv\
						  -batch_size $batch_size\
						  -seed $(openssl rand 1 | od -DAn)\
						  -neighbours 5 | tee -a log.log 
done 
