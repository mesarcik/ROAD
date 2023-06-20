#!/bin/bash
echo "Logging for hyperparams.sh at time: $(date)." >> log.log
limit=None
epochs=100
model=all
latent_dim=64
patch_size=64
batch_size=64
amount=1
resize_amount=0.25
percentage_data=0.5

for backbone in resnet34; do
	python -u main.py -model $model\
					  -limit $limit \
					  -epochs $epochs \
					  -latent_dim $latent_dim \
					  -amount $amount\
					  -backbone $backbone\
					  -percentage_data $percentage_data\
					  -patch_size $patch_size\
					  -data_path /data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_06-04-23.h5\
					  -output_path outputs/LOFAR_validation.csv\
					  -model_path .\
					  -resize_amount $resize_amount\
					  -batch_size $batch_size\
					  -seed 85\ 
					  -neighbours 5 | tee -a log.log 
done 
