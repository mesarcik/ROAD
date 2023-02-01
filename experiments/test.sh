#!/bin/sh
echo "Logging for hyperparams.sh at time: $(date)." >> log.log

limit=None
epochs=50
seed=42 #$(openssl rand -hex 3)
model=position_classifier
latent_dim=128
patch_size=64
batch_size=64 #8192
amount=6

python -u main.py -model $model\
				  -limit $limit \
				  -epochs $epochs \
				  -latent_dim $latent_dim \
				  -amount $amount\
				  -patch_size $patch_size\
				  -data_path /data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_14-01-23.h5\
				  -output_path outputs/LOFAR_decoder.csv\
				  -batch_size $batch_size\
				  -neighbours 3\
				  -seed $seed| tee -a log.log 
