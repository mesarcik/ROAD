#!/bin/sh
echo "Logging for hyperparams.sh at time: $(date)." >> log.log

limit=None
epochs=300
seed=42 #$(openssl rand -hex 3)
model=position_classifier
latent_dim=32
patch_size=64
batch_size=128 #8192
amount=6

python -u main.py -model $model\
				  -limit $limit \
				  -epochs $epochs \
				  -latent_dim $latent_dim \
				  -amount $amount\
				  -patch_size $patch_size\
				  -data_path /data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_18-02-23.h5\
				  -output_path outputs/LOFAR_rfi.csv\
				  -batch_size $batch_size\
				  -neighbours 3\
				  -seed $seed| tee -a log.log 
