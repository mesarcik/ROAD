#!/bin/sh
echo "Logging for hyperparams.sh at time: $(date)." >> log.log

limit=None
epochs=150
seed=42 #$(openssl rand -hex 3)
model=position_classifier

for patch_size in 32 64 128; do
		if   [ $patch_size -eq 32 ]; then batch_size=8196
		elif [ $patch_size -eq 64 ]; then batch_size=4096 
		elif [ $patch_size -eq 128 ]; then batch_size=256
		fi

		for latent_dim in 16 32 64 128 256; do 
				for amount in 1 2 4 6 8; do
						python -u main.py -model $model\
										  -limit $limit \
										  -epochs $epochs \
										  -latent_dim $latent_dim \
										  -amount $amount\
										  -patch_size $patch_size\
										  -data_path /data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_17-11-22.h5\
										  -output_path outputs/LOFAR_hyperparams.csv\
										  -batch_size $batch_size\
										  -neighbours 1 2 5\
										  -seed $seed| tee -a log.log 
				done 
		done
done
