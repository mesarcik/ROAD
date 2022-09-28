#!/bin/sh
echo "Logging for hyperparams.sh at time: $(date)." >> log.log

limit=5000
epochs=150
seed=42 #$(openssl rand -hex 3)
model=VAE

for patch_size in 16 32 64 128
do
		for ld in 2 8 16 32 64 128 256 
		do
				python -u main.py -model $model\
								  -limit $limit \
								  -epochs $epochs \
								  -latent_dim $ld \
								  -patch_size $patch_size\
								  -neighbours 2 4 8 16 32\
								  -seed $seed| tee -a log.log 
		done 
done
