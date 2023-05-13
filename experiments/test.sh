#!/bin/bash
echo "Logging for hyperparams.sh at time: $(date)." >> log.log
limit=None
epochs=50
model=all
latent_dim=64
patch_size=64
batch_size=64
amount=1
resize_amount=0.25
percentage_data=0.5

for backbone in  resnet18 resnet resnet50; do
    for repeat in 1 2 3 ; do 
            python -u main.py -model $model\
                              -limit $limit \
                              -epochs $epochs \
                              -latent_dim $latent_dim \
                              -amount $amount\
                              -backbone $backbone\
                              -percentage_data $percentage_data\
                              -patch_size $patch_size\
                              -data_path /var/scratch/mesarcik/data/constructed_lofar_ad_dataset_06-04-23.h5\
                              -output_path outputs/LOFAR_backbone_test.csv\
                              -model_path /var/scratch/mesarcik\
                              -resize_amount $resize_amount\
                              -batch_size $batch_size\
                              -seed $(openssl rand 1 | od -DAn)\
                              -neighbours 5 | tee -a log.log 
    done 
done 
