import os 
import pandas as pd
from utils import args

def save_results(args:args,**kwargs):
    # get AUROC, AUPRC, F1 and save into csv 
    if not os.path.exists(args.output_path):
        df = pd.DataFrame(columns = ['Model',
                                     'Name',
                                     'Percentage',
                                     'Latent_Dim',
                                     'Epoch',
                                     'Patch_Size',
                                     'Class',
                                     'Amount',
                                     'Neighbour',
                                     'AUROC',
                                     'AUPRC',
                                     'F1'])
    else:  
        df = pd.read_csv(args.output_path)

    df = pd.concat([df, 
                    pd.DataFrame({'Model':args.model,
                    'Name':args.model_name,
                    'Percentage':args.percentage_data,
                    'Latent_Dim':args.latent_dim,
                    'Epoch':kwargs['epoch'],
                    'Patch_Size':args.patch_size,
                    'Class':kwargs['anomaly'],
                    'Amount':args.amount,
                    'Neighbour':kwargs['neighbour'],
                    'AUROC':kwargs['auroc'],
                    'AUPRC':kwargs['auprc'],
                    'F1':kwargs['f1_score']},index=[0])],
                    axis=0, join='outer', ignore_index=True)

    df.to_csv(args.output_path,index=False)

