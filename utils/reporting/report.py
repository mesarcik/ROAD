import os 
import pandas as pd
from utils import args


def save_results(args:args,**kwargs):
    # get AUROC, AUPRC, F1 and save into csv 
    if not os.path.exists('outputs/results_LOFAR_AD.csv'):
        df = pd.DataFrame(columns = ['Model',
                                     'Name',
                                     'Latent_Dim',
                                     'Patch_Size',
                                     'Class',
                                     'Neighbour',
                                     'AUROC',
                                     'AUPRC',
                                     'F1'])
    else:  
        df = pd.read_csv('outputs/results_LOFAR_AD.csv')

    df = pd.concat([df, 
                    pd.DataFrame({'Model':args.model,
                    'Name':args.model_name,
                    'Latent_Dim':args.latent_dim,
                    'Patch_Size':args.patch_size,
                    'Class':kwargs['anomaly'],
                    'Neighbour':kargs['neighbour'],
                    'AUROC':kwargs['auroc'],
                    'AUPRC':kwargs['auprc'],
                    'F1':kwargs['f1_score']},index=[0])],
                    axis=0, join='outer', ignore_index=True)

    df.to_csv('outputs/results_LOFAR_AD.csv',index=False)

