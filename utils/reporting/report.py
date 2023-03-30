import os 
import pandas as pd
from utils import args

def save_results(args:args,**kwargs)->None:
    """
        Gets AUPRC, F-Beta and save into csv 

        Parameters
        ----------
        args:  cmd argument 
        kwargs: 


    """
    if not os.path.exists(args.output_path):
        df = pd.DataFrame(columns = ['Model',
                                     'Name',
                                     'Percentage',
                                     'OOD',
                                     'Latent_Dim',
                                     'Epoch',
                                     'Patch_Size',
                                     'Class',
                                     'Amount',
                                     'Neighbour',
                                     'ErrorType',
                                     'AUPRC',
                                     'Beta',
                                     'F-beta'])
    else:  
        df = pd.read_csv(args.output_path)

    df = pd.concat([df, 
                    pd.DataFrame({'Model':args.model,
                    'Name':args.model_name,
                    'Percentage':args.percentage_data,
                    'OOD': args.ood,
                    'Latent_Dim':args.latent_dim,
                    'Epoch':kwargs['epoch'],
                    'Patch_Size':args.patch_size,
                    'Class':kwargs['anomaly'],
                    'Amount':args.amount,
                    'Neighbour':kwargs['neighbour'],
                    'ErrorType':kwargs['error_type'],
                    'AUPRC':kwargs['auprc'],
                    'Beta':kwargs['beta'],
                    'F-beta':kwargs['f_score']},index=[0])],
                    axis=0, join='outer', ignore_index=True)

    df.to_csv(args.output_path,index=False)

