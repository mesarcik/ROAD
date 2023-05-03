import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd
import os
from utils.data import reconstruct, defaults

def imscatter(z:np.array, inputs:np.array, path:str, epoch:int, ax=None, zoom=1)->None:
    """
        code adapted from: https://gist.github.com/feeblefruits/20e7f98a4c6a47075c8bfce7c06749c2
    """
    fig,ax = plt.subplots(1,1,figsize=(100,100));
    for x, y, image in zip(z[:,0], z[:,1], inputs[:,0,...]):
        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    _dir = '{}/training_outputs'.format(path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    plt.savefig('{}/epoch_embedding_{}'.format(_dir,epoch),dpi=300)
    plt.close('all')

def io(n_plots:int,
       inputs:np.array, 
       reconstructions:np.array,
       path:str, 
       epoch:int, 
       neighbours:int=0, 
       anomaly:str='')->None:
    """
        Plots input and reconstructions
    """
    fig,axs = plt.subplots(n_plots,2,figsize=(5,10));
    for i in range(n_plots):
        r = np.random.randint(len(inputs))
        axs[i,0].imshow(inputs[r,0,...], aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        axs[i,1].imshow(reconstructions[r,0,...], aspect='auto', interpolation='nearest', vmin=0, vmax=1)
   
    _dir = '{}/training_outputs'.format(path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    plt.savefig('{}/{}_epoch_output_{}_{}'.format(_dir, anomaly, epoch, neighbours),dpi=98)
    plt.close(fig)


def knn_io(n_plots:int,
        x_test:np.array, 
        x_train:np.array,
        x_hat_neigh:np.array,
        _D:np.array,
        _I:np.array,
        labels:np.array,
        path:str, 
        epoch:int, 
        args,
        anomaly:str='')->None:
    """
        Plots input and neighbours
    """

    x_recon = reconstruct(x_test, args)
    x_hat_recon= reconstruct(x_hat_neigh[:,0,...], args)

    neighbours = x_train[_I]
    neighbours_recon = np.stack([reconstruct(neighbours[:,i,:],args) 
                                for i in range(neighbours.shape[1])])

    D = np.stack([_D[:,i].reshape(len(_D[:,i])//int(defaults.SIZE[0]//args.patch_size)**2 ,
                                               int(defaults.SIZE[0]//args.patch_size),
                                               int(defaults.SIZE[0]//args.patch_size))
                            for i in range(np.max(args.neighbours))])

    _min, _max = np.percentile(D, [1,99])
    D = np.clip(D,_min, _max)
    D = ((D - D.min()) / (D.max() - D.min()))

    _dir = '{}/training_outputs/{}'.format(path,anomaly)
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    for _ in range(10):
        i = np.random.randint(len(x_recon))
        fig,axs = plt.subplots(3,np.max(args.neighbours)+1,figsize=(10,5));
        axs[0][0].imshow(x_recon[i,0,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
        axs[0][0].set_title("Input", fontsize=5)
        axs[0][0].axis('off')
        axs[1][0].axis('off')

        axs[2][0].imshow(x_hat_recon[i,0,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
        axs[2][0].set_title("0th Recon", fontsize=5)
        axs[2][0].axis('off')
        for n in range(np.max(args.neighbours)):
            axs[0][n+1].imshow(neighbours_recon[n,i,0,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
            axs[0][n+1].set_title("{}".format(labels[i]), fontsize=5)
            axs[0][n+1].axis('off')
            axs[1][n+1].imshow(D[n,i,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
            axs[1][n+1].set_title("{}".format(round(np.mean(D[n,i,...]),3)), fontsize=5)
            axs[1][n+1].axis('off')

            #axs[2][n+1].imshow(D[n,i,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
            #axs[2][n+1].set_title("{}".format(round(np.mean(D[n,i,...]),3)), fontsize=5)
            #axs[2][n+1].axis('off')

        plt.savefig('{}/{}_{}'.format(_dir, anomaly, i),dpi=96)
        plt.close('all')


def loss_curve(path:str,epoch:int, **kwargs)->None:
    plt.title("Train-Validation Accuracy")
    for _name, _loss in kwargs.items():
        if _name == 'descriptor': continue
        plt.plot(np.arange(epoch),_loss, label=_name)

    plt.legend()
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)

    plt.savefig('{}/loss_{}'.format(path,str(kwargs['descriptor'])),dpi=99)

    plt.close('all')

def plot_results(result_path:str, save_path:str, model_name:str, neighbours:int):
    """
        reads in results and plots them
    """
    df = pd.read_csv(result_path)
    df = df[(df.Name == model_name) &
            (df.Neighbour == neighbours)]

    values = df.groupby(['Class']).agg(['mean','std'])['F1'].reset_index()

    pos = np.arange(len(values))
    width = 0.3
    fig, ax = plt.subplots()

    ax.bar(pos, values['mean'].values, yerr=values['std'].values, width=width, label ='SSL')
    ax.set_xticks(pos, list(pd.unique(df.Class)), rotation = 10)

    plt.legend()
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{save_path}/f1_scores',dpi=300)
    plt.close('all')

#def plot_backbones():
#    fig, ax = plt.subplots()
#    df = pd.read_csv('outputs/LOFAR_resnet18_repeats.csv')
#    df = df[df.Class != 'electric_fence']
#
#    for backbone in pd.unique(df.Backbone):
#        _df = df[(df.Backbone == backbone)]
#        means_ssl = _df[_df.ErrorType == 'ssl'].groupby('Class')['F-beta'].mean().values.mean()
#
#        if backbone == 'convnext':
#            marker='o'
#            colour = 'C1'
#        elif 'resnet' in backbone:
#            marker='*'
#            colour = 'C0'
#        elif 'ViT' in backbone:
#            marker= '.'
#            colour = 'C2'
#        model = BackBone(in_channels=4, out_dims=8, model_type=backbone)
#        params = sum(param.numel() for param in model.parameters())/1e6
#
#        ax.scatter(params, means_ssl, marker=marker, color=colour)
#        ax.annotate(backbone, (params, means_ssl))
#
#    ax.set_xlabel('Number of Parameters (Millions)')
#    ax.set_ylabel('Mean F1 Score')
#    plt.grid()
#    plt.tight_layout()
#    plt.savefig('/tmp/temp', dpi=300)
#    plt.close('all')

# def plot_embeddings():
#(train_dataset, val_dataset, test_dataset, supervised_train_dataset, supervised_val_dataset) = get_data(args, transform=None)
#           train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
#           test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
#           
#           backbone = BackBone(in_channels=4,out_dims=args.latent_dim,model_type='resnet18')
#           backbone.load(args,'ssl', True)
#           
#           backbone.to(args.device, dtype=torch.bfloat16)
#           preds,labels,data = [], [],[]
#           for _data, _target, _ ,_ in test_dataloader:
#               _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
#               _z = backbone(_data)
#               Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2,  args.latent_dim*int(defaults.SIZE[0]//args.patch_size)**2])
#               preds.extend(Z.cpu().detach().float().numpy())
#               data.extend(_data.cpu().detach().float().numpy())
#               labels.extend(_target[:,0].numpy().flatten())
#           preds, labels, data  = np.array(preds), np.array(labels), np.array(data)
#           
#           colours = {0:'C0', 1:'C1', 2:'C2', 3:'C3', 4:'C4' ,5:'C5', 6:'C6' , 7:'C7', 8:'C8' , 9:'black'}
#           anomalies = copy.deepcopy(defaults.anomalies)
#           anomalies.append('normal')
#           fig, axs = plt.subplots(1,1, figsize=(7,7))
#           z = TSNE(n_components=2,learning_rate='auto',init='random', perplexity=70).fit_transform(preds)
#           
#           for l in np.unique(labels):
#               ix = np.where(labels== l)
#               axs.scatter(z[ix,0], z[ix,1], c = colours[l], label = anomalies[l], s = 40, alpha=0.8)
#           axs.grid()
#           
#           plt.legend()
#           plt.xlabel('z[0]')
#           plt.xlabel('z[1]')
#           plt.tight_layout()
#           plt.savefig('/tmp/temp', dpi=300)
#           plt.close('all')
