import json
import os 
from lofarReadSnippet import read_hypercube
import numpy as np 
#from read_data import plot_dynamic_spectra_from_file 
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
import sys

assert sys.version_info <= (2, 8), "Due to a bug found in the original reading library python2.7 is requires to run the data generation script"

_json = json.load(open('project-37-at-2022-09-21-13-12-02fc1e63.json'))
#_json = json.load(open('project-37-at-2022-08-16-12-13-add826e4.json'))
file_name = 'LOFAR_AD_dataset_{}'.format(datetime.now().strftime('%d-%m-%y'))
#_sorted = sorted(json_content, key=lambda d: d['data']['image'].split('/')[-1].split('_')[0])
#sorted_json = json.dumps(_sorted, sort_keys=False)

def norm(data):
    data = np.array(data)
    output = np.zeros(data.shape, data.dtype)
    norm_baseline_plot_values = np.zeros(data.shape, data.dtype)
    for pol_idx in range(data.shape[2]):
            for sb_nr in range(data.shape[1]):
                time_series = data[:,sb_nr,pol_idx]
                time_indices = np.arange(time_series.shape[0])
                model = np.polyfit(time_indices, time_series, 3)
                predicted = np.polyval(model, time_indices)

                norm_time_series = time_series - predicted

                norm_baseline_plot_values[:,sb_nr,pol_idx] = norm_time_series
    
    norm_baseline_plot_values = gaussian_filter1d(norm_baseline_plot_values, sigma=1.0, axis=1, truncate=2.5)
    return norm_baseline_plot_values


dirs = [
    'scintillation',
    'data_loss',
    'solar_storm',
    'electric_fence',
    'lightning',
    'high_noise_elements',
    'oscillating_tile',
    ]

data = []
labels = []
source = []
ids = []
stations=[]
frequency_bands =[]

for f in dirs:
    if f == '__pycache__': continue
    for h5s in  os.listdir(f):
        if '.csv' in h5s: continue
        h5s = '785751' 
        print('Reading {}/{}/L{}.MS_extract.h5'.format(f,h5s,h5s))
        h5_file = read_hypercube('{}/{}/L{}.MS_extract.h5'.format(f,h5s,h5s),
                                  visibilities_in_dB=False)
                                  #read_visibilities=True,
                                  #read_flagging=False)
        _file = "L"+h5s
        sublist = [j for j in _json if j['data']['image'].split('/')[-1].split('_')[0] == _file]
        old_sap = '' # done to improve efficiency
        for entry in sublist:
            contents = entry['data']['image'].split('/')[-1].split('_')
            sap, station = int(contents[1][3:]), contents[2]
            if sap != old_sap:
                old_sap = sap
                vis, baselines,freqs  = np.absolute(h5_file['saps'][sap]['visibilities']),np.array(h5_file['saps'][sap]['baselines']), np.array(h5_file['saps'][sap]['central_frequencies'])

                filtered_baselines_dict = {}
                for bl_idx, baseline in enumerate(baselines):
                        if baseline[0] == baseline[1]:
                            filtered_baselines_dict[bl_idx] = baseline

                #keep only baselines and visibilities for filtered_baselines_dict
                # Done this way to be consistent with the bug used for reading
                baselines = np.array([filtered_baselines_dict[idx] for idx in sorted(filtered_baselines_dict.keys())])
                vis = vis[np.array(filtered_baselines_dict.keys()),:] 
                

            indx = int(np.where(baselines[:,0] == station)[0])
            data.append(vis[indx])
            frequency_bands.append(freqs)
            stations.append(station)
            if not entry['annotations'][0]['result']:
                labels.append([''])
            else:
                labels.append(entry['annotations'][0]['result'][0]['value']['choices'])
            source.append(entry['data']['image'])
            ids.append(entry['id'])

        pickle.dump([data,labels,source, ids, frequency_bands],open(file_name, 'wb'), protocol=2)
#        data = np.array(data)
        #for i,_d in enumerate(data):
        #    entry = sublist[i]
        #    contents = entry['data']['image'].split('/')[-1].split('_')
        #    sap, station = int(contents[1][3:]), contents[2]

        #    fig, axs = plt.subplots(2,2,figsize=(16,9))
        #    vmin, vmax = np.percentile(_d[...,0],[1,99])
        #    axs[0,0].imshow(_d[...,0].T, aspect='auto', interpolation='nearest',vmin=vmin, vmax=vmax,cmap='hot')
        #    axs[0,0].invert_yaxis()
        #    vmin, vmax = np.percentile(_d[...,1],[1,99])
        #    axs[1,0].imshow(_d[...,1].T, aspect='auto', interpolation='nearest',vmin=vmin, vmax=vmax,cmap='hot')
        #    axs[1,0].invert_yaxis()
        #    vmin, vmax = np.percentile(_d[...,2],[1,99])
        #    axs[0,1].imshow(_d[...,2].T, aspect='auto', interpolation='nearest',vmin=vmin, vmax=vmax,cmap='hot')
        #    axs[0,1].invert_yaxis()
        #    vmin, vmax = np.percentile(_d[...,3],[1,99])
        #    axs[1,1].imshow(_d[...,3].T, aspect='auto', interpolation='nearest',vmin=vmin, vmax=vmax,cmap='hot')
        #    axs[1,1].invert_yaxis()
        #    print(labels[i] )
        #    plt.suptitle(labels[i] )
        #    plt.savefig('/tmp/temp/{}_{}_{}'.format(_file,station,sap),dpi=96)
        #    plt.close(fig)
        #print('breaking')
        #break



