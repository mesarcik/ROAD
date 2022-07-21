import json
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('training_set_aug2020_reformatted.xlsx - Sheet1.csv')
FILE = 'files.txt'
output = []
with open(FILE,'r') as _file:
    for _i,path in tqdm(enumerate(_file)):
        choices =[]
        SAS_id, SAP, station = path.split('/')[-1].split('_')[0:3]
        SAS_id = int(SAS_id[1:])
        observations = df[df['SAS ID'] == SAS_id]
        for label in observations['Label']:# if multiple anomalies per spectrogram
            observation = observations[observations.Label == label]
            _station = observation['Station'].values[0]
            anomaly = observation['Label'].values[0]
            # All
            if 'All' in _station:    
                choices.append(anomaly)

            # International
            elif 'International' in _station:    
                if 'CS' not in station:
                    choices.append(anomaly)
            # Core
            elif 'Core' in _station:    
                if 'CS' in station:
                    choices.append(anomaly)
            # specific station
            elif station in _station:    
                choices.append(anomaly)

        temp_d = {}
        temp_d["id"] = _i
        temp_d["data"] = {"image":path.rstrip('\n')}
        temp_d["annotations"] = []
        temp_d["predictions"] = [{"result":[{"value": {"choices":choices},
                                           "from_name":"choice",
                                           "to_name":"image",
                                           "type":"choices"}]}]
        temp_d["total_predictions"] =0 
        output.append(temp_d)

#final = json.dumps(output, indent=2)
with open('dataset.json', 'w') as f:
    json.dump(output, f)


