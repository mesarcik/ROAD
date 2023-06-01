# `ROAD`: The Radio Observatory Anomaly Detector


## Labelling with label-studio:
The labelling interface is based on [label-studio](https://labelstud.io/). To get the label server running for the LOFAR_AD project, run the following:
```
  label-studio start LOFAR_AD --sampling uniform &
```
and
```
./webserver /home/mmesarcik/data/LOFAR/compressed/LOFAR_AD/LOFAR_AD_v1/ *.png files 8081
````
