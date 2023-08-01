# Tutorial Run-Wise BKG Fitting

Using the HESS public Crab runs to show how to fit the 3D Bkg template to each run then creating a dataset by stacking all runs. 

Link to public dataset:

https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/

Necessary files:
- `obsid.txt` IDs of the observations to be analysed
- `hess_dl2_dr` path to the datastore (here: public data)

Files created: 
- `threshold.txt` and `bkg_threshold.txt` saving the 10% Bias and BKG Peak thresholds of the runs, respectively.
- `dataset-stacked.fits.gz` resulting stacked dataset

Note:
The note book is a three-in-one, creating the thresholds, creating the datasets and fitting the datasets. 
For many runs, detangling is recommended. 
