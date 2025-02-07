Nuisance Summary

Branches:
- main: results used in the publication 
- archive-branch: Folders containing the data and the plots which where created with other hess datasets/ systematic settings / input spectra

Runs with gammapy-dev(v1.1)/IRF_model
https://github.com/katrinstreil/gammapy/tree/IRF_model




Folder system:
- Analysis: notebooks to fit Asimov datasets, rnd datasets, contours etc. Production of plots saved in the rest of the folders
- Outputfolders: {type of sys}_{IRF}_{model}_{spectral model details}
- Currently used:
  - Eff_area100_PKSflare_crab_cutoff
  - E_reco_PKSflare_crab_cutoff
  - Combined100_PKSflare_crab_cutoff (Combined = Eff_area + E_reco)
  - BKGpl_MSH_mash
 
config.ipynb: used to define systematic and model setup, saves dict to config.yaml which is read in in the Analysis/notebooks
Dataset_Creation, Dataset_load, Dataset_Setup: Helpher modules 

PKS_flare: 
- Storage of the datasets
- Creation of the datasets: threshold estimation, bkg pre-fitting, simulation of datasets with different livetimes

