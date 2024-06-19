Nuisance Summary

Branches:
- master: works with gammapy-dev(v1.1)/IRF_model
- gammapy_IRFmodel1: works with gammapy-dev(v1)/IRF_model
- gammapy_nui: works with gammapy-dev(v0.9)/nui



Folder system:
- Analysis: notebooks to fit Asimov datasets, rnd datasets, contours etc. Production of plots saved in the rest of the folders
- Outputfolders: {type of sys}_{IRF}_{model}_{spectral model details}
- Currently used:
  - Eff_area_PKSflare_crab_cutoff
  - E_reco_PKSflare_crab_cutoff
  - Combined_PKSflare_crab_cutoff (Combined = Eff_area + E_reco)
  - BKGpl_MSH_mash
 
config.ipynb: used to define systematic and model setup, saves dict to config.yaml which is read in in the Analysis/notebooks
Dataset_Creation, Dataset_load, Dataset_Setup: Helpher modules 

PKS_flare: 
- Storage of the datasets
- Creation of the datasets: threshold estimation, bkg pre-fitting, simulation of datasets with different livetimes
