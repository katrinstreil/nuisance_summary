for foreverloop in range(1):

    import random
    
    #TODO: input the initial diffuse emission template parameters
    #different initial parameters will end up in different final parameters after the fit. Continous scenario have the degenaracy problem, while impulsive scenario 
mostly end up at pdelta=0
    
    #List of initial, use import random to pick any of these later
    pindex=[1.5,1.75,2,2.25,2.5] 
    pD0=[0.01,0.1,1,10,100,1000] 
    pdelta=[0,0.25,0.5,0.75,1] 
    pcutoff=[100,1000,10000] 
    ptime=[10**6] #this one is fixed
    pcloudmap=['TS_fine'] #you can also pick 'TY_fine','Y_fine'
    pmode=['imp','impcut','cont','contcut'] 
    
    import sys
    sys.path.append(
    "../I-CR")
    sys.path.append(
    "../V-gammapyhack")
    import diffuse_config 

    import scenario 
    import pickle as pkl

    #load the configuration of diffuse
    def load_diffuse_config():
        minEnergy,maxEnergy,Nspec=diffuse_config.minEnergy,diffuse_config.maxEnergy,diffuse_config.Nspec
        
        BS_TS,BS_TS_fine,BS_Y,BS_Y_fine=diffuse_config.BS_TS,diffuse_config.BS_TS_fine,diffuse_config.BS_Y,diffuse_config.BS_Y_fine
        decayhelpercube,decayhelpertoflux= diffuse_config.decayhelpercube,diffuse_config.decayhelpertoflux
        BN,initial_bin,final_bin= diffuse_config.BN,diffuse_config.initial_bin,diffuse_config.final_bin
        hCloudDist_TS,hCloudDist_TS_fine,hCloudDist_TS_fine_2d,hCloudDist_TY,hCloudDist_TY_fine,hCloudDist_Y,hCloudDist_Y_fine= 
diffuse_config.hCloudDist_TS,diffuse_config.hCloudDist_TS_fine,diffuse_config.hCloudDist_TS_fine_2d,diffuse_config.hCloudDist_TY,diffuse_config.hCloudDist_TY_fine,diffuse_config.hCloudDist_Y,diffuse_config.hCloudDist_Y_fine
        return [minEnergy,maxEnergy,Nspec,BS_TS,BS_TS_fine,BS_Y,BS_Y_fine,\
         decayhelpercube,decayhelpertoflux,BN,initial_bin,final_bin,\
         hCloudDist_TS,hCloudDist_TS_fine,hCloudDist_TS_fine_2d,hCloudDist_TY,hCloudDist_TY_fine,hCloudDist_Y,hCloudDist_Y_fine]

    Para=load_diffuse_config()


    #these number doesn't matter, just to create a frame for save.pkl file
    diff_para=[1.503167,61.555380,0.550192,104.881808,10**6,'contcut','TS_fine',Para]
    with open('../IV-data/save.pkl', 'wb') as f:
        pkl.dump(diff_para, f)

    import gammapyhack
    import analysistree

    import numpy as np
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import csv
    import astropy
    from astropy.convolution import Tophat2DKernel
    from astropy.coordinates import SkyCoord
    from astropy.coordinates.angle_utilities import angular_separation
    import astropy.units as u
    from astropy.io import fits
    from regions import CircleSkyRegion, RectangleSkyRegion
    import gammapy
    from gammapy.datasets import MapDataset, Datasets,FluxPointsDataset
    from gammapy.estimators import FluxPointsEstimator, FluxPoints
    from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
    from gammapy.modeling import Fit,Parameter
    from gammapy.modeling.models import (PointSpatialModel,PowerLawSpectralModel,
        GaussianSpatialModel,SkyModel,FoVBackgroundModel,SpectralModel,
        ExpCutoffPowerLawSpectralModel,Models,SpatialModel,
        TemplateSpectralModel,TemplateSpatialModel,ConstantSpatialModel,PowerLawNormSpectralModel,)
    from regions import CircleSkyRegion
    from gammapy.makers import MapDatasetMaker, SafeMaskMaker
    from gammapy.data import Observation, DataStore
    from gammapy.irf import load_cta_irfs, EDispKernel, PSFKernel
    from gammapy.visualization.utils import plot_contour_line
    from gammapy.estimators import ExcessMapEstimator
    from gammapy.catalog import SourceCatalogHGPS
    from gammapy.modeling.fit import Registry
    from gammapy.modeling.covariance import Covariance
    from gammapy.utils.scripts import make_name
    import copy
    from scipy.stats import norm
    from scipy.integrate import quad
    from itertools import combinations
    import random
    import sys; sys.path.insert(0, '..')
    matplotlib.use('pdf')

    import gammapyhack
    import importlib
    importlib.reload(gammapyhack)


    #randomly pick the initial parameters for the fit
    import glob
    for q in range(1):
        diff_para=[random.choice(pindex),random.choice(pD0),random.choice(pdelta),\
                   random.choice(pcutoff),random.choice(ptime),\
                   random.choice(pmode),random.choice(pcloudmap),Para]
        if not 'cut' in diff_para[5]:
            diff_para[3]=random.choice([100])

    #save the initial parameters in save.pkl
    with open('../IV-data/save.pkl', 'wb') as f:
        pkl.dump(diff_para, f)

    key = key+f'_[{diff_para[0]}_{diff_para[1]}_{diff_para[2]}_{diff_para[3]}_{diff_para[4]}_{diff_para[5]}_{diff_para[6]}]'

    print(key)
    import gammapyhack
    import importlib
    importlib.reload(gammapyhack)
    
    #creat a template frame using the initial parameter
    diffuse_galactic_hess=scenario.gamma_map(diff_para[0],diff_para[1],diff_para[2],diff_para[3],diff_para[4],diff_para[5],diff_para[6],diff_para[7])
    
    #put this template frame to the MyTemplateSpatialMode, will fit the parameters in the template
    #filename doesn't matter, I dont read that file anyway
    if diff_fit == 'loose':
        template_diffuse = gammapyhack.MyTemplateSpatialModel(diffuse_galactic_hess, normalize=False, filename='dataset_fit_diffuse.fits')
        
    #strict means I set some boundaries for diffuse template
    #filename doesn't matter, I dont read that file anyway
    if diff_fit == 'strict':
        template_diffuse = gammapyhack.MyTemplateSpatialModel_s(diffuse_galactic_hess, normalize=False, filename='dataset_fit_diffuse.fits')
        
    #Define the diffuse_model
    diffuse_model = SkyModel(spectral_model=PowerLawNormSpectralModel(),spatial_model=template_diffuse,name="diffuse")
    diffuse_model.spectral_model.norm.min = 0




    #Define the fg_model, standard procedure
    filename = "../IV-data/gammarays-10GeV-Fornieri20-Remy18-pi0-CO-GC_allRings_wo_CMZ_CAR_normlalized.fits"
    if '3s' in tree:
        filename='../IV-data/Hermes_SimpleCRdensity_CO_allRings_wo_CMZ_02deg.fits'
    if '3n' in tree:
        filename='../IV-data/Hermes_CO_allRings_wo_CMZ_02deg.fits'

    m = Map.read(filename)
    m.unit = "sr^-1"
    fg_spatial_model = TemplateSpatialModel(m, filename=filename,normalize=True)
    fg_spectral_model = PowerLawSpectralModel(index=2, amplitude="8e-7 cm-2 s-1 TeV-1", reference="0.01 TeV")
    if cutoff_fg==True:
        fg_spectral_model = ExpCutoffPowerLawSpectralModel(index=2, amplitude="8e-7 cm-2 s-1 TeV-1", lambda_="0.1 TeV-1",reference="0.01 TeV")
        
    fg_model = SkyModel(spectral_model=fg_spectral_model,spatial_model=fg_spatial_model,name="foreground")

    
