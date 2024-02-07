<<<<<<< HEAD
from gammapy.datasets import MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
=======
import numpy as np
from gammapy.datasets import MapDataset
from gammapy.modeling.models import (
    CompoundNormSpectralModel,
    FoVBackgroundModel,
    Models,
    PowerLawNormPenSpectralModel,
>>>>>>> 161c2451f9e3a02f82151689962bd67209e6859e
    PowerLawNormSpectralModel,
)
from gammapy.modeling.models.IRF import (  # ,IRFModel
    EffAreaIRFModel,
    ERecoIRFModel,
    IRFModels,
)

<<<<<<< HEAD

def load_config():
    import json
    import os

    path = os.getcwd()
    substring = "nuisance_summary"
    path = path[: path.find(substring)] + substring + "/"
    with open(path + "config.json") as json_file:
        config = json.load(json_file)
    return config


config = load_config()
case = config["case"]
path = config[case]["path"]
figformat = config["figformat"]
path_pksflare = config[case]["path_pksflare"]


def get_path(source):
    return path_pksflare


def create_asimov(model, source, parameters=None, livetime=None):
    path = get_path(source)
    models = set_model(path, model)

    if livetime is not None:
        model = livetime
    dataset = MapDataset.read(f"{path}/HESS_public/dataset-simulated-{model}.fits.gz")
    print("loaded dataset:")
    print(f"{path}/HESS_public/dataset-simulated-{model}.fits.gz")
    if parameters is not None:
        for p in parameters:
            models.parameters[p.name].value = p.value
            models.parameters[p.name].error = p.error

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    bkg_model.parameters["tilt"].frozen = False
    models.append(bkg_model)
    dataset.models = models
    dataset.counts = dataset.npred()
    return dataset


def set_model(path, model):
    return Models.read(f"{path}/HESS_public/model-{model}.yaml").copy()


def load_dataset_N(dataset_empty, path, bkg_sys=False):
    models_load = Models.read(path).copy()
    Source = models_load.names[0]
    models = Models(models_load[Source].copy())
    dataset_read = dataset_empty.copy()

    if bkg_sys:
        import operator

        compoundnorm = CompoundNormSpectralModel(
            model1=PowerLawNormSpectralModel(),
            model2=PowerLawNormPenSpectralModel(),
            operator=operator.mul,
        )

        bkg = FoVBackgroundModel(
            dataset_name=dataset_read.name, spectral_model=compoundnorm
        )

    else:
        bkg = FoVBackgroundModel(dataset_name=dataset_read.name)
    models.append(bkg)

    for p in bkg.parameters:
        p.value = models_load.parameters[p.name].value
        p.error = models_load.parameters[p.name].error
    for m in models_load:
        if m.type == "irf":
            irf = IRFModels(
                e_reco_model=ERecoIRFModel(),
                eff_area_model=EffAreaIRFModel(),
                datasets_names=dataset_read.name,
            )
            for p in irf.parameters:
                p.frozen = False
                p.value = models_load.parameters[p.name].value
                p.error = models_load.parameters[p.name].error
                p.frozen = models_load.parameters[p.name].frozen

            models.append(irf)
    dataset_read.models = models
    return dataset_read


def load_dataset(dataset_empty, path):
    models_load = Models.read(path).copy()
    Source = models_load.names[0]
    models = Models(models_load[Source].copy())
    dataset_read = dataset_empty.copy()

    bkg = FoVBackgroundModel(dataset_name=dataset_read.name)
    for p in bkg.parameters:
        p.value = models_load.parameters[p.name].value
        p.error = models_load.parameters[p.name].error

    models.append(bkg)
    dataset_read.models = models
    return dataset_read
=======
path_crab = "/home/katrin/Documents/Crab"


class sys_dataset:
    def __init__(
        self,
        dataset_asimov=None,
        shift=0,
        tilt=0,
        bias=0,
        resolution=0,
        bkg_norm=None,
        bkg_tilt=None,
        rnd=False,
        e_reco_creation=10,
        cutoff=False,
        gun=False,
    ):
        self.dataset_asimov = dataset_asimov
        self.shift = shift
        self.tilt = tilt
        self.bias = bias
        self.resolution = resolution
        self.bkg_norm = bkg_norm
        self.bkg_tilt = bkg_tilt
        self.rnd = rnd
        self.e_reco_creation = e_reco_creation
        self.cutoff = cutoff
        self.gun = gun

    def create_dataset(self):
        dataset = self.dataset_asimov.copy()
        models = Models(self.dataset_asimov.models.copy())
        # bkg model
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        bkg_model.parameters["tilt"].frozen = False
        models.append(bkg_model)
        dataset.models = models

        if self.rnd:
            counts_data = np.random.poisson(dataset.npred().data)
        else:
            counts_data = dataset.npred().data

        dataset.counts.data = counts_data

        # irf model
        dataset.e_reco_n = self.e_reco_creation
        effareamodel = EffAreaIRFModel()
        ereco = ERecoIRFModel()
        IRFmodels = IRFModels(
            eff_area_model=effareamodel, e_reco_model=ereco, datasets_names=dataset.name
        )
        models.append(IRFmodels)
        dataset.models = models
        dataset.models.parameters["norm_nuisance"].value = self.shift
        dataset.models.parameters["tilt_nuisance"].value = self.tilt
        dataset.models.parameters["bias"].value = self.bias
        dataset.models.parameters["resolution"].value = self.resolution
        dataset.exposure = dataset.npred_exposure()
        dataset.edisp = dataset.npred_edisp()

        # set models without the IRF model
        models = Models(self.dataset_asimov.models.copy())
        models.append(FoVBackgroundModel(dataset_name=dataset.name))
        models.parameters["tilt"].frozen = False
        dataset.models = models

        return dataset

    def create_dataset_N(self, e_reco_n, counts=None):
        dataset_ = self.create_dataset()
        # this is to ensure the same Poission statistic
        if counts is None:
            counts = dataset_.counts.copy()
        dataset_N = MapDataset(
            counts=counts,
            exposure=dataset_.exposure.copy(),
            background=dataset_.background.copy(),
            psf=dataset_.psf.copy(),
            edisp=dataset_.edisp.copy(),
            mask_safe=dataset_.mask_safe.copy(),
            gti=dataset_.gti.copy(),
            name="dataset N",
        )
        models = Models(self.dataset_asimov.models.copy())
        # bkg model
        if self.bkg_norm is not None or self.bkg_tilt is not None:
            import operator

            model2 = PowerLawNormPenSpectralModel()
            compoundnorm = CompoundNormSpectralModel(
                model1=PowerLawNormSpectralModel(), model2=model2, operator=operator.mul
            )

            bkg_model = FoVBackgroundModel(
                dataset_name=dataset_N.name, spectral_model=compoundnorm
            )
            if self.bkg_norm is not None:
                bkg_model.parameters["norm_nuisance"].value = self.bkg_norm
            if self.bkg_tilt is not None:
                bkg_model.parameters["tilt_nuisance"].value = self.bkg_tilt

        else:
            bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name)

        bkg_model.parameters["tilt"].frozen = False
        models.append(bkg_model)
        # irf model
        effareamodel = EffAreaIRFModel()
        ereco = ERecoIRFModel()
        IRFmodels = IRFModels(
            eff_area_model=effareamodel,
            e_reco_model=ereco,
            datasets_names=dataset_N.name,
        )

        models.append(IRFmodels)
        dataset_N.models = models
        dataset_N.e_reco_n = e_reco_n
        return dataset_N
>>>>>>> 161c2451f9e3a02f82151689962bd67209e6859e
