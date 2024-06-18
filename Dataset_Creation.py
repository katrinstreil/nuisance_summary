from gammapy.datasets import MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    PowerLawNormSpectralModel,
)
from gammapy.modeling.models.IRF import (  # ,IRFModel
    EffAreaIRFModel,
    ERecoIRFModel,
    IRFModels,
)


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