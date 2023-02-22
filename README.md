# Model-fingerprinting-and-ownership-verification

This repo contains essential code for paper: fingerprinting deep neural network globally via universal adversarial perturbations.

## Prerequisite:
1. Pytorch Version > 1.6.0
2. Python Version > 3.6.9
3. Preparing for necessary experiment materials: 
    - Determining model architectures for victim/extraction/homologous models, save your architecture in root path of your project under filename "./model_architecture.py", provided a get_model function in this .py file which takes a single parameter (Key) and outputs a corresponding architecture (Value).
    - Victim model
    - UAP for victim model: if you are not familiar with UAP generation process, using https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#universal-perturbation-attack helps you speed up the process. Keep your attack success rate > 70%
    - Shadow models:
        - Extraction model: follow this paper to produce your extraction model https://arxiv.org/abs/1602.02697 (either using prediction class or prediction label works fine, make sure your extraction agreement > 70%). You may use the extraction package: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/extraction.html as well.
        - Homologous model: change hyperparameters, mixing test set into your trainset, try another model structure etc. Produce your homologous model the same way as producing your victim model. 
    - Test model: Models to test the verification framework. Ideally, we recommend that you train your test models expicilty from shadow models.
  
## How to start:
To begin with, one must understand that this repository helps you build up a verification framework to protect your victim model, meaning that given any suspect models, the verification framework helps you figure out whether it violates your IP. As the whole verification process requires enormous run-time memory, we breaks it up into several steps and each step saves its internal result to enable any restarting after interruption. The following steps show you how to build this framework:

1. Run function fingerprintPointSelection in fingerprint_point_selection.py. This function returns selected fingerprint generation datapoints and save them in ./result/split_sets_" + str(n_clusters) + "_points_" + str(n_neighbors) + "views.pkl"
    - Configure your n_cluster, n_neighbors coressponding to your datasets. Remember that these two parameters are tunable hyper-parameters (the recommeded choice and the effect of these parameters are discussed in ablation study)
    - Load your victim model into variable net
    - Load your victim model's training data into data_loader (See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for detail)
3. Run function framework_building function in verification_framework.py. This function returns a trained encoder that outputs similarities between models:
    - Load your victim model into variable fv
    - Load your UAP for victim model into variable v
    - Load fingerprint generation points from ./result/split_sets_" + str(n_clusters) + "_points_" + str(n_neighbors) + "views.pkl" into variable fp_gene_points
    - Structure your models prepared in prerequisite:
        - put all the extraction models into one folder, copy path of this folder into indep_models_path. put all the homologous models into one folder, copy path of this folder into homo_models_path.
        - specifying models architecture in indep_models_arch and homo_models_archi: A list contains architecture keys, length equals number of models.
4. Run framework_testing to test the performance of the trained encoder on your test models.

