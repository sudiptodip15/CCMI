# CCMI

Code for reproducing key results in the paper [CCMI : Classifier based Conditional Mutual Information Estimation](https://arxiv.org/abs/1906.01824) by Sudipto Mukherjee, Himanshu Asnani and Sreeram Kannan. If you use the code, please cite our paper. The code can be used for mutual information and conditional mutual information estimation; conditional independence testing.

## Dependencies 

The code has been tested with the following versions of packages.
- Python 3.6.5
- Tensorflow 1.11.0
- xgboost 0.80 (Optional : To run CCIT baseline for conditional independence testing)

## Running CCMI on your own data-sets 

First cd to the folder contain CCMI code,
```bash
$ cd CIT
$ python
``
and then can run CCMI as shown in the example below :

```bash
>> from CCMI import CCMI
>> import numpy as np
>> X = np.random.randn(5000, 1)
>> Y = np.random.randn(5000, 1)
>> Z = np.random.randn(5000, 1)
>> model_indp = CCMI(X, Y, Z, tester = 'Classifier', metric = 'donsker_varadhan', num_boot_iter = 10, h_dim = 64, max_ep = 20)
>> cmi_est_indp = model_indp.get_cmi_est()
>> print(cmi_est_indp)
-0.0003
>> Y = 0.5*X + np.random.normal(loc = 0, scale = 0.2, size = (X.shape))
>> model_dep = CCMI(X, Y, Z, tester = 'Classifier', metric = 'donsker_varadhan', num_boot_iter = 10, h_dim = 64, max_ep = 20)
>> cmi_est_dep = model_dep.get_cmi_est()
>> print(cmi_est_dep)
0.9707
```

## Reproducing Results of the paper

### CMI Estimation : Synthetic data generation

./data/gen_cmi_data.py - Contains several categories of synthetic data generators that have ground truth CMI values known. The models have X and Y are 1-dimensional variables, which dimension of Z can scale. Model-I in the paper corresponds to 'Category F' and Model-II to 'Category G'. To generate data from any category with given dimension (say dz = 20) and number of samples (say N = 5000), run the following from the 'data' folder:

```bash
PYTHONPATH='..' python gen_cmi_data.py --cat F --num_th 5 --dz 20
```

(Note: PYTHONPATH='..' is required because NPEET code is the parent folder, but gen_cmi_data.py is run from ./data/)

For ease of use, we have provided a bash script './data/gen_synthetic_data_bash.sh' which will generate all the data-sets used for linear and non-linear CMI estimation experiments in the paper. So, alternatively, to generate all data-sets used in the paper, 
```bash
$ chmod +x gen_synthetic_data_bash.sh
$./gen_synthetic_data_bash.sh
```

(Note: Due to random functions used to simulate data and different random seeds, the exact values of true and estimated CMI for generated data-sets will be different from those in the paper.)


### CMI Estimation : Running the Estimators

To run Generator+Classifier estimators, first cd to CMI_Est and then run :
```
bash
python main_CMI_Est.py --mimic cgan --tester Classifier --metric donsker_varadhan --cat F --num_th 5 --dz 20
```

Similarly for other Generators,
```
bash
python main_CMI_Est.py --mimic cvae --tester Classifier --metric donsker_varadhan --cat F --num_th 5 --dz 20
python main_CMI_Est.py --mimic knn --tester Classifier --metric donsker_varadhan --cat F --num_th 5 --dz 20
```

For difference-based CMI estimates, run the following (for Classifier-MI and f-MINE respectively) :
```
bash
python main_CMI_Est.py --mimic mi_diff --tester Classifier --metric donsker_varadhan --cat F --num_th 5 --dz 20
python main_CMI_Est.py --mimic mi_diff --tester Neural --metric f_divergence --cat F --num_th 5 --dz 20
```

For ease of use, we have provided bash scripts './CMI_Est/run_<estimator>_mimic.sh' which will run the corresponding estimator on all linear and non-linear CMI estimation experiments in the paper. So, alternatively, to obtain estimates from CGAN+Classifier, run the following 
```bash
$ chmod +x run_cgan_mimic.sh
$./run_cgan_mimic.sh
```
Similary, run_cvae_mimic.sh, run_knn_mimic.sh, run_mi_diff_mimic.sh, run_mi_diff_mimic_neural.sh, run_ksg_baseline.sh .

(Note : Make sure to run './data/ggen_synthetic_data_bash.sh' before the estimation scripts to ensure data-sets are first created.)

