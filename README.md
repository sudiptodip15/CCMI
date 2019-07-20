# CCMI

Code for reproducing key results in the paper [CCMI : Classifier based Conditional Mutual Information Estimation](https://arxiv.org/abs/1906.01824) by Sudipto Mukherjee, Himanshu Asnani and Sreeram Kannan. If you use the code, please cite our paper. The code can be used for mutual information and conditional mutual information estimation; conditional independence testing.

## Dependencies 

The code has been tested with the following versions of packages.
- Python 3.6.5
- Tensorflow 1.11.0
- xgboost 0.80 (Optional : To run CCIT baseline for conditional independence testing)

## Running CCMI as a standalone module

```bash
$ cd CIT
$ python
```
You can run CCMI as a standalone module as shown in the example below :

```bash
>> from CCMI import CCMI
>> import numpy as np
>> X = np.random.randn(5000, 1)
>> Y = np.random.randn(5000, 1)
>> Z = np.random.randn(5000, 1)
>> model_indp = CCMI(X, Y, Z, tester = 'Classifier', metric = 'donsker_varadhan', num_boot_iter = 10, h_dim = 64, max_ep = 20)
>> cmi_est = model_indp.get_cmi_est()
>> print(cmi_est)
-0.0003
>> Y = 0.5*X + np.random.normal(loc = 0, scale = 0.2, size = (X.shape))
>> model_dep = CCMI(X, Y, Z, tester = 'Classifier', metric = 'donsker_varadhan', num_boot_iter = 10, h_dim = 64, max_ep = 20)
>> cmi_est = model_indp.get_cmi_est()
>> print(cmi_est)
0.9707
```

## Documentation 

Detailed documentation coming up soon ...
