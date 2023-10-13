# A Unified Solution for Privacy and Communication Efficiency in Vertical Federated Learning

Published in NeurIPS-2023. 


## Authors:
Ganyu Wang (Western University)\
Bin Gu (Jilin University and MBZUAI)\
Qingsong Zhang (Xidian University)\
Xiang Li (Western University)\
Boyu Wang (Western University)\
Charles X. Ling (Western University)


# How to run this project

## Install the required library
Option 1:\
Install pytorch from https://pytorch.org/.

Other package can be installed via debugging (csv, copy). 

Option 2:\
Use the requirements.txt for config the environment. Or use the file as reference.
```
$ pip install -r requirements.txt
```

## Run with command line.

```
$ python main.py --[arguments below] [argument value below]  
```

| Argument                  | Description                                                             | Default | Options                      |
|---------------------------|-------------------------------------------------------------------------|---------|------------------------------|
| random_seed               | Random seed for repeat experiment                                       | 12341   | -                            |
| framework_type            | The framework of the VFL (VFL-CZOFO, VAFL, ZOO-VFL)                     | ZOFO    | ZOFO, FO, ZO                 |
| dataset_name              | dataset                                                                 | MNIST   | MNIST, CIFAR10               |
| model_type                | The ML model used for the framework.                                    | MLP     | MLP, SimpleResNet18          |
| n_party                   | number of participants, including the server and clients                | 3       | >1                           |
| client_output_size        | the dimension of the embedding.                                         | 64      | MLP: 64, SimpleResNet18: 10. |
| server_embedding_size     | The embedding size of the server's model. Only for MNIST experiment     | 128     |                              |
| client_lr                 | The learning rate of the client                                         | 0.02    |                              |
| server_lr                 | The learning rate of the server                                         | 0.02    |                              |
| batch_size                | Batch size                                                              | 64      |                              |
| n_epoch                   | Number of epoch training                                                | 100     |                              |
| u_type                    | The type of the random perturbation of ZOO                              | Uniform | Uniform, Normal, Coordinate  |
| mu                        | ZOO parameter                                                           | 0.001   |                              |
| d                         | ZOO parameter                                                           | 1       |                              |
| sample_times              | Avg-RandGradEst sampling times (q in the paper)                         | 1       |                              |
| compression_type          | The forward message compression type. No compression/Uniform Scale      | None    | None, Scale                  |
| compression_bit           | The compression bit of the Uniform Scale Compressor                     | 8       | 2, 4, 8                      |
| response_compression_type | The backward message compression type, No compression/Uniform Scale     | None    | None, Scale                  |
| response_compression_bit  | The compression bit of the Uniform Scale Compressor                     | 8       | 2, 4, 8                      |
| log_file_name             | The file name of the log file. The log will be saved at the main folder | None    |                              |


# Result
The result you will get is the log ouput with the training accuracy at each epoch, the commmunication cost for each epoch, and the test accuracy at the last row. 




