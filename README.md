# SpatialSim howto

This file provides information on replicating he experiments of the accompanying paper, and instructions on how to use the provided datasets.

## Replicating experiments

This is done in two steps: first, generate the configuration file for the wanted experiment; second, launch the created runfile.

### Identification

For Identification, please use the command line command

```
python generate_config.py --mode simple
```

The default datasets used are IDS_3 to IDS_8, containing 3 to 8 objects. For changing this behavior, one can specify the minimum number of objects and maximum number of objects, respectively, by using the `-No` (resp. `-Nm`) flag. Thus, the above command is equivalent to

```
python generate_config.py --mode simple -No 3 -Nm 8
```

After typing the command, a number is printed in the terminal; it corresponds to the identifying index of the configuration file and of the experiment. The configuration file is stored in the `configs` folder under `config<index>`. To launch the experiment, please type (if using bash):

```
bash launch<index>.sh
```

The results are stored in the `experimental_results` folder, under `expe<index>`. Inside the folder, one can find the log file, and a folder for each of the trained models, inside which there is a folder `data` for the train and test accuracies and a folder `models` where the weights for the different seeds of the models, at the end of training, are stored. The format is the one provided by pytorch.

### Comparison

For Comparison, the process is similar. To replicate the experiments in the paper, please type:

```
python generate_configs.py --mode double
```

As in the previous case, the `-No` and `-Nm` flags allow to control the minimum and maximum numbers of objects. The different allowed combinations of these two flags are `-No 3 -Nm 8` (default), `-No 9 -Nm 20`, and `-No 21 -Nm 30`.

Once the configuration file is generated (its index is printed in the terminal), please execute 

```
bash launch<index>.sh
```

to launch the experiment.

### Getting the metrics

To access the results of training (test accuracies), please enter the python interpreter, and then execute:

python```
>>> from run_utils import *
>>> model_metrics(<index>)
```

## Using the provided datasets

### Data format

The data are provided in the data.zip folder in the Google Drive folder of the experiment. All details about the different datasets are given in the Supplementary Material of the paper.

Each dataset is provided as a JSON dump of a dictionnary. This dictionnary has two entries: `'data'` for the data, and `'labels'` for the labels. The labels entry is a list of length <N_SAMPLES>, with 0 for the negative class and 1 for the positive class. The data entry is: 

- For the Identification task: a list of order 3: for each sample, a configuration is represented as list of all object vectors, the vector of features for each object being itself represented by a list. The `'data'` entry is the list of all configurations.

- For the Comparison task: a list of order 4: for each sample, a list of length 2 gives the two configurations to compare. Each of those configurations is represented in the same way as for Identification, as a list of lists. The `'data'` entry is the list of all couples of configurations.

### Pytorch Dataloaders

We also provide a subclass of the pytorch `Dataset` and `DataLoader` for our datasets. They are given in the `data_utils.py` module, which is standalone and only requires pytorch to run. The module provides the `SpatialSimDataset` and `SpatialSimDataLoader` classes. Example use:

python```
>>> # Identification
>>> dsname = 'IDS_5'
>>> ds = SpatialSimDataset(os.path.join(DATAPATH, dsname))
>>> dl = SpatialSimDataLoader(ds)
>>>
>>> # Comparison
>>> dsname = 'CDS_3_8_0'
>>> ds2 = SpatialSimDataset(os.path.join(DATAPATH, dsname))
>>> dl2 = SpatialSimDataLoader(ds2)
```

The output of `ds[idx]` for Identification is a 2d `torch.Tensor` with the 0-th dimension representing the different objects and the first dimension representing the object features, and a tensor holding the scalar label. For Comparison it is a 3-tuple of `torch.Tensor`s, one for each configuration, followed by the label.

The output of one iteration of the dataloader is, for Identification, a 3-tuple. The first element T is a 2d tensor representation of the data for `BATCH_SIZE` samples, concatenated in the batch (0-th) dimension. The second element is the concatenation of the labels for the batch. The third element is a `torch.LongTensor` of indices, of the same length as T that gives, for each element of T (each object in the batch) the index of the scene the object belongs to; it thus contains elements in `[0..BATCH_SIZE - 1]`.

For Comparison, the data is given in a similar manner. the output of one iteration is a 5-tuple of T1, T2, L, Bi1, Bi2. T1 and T2 are the object tensors for the first and second configurations, where all objects are concatenated alonf the 0-th dimension regardless of the sample they belon to in the batch. L is the tensor of labels, and Bi1/Bi2 are the batch index tensors for the objects in T1 and T2 respectively.

The batch index tensors are necessary because all configurations may not contain the same numbers of objects, which makes it impossible to create a separate dimension for the numbers of objects without using some king of padding.