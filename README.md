# README

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

The data are provided in the data.zip folder in the Google Drive folder of the experiment. All details about the different datasets are given in the Supplementary Material of the paper.

To use