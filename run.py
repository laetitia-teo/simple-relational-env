"""
This is an executable script for running the Parts experiment (may be updated 
to include other experiments as well).

In the Parts challenge, a model has to learn to discriminate when a
configuration of objects is present in a bigger scene. 
"""
import os.path as op
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

import gen
import baseline_models as bm
import graph_models as gm

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from argparse import ArgumentParser

# data utilities

from dataset import collate_fn
from baseline_utils import data_to_obj_seq_parts
from graph_utils import tensor_to_graphs, data_to_graph_parts

# training 

# visualization/image generation

from env import Env
from test_utils import ModelPlayground

# data path

pretrain_path = op.join('data', 'simple_task', 'train.txt')
train_path = op.join('data', 'parts_task', 'train1.txt')
overfit_path = op.join('data', 'parts_task', 'overfit10000_32.txt')

# script arguments

parser = ArgumentParser()
parser.add_argument('-n', '--nepochs',
                    dest='n',
                    help='number of epochs',
                    default='20')
parser.add_argument('-s', '--seed-number',
                    dest='s',
                    help='number of seeds',
                    default='10')
parser.add_argument('-t', '--task',
                    dest='task',
                    help='task to train and test on',
                    default='parts_task')
parser.add_argument('-c', '--cuda',
                    dest='cuda',
                    help='whether or not to use the GPU',
                    default='True')
parser.add_argument('-d', '--directory',
                    dest='direc',
                    help='integer identifying the directory for saving' \
                    + 'results and models')
args = parser.parse_args()

task = args.task
s = int(args.s)
n = int(args.n)
if args.cuda in ['True', 'true', 't', 'yes', 'y', '1']:
    cuda = True
elif args.cuda in ['False', 'false', 'f', 'no', 'n', '0']:
    cuda = False
direc = args.direc

# task handling

task_list = ['parts_task',
             'similarity_objects',
             'count',
             'select']
if task == 'parts_task':
    task_type = 'scene'
    f_out = 2
    criterion = torch.nn.CrossEntropyLoss()
    directory = op.join('data', 'parts_task', 'idx')
if task == 'similarity_objects':
    task_type = 'objects'
    f_out = 2
    criterion = torch.nn.CrossEntropyLoss()
    directory = op.join('data', 'similarity_objects')
if task == 'count':
    task_type = 'scene'
    f_out = 1
    criterion = torch.nn.MSELoss() # for now
    directory = op.join('data', 'count')
if task == 'select':
    task_type = 'objects'
    f_out = 2
    criterion = torch.nn.CrossEntropyLoss()
    directory = op.join('data', 'select')

# task dict

t_dict = {
    'parts_task': gen.PartsGen,
    'similarity_objects': gen.SimilarityObjectsGen,
    'count': gen.CountGen,
    'select': gen.SelectGen
}

data_path_dict = {
    'parts_task': 'data/parts_task',
    'similarity_objects': 'data/similarity_objects',
    'count': 'data/count',
    'select': 'data/select',
}

# hparams

N_SH = 3
N_OBJ = 3
B_SIZE = 128
L_RATE = 1e-3
N_EPOCHS = 1
F_OBJ = 10
H = 16

f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'f_out': f_out}

image_viz_path = op.join('data')

### General utils ###

def nparams(model):
    """
    Returns the number of trainable parameters in a pytorch model.
    """
    return sum(p.numel() for p in model.parameters())

### Data transformation functions ###

def data_to_clss_parts(data):
    """
    Get the ground truth from the Parts task dataloader.
    """
    return data[2]

def data_to_clss_obj(data):
    """
    Get the ground truth an object dataloader.
    Since the labels are produced in the same way as objects, they are of size
    (N, 1) : we have to squeeze them to (N)
    """
    return data[2].squeeze(-1)

def data_to_clss_count(data):
    return data[2].unsqueeze(-1)

# def data_to_clss_count(data):
#     return data[2].squeeze(-1)

def data_to_clss_simple(data):
    """
    Get the ground truth from the Simple task dataloader.
    """
    return data[1].long()[:, 1]

def data_fn_naive(data):
    return (torch.reshape(data, [-1, 60]),)

def data_fn_scene(data):
    data = torch.reshape(data, (-1, 2, F_OBJ*N_OBJ))
    return data[:, 0, :], data[:, 1, :]

def data_fn_naivelstm(data):
    return (data.permute(1, 0, 2),)

def data_fn_scenelstm(data):
    # TODO : optimize this
    data = torch.reshape(data, (-1, 2, N_OBJ, F_OBJ))
    d1, d2 = data[:, 0, ...], data[:, 1, ...]
    return d1.permute(1, 0, 2), d2.permute(1, 0, 2)

def data_fn_graphs(n):
    """
    Transforms object data in 2 graphs for graph models.

    n is number of objects. 
    """
    def data_fn_gr(data):
        return tensor_to_graphs(data[0])
    return data_fn_gr

def data_to_state_lists(data):
    """
    Used to transform data given by the parts DataLoader into a list of state
    lists usable by env.Env to generate the configuration image corresponding
    to this state of the environment.
    There is one state list per graph in the batch, and there are two scenes
    per batch (the target, smaller scene, and the reference scene).
    """
    targets, refs, labels, t_batch, r_batch = data
    f_x = targets.shape[-1]
    t_state_lists = []
    r_state_lists = []
    n = int(t_batch[-1] + 1)
    for i in range(n):
        t_idx = (t_batch == i).nonzero(as_tuple=True)[0]
        r_idx = (r_batch == i).nonzero(as_tuple=True)[0]
        target = list(targets[t_idx].numpy())
        ref = list(refs[r_idx].numpy())
        t_state_lists.append(target)
        r_state_lists.append(ref)
    return t_state_lists, r_state_lists, list(labels.numpy())

data_fn_graphs_three = data_fn_graphs(3)

### Data loading utilities ###

def load_dl_legacy(name):
    """
    Loads a DataLoader, for the old data format in SimpleTask.
    """
    path = op.join('data', 'simple_task', 'dataset_binaries', name)
    print('loading dataset...')
    with open(path, 'rb') as f:
        ds = pickle.load(f)
    print('done')
    dataloader = DataLoader(ds, batch_size=B_SIZE, shuffle=True)
    return dataloader

def load_dl_parts(name, bsize=128):
    """
    Loads a DataLoader in the Parts Task data format.
    """
    path = op.join('data', 'parts_task', 'old', name)
    print('loading data ...')
    p = gen.PartsGen()
    p.load(path)
    dataloader = DataLoader(p.to_dataset(),
                            batch_size=bsize,
                            shuffle=True,
                            collate_fn=collate_fn)
    print('done')
    return dataloader

def load_dl(name, bsize=128, cuda=False):
    print('loading data...')
    path = data_path_dict[task]
    path = op.join(path, name)
    p = t_dict[task]()
    p.load(path)
    dataloader = DataLoader(p.to_dataset(cuda=cuda),
                            batch_size=bsize,
                            shuffle=True,
                            collate_fn=collate_fn)
    print('done')
    return dataloader

### Model evaluation utilities ###

# binary classification metrics

def compute_accuracy(pred_clss, clss):
    """
    Computes accuracy on one batch prediction.
    Assumes pred_clss is detached from the computation graph.
    """
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    accurate = np.logical_not(np.logical_xor(pred_clss, clss))
    return torch.sum(accurate).item()/len(accurate)

def compute_accuracy_objs(pred, clss):
    """
    Computes accuracy when the result is 
    """
    ...

def compute_precision(pred_clss, clss):
    """
    Computes precision on one batch prediction.
    Assumes pred_clss is detached from the computation graph.
    """
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    tp = torch.sum((pred_clss == 1) and (clss == 1))
    fp = torch.sum((pred_clss == 1) and (clss == 0))
    return tp / (tp + fp)

def compute_recall(pred_clss, clss):
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    tp = torch.sum((pred_clss == 1) and (clss == 1))
    fn = torch.sum((pred_clss == 0) and (clss == 1))
    return tp / (tp + fn)

def compute_f1(precision, recall):
    """
    Computes the F1 score when given the precision and recall.
    """
    return 2 / ((1 / precision) + (1 / recall))

# misc

def compute_close_to_int(pred, true):
    """
    Computes the best metric I could find for the count task.
    The function counts +1 if the predicted number is closest to the true
    integer that to any other integer.
    """
    accurate = (torch.abs(pred - true) <= 0.5).long()
    return torch.sum(accurate).item()/len(accurate)

### Training functions ###

# loss for the counting task, this may be refined
def count_loss():
    return 

def one_step(model,
             dl,
             optimizer,
             criterion=criterion,
             task=task,
             train=True,
             cuda=False):
    accs = []
    losses = []
    n_passes = 0
    cum_loss = 0
    cum_acc = 0
    if task == 'parts_task':
        metric = compute_accuracy
    elif task == 'similarity_objects':
        metric = compute_accuracy
    elif task == 'count':
        metric = compute_close_to_int
    elif task == 'select':
        metric = compute_accuracy
    if isinstance(model, gm.GraphModel):
        data_fn = data_to_graph_parts
        if task == 'parts_task':
            clss_fn = data_to_clss_parts
        elif task == 'similarity_objects':
            clss_fn = data_to_clss_obj
        elif task == 'count':
            clss_fn = data_to_clss_count
        elif task == 'select':
            clss_fn = data_to_clss_obj
    else:
        # handle baselines here
        ...
    for data in tqdm(dl):
        optimizer.zero_grad()
        # ground truth, model prediction
        clss = clss_fn(data)
        if cuda:
            clss = clss.cuda()
        pred_clss = model(*data_fn(data))

        if type(pred_clss) is list:
            # we sum the loss of all the outputs of the model
            loss = sum([criterion(pred, clss) for pred in pred_clss])

        else:
            loss = criterion(pred_clss, clss)

        loss.backward()

        if train:
            optimizer.step()

        l = loss.detach().cpu().item()
        if type(pred_clss) is list:
            # we evaluate accuracy on the last prediction
            a = metric(pred_clss[-1].detach().cpu(), clss.cpu())
        else:
            a = metric(pred_clss.detach().cpu(), clss.cpu())
        cum_loss += l
        cum_acc += a
        losses.append(l)
        accs.append(a)
        n_passes += 1
    return losses, accs

def run(n_epochs, model, dl, data_fn, optimizer, criterion):
    for epoch in range(n_epochs):
        mean_loss, mean_acc = one_step(model,
                                       dl,
                                       data_fn,
                                       optimizer,
                                       criterion)
        print('Epoch : {}, Mean loss : {}, Mean Accuracy {}'.format(
            epoch, mean_loss, mean_acc))

def several_inits(seeds, dl, data_fn):
    """
    Does several runs of one step on different models, initialized with the
    random seeds provided.
    """
    metrics = {}
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        g_model = gm.GraphEmbedding([16], 16, 5, f_dict)
        opt = torch.optim.Adam(g_model.parameters(), lr=L_RATE)
        metrics[seed] = one_step(g_model, dl, data_fn, opt)
    return metrics

# model related funtions

def make_model():
    """
    For rapid creation of models and optimizers.
    """
    model = gm.GraphMatchingv2_U([16, 16], 10, 1, f_dict, task_type)
    opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
    return model, opt

def save_model(m, path):
    """
    Saves the model m with path path in the model save folder.
    """
    torch.save(m.state_dict(), path)

def load_model(m, path):
    """
    Loads the model parameters with path path into model m.
    """
    m.load_state_dict(torch.load(path))
    return m

### Visualization/Plotting/Image generation utilities ###

def batch_to_images(data, path):
    """
    Transforms a batch of data into a set of images.
    """
    t_state_lists, r_state_lists, labels = data_to_state_lists(data)
    env = Env(16, 20)
    for tsl, rsl, l in zip(t_state_lists, r_state_lists, labels):
        # state to env
        # generate image
        # save it, name is hash code
        env.reset()
        env.from_state_list(tsl, norm=True)
        t_img = env.render(show=False)
        env.reset()
        env.from_state_list(rsl, norm=True)
        r_img = env.render(show=False)
        # separator is gray for false examples and white for true examples
        sep = np.ones((2, t_img.shape[1], 3)) * 127.5  + (l * 127.5)
        img = np.concatenate((t_img, sep, r_img)) # concatenate on what dim ?
        img_name = str(hash(img.tostring())) + '.jpg'
        cv2.imwrite(op.join(path, img_name), img)

def plot_metrics(losses, accs, i, path):
    """
    Utility for plotting training loss and accuracy in a run.
    Also saves the numpy arrays corresponding to the training metrics in the
    same folder. 
    """
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(losses)
    axs[0].set_title('loss')
    axs[0].set_xlabel('steps (batch size %s)' % B_SIZE)
    fig.suptitle('Training metrics for seed {}'.format(i))

    axs[1].plot(accs)
    axs[1].set_title('accuracy')
    axs[1].set_ylabel('steps (batch size %s)' % B_SIZE)

    filename = op.join(
        path,
        (str(i) + '.png'))
    plt.savefig(filename)
    plt.close()
    # save losses and accuracies as numpy arrays
    np.save(
        op.join(path,
                (str(i) + 'loss.npy')),
        np.array(losses))
    np.save(
        op.join(path,
                (str(i) + 'acc.npy')),
        np.array(accs))

# model = gm.Simplified_GraphEmbedding([16, 16], 16, f_dict)
# model = gm.AlternatingSimple([16, 16], 2, f_dict)
# model = gm.GraphMatchingSimple([16, 16, 16], 10, 1, f_dict)
model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
opt = torch.optim.Adam(model.parameters(), lr=L_RATE)

def several_steps(n, dl, model, opt, cuda=False):
    losses, accs = [], []
    for _ in range(n):
        l, a = one_step(model,
                        dl,
                        opt, 
                        cuda=cuda)
        losses += l
        accs += a
    return losses, accs


def pre_train(n):
    losses, accs = [], []
    for i in range(n):
        print('Epoch %s' % i)
        l, a = one_step(model,
                        pretrain_dl,
                        opt, 
                        criterion,
                        task)
        losses += l
        accs += a
    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.plot(accs)
    plt.show()

def overfit(n):
    overfit_dl = load_dl_parts('overfit10000_32.txt', bsize=B_SIZE)
    losses, accs = [], []
    for i in range(n):
        print('Epoch %s' % i)
        l, a = one_step(model,
                        overfit_dl,
                        opt, 
                        criterion,
                        task)
        losses += l
        accs += a
    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.plot(accs)
    plt.show()
    return losses, accs

def run(n=int(args.n)):
    losses, accs = [], []
    for i in range(n):
        print('Epoch %s' % i)
        l, a = one_step(model,
                        train_dl,
                        opt, 
                        criterion,
                        task)
        losses += l
        accs += a
    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.plot(accs)
    plt.show()

def run_curriculum(retrain=True):
    """
    In this experiment, we train on different datasets, parametrized by the 
    number of distractor objects in the reference image.

    The number of distractors goes from 0 to 5, so there are 6 different
    datasets to train on.

    The argument retrain controls whether to start a different network for 
    each dataset (in which case we control the learning ability of our model as
    a function of the number of distractors), or to keep the same parameters
    and optimizer throughout the different datasets (in which case we monitor
    the effect of a curriculum of training data on the optimization process).

    This last setting should be contrasted with learning on a dataset of 0-5
    distractors with no curriculum for the same amounts of steps.

    Don't forget to test the overfitting capacity of the model beforehand.

    We save the resulting plots in the directory specified at save_path.
    """
    ds_names = ['curriculum0.txt',
                'curriculum1.txt',
                'curriculum2.txt',
                'curriculum3.txt',
                'curriculum4.txt',
                'curriculum5.txt']
    model = None
    path =  op.join('experimental_results',
                    'parts_curriculum',
                    'retrain')
    for name in ds_names:
        print('Training %s' % name)
        if retrain or (model is None):
            model = gm.ConditionByGraphEmbedding([16, 16], 10, 20, 3, f_dict)
            opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        c_dl = load_dl_parts(name, bsize=B_SIZE)
        l, a = one_step(model,
                        c_dl,
                        opt)
        # plot and save training metrics
        plot_metrics(l, a, i, path)
        # checkpoint model
        save_model(
            model,
            op.join(
                'parts_curriculum',
                'retrain',
                (name[:-4] + '.pt')))

def curriculum_diffseeds(s, n, cur_n=0, training=None, cuda=False):
    """
    n : number of epochs;
    s : number of seeds;
    cur_n : number of distractors
    """
    dl_train = load_dl('curriculum%s.txt' % cur_n)
    dl_test = load_dl('curriculum%stest.txt' % cur_n)
    for i in range(s):
        if training is None:
            model, opt = make_model()
            if cuda:
                model.cuda()
        else:
            model, opt = training
        l, a = several_steps(n, dl_train, model, opt, cuda=cuda)
        # path = 'experimental_results/count/cur_run1/curriculum%s' % cur_n
        path = op.join(
            'experimental_results',
            'all_tasks',
            task,
            'run4',
            'curriculum%s' % cur_n)
        plot_metrics(l, a, i, path)
        # checkpoint model
        save_model(
            model,
            op.join(
                path,
                ('model' + str(i) + '.pt')))
        plot_metrics(l, a, i, path)
        # test 
        model.eval()
        l_test, a_test = one_step(model,
                                  dl_test,
                                  opt, 
                                  train=False,
                                  cuda=cuda)
        model.train()
        print('Test accuracy %s' % np.mean(a_test))

def run_one_diffseeds(s, n, cuda=False):
    """
    Runs one experiment with s seeds for n epochs.
    """
    dl_train = load_dl('data/parts_task/idx/mixed2-5_0-6_100000.txt')
    dl_test = load_dl('data/parts_task/idx/test1.txt')
    path = 'experimental_results/run1'
    for i in range(s):
        model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
        if cuda:
            model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        l, a = several_steps(n, dl_train, model, opt, task, cuda=cuda)
        plot_metrics(l, a, i, path)
        # checkpoint model
        save_model(
            model,
            op.join(
                'run1',
                (str(i) + '.pt')))
        # test 
        model.eval()
        l_test, a_test = one_step(model,
                                  dl_test,
                                  opt, 
                                  train=False,
                                  cuda=cuda)
        model.train()
        print('Test accuracy %s' % np.mean(a_test))

# run_curriculum()

def try_all_cur_n(s, n, cuda=False):
    """
    Performs computation with different initializations on all possible numbers
    of distractors.

    The goal of this test 
    """
    cur_list = [1, 2, 3, 4, 5]
    for cur_n in cur_list:
        curriculum_diffseeds(s, n, cur_n=cur_n, cuda=cuda)

def try_full_cur(s, n):
    # try all seeds
    dl_train_list = []
    dl_test_list = []
    path = 'experimental_results/curriculum_full'
    for cur_n in range(6):
        dl_train_list.append(load_dl_parts('curriculum%s.txt' % cur_n))
        dl_test_list.append(load_dl_parts('curriculum%stest.txt' % cur_n))
    for i in range(s):
        model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
        opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        losses = []
        accs = []
        for cur_n in range(6):
            dl_train = dl_train_list[cur_n]
            # dl_test = dl_test_list[cur_n]
            l, a = several_steps(n, dl_train, model, opt, task)
            losses += l
            accs += a
        # save data
        plot_metrics(losses, accs, i, path)
        # checkpoint model
        save_model(
            model,
            op.join(
                'curriculum_full',
                (str(i) + '.pt')))

        dl_test = dl_test_list[-1]
        l_test, a_test = one_step(model,
                                  dl_test,
                                  opt, 
                                  train=False)
        print('Test accuracy %s' % np.mean(a_test))

def mix_all_cur(s, n):
    """
    This is the experiment that examines the relevance of training with a
    curriculum on the number of distractors. Instead of presenting our model
    with first only 0 distractors, then 1, then 2, we mix all distractor
    numbers by merging all those datasets, and train for the same number of
    iterations.
    """
    names = ['curriculum%s.txt' % i for i in range(6)]
    test_name = 'curriculum5test.txt'
    p = gen.PartsGen()
    for name in names:
        p.load(op.join('data', 'parts_task', name), replace=False)
    dl_test = load_dl_parts(test_name)
    # load all the datasets
    dl = DataLoader(p.to_dataset(),
                    batch_size=B_SIZE,
                    shuffle=True,
                    collate_fn=collate_fn)
    for i in range(s):
        model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
        opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        losses, accs = several_steps(n, dl, model, opt)
        # plot
        plot_metrics(losses, accs, i, 'experimental_results/full')
        # checkpoint model
        save_model(
            model,
            op.join(
                'full',
                (str(i) + '.pt')))
        # test 
        l_test, a_test = one_step(model,
                                  dl_test,
                                  data_to_graph_parts,
                                  data_to_clss_parts,
                                  opt, 
                                  criterion,
                                  train=False)
        print('Test accuracy %s' % np.mean(a_test))

def load_model_playground():
    model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
    model.load_state_dict(torch.load('saves/models/cur_run10/curriculum3/5.pt'))
    pg = ModelPlayground(16, 20, model)
    maps = pg.model_heat_map(4, show=True)
    return maps

############################ Running the script ###############################

if __name__ == '__main__':
    try_all_cur_n(s, n, cuda)