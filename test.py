"""
Small test script for quick debugging.
"""

from torch.utils.data import DataLoader
from dataset import PartsDataset, collate_fn
from graph_utils import data_to_graph_parts

from gen import PartsGen
import os.path as op

p = PartsGen()
path= op.join('data', 'parts_task', 'test.txt')
p.load(path)
ds = p.to_dataset()
dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=collate_fn)
a = iter(dl)
data = next(a)
# how to get data from the DataLoader output
# first get the labels
labels = data[2]
# then build input graphs from the data
# first graph is always the target, second graph is always the reference 
graph1, graph2 = data_to_graph_parts(next(a))
# get data for the baseline models
