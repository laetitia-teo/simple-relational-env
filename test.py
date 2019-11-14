from torch.utils.data import DataLoader
from dataset import PartsDataset, collate_fn
from graph_models import data_to_graph_parts

from gen import PartsGen
import os.path as op

p = PartsGen()
path= op.join('data', 'parts_task', 'test.txt')
p.load(path)
ds = p.to_dataset()
dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=collate_fn)
a = iter(dl)