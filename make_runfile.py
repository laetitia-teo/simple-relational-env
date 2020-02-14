import os
import re

paths = os.listdir('configs/')
configlist = sorted([re.search(r'config([0-9]+)', p)[1] for p in paths])

s = """#!/bin/sh
#SBATCH --mincpus 12
#SBATCH -p routage
#SBATCH -t 60:00:00
#SBATCH -e experimental_results/expe{0}.err
#SBATCH -o experimental_results/expe{0}.out
python run_experiment.py -c config{0}
wait"""

for c in configlist:
    with open('launch%s.sh' % c, 'w') as f:
        f.write(s.format(c))