# GraphGym
## Installation

**Requirements**

- CPU or NVIDIA GPU, Linux, Python3
- PyTorch, various Python packages; Instructions for installing these dependencies are found below


**1. Python environment (Optional):**
We recommend using Conda package manager

```bash
conda create -n graphgym python=3.7
source activate graphgym
```

**2. Pytorch:**
Install [PyTorch](https://pytorch.org/). 
We have verified GraphGym under PyTorch 1.8.0, and GraphGym should work with PyTorch 1.4.0+. For example:

```bash
# CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Pytorch Geometric:**
Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), 
follow their instructions. For example:

```bash
# CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
# TORCH versions: 1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.8.0
CUDA=cu101
TORCH=1.8.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

**4. GraphGym and other dependencies:**


```bash
git clone https://github.com/snap-stanford/GraphGym
cd GraphGym
pip install -r requirements.txt
pip install -e .  # From latest verion
pip install graphgym # (Optional) From pypi stable version
```




## GraphGym In-depth Usage

### 1 Run a single GNN experiment
A full example is specified in [`run/run_single.sh`](run/run_single.sh).

**1.1 Specify a configuration file.**
In GraphGym, an experiment is fully specified by a `.yaml` file.
Unspecified configurations in the `.yaml` file will be populated by the default values in 
[`graphgym/config.py`](graphgym/config.py).
For example, in [`run/configs/example.yaml`](run/configs/example.yaml), 
there are configurations on dataset, training, model, GNN, etc.
Concrete description for each configuration is described in 
[`graphgym/config.py`](graphgym/config.py).

**1.2 Launch an experiment.**
For example, in [`run/run_single.sh`](run/run_single.sh):

```bash
python main.py --cfg configs/example.yaml --repeat 3
```
You can specify the number of different random seeds to repeat via `--repeat`.

**1.3 Understand the results.**
Experimental results will be automatically saved in directory `run/results/${CONFIG_NAME}/`; 
in the example above, it is `run/results/example/`.
Results for different random seeds will be saved in different subdirectories, such as `run/results/example/2`.
The aggregated results over all the random seeds are *automatically* generated into `run/results/example/agg`,
including the mean and standard deviation `_std` for each metric.
Train/val/test results are further saved into subdirectories, such as `run/results/example/agg/val`; here, 
`stats.json` stores the results after each epoch aggregated across random seeds, 
`best.json` stores the results at *the epoch with the highest validation accuracy*.

### 2 Run a batch of GNN experiments
A full example is specified in [`run/run_batch.sh`](run/run_batch.sh).

**2.1 Specify a base file.**
GraphGym supports running a batch of experiments.
To start, a user needs to select a base architecture `--config`.
The batch of experiments will be created by perturbing certain configurations of the base architecture.

**2.2 (Optional) Specify a base file for computational budget.**
Additionally, GraphGym allows a user to select a base architecture to *control the computational budget* for the grid search, `--config_budget`.
The computational budget is currently measured by the number of trainable parameters; the control is achieved by auto-adjust
the hidden dimension size for GNN.
If no `--config_budget` is provided, GraphGym will not control the computational budget.

**2.3 Specify a grid file.**
A grid file describes how to perturb the base file, in order to generate the batch of the experiments.
For example, the base file could specify an experiment of 3-layer GCN for Cora node classification.
Then, the grid file specifies how to perturb the experiment along different dimension, such as number of layers,
model architecture, dataset, level of task, etc.


**2.4 Generate config files for the batch of experiments,** based on the information specified above.
For example, in [`run/run_batch.sh`](run/run_batch.sh):
```bash
python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
  --config_budget configs/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --out_dir configs
```

**2.5 Launch the batch of experiments.**
For example, in [`run/run_batch.sh`](run/run_batch.sh):
```bash
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
```
Each experiment will be repeated for `$REPEAT` times. 
We implemented a queue system to sequentially launch all the jobs, with `$MAX_JOBS` concurrent jobs running at the same time.
In practice, our system works great when handling thousands of jobs.

**2.6 Understand the results.**
Experimental results will be automatically saved in directory `run/results/${CONFIG_NAME}_grid_${GRID_NAME}/`; 
in the example above, it is `run/results/example_grid_example/`.
After running each experiment, GraphGym additionally automatically averages across different models, saved in
`run/results/example_grid_example/agg`. 
There, `val.csv` represents validation accuracy for each model configuration at the *final* epoch; 
`val_best.csv` represents the results at the epoch with the highest average validation error;
`val_best_epoch.csv` represents the results at the epoch with the highest validation error, averaged over different random seeds.
When test set split is provided, `test.csv` represents test accuracy for each model configuration at the *final* epoch; 
`test_best.csv` represents the test set results at the epoch with the highest average validation error;
`test_best_epoch.csv` represents the test set results at the epoch with the highest validation error, averaged over different random seeds.





### 3 Analyze the results
We provides a handy tool to automatically provide an overview of a batch of experiments in
[`analysis/example.ipynb`](analysis/example.ipynb).
```bash
cd analysis
jupyter notebook
example.ipynb   # automatically provide an overview of a batch of experiments
```



### 4 User customization
A highlight of GraphGym is that it allows users to easily register their customized modules.
The supported customized modules are provided in directory 
[`graphgym/contrib/`](graphgym/contrib), including:
- Activation [`graphgym/contrib/act/`](graphgym/contrib/act), 
- Customized configurations [`graphgym/contrib/config/`](graphgym/contrib/config), 
- Feature augmentation [`graphgym/contrib/feature_augment/`](graphgym/contrib/feature_augment), 
- Feature encoder [`graphgym/contrib/feature_encoder/`](graphgym/contrib/feature_encoder),
- GNN head [`graphgym/contrib/head/`](graphgym/contrib/head), 
- GNN layer [`graphgym/contrib/layer/`](graphgym/contrib/layer), 
- Data loader [`graphgym/contrib/loader/`](graphgym/contrib/loader),
- Loss function [`graphgym/contrib/loss/`](graphgym/contrib/loss), 
- GNN network architecture [`graphgym/contrib/network/`](graphgym/contrib/network), 
- Optimizer [`graphgym/contrib/optimizer/`](graphgym/contrib/optimizer),
- GNN global pooling (graph classification only) 
  [`graphgym/contrib/pooling/`](graphgym/contrib/pooling), 
- GNN stage [`graphgym/contrib/stage/`](graphgym/contrib/stage),
- GNN training pipeline [`graphgym/contrib/train/`](graphgym/contrib/train), 
- Data transformations [`graphgym/contrib/transform/`](graphgym/contrib/transform).

Within each directory, (at least) an example is provided, showing how to register user customized modules.
Note that new user customized modules may result in new configurations; in these cases, new configuration fields
can be registered at [`graphgym/contrib/config/`](graphgym/contrib/config).

**Note: Applying to your own datasets.**
A common use case will be applying GraphGym to your favorite datasets.
To do so, you may follow our example in 
[`graphgym/contrib/loader/example.py`](graphgym/contrib/loader/example.py).
GraphGym currently accepts a list of [NetworkX](https://networkx.org/documentation/stable/index.html) graphs 
or [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) datasets.



### Use case: Identity-aware Graph Neural Networks (AAAI 2021)

Reproducing experiments in *[Identity-aware Graph Neural Networks](https://arxiv.org/abs/2101.10320)*, Jiaxuan You, Jonathan Gomes-Selman, Rex Ying, Jure Leskovec, **AAAI 2021**.
You may refer to the [paper](https://arxiv.org/abs/2101.10320) or [project webpage](http://snap.stanford.edu/idgnn/) for more details. 

```bash
# NOTE: We include the raw results for ID-GNN in analysis/idgnn.csv
cd run/scripts/IDGNN/
bash run_idgnn_node.sh   # Reproduce ID-GNN node-level results
bash run_idgnn_edge.sh   # Reproduce ID-GNN edge-level results
bash run_idgnn_graph.sh   # Reproduce ID-GNN graph-level results
```

<div align="center">
  <img align="center" src="https://github.com/snap-stanford/GraphGym/raw/master/docs/IDGNN.png" width="900px" />
  <b><br>Figure 5: Overview of Identity-aware Graph Neural Networks (ID-GNN).</b>
</div>
