# Goal-Based Reinforcement Learning


## Reproducing the experiments
In this directory you can find the code to reproduce the runs of SAC.
The code base is taken from https://github.com/microsoft/oac-explore.

### Requirements
A python version > 3.6 is required.
Required libraries, which are listed in requirements.txt file, can be installed by running
```bash
pip install -r environment/requirements.txt
```
A Mujoco license is needed to run the Mujoco experiments.

### Launch scripts
Before launching the experiments set the PYTHONPATH environment variable:
```bash
export PYTHONPATH=$(pwd)
```

In the following scripts, we provide a default seed of 0.

#### GridWorld (Goal 1)
```bash
python main.py --seed 0 --domain "GridGoal1" --num_epochs 100 --num_expl_steps_per_train_loop 12000 --num_eval_steps_per_epoch 12000 --num_trains_per_train_loop 12000 --max_path_length 1200
```

#### GridWorld (Goal 2)
```bash
python main.py --seed 0 --domain "GridGoal2" --num_epochs 100 --num_expl_steps_per_train_loop 12000 --num_eval_steps_per_epoch 12000 --num_trains_per_train_loop 12000 --max_path_length 1200
```

#### GridWorld (Goal 3)
```bash
python main.py --seed 0 --domain "GridGoal3" --num_epochs 100 --num_expl_steps_per_train_loop 12000 --num_eval_steps_per_epoch 12000 --num_trains_per_train_loop 12000 --max_path_length 1200
```

#### AntEscape
```bash
python main.py --seed 0 --domain "AntEscape" --num_epochs 500 --num_expl_steps_per_train_loop 5000 --num_eval_steps_per_epoch 5000 --num_trains_per_train_loop 5000 --max_path_length 500
```

#### AntJump
```bash
python main.py --seed 0 --domain "AntJump" --num_epochs 1000 --num_expl_steps_per_train_loop 5000 --num_eval_steps_per_epoch 5000 --num_trains_per_train_loop 5000 --max_path_length 500
```

#### AntNavigate
```bash
python main.py --seed 0 --domain "AntNavigate" --num_epochs 1000 --num_expl_steps_per_train_loop 5000 --num_eval_steps_per_epoch 5000 --num_trains_per_train_loop 5000 --max_path_length 500
```

#### HumanoidUp
```bash
python main.py --seed 0 --domain "HumanoidUp" --num_epochs 2000 --num_expl_steps_per_train_loop 6000 --num_eval_steps_per_epoch 6000 --num_trains_per_train_loop 6000 --max_path_length 2000 &
```

### Results visualization
Statistics are logged under the data folder.