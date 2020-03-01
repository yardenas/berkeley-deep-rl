import os
import subprocess
import numpy as np
from collections import defaultdict
import uuid
import shlex
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mean_std(means, stds, x=None, label=None, c=None):
    sns.set(style="darkgrid")
    means = np.array(means)
    stds = np.array(stds)
    x = np.array(x) if x is not None else np.arange(len(means))
    if c is None:
        c = np.random.rand(3,)
    plt.plot(x, means, label=label, c=c)
    plt.fill_between(x, means - stds, means + stds, alpha=0.3, facecolor=c)
    plt.legend()


def run_problem_1():
    print("=========================================================\n"
          "               Running Problem 1                         \n"
          "=========================================================")
    commands = ['python run_hw4_mb.py --exp_name cheetah_n500_arch1x32'
                ' --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000'
                ' --num_agent_train_steps_per_iter 500 --n_layers 1'
                ' --size 32 --scalar_log_freq -1 --video_log_freq -1',
                'python run_hw4_mb.py --exp_name cheetah_n5_arch2x250 --env_name cheetah-cs285-v0'
                ' --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5'
                ' --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1',
                'python run_hw4_mb.py --exp_name cheetah_n500_arch2x250 --env_name cheetah-cs285-v0'
                ' --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 '
                ' --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1']
    # Launch commands and await until both done.
    processes = [subprocess.Popen(shlex.split(command)) for command in commands]
    for process in processes:
        process.wait()


def run_problem_2():
    print("=========================================================\n"
          "               Running Problem 2                         \n"
          "=========================================================")
    commands = ['python run_hw4_mb.py --exp_name obstacles_singleiteration'
                ' --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20'
                ' --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10']
    # Launch commands and await until both done.
    processes = [subprocess.Popen(shlex.split(command)) for command in commands]
    for process in processes:
        process.wait()

def run_problem_3():
    print("=========================================================\n"
          "               Running Problem 3                         \n"
          "=========================================================")
    commands = ['python run_hw4_mb.py --exp_name obstacles --env_name obstacles-cs285-v0'
                ' --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000'
                ' --batch_size 1000 --mpc_horizon 10 --n_iter 12',
                'python run_hw4_mb.py --exp_name reacher --env_name reacher-cs285-v0 --add_sl_noise'
                ' --mpc_horizon 10 --num_agent_train_steps_per_iter 1000'
                ' --batch_size_initial 5000 --batch_size 5000 --n_iter 15',
                'python run_hw4_mb.py --exp_name cheetah --env_name cheetah-cs285-v0'
                ' --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500'
                ' --batch_size_initial 5000 --batch_size 5000 --n_iter 20']
    # Launch commands and await until both done.
    processes = [subprocess.Popen(shlex.split(command)) for command in commands]
    for process in processes:
        process.wait()


def main(task):
    # Verify task
    tasks = {
        'problem-1': run_problem_1,
        'problem-2': run_problem_2,
        'problem-3': run_problem_3
    }
    assert task in tasks.keys(), "Invalid task"
    tasks[task]()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    if (args.task == ''):
        raise RuntimeError('No task provided')
    main(args.task)