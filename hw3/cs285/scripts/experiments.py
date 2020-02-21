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


def run_problem_4():
    print("=========================================================\n"
          "               Running Problem 4                         \n"
          "=========================================================")
    commands = ['python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100'
                ' -b 1000 --exp_name 1_1 -ntu 1 -ngsptu 1',
                'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100'
                ' -b 1000 --exp_name 100_1 -ntu 100 -ngsptu 1',
                'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100'
                ' -b 1000 --exp_name 1_100 -ntu 1 -ngsptu 100',
                'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100'
                ' -b 1000 --exp_name 10_10 -ntu 10 -ngsptu 10']

    # Launch commands and await until both done.
    processes = [subprocess.Popen(shlex.split(command)) for command in commands]
    for process in processes:
        process.wait()


def run_problem_5():
    print("=========================================================\n"
          "               Running Problem 5                         \n"
          "=========================================================")
    commands = ['python run_hw3_actor_critic.py --env_name InvertedPendulum-v2'
                ' --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01'
                ' --exp_name inverted_pendulum -ntu 10 -ngsptu 10 --video_log_freq 10',
                'python run_hw3_actor_critic.py --env_name HalfCheetah-v2'
                ' --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2'
                ' -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name half_cheetah -ntu 10'
                ' -ngsptu 10 --video_log_freq 10']
    # Launch commands and await until both done.
    processes = [subprocess.Popen(shlex.split(command)) for command in commands]
    for process in processes:
        process.wait()


def main(task):
    # Verify task
    tasks = {
        'problem-4': run_problem_4,
        'problem-5': run_problem_5
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