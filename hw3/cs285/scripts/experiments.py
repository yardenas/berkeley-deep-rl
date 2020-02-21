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
                ' -b 1000 --exp_name 1_1 -ntu 100 -ngsptu 1',
                'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100'
                ' -b 1000 --exp_name 1_1 -ntu 1 -ngsptu 100',
                'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100'
                ' -b 1000 --exp_name 1_1 -ntu 10 -ngsptu 10']

    # Launch commands and await until both done.
    processes = [subprocess.Popen(shlex.split(command)) for command in commands]
    for process in processes:
        process.wait()


def vis_problem_4():
    # Collect data and summarize.
    folders = [folder for folder in glob.glob('cs285/data/*')]
    lb_results = defaultdict(lambda: defaultdict(list))
    sb_results = defaultdict(lambda: defaultdict(list))
    for folder in folders:
        if 'lb_' in folder:
            for i in range(100):
                metric_file = os.path.join(folder, 'metrics_{}.json'.format(i))
                with open(metric_file, 'r') as file:
                    logs = json.load(file)
                    lb_results[folder]['Eval_AverageReturn'].append(logs['Eval_AverageReturn'])
                    lb_results[folder]['Eval_StdReturn'].append(logs['Eval_StdReturn'])
        elif 'sb_' in folder:
            for i in range(100):
                metric_file = os.path.join(folder, 'metrics_{}.json'.format(i))
                with open(metric_file, 'r') as file:
                    logs = json.load(file)
                    sb_results[folder]['Eval_AverageReturn'].append(logs['Eval_AverageReturn'])
                    sb_results[folder]['Eval_StdReturn'].append(logs['Eval_StdReturn'])
    plt.figure(1)
    for folder, item in sb_results.items():
        plot_mean_std(item['Eval_AverageReturn'], item['Eval_StdReturn'], label=folder)
    plt.figure(2)
    for folder, item in lb_results.items():
        plot_mean_std(item['Eval_AverageReturn'], item['Eval_StdReturn'], label=folder)
    plt.show()


def run_problem_6():
    print("=========================================================\n"
          "                     Running Problem 6                   \n"
          "=========================================================")
    command = 'python run_hw2_policy_gradient.py --env_name LunarLanderContinuous-v2 ' \
              '--ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg' \
              ' --nn_baseline --exp_name ll_b40000_r0.005'

    process = subprocess.Popen(shlex.split(command))
    process.wait()

def main(task):
    # Verify task
    tasks = {
        'problem-4': run_problem_4,
        'vis-problem-4': vis_problem_4,

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