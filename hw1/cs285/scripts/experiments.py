import os
import subprocess
import numpy as np
from collections import defaultdict
import uuid
import shlex
import glob
import json
import matplotlib.pyplot as plt
# import seaborn as sns


def plot_mean_std(means, stds, x=None, label=None, c='y'):
    # sns.set(style="darkgrid")
    means = np.array(means)
    stds = np.array(stds)
    x = np.array(x) if x is not None else np.arange(len(means))
    plt.plot(x, means, label=label, c=c)
    plt.fill_between(x, means - stds, means + stds, alpha=0.3, facecolor=c)
    plt.legend()


def run_bc_2():
    print("=========================================================\n"
          "               Running Behavioral Cloning 2              \n"
          "=========================================================")
    session_id = uuid.uuid1()
    commands = ['python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl'
                ' --env_name Ant-v2 --exp_name {}_bc_2_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl'
                ' --eval_batch_size 50000 --num_agent_train_steps_per_iter 10000'.format(session_id),
                'python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Hopper.pkl'
                ' --env_name Hopper-v2 --exp_name {}_bc_2_hopper --n_iter 1'
                ' --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 50000'
                ' --num_agent_train_steps_per_iter 10000'.format(session_id)]
    # Launch commands and await until both done.
    processes = [subprocess.Popen(shlex.split(command)) for command in commands]
    for process in processes:
        process.wait()
    # Collect data and summarize.
    folders = [folder for folder in glob.glob('cs285/data/*') if str(session_id) in folder]
    results = dict()
    for folder in folders:
        metric_file = os.path.join(folder, 'metrics_0.json')
        with open(metric_file, 'r') as file:
            logs = json.load(file)
            results[folder] = {
                'Eval_AverageReturn': logs['Eval_AverageReturn'],
                'Eval_StdReturn': logs['Eval_StdReturn'],
                'Train_AverageReturn': logs['Train_AverageReturn'],
                'Train_StdReturn': logs['Train_StdReturn']
            }
    print("{:<8} {:<20} {:<20} {:<20} {:<20}"
          .format('Agent', 'Eval_AverageReturn', 'Eval_StdReturn', 'Train_AverageReturn', 'Train_StdReturn'))
    for key, value in results.items():
        agent = key.split('_')[4]
        print("{:<8} {:<20} {:<20} {:<20} {:<20}"
              .format(agent, value['Eval_AverageReturn'], value['Eval_StdReturn'],
                      value['Train_AverageReturn'], value['Train_StdReturn']))


def run_bc_3():
    print("=========================================================\n"
          "               Running Behavioral Cloning 3              \n"
          "=========================================================")
    iterations = range(1000, 10001, 1000)
    seeds = range(5)
    template = 'python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl ' \
               '--env_name Ant-v2 --exp_name 3_{seed}_{iteration_number}' \
               '_ --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl' \
               ' --eval_batch_size 50000 --num_agent_train_steps_per_iter {iteration_number} --seed {seed}'
    for iteration_number in iterations:
        # Launch commands and await until both done.
        processes = [subprocess.Popen(shlex.split(
            template.format(seed=seed, iteration_number=iteration_number))) for seed in seeds]
        for process in processes:
            process.wait()


def visualize_bc_3():
    # Collect relevant and organize experiment data.
    folders = [folder for folder in glob.glob('cs285/data/*') if 'bc_3' in folder]
    results = defaultdict(lambda: defaultdict(list))
    for folder in folders:
        metric_file = os.path.join(folder, 'metrics_0.json')
        num_iterations = folder.split('_')[3]
        with open(metric_file, 'r') as file:
            logs = json.load(file)
            results[num_iterations]['Eval_AverageReturn'].append(logs['Eval_AverageReturn'])
    iterations = range(1000, 10001, 1000)
    means = [None] * len(results)
    stds = [None] * len(results)
    assert len(iterations) == len(results), "Wrong number of results."
    for i, iteration_number in enumerate(iterations):
        iteration_number_means = results[str(iteration_number)]['Eval_AverageReturn']
        means[i] = (np.mean(iteration_number_means))
        stds[i] = (np.std(iteration_number_means))
    plot_mean_std(means, stds, x=iterations, label='bc_3')
    plt.show()


def run_dagger():
    print("=========================================================\n"
          "                     Running DAgger                      \n"
          "=========================================================")


def main(task):
    # Verify task
    tasks = {
        'bc-2': run_bc_2,
        'bc-3': run_bc_3,
        'visualize-bc-3': visualize_bc_3,
        'dagger': run_dagger
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