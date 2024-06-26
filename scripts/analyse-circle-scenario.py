#!/usr/bin/env nix-shell
#! nix-shell -i python3
#! nix-shell -p python3Packages.numpy
#! nix-shell -p python3Packages.scipy
#! nix-shell -p python3Packages.rich
#! nix-shell -p python3Packages.tabulate
#! nix-shell -p python3Packages.matplotlib
#! nix-shell -p python3Packages.toolz
#! nix-shell -p python3Packages.seaborn
#! nix-shell -p python3Packages.result
#! nix-shell -p python3Packages.pretty-errors
#! nix-shell -p python3Packages.seaborn
#! nix-shell -p python3Packages.catppuccin
#! nix-shell -p texliveFull

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import re
import json
import argparse
import itertools
from pathlib import Path
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rich import print, pretty
from typing import  Iterable
import pretty_errors
from catppuccin import PALETTE

# import .scripts.ldj
from ldj import ldj

# use LaTeX for text with matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

sns.set_theme()
pretty.install()

RESULTS_DIRS = [
    Path('./experiments/circle-experiment-lm-3-th-5'),
    Path('./experiments/circle-experiment-lm-3-th-13'),
    Path('./experiments/circle-experiment-lm-1-th-13')
]

for RESULTS_DIR in RESULTS_DIRS:
    assert RESULTS_DIR.is_dir() and RESULTS_DIR.exists()

# RESULTS_DIR = Path('./experiments/circle-experiment-lm-3-th-5')
# assert RESULTS_DIR.is_dir() and RESULTS_DIR.exists()

flavor = PALETTE.latte.colors
# num-robots-10-seed-0.json
RE = re.compile(r"num-robots-(\d+)-seed-(\d+).json")

def flatten(lst: Iterable) -> list:
    return list(itertools.chain.from_iterable(lst))

def process_file(file):
    match = RE.match(file.name)
    assert match is not None
    num_robots = int(match.group(1))
    seed = int(match.group(2))

    with open(file, 'r') as file:
        data = json.load(file)

    distance_travelled_of_each_robot: list[float] = []
    ldj_of_each_robot: list[float] = []

    for _, robot_data in data['robots'].items():
        positions = np.array(robot_data['positions'])
        # print(f"{positions.shape=}")
        # assert positions.shape == (num_robots, 2)
        # n x 2 matrix
        # sum the length between each pair of points
        distance_travelled = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        # print(f"{distance_travelled=}")
        distance_travelled_of_each_robot.append(distance_travelled)
        mission = robot_data['mission']
        # mission = robot_data.get("mission", None)
        # if not mission:
        #     continue
        t_start: float = mission['started_at']
        t_final: float = mission['finished_at'] if mission['finished_at'] else mission['duration'] + t_start
        timestamps: np.ndarray = np.array([measurement['timestamp'] for measurement in robot_data['velocities']])
        velocities3d_bevy: np.ndarray = np.array([measurement['velocity'] for measurement in robot_data['velocities']])
        velocities = velocities3d_bevy[:, [0, 2]]

        metric = ldj(velocities, timestamps)
        ldj_of_each_robot.append(metric)

    makespan: float = data['makespan']
    return num_robots, distance_travelled_of_each_robot, makespan, ldj_of_each_robot

    # for robot_id, robot_data in data['robots'].items():
    #     # print(f"{robot_data=}")
    #     # sys.exit(0)
    #     mission = robot_data['mission']
    #     # mission = robot_data.get("mission", None)
    #     # if not mission:
    #     #     continue
    #     t_start: float = mission['started_at']
    #     t_final: float = mission['finished_at'] if mission['finished_at'] else mission['duration'] + t_start
    #     timestamps: np.ndarray = np.array([measurement['timestamp'] for measurement in robot_data['velocities']])
    #     velocities3d_bevy: np.ndarray = np.array([measurement['velocity'] for measurement in robot_data['velocities']])
    #     velocities = velocities3d_bevy[:, [0, 2]]
    #
    #     metric = ldj(velocities, timestamps)
    #     ldj_of_each_robot[robot_id] = metric


def main():
    print(f"{sys.executable = }")
    print(f"{sys.version = }")

    data = []
    for RESULTS_DIR in RESULTS_DIRS:
        with ProcessPoolExecutor() as executor:
            results = executor.map(process_file, RESULTS_DIR.glob('*.json'))

        # Aggregate results in a single-threaded manner to avoid data
        aggregated_data_distance_travelled: dict[int, list[float]] = collections.defaultdict(list)
        aggregated_data_makespan: dict[int, list[float]] = collections.defaultdict(list)
        aggregated_data_ldj: dict[int, list[float]] = collections.defaultdict(list)

        for num_robots, distance_travelled_for_each_robot, makespan, ldj_for_each_robot in results:
            aggregated_data_distance_travelled[num_robots].extend(distance_travelled_for_each_robot)
            aggregated_data_makespan[num_robots].append(makespan)
            aggregated_data_ldj[num_robots].extend(ldj_for_each_robot)

        data_distance = [aggregated_data_distance_travelled[key] for key in sorted(aggregated_data_distance_travelled.keys())]
        labels_distance = sorted(aggregated_data_distance_travelled.keys())

        data.append(
            {
                'distance': data_distance,
                'labels_distance': labels_distance,
                'makespan': aggregated_data_makespan,
                'ldj': aggregated_data_ldj
            }
        )

    # plot distance travelled
    fig, ax = plt.subplots(figsize=(4, 3))

    boxplot_opts = dict(
        showmeans=False, showfliers=True,
        # medianprops={"color": flavor.blue.hex, "linewidth": 0.5},
        medianprops=dict(color=flavor.blue.hex, linewidth=0.5),
        boxprops=dict(linestyle='-', linewidth=2, color=flavor.lavender.hex),
        whiskerprops=dict(color=flavor.lavender.hex, linewidth=1.5),
        capprops=dict(color=flavor.lavender.hex, linewidth=1.5),
        flierprops=dict(marker='D', color=flavor.lavender.hex, markersize=8)
    )

    for i, d in enumerate(data):
        distance_data = d['distance']
        labels = d['labels_distance']
        ax.boxplot(distance_data, labels=labels, **boxplot_opts)
    # ax.boxplot(data, labels=labels, **boxplot_opts)

    # violin_parts = plt.violinplot(data, showmeans=False, showmedians=True)

    ax.set_xlabel(r'Number of Robots $N_R$', fontsize=12)
    ax.set_ylabel(r'Distance Travelled $[m]$', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set_ylim(0, 250)

    # Draw optimal line
    ax.axhline(y=100, color=flavor.overlay2.hex, linestyle='--', linewidth=1.5)

    fig.tight_layout()
    fig.savefig('circle-experiment-distance-travelled.svg')

    # plot makespan
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, d in enumerate(data):
        aggregated_data_makespan = d['makespan']
        makespan_data = [aggregated_data_makespan[key] for key in sorted(aggregated_data_makespan.keys())]
        labels = sorted(aggregated_data_makespan.keys())
        makespan_data = [np.mean(makespan) for makespan in makespan_data]
        ax.plot(labels, makespan_data, marker='o', color=flavor.lavender.hex, label='Circle Scenario')

    ax.set_ylim(0, 200)
    # ax.set_aspect(1 / 1.414) # A4 paper
    # ax.boxplot(data, labels=labels, flierprops=dict(marker='D', color='r', markersize=8))

    ax.set_xlabel(r'Number of Robots $N_R$', fontsize=12)
    ax.set_ylabel(r'Makespan $[s]$', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend = ax.legend(borderpad=0.5, framealpha=0.8, frameon=True)
    legend.get_frame().set_facecolor(flavor.surface0.hex)  # Change background color
    # plt.tight_layout()
    fig.savefig('circle-experiment-makespan.svg')
    

    # plot ldj
    fig, ax = plt.subplots(figsize=(4, 3))

    for i, d in enumerate(data):
        aggregated_data_ldj = d['ldj']
        data_ldj = [aggregated_data_ldj[key] for key in sorted(aggregated_data_ldj.keys())]
        labels = sorted(aggregated_data_ldj.keys())
        ax.boxplot(data_ldj, labels=labels, **boxplot_opts)

    ax.set_ylim(-25, 0)
    # ax.set_aspect(1 / 1.414) # A4 paper
    ax.set_xlabel(r'Number of Robots $N_R$', fontsize=12)
    ax.set_ylabel(r'Log Dimensionless Jerk $[m/s^3]$', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    fig.savefig('circle-experiment-ldj.svg')

    fig.show()


if __name__ == '__main__':
    sys.exit(main())
