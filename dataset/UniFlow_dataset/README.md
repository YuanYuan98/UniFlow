# Dataset

This repository contains grid-based and graph-based datasets used in our research.

## Download Instructions

To download the datasets, please follow these steps:

1. Use one of the following links to download the dataset as a zip file:

- Google Drive: [Data Download Link](https://drive.google.com/drive/folders/18W7G2CFv_HQNbFtPdxmG91eE9N4MdI35?usp=drive_link).
- Baidu Netdisk: [Data Download Link](https://pan.baidu.com/s/1CBLSAp8ZyJBsEIt86t4ikA?pwd=wgwm).

2. Create a folder named ``dataset`` within your UniFlow project directory.

3. After downloading the dataset files, move them to the following path:  ``UniFlow/dataset/``.


## Data Structure Overview

### Grid-based data

Each dataset contains two files: `<name>.npy` and `<name>_ts.npy`, representing the spatio-temporal data and the corresponding timestamps, respectively.


- `X` from `<name>.npy`: This file contains the spatio-temporal data with a shape of $\(T \times H \times W \times 1\)$, where
  -  $T$ is the total number of time steps,
  -  $H$ and $W$ represent the grid partition dimensions.
- `timestamps` from `<name>_ts.npy`: This file includes timestamp information, detailing the day of the week and time of day, with a shape of $T \times 2$.

### Graph-based data


Each dataset includes three files: `<name>.npy`, `<name>_ts.npy`, and `matrix_<name>.json`. The graph-based data differs from the grid-based data by including an additional file containing spatial topology information.

- `X` from `<name>.npy`: This file contains spatio-temporal data with a shape of  $T \times N \times 1$,  where:
  -  $T$ is the total number of time steps,
  -  $N$ is the number of nodes.
- `timestamps` from  `<name>_ts.npy`: This file includes timestamp information, detailing the day of the week and time of day, with a shape of $T \times 2$.

- `matrix_<name>.json`: This file stores spatial topology data in a dictionary format with the following keys:

  - `nodes`: List of node indices.
  - `edges`: Edge information.
  - `adj`: Adjacency list.

## Dataset Name Desciption

Each dataset is named using the format `<TypeCity>_<time_steps>`.

For example

- `TaxiBJ13_48.npy`: Here, `TaxiBJ13`  indicates the type (Taxi) and city (BJ) information,  while `48` signifies that the dataset contains 48 time steps per day. This implies that the temporal data is recorded every 30 minutes.

For some datasets, the duration of the data is also included in the name. For instance:

- `GraphBJ_28_96.npy`: In this example, `28`  indicates that the data spans a duration of 28 days.
