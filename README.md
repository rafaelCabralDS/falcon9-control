
# RL Falcon 9 Controller 

## Overview

This project implements a reinforcement learning (RL) simulation environment for optimizing a booster trajectory. The project includes training scripts, environment configurations, and notebooks for analysis. 

## Table of Contents

1. [Project Structure](#project-structure)
   - [src](#src-project-files)
     - [common](#common-helpers)
     - [environment](#environment-rl-simulation-env)
     - [training](#training-helpers-for-the-training)
   - [notebooks](#notebooks-useful-analysis-for-the-project)
   - [results](#results-training-results-folder)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Contributing](#contributing)
6. [License](#license)

## Project Structure

### 1. src (Project Files)

- **main.py**: The main training script for running the reinforcement learning model.
- **fine_tuning.py**: Script for hyperparameter search using Optuna.
- **playground.py**: An interactive script that allows users to experiment with the environment using keyboard inputs.
- **env.yaml**: Configuration file for the environment settings. Must be modified to adjust parameters.
- **train.yaml**: Configuration file for the reinforcement learning model settings.
- **crs_11_env.yaml**: Configuration file for the benchmark environment.


#### 1.1 common (Helpers)
This directory contains common utility functions and classes used throughout the project. These helpers streamline various tasks such as loading configurations, monitoring progress, and logging.

#### 1.2 environment (RL Simulation Env)
This folder defines the reinforcement learning environment where the booster trajectory is simulated. It includes the necessary components to create and manage the environment, as well as the reward structures.

#### 1.3 training (Helpers for the Training)
This folder contains scripts and functions to facilitate training and hyperparameter tuning.


### 2. notebooks (Useful Analysis for the Project)
This folder contains Jupyter notebooks with analysis, visualizations, and exploratory data analysis related to the project. These notebooks can be used to gain insights into model performance and trajectory simulations.

### 3. results (Training Results Folder)
This directory is designated for saving all results related to the training process, including logs, model checkpoints, and performance metrics.

## Installation

To set up the project, ensure you have Python installed along with the necessary libraries. You can use the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Running the Training Script**:
   To start the training process, execute the main training script:

   ```bash
   python src/main.py
   ```

2. **Hyperparameter Tuning**:
   To perform hyperparameter optimization, use the fine-tuning script:

   ```bash
   python src/fine_tuning.py
   ```

3. **Interactive Playground**:
   To interact with the environment, run the playground script:

   ```bash
   python src/playground.py
   ```

## Configuration

Modify the following YAML files to customize your environment and training settings:

- `env.yaml`: Adjust the environment parameters for the RL simulation.
- `train.yaml`: Set the hyperparameters and model configurations for training.
- `crs_11_env.yaml`: Default settings for the SpaceX CRS-11 mission, used as benchmark for this study.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

---

Feel free to modify any sections to better fit your project's specifics or to add any additional information you find necessary!