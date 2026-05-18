# NEAT-line-follower

## Overview

This repository implements the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm for training neural network controllers in line-following robots. NEAT is a genetic algorithm that evolves both the weights and structure of neural networks, making it particularly effective for robotic control tasks.

## Description

Using NEAT algorithm in line following robots

## 📄 Research Paper

A comprehensive research paper on this topic was developed and is available in the repository:

**[NEAT-Based Neural Controller for Robust Line Following Robots](https://github.com/abstract-inf/NEAT-line-follower/blob/main/NEAT_Based_Neural_Controller_for_RobustLine_Following_Robots.pdf)**

This paper provides detailed analysis and theoretical foundations for applying NEAT to line-following robotic systems.

## Key Features

- NEAT-based neural network evolution for robot control
- Line-following task optimization
- Jupyter notebook demonstrations and tutorials
- Customizable evolutionary parameters
- Research-backed implementation

## Getting Started

### Prerequisites
- Python 3.x
- NEAT-Python library
- Jupyter Notebook
- Required Python dependencies (see notebooks for details)

### Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/abstract-inf/NEAT-line-follower.git
   cd NEAT-line-follower
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the notebooks**
   Open the Jupyter notebooks to see demonstrations and run experiments with the NEAT algorithm.

## How NEAT Works

NEAT evolves artificial neural networks by:
1. Starting with simple networks
2. Gradually adding nodes and connections through mutation
3. Selecting the best-performing networks for reproduction
4. Creating a population of increasingly sophisticated controllers

This approach is particularly suited for robotic tasks like line following, where the optimal network structure is unknown a priori.

## Related Resources

- [NEAT Research Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [NEAT-Python Documentation](https://neat-python.readthedocs.io/)
- [Research Paper - NEAT-Based Neural Controller for Robust Line Following Robots](https://github.com/abstract-inf/NEAT-line-follower/blob/main/NEAT_Based_Neural_Controller_for_RobustLine_Following_Robots.pdf)

## License

[MIT License]

---

**Author:** [@abstract-inf](https://github.com/abstract-inf)
