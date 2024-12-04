# Traveling Salesman Problem Solver

## Overview
This repository contains a Python implementation of a Genetic Algorithm (GA) solution for the Traveling Salesman Problem (TSP). The project provides a flexible and configurable approach to solving TSP using evolutionary computation techniques.

## Project Structure
```
.
├── input/             # Input files with distance matrices
├── output/            # Generated output files and charts
├── plotting/          # Jupyter notebooks for data visualization
└── src/               # Source code for the TSP solver
```

### Key Components
- `src/base_tsp_solver.py`: Base solver implementation
- `src/tsp_factory.py`: Factory for creating TSP solver instances
- `src/tsp_solver.py`: Main TSP solver logic
- `src/config_parameters.py`: Configuration management
- `src/selection_methods.py`: Selection strategies
- `src/mutation_methods.py`: Mutation techniques
- `src/crossover_methods.py`: Crossover operations

## Prerequisites
- Python 3.12+
- Poetry (dependency management)

## Installation
1. Clone the repository
2. Install dependencies:
```bash
poetry install
```

## Configuration
Modify `config_parameters.py` to adjust genetic algorithm parameters:
- Population size
- Number of generations
- Mutation rates
- Selection methods
- Crossover techniques

## Visualization
Jupyter notebooks in the `plotting/` directory provide:
- Performance charts
- Mutation rate analysis
- Comparative visualizations

## Input Data
The `input/` directory contains distance matrices:
- `15.in`: 15-city problem
- `29.in`: 29-city problem

## Output
Generated in the `output/` directory:
- Performance graphs
- Mutation rate plots
