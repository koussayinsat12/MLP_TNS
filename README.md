# MLP Neural Network Training Project

This MATLAB project involves training a Multi-Layer Perceptron (MLP) neural network using synthetic data generated from a nonlinear filter.

## Overview

The project includes MATLAB scripts and an `MLP` class that implements the MLP network. It trains the network on synthetic data, observes the Mean Squared Error (EQM) during training, and visualizes the learned weight matrices for different parameters and settings.

## Project Structure

- **`MLP.m`**: MATLAB class defining the Multi-Layer Perceptron neural network.
- **`calcul_eqmmin.m`**: MATLAB function to calculate the minimum number of training iterations required to reach a plateau in the mean squared error during training.
- **`test_epochs.m`**: MATLAB script containing the main code to generate synthetic data, train the MLP for different epochs, and visualize results.
- **`test_batch_size.m`**:train the MLP for different batch_size, and visualize results.
  
## Data Generation

- The project generates synthetic data using a sinusoidal function with added noise and applies a nonlinear filter to create a dataset for training and testing the MLP.

## Training Process

- **Parameters**:
  - Number of neurons, order, step size, initialization methods, batch size, and epochs can be adjusted in the `main_training_script.m` file.
- **Training Loop**:
  - Utilizes the `MLP` class to train the network with varying parameters (e.g., batch size, epochs).
  - Records the EQM during training for different parameter settings.
  - Visualizes the EQM changes over epochs and displays the trained weight matrices.

## Usage

1. **Setup**:
   - Ensure MATLAB is installed on your system.
2. **Run**:
   - Execute the `test_epochs.m` MATLAB script to train the MLP network with different epochs and visualize the results.
3. **Experiment**:
   - Adjust hyperparameters and settings in the script to observe their effects on training convergence and weight matrices.

## Visualizations

- The project generates plots showing EQM changes with epochs for different settings and displays trained weight matrices using MATLAB's visualization tools.

## Further Improvements

- Experiment with different architectures, activation functions, and optimization algorithms to enhance network performance.
- Incorporate real-world datasets and adapt the MLP for specific tasks.

## Credits

This project was created by kousai Ghaouari as a part of National Institute of Applied Science and Technology.
