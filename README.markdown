# DCGAN on MNIST Dataset

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST dataset to generate handwritten digit images.

## Overview

The DCGAN consists of a Generator and a Discriminator, both built using PyTorch. The Generator creates synthetic images from random noise, while the Discriminator evaluates whether images are real (from the dataset) or fake (generated). After 130 epochs and approximately 10 hours of training, the model produced the attached sample images showcasing generated digits.

## Features

- Trains on the MNIST dataset (handwritten digits).
- Generates 28x28 grayscale images.
- Saves model checkpoints every 5 epochs and generated images every 10 epochs.
- Utilizes GPU acceleration if available.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

Install dependencies using:

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

1. Clone the repository:

   ```bash
   git clone [repository-url]
   cd [repository-folder]
   ```
2. Download the MNIST dataset automatically during the first run.
3. Run the training script:

   ```bash
   python DCGAN_MNIST.ipynb
   ```

## Training Details

- **Epochs**: 150
- **Batch Size**: 128
- **Learning Rate**: 0.0002
- **Optimizer**: Adam with betas (0.5, 0.999)
- **Loss Function**: Binary Cross-Entropy with Logits

## Generated Outputs

- Generated images are saved as `image_at_epoch_0130.png`  for epoch 130).

## Results

Check the attached image for a sample of generated digits after 130 epochs. The training process was computationally intensive, taking around 10 hours, highlighting the challenge of stabilizing the Generator-Discriminator balance.

## Contributing

Feel free to fork this repository, submit issues, or create pull requests to improve the model or documentation.

## Acknowledgments

Inspired by the PyTorch DCGAN tutorial and the broader machine learning community. Special thanks to the open-source tools that made this project possible!
