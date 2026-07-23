# Bravais LEED Classifier

This project addresses the machine-learning exercise in `Exercise-ML.pdf`: generate synthetic LEED-like images for the five 2D Bravais lattices, train a neural network to classify them, and inspect the resulting test performance.

## Run

Install dependencies:

```bash
uv sync
```

Train and evaluate the CNN:

```bash
uv run python BravaisCNN.py
```

For a faster smoke test:

```bash
uv run python BravaisCNN.py --train-samples 250 --test-samples 100 --epochs 1 --no-save
```

Useful options:

```bash
uv run python BravaisCNN.py --method reciprocal --epochs 5
uv run python BravaisCNN.py --method fft --epochs 5
uv run python BravaisCNN.py --device cpu
```

The script prints train/test accuracy and a confusion matrix. By default it saves `bravais_cnn.pt`.

## Files

- `LEEDGenerator.py`: synthetic LEED image generator.
- `BravaisCNN.py`: PyTorch dataset, CNN, training loop, evaluation, and model saving.
- `leed.py`: smaller exploratory helper code for lattice and reciprocal-lattice vectors.

## Network Design

The input is one grayscale LEED image of shape `1 x 128 x 128`, normalized to the range `[0, 1]`. The output is a five-component class score vector for:

```text
oblique, rectangular, centered-rectangular, square, hexagonal
```

`BravaisCNN.py` uses a compact convolutional neural network. Convolution layers are a good fit because the relevant information is spatial: spot positions, symmetry, angles, and systematic absences. Pooling and global average pooling make the classifier less sensitive to small shifts and reduce the number of parameters, keeping it fast on a laptop CPU.

Chosen hyperparameters include image size, number of convolution channels, kernel sizes, batch size, learning rate, number of epochs, optimizer, and training-set size. The class count and output activation are fixed by the problem. Image size and a small CNN can be chosen from speed constraints. Learning rate, number of epochs, amount of noise, and the balance between reciprocal-space and FFT-generated data are best checked by trial and error.

## Training Data

The data are synthetic, generated on demand. The generator samples real-space lattice vectors for each Bravais class, constructs reciprocal-lattice spot positions, adds a simple intensity falloff, blur/noise, and random global rotation. It can also generate images from an FFT of a real-space lattice grid.

This deliberately does not use first-principles electronic-structure methods. For the assignment goal, the classifier mainly needs Bravais-lattice symmetry information, so simple kinematic diffraction-style simulation is enough and is much faster. The model includes lattice geometry, reciprocal-space periodicity, systematic absences for centered rectangular lattices, finite spot width, random orientation, and noise.

Neglected physics includes dynamical multiple scattering, detailed atomic scattering factors, surface relaxation/reconstruction, inelastic background, detector distortions, energy dependence, and realistic experimental artifacts. These simplifications should be mentioned when interpreting accuracy: the trained model is learning the synthetic generator distribution, not the full complexity of experimental LEED.

## Expected Failure Cases

Common confusions are physically reasonable:

- Oblique lattices with angles close to 90 degrees can look rectangular.
- Rectangular lattices with similar side lengths can look square.
- Centered rectangular lattices can be confused with rectangular or oblique patterns depending on visible extinctions.
- Noise, blur, and limited crop size can hide weak spots needed for classification.

The current generator avoids the worst boundary cases during sampling, but the confusion matrix printed after training is the place to discuss which errors remain.

## Example Result

On this machine, the command

```bash
uv run python BravaisCNN.py --train-samples 1000 --test-samples 250 --epochs 3 --method reciprocal --device cpu --no-save
```

reached a best test accuracy of `90.8%`. The weakest class was `oblique`, mostly confused with `hexagonal` in that run. This is plausible because both can contain non-orthogonal spot geometry; with noise, finite crop size, and random parameter sampling, the classifier can miss the lower-symmetry distortions that distinguish a general oblique lattice.
