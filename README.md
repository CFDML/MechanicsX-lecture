# MechanicsX-lecture

Courseware and learning materials

## Datasets preparation

A `download.sh` script is provided to download the MNIST dataset into the `mnist/` folder.
Run it from the repository root:

```bash
source download.sh
```

If you encounter any issue with dataset downloading during the first run of the demos, you can manually move the files in the assets/ directory to the mnist/ folder.

## Environment setup

This repository is a Julia project. The `Project.toml` declares all dependencies required to run the demos.

Prerequisites:
- Julia (tested with version 1.12)
- Git (to clone the repo)
- Internet access (optional, to download the MNIST dataset)

Get the sources and instantiate the environment:

```bash
# Clone and enter the repo
git clone https://github.com/CFDML/MechanicsX-lecture.git
cd MechanicsX-lecture

# Start Julia with this project and install dependencies
julia --project=.

# In the Julia REPL (Pkg mode)
# press ] to enter pkg mode, then run:
(MechanicsX-lecture) pkg> instantiate
(MechanicsX-lecture) pkg> status
```

## Run the MNIST demos

The `mnist/` folder contains two runnable examples:
- `mnist/mlp.jl`: a simple multilayer perceptron working on flattened 28×28 images.
- `mnist/conv.jl`: a small CNN (LeNet-style) operating on 28×28 images.

Both scripts will:
- `cd` into their own directory, so all paths are relative to `mnist/`.
- Download the MNIST dataset into `mnist/` on first run using `MLDatasets`.
- Train for a small number of epochs and print periodic metrics.
- Render a sample image using `ImageCore`.

Simply run from the repo root:

```bash
# MLP demo
julia --project=. mnist/mlp.jl

# CNN demo
julia --project=. mnist/conv.jl
```
Afterwards, you are encouraged to run the scripts using Visual Studio Code with the Julia extension.
You may modify hyperparameters, model architecture, and training loops to experiment with different settings.

## What to expect on first run

- Terminal image not displayed: ensure your terminal supports true color; you
	can comment out the visualization lines without affecting training.
- Version conflicts: run `pkg> instantiate` again to re-sync with `Project.toml`.
- The first epoch may be a bit slower due to JIT compilation.

## License

See `LICENSE` in this repository.
