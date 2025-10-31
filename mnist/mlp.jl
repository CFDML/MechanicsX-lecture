#=
MNIST MLP training demo (with Flux.jl)

Principle
---------
We train a simple multilayer perceptron (MLP) to classify MNIST digits.
Each 28×28 grayscale image is flattened to a 784-dimensional vector,
passed through two Dense layers with nonlinear activations,
and mapped to 10 class logits which are converted to probabilities via softmax.
We optimize cross-entropy loss with Adam.
=#

using Flux, MLDatasets, Statistics

# Ensure the working directory is the folder with this script (so relative paths work)
cd(@__DIR__)
# Download/load MNIST to the current directory and pick the split
train_data = MNIST(; dir=".", split=:train)
test_data = MNIST(; dir=".", split=:test)

# Quick sanity checks on the dataset layout and sizes
@show typeof(train_data.features)
@show size(train_data.features)
@show size(test_data.features)

# MLP classifier: 784 -> 32 (sigmoid) -> 10 (softmax probabilities)
# The last softmax makes the model output class probabilities per column
model = Chain(Dense(28^2 => 32, sigmoid), Dense(32 => 10), softmax)

# Sanity forward passes: single sample vector (784) and a mini-batch of 3 (784×3)
p1 = model(rand(Float32, 28^2)) # run model on random data shaped like an image
p3 = model(rand(Float32, 28^2, 3)) # run model on a batch of 3 fake, random "images"

"""
    load_data(data::MNIST; batchsize::Int=64)

Create a `Flux.DataLoader` for MNIST.

Steps:
- Flatten images to shape (784, N) to match the MLP input.
- Convert integer labels (0–9) to one-hot targets of shape (10, N).
- Build a shuffled mini-batch iterator of tuples (x, y).

Returns a `DataLoader` that yields `(x, y)` where `x::Array{Float32,2}` with
size (784, B) and `y` is one-hot with size (10, B).
"""
function load_data(data::MNIST; batchsize::Int=64)
    # Flatten all spatial dims to a matrix: (pixels, N)
    x2dim = reshape(data.features, 28^2, :)
    # One-hot encode labels into 10 classes (digits 0 through 9)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    # Create shuffled mini-batches
    loader = Flux.DataLoader((x2dim, yhot); batchsize, shuffle=true)

    return loader
end

# Training data loader (100 samples per batch)
dl = load_data(train_data; batchsize=100)

# Peek at the first batch and a quick loss number as a smoke test
x1, y1 = first(dl)
model(x1), y1
@show Flux.crossentropy(model(x1), y1)

"""
    get_accuracy(model, data::MNIST=test_data) -> Float64

Compute classification accuracy (%) on an MNIST split.

Implementation details:
- Loads the entire split in one batch for simplicity (OK for MNIST size).
- Uses `Flux.onecold` to convert probability vectors and one-hot targets to
  integer labels, then compares equality per sample.
"""
function get_accuracy(model, data::MNIST=test_data)
    # One big batch containing the whole split (x: 784×N, y: 10×N)
    (x, y) = only(load_data(data; batchsize=length(data)))
    y_hat = model(x)
    # Compare predicted labels vs. ground truth
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)
    acc = round(100 * mean(iscorrect); digits=2)

    return acc
end

# Initialize optimizer state (Adam with learning rate 3e-4) and train for 30 epochs
opt_state = Flux.setup(Adam(3e-4), model)
for epoch in 1:30
    # Periodic evaluation before each 3-epoch block
    if mod(epoch, 3) == 1
        train_accuracy = get_accuracy(model, train_data)
        test_accuracy = get_accuracy(model, test_data)
        @info "Before epoch = $epoch" train_accuracy test_accuracy
    end

    loss = 0.0
    for (x, y) in dl
        # Compute the loss and the gradients w.r.t. the model parameters
        l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
        # Apply Adam update to all parameters using computed gradients
        Flux.update!(opt_state, model, gs[1])
        # Accumulate average loss for logging/monitoring
        loss += l / length(dl)
    end
end

# Optional visualization in the terminal
using ImageCore, ImageInTerminal

# Load the entire test set as one batch for visualization/inference
xtest, ytest = only(load_data(test_data; batchsize=length(test_data)))

idx = 34                              # fixed example index
#idx = rand() * 10000 |> round |> Int  # or a random test sample index

# Reshape a test vector (784) back to 28×28 for display. Transpose to match
# expected orientation in the terminal renderer.
reshape(xtest[:, idx], 28, 28) .|> Gray |> transpose

# Ground-truth vs. predicted digit for the selected example
@show Flux.onecold(ytest, 0:9)[idx]
@show Flux.onecold(model(xtest[:, idx]), 0:9)
