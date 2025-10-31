#=
MNIST CNN training demo (Flux.jl)

Principle
---------
We train a small convolutional neural network (CNN) to classify MNIST digits.
Each 28×28 grayscale image is treated as a 1-channel input and processed by two
Conv→ReLU→MaxPool stages to extract spatial features, then flattened and passed
through a small MLP head to predict 10 class logits. We use logit cross-entropy
on raw logits (no softmax layer in the model) and optimize with Adam plus weight
decay.
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

# CNN classifier (LeNet-style): two Conv+Pool blocks, then a small MLP head.
# Output layer returns raw logits (no final softmax); loss uses logitcrossentropy.
model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
)

# Sanity forward passes: single sample (batch B=1) and a mini-batch of 3
p1 = model(rand(Float32, 28, 28, 1, 1)) # run model on random data shaped like an image
p3 = model(rand(Float32, 28, 28, 1, 3)) # run model on a batch of 3 fake, random "images"

"""
    load_data(data::MNIST; batchsize::Int=64)

Create a `Flux.DataLoader` for CNN training on MNIST.

Steps:
- Reshape images to 4D tensors (28, 28, 1, N) inserting a channel dimension.
- One-hot encode labels into 10 classes (0–9) with shape (10, N).
- Build a shuffled mini-batch iterator of tuples (x, y).

Returns a `DataLoader` that yields `(x, y)` where `x::Array{Float32,4}` with
size (28, 28, 1, B) and `y` is one-hot with size (10, B).
"""
function load_data(data::MNIST; batchsize::Int=64)
    # Insert a trivial channel dimension for grayscale: (H,W,1,N)
    x4dim = reshape(data.features, 28, 28, 1, :)  # insert trivial channel dim
    # Convert integer labels to one-hot class matrix (10×N)
    yhot = Flux.onehotbatch(data.targets, 0:9)  # make a 10×N OneHotMatrix
    # Create shuffled mini-batches
    return Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

# Training data loader (100 samples per batch)
dl = load_data(train_data; batchsize=100)

"""
    loss_and_accuracy(model, data::MNIST)

Compute the logit cross-entropy loss and accuracy (%) on a full MNIST split.

Implementation details:
- Forms one big batch (OK for MNIST size) to simplify evaluation.
- Uses `Flux.logitcrossentropy` since the model outputs logits (no softmax).
- `Flux.onecold` converts predictions/targets to integer labels for accuracy.

Returns a `NamedTuple`: (loss, acc, split).
"""
function loss_and_accuracy(model, data::MNIST)
    # Make one big batch for the entire split (x: 28×28×1×N, y: 10×N)
    (x, y) = only(load_data(data; batchsize=length(data)))
    ŷ = model(x)
    # Logit cross-entropy: numerically stable softmax + NLL in one op
    loss = Flux.logitcrossentropy(ŷ, y)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    # Return a compact NamedTuple of metrics for logging
    return (; loss, acc, split=data.split)
end

@show loss_and_accuracy(model, train_data)

# Hyperparameters and training settings
settings = (;
    eta=3e-4,     # learning rate
    lambda=1e-2,  # for weight decay
    batchsize=100,
    epochs=10,
)

# Optimizer: Adam preceded by WeightDecay (applied to parameters each step)
opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
opt_state = Flux.setup(opt_rule, model)

for epoch in 1:settings.epochs
    for (x, y) in dl
        # Compute gradients of the loss w.r.t. model parameters for this batch
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
        Flux.update!(opt_state, model, grads[1])
    end

    # Logging & saving, but not on every epoch (here: every odd epoch)
    if epoch % 2 == 1
        loss, acc, _ = loss_and_accuracy(model, test_data)
        test_loss, test_acc, _ = loss_and_accuracy(model, test_data)
        @info "logging:" epoch acc test_acc
    end
end

# Optional visualization in the terminal
using ImageCore, ImageInTerminal

# Load the entire test set as one batch for visualization/inference
xtest, ytest = only(load_data(test_data; batchsize=length(test_data)))

idx = 34                              # fixed example index
#idx = rand() * 10000 |> round |> Int  # or a random test sample index

# Show the selected test image as grayscale (taking the single channel) and transpose
# for a more conventional terminal orientation.
xtest[:, :, 1, idx] .|> Gray |> transpose

# Ground-truth vs. predicted digit for the selected example
@show Flux.onecold(ytest, 0:9)[idx]
@show Flux.onecold(model(xtest)[:, idx], 0:9)
