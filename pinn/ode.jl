# =========================================================
#  Physics-Informed Neural Network (PINN) Example in Julia
# =========================================================
#
# This script demonstrates a minimal PINN implementation
# using Lux.jl, Zygote.jl, and SciML's optimization tools.
#
# The goal is to solve the simple first-order ODE:
#
#        du/dx = -u(x),     x ∈ [0, 1]
#        u(0) = 1
#
# whose exact solution is:
#
#        u(x) = exp(-x)
#
# --------------------------------------------------------------------
# Program workflow:
# --------------------------------------------------------------------
#  1. Generate collocation points on the interval [0,1].
#
#  2. Construct a fully-connected neural network ũ(x) and define the
#     physical solution as:
#
#          u(x) = 1 + ũ(x)
#
#     so that the boundary condition u(0) = 1 is automatically satisfied.
#
#  3. Compute the PDE residual:
#
#          R(x) = u(x) + du/dx
#
#     where du/dx is obtained using Zygote's automatic differentiation.
#
#  4. Define the PINN loss as sum(R(x)²) over all collocation points.
#
#  5. Train the neural network in two stages:
#        • AdamW optimizer — fast initial convergence
#        • LBFGS optimizer — high-accuracy refinement
#
#  6. Evaluate the trained model on test points and compare with the
#     exact analytical solution exp(-x).
#
#  7. Plot and visualize both the exact and PINN-predicted solutions.
#
# This program illustrates the essential components of a PINN:
#     - Neural approximation of the solution
#     - Derivative calculation via AD
#     - Physics-based loss construction
#     - Coupled gradient-based optimization
#
# =========================================================

using Lux, Zygote, Solaris, Plots, Random, LinearAlgebra

# ---------------------------------------------------------
# 1. Generate training points on the 1D domain
# ---------------------------------------------------------
nx = 100                                 # number of training points
X = collect(range(0, 1; length=nx))      # uniform grid in [0,1]
X = permutedims(X)                       # convert to a row vector (1 × nx)
Y = zeros(axes(X))                       # dummy array; not used in this PINN

# ---------------------------------------------------------
# 2. Build a neural network using Lux
# ---------------------------------------------------------
# A 3-layer fully-connected network:
#   input layer:  1  → 20
#   hidden layer: 20 → 20
#   output layer: 20 → 1
# No biases are used; tanh activation for hidden layers.
nn = Chain(
    Dense(1 => 20, tanh; use_bias=false),
    Dense(20 => 20, tanh; use_bias=false),
    Dense(20 => 1; use_bias=false),
)

# Initialize NN parameters (ps) and internal state (st)
ps, st = Lux.setup(Xoshiro(0), nn)

# ---------------------------------------------------------
# 3. Define the physics-informed loss function
# ---------------------------------------------------------
# Here the PDE being solved looks like:
#     u + du/dx = 0
# with the analytic solution u(x) = C * exp(-x)
# Your PINN enforces the residual R = u + du/dx = 0
#
# The NN outputs ũ(x), and the final prediction is:
#     u(x) = 1 + ũ(x)
# presumably enforcing u(0)=1 (or similar constraint).
# ---------------------------------------------------------

function loss(p)
    # Build a stateful wrapper so Lux + Zygote works properly
    model = StatefulLuxLayer{true}(nn, p, st)

    # Forward evaluation of ũ(x); offset by +1 for boundary condition
    u = 1 .+ model(X)

    # Compute du/dx using Zygote's jacobian
    # jacobian returns a matrix, take only the diagonal elements
    # since X is 1D and each input is independent
    ux = Zygote.jacobian(model, X)[1] |> diag |> permutedims

    # Physics-informed residual R = u + u_x
    pred = u .+ ux

    # MSE loss over all collocation points
    l = sum(abs2, pred)

    return l
end

# ---------------------------------------------------------
# 4. Train using two-stage optimization
# ---------------------------------------------------------
# First: AdamW → fast but noisy; good for initial convergence.
res = sci_train(loss, ps, AdamW(); cb=default_callback, maxiters=200, ad=AutoZygote())

# Second: L-BFGS → quasi-Newton method; improves final accuracy.
res = sci_train(loss, res.u, LBFGS(); cb=default_callback, maxiters=200, ad=AutoZygote())
# 'res.u' is the optimized parameter vector.

# ---------------------------------------------------------
# 5. Evaluate the trained model on a test grid
# ---------------------------------------------------------
xTest = Vector(range(0.0, 1.0; length=33)) |> permutedims

# Exact solution u(x) = exp(-x)
yTest = exp.(-xTest)

# PINN prediction: u(x) = 1 + ũ(x)
yPred = 1 .+ nn(xTest, res.u, st)[1]

# ---------------------------------------------------------
# 6. Plot exact vs. predicted solution
# ---------------------------------------------------------
plot(xTest', yTest'; label="exact")
scatter!(xTest', yPred'; label="NN")
