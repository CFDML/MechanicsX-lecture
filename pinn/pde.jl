################################################################################
#  Physics-Informed Neural Network (PINN) Solver for a 1D Transport Equation
#
#  This program implements a PINN to solve a simple 1D advection/transport PDE:
#
#        u_t + u_x = 0
#
#  on the space–time domain  x ∈ [-1, 1],  t ∈ [0, 0.09].
#
#  ----------------------  Problem Specification  ------------------------------
#  • PDE:         u_t + u_x = 0           (first-order linear hyperbolic PDE)
#  • Initial:     u(x, 0) = sin(π(x - 0) + π)
#  • Boundary:    u(-1, t) = u(1, t)      (periodic boundary condition)
#
#  ----------------------  PINN Method Overview  -------------------------------
#  Instead of discretizing the PDE with finite differences, this PINN enforces:
#
#      • PDE residual     r(x,t) = u_t + u_x → 0
#      • Initial condition residual
#      • Boundary condition residual
#
#  The neural network u(x,t; θ) is trained by minimizing:
#
#      L(θ) = ||u_t + u_x||²  +  ||u(x,0) − u₀(x)||²  +  ||u(-1,t) − u(1,t)||²
#
#  Automatic differentiation (ForwardDiff + Zygote) is used to compute
#  derivatives with respect to x and t. Solaris provides a sci_train interface
#  for easy optimization loops and parameter/state handling for Lux networks.
#
#  ----------------------  Workflow Summary  -----------------------------------
#  1. Generate a tensor grid of space–time training points (x,t).
#  2. Build a feed-forward Lux neural network approximating u(x,t).
#  3. Construct additional data:
#        – initial condition samples
#        – boundary points at x = -1 and x = 1
#        – derivative masks for selecting u_x and u_t from the Jacobian
#  4. Define the physics-informed loss function combining PDE, IC, and BC terms.
#  5. Train with AdamW multiple times for refinement.
#  6. Evaluate the trained model on a chosen time slice and compare with the
#     exact analytic solution.
#
#  This script demonstrates the full implementation of a PINN in Julia using:
#         Lux – neural network definition
#         Solaris – training utilities
#         DI (DifferentiationInterface) – computing derivatives
#         Plots – visualization
#
################################################################################

using Solaris, Plots
import Flux, Lux
import DifferentiationInterface as DI

###############################################################
# meshgrid — construct Cartesian grid just like MATLAB/NumPy
###############################################################
function meshgrid(x, y)
    # X repeats x along rows, Y repeats y along columns
    X = [i for j in y, i in x]
    Y = [j for j in y, i in x]
    return X, Y
end

###############################################################
# 1. Generate training (collocation) points in space and time
###############################################################

# Time points t = 0, 0.01, ..., 0.09
tsteps = range(0.0; step=0.01, length=10) |> collect

# Spatial points x ∈ [-1, 1]
xsteps = range(-1.0, 1.0; length=256) |> collect

# Cartesian grid of (x,t)
tm, xm = meshgrid(tsteps, xsteps)

# Flatten (x,t) grid into column vectors for batching
xMesh1D = reshape(xm, (1, :))   # 1 × N
tMesh1D = reshape(tm, (1, :))   # 1 × N

# PINN input matrix: X = [x; t], shape 2 × N
X = cat(xMesh1D, tMesh1D; dims=1) .|> Float32

# Store true solution (used only for comparison, not training)
Y = zeros(Float32, 1, size(X, 2))
for j in axes(Y, 2)
    # Exact solution of u_t + u_x = 0 is u(x,t) = sin(π(x−t)+π)
    Y[1, j] = sin(π * (X[1, j] - X[2, j]) + π)
end

# Build dataloader: physics collocation points are the "training data"
train_loader = Flux.DataLoader((X, Y); batchsize=256, shuffle=false)

###############################################################
# 2. Build deep fully-connected neural network u(x,t)
###############################################################
nn = Lux.Chain(
    Lux.Dense(2, 20, tanh),   # input: (x,t)
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 1),         # output: u(x,t)
)

# Initialize parameters p0 and NN state st
p0, st = Solaris.setup(nn)

###############################################################
# 3. Initial condition (t = 0)
###############################################################

# For first 256 points, t = 0 and x spans [-1,1]
X0 = X[:, 1:256]     # 2 × 256
Y0 = Y[:, 1:256]     # 1 × 256

###############################################################
# 4. Boundary condition: periodic u(-1,t) = u(1,t)
###############################################################
xl = -1 .* ones(Float32, 1, length(tsteps))  # x = -1 for all t
xr =  1 .* ones(Float32, 1, length(tsteps))  # x =  1 for all t

# Construct boundary point matrices
XL = vcat(xl, tsteps')   # left boundary (x=-1, t=tsteps)
XR = vcat(xr, tsteps')   # right boundary (x=1 , t=tsteps)

###############################################################
# 5. Selector vectors for automatic differentiation (AD)
# 
# DI.jacobian gives ∂u/∂(x,t) for each element in a batch.
# To extract ∂u/∂x and ∂u/∂t for the whole batch in matrix form:
#     mdx = [1,0,1,0,1,0,...]ᵀ
#     mdt = [0,1,0,1,0,1,...]ᵀ
###############################################################
_l = train_loader.batchsize   # batch size

mdx = zeros(Float32, 2 * _l, 1)
mdt = zeros(Float32, 2 * _l, 1)

for i in 1:_l
    mdx[2(i-1)+1] = 1.0f0     # pick ∂u/∂x for sample i
    mdt[2i]       = 1.0f0     # pick ∂u/∂t for sample i
end

###############################################################
# 6. Physics-Informed Loss Function
#    Contains:
#      (1) PDE residual: u_t + u_x = 0
#      (2) Initial condition at t=0
#      (3) Periodic boundary condition
###############################################################
function loss(p, dl)
    x, y = dl                  # x: input batch; y: exact (not used)
    model = SR.stateful(nn, p, st)

    ### (1) PDE Residual u_t + u_x = 0 -----------------------------
    jac = DI.jacobian(model, AutoForwardDiff(), x)   # compute ∂u/∂x and ∂u/∂t

    ux = jac * mdx |> permutedims   # extract ∂u/∂x
    ut = jac * mdt |> permutedims   # extract ∂u/∂t

    l1 = @. ut + ux                 # PDE residual R = u_t + u_x

    ### (2) Initial condition u(x,0) = sin(π(x−0)+π) ----------------
    u0 = model(X0)
    l2 = u0 - Y0                    # mismatch from exact initial condition

    ### (3) Boundary condition u(-1,t) = u(1,t) ----------------------
    uL = model(XL)
    uR = model(XR)
    l3 = uL - uR                    # periodic boundary mismatch

    ### Total PINN Loss ---------------------------------------------
    loss = sum(abs2, l1) + sum(abs2, l2) + sum(abs2, l3)
    return loss
end

###############################################################
# 7. Train neural network using SciML's sci_train
###############################################################

# Stage 1: larger learning rate for rapid descent
res = sci_train(
    loss,
    p0,
    train_loader,
    AdamW(0.05);
    cb=default_callback,
    maxiters=100,
    ad=AutoZygote(),
)

# Stage 2: smaller learning rate for refinement
res = sci_train(
    loss,
    res.u,
    train_loader,
    AdamW(0.01);
    cb=default_callback,
    maxiters=200,
    ad=AutoZygote(),
)

###############################################################
# 8. Test and visualize solution at specific time index
###############################################################
let idx = 10    # choose the 10th time slice for plotting
    # Build test matrix [x;t_fixed]
    Xtest = hcat(xm[:, idx], tm[:, idx]) |> permutedims

    # NN prediction
    Ytest = nn(Xtest, SR.cpu(res.u), st)[1] |> permutedims

    # Exact reference solution at selected t
    _t = Xtest[2, :] |> first
    Yref = @. sin(π * (Xtest[1, :] - _t) + π)

    # Plot exact vs NN prediction
    plot(xm[:, idx], Yref; label="exact")
    plot!(xm[:, idx], Ytest; label="NN")
end
