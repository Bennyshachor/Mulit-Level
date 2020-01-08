# full code routines
# Run MyModule once
LOAD_PATH
path = push!(LOAD_PATH, "C:/Users/.....") #path of the module and geometry and parameters files

using Module2D
using DelimitedFiles, LinearAlgebra, Optim, JuliaFEM, SparseArrays, JLD2, Distributed

# Creating parameters for the problem - Mesh, Design variables,
Hmax,k_exact,Nodes,Elem,BoundaryNodes,Lnodes,tri,C,d,
        alphaRegInit,tau,dAdk,source_loc,obs_loc,DirNodes = CreateParameters()

MaxLevel = 4
Iterations = [3,3,2,8] #number of iterations per levels
MaxIter = maximum(Iterations)
cost = zeros(MaxLevel*(MaxIter+1))

# the algorithm
for level = 1:MaxLevel
    # declearing global constants
    global alphaReg
    global res
    global cost

    alphaReg = alphaRegInit/(3^(level-1))
    File = "MLParametersHmax" * Hmax * "level" * string(level) * ".jld2"

# Creating the initial guess
    if level == 1
        @load File L idx
        initial_x = 0.5 * ones(4^level)
    elseif level == MaxLevel
        L = FEMmodel(Lnodes,tri,ones(1,size(tri,2)),Lnodes[:,1],0)
        P = zeros(size(Elem,2),4^(level-1)) # In the Max Level the bisection is done using the operator P
        [P[j,idx[j]] = 1 for j=1:size(Elem,2)]
        initial_x = P*Optim.minimizer(res)

    else
        @load File L idx Levelidx
        B = zeros(4^level,4^(level-1))
        [B[j,Levelidx[j]] = 1 for j=1:4^level]
        initial_x = B*Optim.minimizer(res)
    end

    if level == MaxLevel
        P = zeros(length(initial_x),length(initial_x)) + I(length(initial_x))
        dAdkML = dAdk
    else
        P = zeros(size(Elem,2),4^level)
        [P[j,idx[j]] = 1 for j=1:size(Elem,2)]
        dAdkML      = zeros(size(Nodes,2),size(Nodes,2),4^level)
        for i = 1:length(idx)
            dAdkML[:,:,idx[i]] += dAdk[:,:,i]
        end
    end

    res = Optim.optimize(k -> ObjectiveML(k,Nodes,Elem,source_loc,DirNodes,L,alphaReg,C,d,P),
                k -> GradML(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdkML,L,alphaReg,P),
                k -> HessML(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdkML,L,alphaReg,tau,P),
                initial_x, method=NewtonTrustRegion(; initial_delta = 1.0,
                delta_hat = 100.0, eta = 0.1, rho_lower = 0.25,
                rho_upper = 0.75) ;show_trace=true,store_trace = true,
                extended_trace = false,
                x_tol = 1e-6, iterations = Iterations[level], inplace = false)

    CostLoc = (level-1)*MaxIter + 1
    cost[CostLoc : CostLoc + length(Optim.f_trace(res))-1] = Optim.f_trace(res)
    global idx
end

k = Optim.minimizer(res)
