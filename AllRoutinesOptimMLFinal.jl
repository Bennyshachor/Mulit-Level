# full code routines
# Run MyModule once
LOAD_PATH
path = push!(LOAD_PATH, "C:/Users/benis/OneDrive/delft/ThesisCodeToGitHub")

using TheModule
using DelimitedFiles, LinearAlgebra, Optim, JuliaFEM, SparseArrays, JLD2

start = time()

function ObjectiveML(k,Nodes,Elem,source_loc,DirNodes,L,alphaReg,C,d,P,problem)
    print("Inside Objective")
    if P == zeros(length(k),length(k)) + I(length(k))
        k_fine = k
    else
        k_fine = P*k
    end
    LS = 0.0
    F = zeros(1)
    for i = 1:size(source_loc,2)
        u,~,~,~ = FEMmodelOptim2(Nodes,Elem,k_fine,source_loc[:,i],DirNodes,problem)
        ls = norm(C[:,:,i]*u - d[:,i])^2 # denotes least squares
        LS += ls
    end
    reg = alphaReg/2*transpose(k)*L*k
    F = 1/2 * LS + reg
    return F
end

function GradOptimML!(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,P,problem)

    k_fine = P*k
    grad  = zeros(length(k))
    gls   = zeros(length(k))
    #print("In grad size of k_fine = ",sizeof(k_fine))
    for ii in 1:size(source_loc,2)
        u,A,f,LU = FEMmodelOptim2(Nodes,Elem,k_fine,source_loc[:,ii],DirNodes,problem)
        F     = C[:,:,ii]*u
        r     = F - d[:,ii]
        z     = zeros(length(u))
        r1     = -C[:,:,ii]'*r
        z[length(DirNodes)+1:end]  = LU'\r1[length(DirNodes)+1:end]
        gls1   = zeros(length(k))

        for i in 1:length(k)
            gls1[i]  = (dAdk[:,:,i]*u)'*z
        end
        gls += gls1
    end

    greg = alphaReg * L * k
    grad = gls + greg
    return grad
end

function HessOptimML!(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,tau,P,problem)

    Hls     = zeros(length(k),length(k))
    gls     = zeros(length(k))
    gls1    = zeros(length(k))
    gls     = GradOptimML!(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,P,problem)

    for i in 1:length(gls1)
        ei      = zeros(length(gls)); #creating the standard i-th unit vector
        ei[i]   = 1;
        gls2    = zeros(length(gls))
        gls2    = GradOptimML!(k+tau*ei,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,P,problem)
        gls1    = gls - alphaReg*L*(k)
        gls2    = gls2 - alphaReg*L*(k+tau*ei)
        Hei     = (gls2 - gls1)/tau; # Hessian matrix - vector multiplication approximation
        for j in 1:length(gls1)
            ej       = zeros(length(gls1)); #creating the standard i-th unit vector
            ej[j]    = 1
            Hls[i,j]  = dot(Hei,ej)
        end
    end

    Hreg = alphaReg * L
    Hess = Hls + Hreg
    return Hess
end

#function FEMmodelOptim2(Nodes,Elem,k,source_loc,DirNodes)
function FEMmodelOptim2(Nodes,Elem,k,source_loc,DirNodes,problem)

       #=
       print("Starting FEMOptim")
       [print(" \n Element " * string(j)*" = ", problem.elements[j].fields[ "thermal conductivity"].data) for j = 1:size(Elem,2)]
       print("\n k size = ",  sizeof(k))
       [update!(problem.elements[j], "thermal conductivity", k[j]) for j = 1:size(Elem,2)]
       print("\n After Update")
       [print(" \n Element " * string(j)*" = ", problem.elements[j].fields[ "thermal conductivity"].data) for j = 1:size(Elem,2)]
       update!(el, "geometry", Y)
       [problem.elements[j].fields["thermal conductivity"].data = k[j]  for j = 1:size(Elem,2)]
       =#
       #=
       Y = Dict(j => convert(Array{Float64,1},Nodes[:,j]) for j in 1:size(Nodes,2))
       for j = 1:size(Elem,2)
           update!(problem.elements[j], "geometry", Y)
           update!(problem.elements[j], "thermal conductivity", k[j])
       end
       =#
        #
        problem = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(stdout, Base.CoreLogging.Warn)) do
                        problem = Problem(Heat, "example Heat", 1)
                        end

        # add triangle elements to problem
        Y = Dict(j => convert(Array{Float64,1},Nodes[:,j]) for j in 1:size(Nodes,2))
        [add_element!(problem, MakeElement(Elem[:,i],Y,k[i])) for i in 1:size(Elem,2)]
        #

        time1 = 0.0
        B  =Tri3
        assemble_stiffness!(B, problem, problem.assembly, problem.elements , time1)
        sourceNode = FindNode(source_loc[1],source_loc[2],Nodes)
        K = sparse(problem.assembly.K.I, problem.assembly.K.J, problem.assembly.K.V)
        f = zeros(K.n)
        f[sourceNode] = 1
        LU = lu(K[length(DirNodes)+1:end,length(DirNodes)+1:end])
        u = zeros(size(Nodes,2))
        u[length(DirNodes)+1:length(u)] = LU\f[length(DirNodes)+1:length(u)]

        return u,K,f,LU
end

function CreateParameters()
    Hmax = 0.3
    Hmax = string(Hmax)
    File = "ParametersHmax" * Hmax * ".jld2"
    #Lnodes is the nodes of the design variables - The centers of the state space elements
    @load File Nodes Elem BoundaryNodes Lnodes tri
    k,center = DesignVariables(Elem,Nodes,Lnodes)

    # Locaions and observations
    loc = [0.5 0.5 0.0 -0.5 -0.5 ; 0.5 -0.5 0.0 0.5 -0.5]
    sourcelocation = findClosestNodes(loc,Nodes)
    source_loc = Nodes[:,sourcelocation] #+ ones(2,length(sourcelocation))/1000
    obs_loc    = Nodes[:,sourcelocation]
    k_exact = copy(k)

    ## initialize paremetrs
    u = zeros(size(Nodes,2),size(source_loc,2))#tri = convert(Array{Int32,2},tri)
    A = zeros(size(Nodes,2),size(Nodes,2),size(source_loc,2))
    f = zeros(size(Nodes,2),size(source_loc,2))
    C = zeros(size(obs_loc,2),size(Nodes,2),size(source_loc,2))
    d = zeros(size(obs_loc,2),size(source_loc,2))

    DirNodes = BoundaryNodes # Defining the boundary Dirichlet nodes
    #Solving the forward problem using FEM method and creating the observations
    for i in 1:size(source_loc,2)
        #global u[:,i],A[:,:,i],f[:,i] = FEMmodel(Nodes,Elem,k,source_loc[:,i],DirNodes)
        global u[:,i],~,~,problem = FEMmodel(Nodes,Elem,k,source_loc[:,i],DirNodes)
        global C[:,:,i], d[:,i] = MapToObservations(obs_loc,Nodes,u[:,i])
    end

    # Defining regularization and derivatives parameters
    alphaRegInit    = 1e-4
    tau             = 1e-4
    dAdk = assembleStifnessDerivative(Nodes,Elem,ones(1,size(Elem,2)),DirNodes)

    return Hmax,k_exact,Nodes,Elem,BoundaryNodes,Lnodes,tri,u,A,f,C,d,
                alphaRegInit,tau,dAdk,source_loc,obs_loc,DirNodes,problem
end

Hmax,k_exact,Nodes,Elem,BoundaryNodes,Lnodes,tri,u,A,f,C,d,
        alphaRegInit,tau,dAdk,source_loc,obs_loc,DirNodes,problem = CreateParameters()
# Generating the mesh from a predefined file
#=
Hmax = 0.3
Hmax = string(Hmax)
File = "ParametersHmax" * Hmax * ".jld2"
#Lnodes is the nodes of the design variables - The centers of the state space elements
@load File Nodes Elem BoundaryNodes Lnodes tri
k,center = DesignVariables(Elem,Nodes,Lnodes)

# Locaions and observations
loc = [0.5 0.5 0.0 -0.5 -0.5 ; 0.5 -0.5 0.0 0.5 -0.5]
sourcelocation = findClosestNodes(loc,Nodes)
source_loc = Nodes[:,sourcelocation] #+ ones(2,length(sourcelocation))/1000
obs_loc    = Nodes[:,sourcelocation]
k_exact = copy(k)

## initialize paremetrs
u = zeros(size(Nodes,2),size(source_loc,2))#tri = convert(Array{Int32,2},tri)
A = zeros(size(Nodes,2),size(Nodes,2),size(source_loc,2))
f = zeros(size(Nodes,2),size(source_loc,2))
C = zeros(size(obs_loc,2),size(Nodes,2),size(source_loc,2))
d = zeros(size(obs_loc,2),size(source_loc,2))

DirNodes = BoundaryNodes # Defining the boundary Dirichlet nodes
#Solving the forward problem using FEM method and creating the observations
for i in 1:size(source_loc,2)
    global u[:,i],A[:,:,i],f[:,i] = FEMmodel(Nodes,Elem,k,source_loc[:,i],DirNodes)
    global C[:,:,i], d[:,i] = MapToObservations(obs_loc,Nodes,u[:,i])
end

# Defining regularization and derivatives parameters
alphaRegInit    = 1e-4
tau             = 1e-4
dAdk = assembleStifnessDerivative(Nodes,Elem,ones(1,size(Elem,2)),DirNodes)
=#

MaxLevel = 4
MaxIter  = 100
Iterations = [3,3,2,8] #number of iterations per level
cost = zeros(MaxLevel*MaxIter)

# the algorithm
for level = 1:MaxLevel#-1
    # declearing global constants
    global alphaReg
    global res
    global cost
    global problem

    alphaReg = alphaRegInit/(3^(level-1))
    File = "MLParametersHmax" * Hmax * "level" * string(level) * ".jld2"

    if level == 1
        @load File L idx
        initial_x = 0.5 * ones(4^level)
    elseif level == MaxLevel
        ~,L,~,~ = FEMmodel(Lnodes,tri,ones(1,size(tri,2)),Lnodes[:,1],0)
        P = zeros(size(Elem,2),4^(level-1))
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
        dAdkML = dAdk[:,:]*P
        #for i = 1:length(idx)
        #    dAdkML[:,:,idx[i]] += dAdk[:,:,i]
        #end
    end

    res = Optim.optimize(k -> ObjectiveML(k,Nodes,Elem,source_loc,DirNodes,L,alphaReg,C,d,P,problem),
                k -> GradOptimML!(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdkML,L,alphaReg,P,problem),
                k -> HessOptimML!(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdkML,L,alphaReg,tau,P,problem),
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

writedlm("C:/Users/benis/OneDrive/delft/julia/MATLAB/delim_file_Lambda.txt",k)
writedlm("C:/Users/benis/OneDrive/delft/julia/MATLAB/delim_file_kexact.txt",k_exact)
writedlm("C:/Users/benis/OneDrive/delft/julia/MATLAB/delim_file_SourceLoc.txt",source_loc)
writedlm("C:/Users/benis/OneDrive/delft/julia/MATLAB/delim_file_CostOptim.txt",cost)
