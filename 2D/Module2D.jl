module Module2D

using JuliaFEM, LinearAlgebra, DelimitedFiles, JLD2, SparseArrays, Distributed

export DesignVariables, MakeElement,FEMmodel, assemble_stiffness,MapToObservations
export assembleStifnessDerivative, findClosestNodes, FindNode, CreateL
export ObjectiveML, GradML, HessML, CreateParameters

function DesignVariables(Elem,Nodes,Lnodes)
        # Create the design variables of each element, based on function and center of elements
        # function: 1.2D Gaussian function
        k = zeros(size(Elem,2))
        center = Lnodes

        for i in 1:size(Elem,2)

         if ((center[1,i] - 0)^2 + (center[2,i] - 0)^2 ) <= (0.4^2)

                 k[i] = 2

          else
                 k[i] = 1
          end
                #=

                if ((center[1,i] + 0.5)^2 + (center[2,i] + 0.5)^2 ) <= (0.3^2)
                       k[i] = 15
                elseif ((center[1,i] - 0.5)^2 + (center[2,i] - 0.5)^2 ) <= (0.3^2)
                       k[i] = 300
                else
                       k[i] = 10
                end
                =#


               #k[i] = 3

        end
        return k,center
end

function MakeElement(Elem,Y,k)
        # node numbers
        a = convert(Array{Int64,1},Elem)
        if length(a) == 3
                el = Element(Tri3,a)
        else
                el = Element(Seg2,a)
                update!(el, "heat flux", 0.0)
        end
        #coordinates of element nodes
        # Y = Dict(j => convert(Array{Float64,1},Nodes[:,Elem[j,i]]) for j in 1:length(a))
        update!(el, "geometry", Y)
        update!(el, "thermal conductivity", k)
        update!(el, "density", 1)
        return el
end

function assemble_stiffness(B::DataType, problem::Problem{Heat}, assembly::Assembly,
                                        elements::Vector{Element}, time::Float64)

    bi = BasisInfo(B)
    ndofs = length(B)
    Ke = zeros(ndofs, ndofs)
    fe = zeros(ndofs)
    iii = 0
    for element in elements
        fill!(Ke, 0.0)
        fill!(fe, 0.0)
        for ip in get_integration_points(element)
            J, detJ, N, dN = element_info!(bi, element, ip, time)
            k = element("thermal conductivity", ip, time)
            Ke += ip.weight * k*dN'*dN * detJ
            if haskey(element, "heat source")
                f = element("heat source", ip, time)
                fe += ip.weight * N'*f * detJ
            end
        end
        gdofs = get_gdofs(problem, element)
        add!(assembly.K, gdofs, gdofs, Ke)
        add!(assembly.f, gdofs, fe)
    end
end

function MapToObservations(obs_loc,nodes,u)
        #obs_loc is a matrix where row 1 is x coord and row 2 is Y coord,
        #nodes is the nodes list, u is the solution u.
        # creates a matrix C where (Cu)_i = u(x_i), u is the forward model solution.
        # Creates a vector d where d = u(x_i)
        loc = zeros(Int32,1,size(obs_loc,2))
        [loc[i] = FindNode(obs_loc[1,i],obs_loc[2,i],nodes) for i  in 1:size(obs_loc,2)]
        C = zeros(Int32,length(loc),length(u))
        d = zeros(1,length(loc))
        for i = 1:length(loc)
                C[i,loc[i]] = 1
                d[i] = u[loc[i]]
        end
        return C,d
end

function assembleStifnessDerivative(Nodes,Elem,k,DirNodes)
        # create the derivative of the stifness matrix
        #println("start assembling derivative")
        problem = Problem(Heat, "example Heat", 1)

        Y = Dict(j => convert(Array{Float64,1},Nodes[:,j]) for j in 1:size(Nodes,2))
        [add_element!(problem, MakeElement(Elem[:,i],Y,k[i])) for i in 1:size(Elem,2)]

        time = 0.0
        B  =Tri3

        #### assembling the desired Matrix - Using assemble_stiffness! code
        dAdk = zeros(size(Nodes,2),size(Nodes,2),size(Elem,2))
        bi = BasisInfo(B)
        ndofs = length(B)
        Ke = zeros(ndofs, ndofs)
        counter = 0
        for element in problem.elements
            counter = counter+1;
            fill!(Ke, 0.0)
            for ip in get_integration_points(element)
                J, detJ, N, dN = element_info!(bi, element, ip, time)
                k = element("thermal conductivity", ip, time)
                Ke += ip.weight * k*dN'*dN * detJ
            end
            dAdk[element.connectivity,element.connectivity,counter] =  Ke
        end

        if !=(DirNodes, 0)
                dAdk[DirNodes,:,:]           = zeros(length(DirNodes),size(dAdk,2),size(Elem,2))
                dAdk[:,DirNodes,:]           = zeros(size(dAdk,2),length(DirNodes),size(Elem,2))
        end

        return dAdk
end

function findClosestNodes(loc,Nodes)
    NodeLocation = Array{Int64,1}(undef, size(loc,2))
    for i = 1:size(loc,2)
        dist = [norm(loc[:,i]-Nodes[:,j]) for j in 1: size(Nodes,2)]
        NodeLocation[i] = findall(x->x==minimum(dist), dist)[1]
    end
    return NodeLocation
end

function FEMmodel(Nodes,Elem,k,source_loc,DirNodes)

        problem = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(stdout, Base.CoreLogging.Warn)) do
                        problem = Problem(Heat, "example Heat", 1)
                        end

        # add triangle elements to problem
        Y = Dict(j => convert(Array{Float64,1},Nodes[:,j]) for j in 1:size(Nodes,2))
        [add_element!(problem, MakeElement(Elem[:,i],Y,k[i])) for i in 1:size(Elem,2)]
        #

        time1 = 0.0
        B  =Tri3
        assemble_stiffness(B, problem, problem.assembly, problem.elements , time1)
        if !=(DirNodes, 0)
            sourceNode = FindNode(source_loc[1],source_loc[2],Nodes)
            K = sparse(problem.assembly.K.I, problem.assembly.K.J, problem.assembly.K.V)
            f = zeros(K.n)
            f[sourceNode] = 1
            LU = lu(K[length(DirNodes)+1:end,length(DirNodes)+1:end])
            u = zeros(size(Nodes,2))
            u[length(DirNodes)+1:length(u)] = LU\f[length(DirNodes)+1:length(u)]
            return u,K,f,LU
        else
            K = sparse(problem.assembly.K.I, problem.assembly.K.J, problem.assembly.K.V)
            return K
        end
end

function FindNode(x,y,node)
        nnodes = size(node,2)
        for i in 1:nnodes
                if x == node[1,i] && y == node[2,i]
                        return i
                end
        end
end

function CreateL(level)

    x = range(-1 + (0.5)^level, stop= 1 - (0.5)^level , step=(1/2)^(level-1))
    y = x

    nnodes = Int32(length(x)*length(y))
    Lnodes = Matrix(undef, 2,nnodes )
    for i in 1:length(x) , j in 1:length(y)
                    Lnodes[1, j + (i-1)*length(y)] = x[i]
                    Lnodes[2, j + (i-1)*length(y)] = y[j]
    end
     # from left bottom to up and then bottom right and so on
    Lnodes = convert(Array{Float64,2},Lnodes)
    tri= mxcall(:delaunay,1,Lnodes[1,:],Lnodes[2,:])
    tri = convert(Array{Int32,2},tri')
    Lnodes = convert(Array{Any,2},Lnodes)
    ~,L,~ = FEMmodel(Lnodes,tri,ones(1,size(tri,2)),Lnodes[:,1],0)
    return L
end

function ObjectiveML(k,Nodes,Elem,source_loc,DirNodes,L,alphaReg,C,d,P)
    if P == zeros(length(k),length(k)) + I(length(k))
        k_fine = k
    else
        k_fine = P*k
    end
    LS = 0.0
    F = zeros(1)
    for i = 1:size(source_loc,2)
        u,~,~,~ = FEMmodel(Nodes,Elem,k_fine,source_loc[:,i],DirNodes)
        ls = norm(C[:,:,i]*u - d[:,i])^2 # denotes least squares
        LS += ls
    end
    reg = alphaReg/2*transpose(k)*L*k
    F = 1/2 * LS + reg
    return F
end

function GradML(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,P)

    k_fine = P*k
    grad  = zeros(length(k))
    gls   = zeros(length(k))
    #print("In grad size of k_fine = ",sizeof(k_fine))
    for ii in 1:size(source_loc,2)
        u,A,f,LU = FEMmodel(Nodes,Elem,k_fine,source_loc[:,ii],DirNodes)
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

function HessML(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,tau,P)

    Hls     = zeros(length(k),length(k))
    gls     = zeros(length(k))
    gls1    = zeros(length(k))
    gls     = GradML(k,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,P)

   for i in 1:length(gls1)
        ei      = zeros(length(gls)); #creating the standard i-th unit vector
        ei[i]   = 1;
        gls2    = zeros(length(gls))
        gls2    = GradML(k+tau*ei,Elem,Nodes,source_loc,DirNodes,C,d,dAdk,L,alphaReg,P)
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
        global u[:,i],~,~,~ = FEMmodel(Nodes,Elem,k,source_loc[:,i],DirNodes)
        global C[:,:,i], d[:,i] = MapToObservations(obs_loc,Nodes,u[:,i])
    end

    # Defining regularization and derivatives parameters
    alphaRegInit    = 1e-4
    tau             = 1e-4
    # Defining the deriative dA/dk
    dAdk = assembleStifnessDerivative(Nodes,Elem,ones(1,size(Elem,2)),DirNodes)

    return Hmax,k_exact,Nodes,Elem,BoundaryNodes,Lnodes,tri,C,d,
                alphaRegInit,tau,dAdk,source_loc,obs_loc,DirNodes
end


end
