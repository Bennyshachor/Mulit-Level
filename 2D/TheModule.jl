module TheModule

using VoronoiDelaunay
using Gadfly
using GeometricalPredicates
using JuliaFEM
using LinearAlgebra
using DelimitedFiles
using MATLAB

export MakeElement
export assemble_stiffness!
export FEMmodel
export MapToObservations
export DesignVariables
export assembleStifnessDerivative
export findClosestNodes
export FindNode
export CreateL

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

function assemble_stiffness!(B::DataType, problem::Problem{Heat}, assembly::Assembly,
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

function FEMmodel(Nodes,Elem,k,source_loc,DirNodes)
        # Solves FEM model with Dirichlet/Neuman BC
        # inputs:
        # Nodes - cordinate list, x is first row, y is second row
        # Elem - element list where for each lement the nodes number is specified
        # k - thermal conductivity in each element - Array
        # f - source location - [x,y]
        # DirNodes - The nodes which have dirichlet conditions on - shows the number of the node...
        # output: solution u to the nest equationto -div(k grad u) = f ;
        ############ -The code doesn't take into account flux i.e. du/dn  ~= 0 as BC ############
        problem = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(stdout, Base.CoreLogging.Warn)) do
                        problem = Problem(Heat, "example Heat", 1)
                        end


        # add triangle elements to problem
        Y = Dict(j => convert(Array{Float64,1},Nodes[:,j]) for j in 1:size(Nodes,2))
        [add_element!(problem, MakeElement(Elem[:,i],Y,k[i])) for i in 1:size(Elem,2)]


        time = 0.0
        B  =Tri3
        assemble_stiffness!(B, problem, problem.assembly, problem.elements , time)

        K = Matrix(problem.assembly.K) # before adjusting BC
        f = Vector(problem.assembly.f) # before adjusting BC

        sourceNode = FindNode(source_loc[1],source_loc[2],Nodes)
        f = zeros(size(Nodes,2))
        f[sourceNode] = 1
        if !=(DirNodes, 0)
                K[DirNodes,:]           = zeros(length(DirNodes),size(K[:,:,1],2))
                K[:,DirNodes]           = zeros(size(K[:,:,1],2),length(DirNodes))
                K[DirNodes,DirNodes]    = I + zeros(length(DirNodes),length(DirNodes))
                f[DirNodes]             = zeros(length(DirNodes))
        end
        u = zeros(size(Nodes,2))
        u = K\f

        return u,K,f,problem
end

function MapToObservations(obs_loc,nodes,u)
        #obs_loc is a matrix where row 1 is x coord and row 2 is Y coord,
        #nodes is the nodes list, u is the solution u.
        # creates a matrix C where (Cu)_i = u(x_i), u is the forward model solution.
        # Creates a vector d where d = u(x_i)
        loc = zeros(Int32,1,size(obs_loc,2))
        [loc[i] = FindNode(obs_loc[1,i],obs_loc[2,i],nodes) for i  in 1:size(obs_loc,2)]
        println(loc)
        C = zeros(Int32,length(loc),length(u))
        d = zeros(1,length(loc))
        for i = 1:length(loc)
                C[i,loc[i]] = 1
                d[i] = u[loc[i]]
        end
        return C,d
end

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

end
