# set boundary conditions
@parallel_indices (iy) function bc_expand_x!(A::T) where {T<:Data.Array}
    A[  1, iy] = A[    2, iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_expand_y!(A::T) where {T<:Data.Array}
    A[ix,   1] = A[ix,     2]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel_indices (ix) function bc_VxTopBot!(A::T, Vtop::N, Vbot::N) where {T<:Data.Array, N<:Number}
    A[ix, 1]   = 2 * Vbot - A[ix, 2]
    A[ix, end] = 2 * Vtop - A[ix, end-1]
    return
end

@parallel_indices (iy) function periodic_bound_x!(A::T) where {T<:Data.Array}
    A[  1, iy] = A[end-1, iy]
    A[end, iy] = A[    2, iy]
    return
end

@parallel_indices (ix) function bc_setTopBot!(A::T, valTop::N, valBot::N) where {T<:Data.Array, N<:Number}
    A[ix,   1] = valBot
    A[ix, end] = valTop
    return
end

@parallel_indices (iy) function bc_setLeftRight!(A::T, valLeft::N, valRight::N) where {T<:Data.Array, N<:Number}
    A[  1, iy] = valLeft
    A[end, iy] = valRight
    return
end

@parallel_indices (iy) function bc_LeftIsRight(A::T) where {T<:Data.Array}
    A[1, iy] = A[end, iy]
    return
end

# compute boundary velocities based on strain rate and set initial velocity field
function InitBC!(Vx, Vy, Lx, Ly, nx, ny, yc, xn, yn, εbg, bcType=1)
    # 1: Simple shear right, periodic
    # 2: Simple shear left, periodic
    # 3: Simple shear right, open
    # 4: Simple shear left, open
    # 5: Pure shear vertical
    # 6: Pure shear horizontal

    # boundary values
    VxBot, VxTop, VxLeft, VxRight = 0.0, 0.0, 0.0, 0.0
    VyBot, VyTop, VyLeft, VyRight = 0.0, 0.0, 0.0, 0.0
    if bcType == 1 || bcType == 3
        VxTop   =  2.0 * Ly * εbg
    elseif bcType == 2 || bcType == 4
        VxTop   = -2.0 * Ly * εbg
    elseif bcType == 5
        VyTop   = -Ly * εbg / 2.0
        VyBot   =  Ly * εbg / 2.0
        VxRight =  Lx * εbg / 2.0
        VxLeft  = -Lx * εbg / 2.0
    elseif bcType == 6
        VyTop   =  Ly * εbg / 2.0
        VyBot   = -Ly * εbg / 2.0
        VxRight = -Lx * εbg / 2.0
        VxLeft  =  Lx * εbg / 2.0
    else
        @printf("!!!!!!!!!!!!!!!!!!!!!!!\n bcType not recognized \n!!!!!!!!!!!!!!!!!!!!!!!\n")
    end

    # initial field
    if 0 < bcType < 5
        yVx         = yc[1,:]
        Vxlin       = (yVx .+ Ly/2) ./ Ly .* VxTop
        for i = 1 : nx+1
            Vx[i,:] .= Vxlin
        end
    else
        xVx         = xn[:,1]
        yVy         = yn[1,:]
        Vxlin       = (xVx .+ Lx/2) ./ Lx .* (VxRight - VxLeft) .+ VxLeft
        Vylin       = (yVy .+ Ly/2) ./ Ly .* (VyTop   - VyBot)  .+ VyBot
        for i = 1 : ny+2
            Vx[:,i] .= Vxlin
        end
        for i = 1 : nx+2
            Vy[i,:] .= Vylin
        end
    end
    V0          = max(maximum(Vx) - minimum(Vx), maximum(Vy) - minimum(Vy))

    return VxBot, VxTop, VxLeft, VxRight, VyBot, VyTop, VyLeft, VyRight, V0
end

# apply viscosity boundary conditions
function applyEtaBC!(η_nodes, η_dx, bcType, ny)
    if bcType == 1 || bcType == 2
        @parallel (1:ny+1) bc_LeftIsRight(η_nodes)
        @parallel (1:ny)   bc_LeftIsRight(η_dx)
    end
end

# apply stress boundary conditions
function applyTauBC!(τxx, τxy, bcType, nx, ny)
    if bcType == 1 || bcType == 2
        @parallel (1:ny)   periodic_bound_x!(τxx)
    elseif bcType == 5 || bcType == 6
        @parallel (1:nx+1) bc_setTopBot!(τxy, 0.0, 0.0)
        @parallel (1:ny+1) bc_setLeftRight!(τxy, 0.0, 0.0)
    end
end

# apply heat flux boundary conditions
function applyFluxBC!(qx, qy, bcType, nx, ny)
    if bcType == 1 || bcType == 2
        @parallel (1:nx)  bc_setTopBot!(qy, 0.0, 0.0)
    elseif bcType == 3 || bcType == 4 || bcType == 5 || bcType == 6
        @parallel (1:ny)  bc_setLeftRight!(qx, 0.0, 0.0)
        @parallel (1:nx)  bc_setTopBot!(qy, 0.0, 0.0)
    end
end

# apply velocity boundary conditions
function applyVelBC!(Vx, Vy, bcType, VxTop, VxBot, VxLeft, VxRight, VyTop, VyBot, nx, ny)
    if 0 < bcType < 5
        if bcType == 1 || bcType == 2 
            @parallel (1:ny+1) periodic_bound_x!(Vy)
        else
            @parallel (1:ny)   bc_expand_x!(Vx)
            @parallel (1:ny+1) bc_expand_x!(Vy)
        end
        @parallel (1:nx+1) bc_VxTopBot!(Vx, VxTop, VxBot)
        @parallel (1:nx+2) bc_setTopBot!(Vy, 0.0, 0.0)
    else
        @parallel (1:nx+2) bc_setTopBot!(Vy, VyTop, VyBot)
        @parallel (1:ny+2) bc_setLeftRight!(Vx, VxLeft, VxRight)
    end
end

# apply pressure and temperature boundary conditions
function applyPTBC!(P, T, bcType, ny)
    if bcType == 1 || bcType == 2
        @parallel (1:ny) periodic_bound_x!(T)
        @parallel (1:ny) periodic_bound_x!(P) 
    end
end