export nonDimInput, nonDimMatParam, preAllocVel, preAllocEps, preAllocTau, preAllocEta, preAllocTemp, preAllocErr, preAllocRest, Sol2PS, Rheo2PS, Mat2PS, Coord2PS
export extra!, center2vertex!, center2vertex_x!, vertex2center_x!

# nondimensionalizes input parameters
function nonDimInput(t_end, Lx, Ly, εbg, T0, T1, P0, dt0, gs0, dT_ref0, dτ_crit, rad_x, rad_y, x_off, y_off, dmin, restartName, restartFlag, saveName)
    if restartFlag
        if length(restartName) == 0
            restartName = saveName*"_restart.jld2"
        end
        CD = load(restartName, "CD")
    else
        CD = GEO_units(length=10km, temperature=600C, stress=1000MPa, viscosity=1e20Pas)
    end
        # setup
    t_end   = nondimensionalize(t_end,   CD)
    Lx      = nondimensionalize(Lx,      CD)
    Ly      = nondimensionalize(Ly,      CD)
    εbg     = nondimensionalize(εbg,     CD)
    T0      = nondimensionalize(T0,      CD)
    T1      = nondimensionalize(T1,      CD)
    P0      = nondimensionalize(P0,      CD)
    dt0     = nondimensionalize(dt0,     CD)
    gs0     = nondimensionalize(gs0,     CD)
        # solver
    dT_ref0 = nondimensionalize(dT_ref0, CD)
    dτ_crit = nondimensionalize(dτ_crit, CD)
        # anomaly
    rad_x   = nondimensionalize(rad_x,   CD)
    rad_y   = nondimensionalize(rad_y,   CD)
    x_off   = nondimensionalize(x_off,   CD)
    y_off   = nondimensionalize(y_off,   CD)
    dmin    = nondimensionalize(dmin,    CD)

    return CD, t_end, Lx, Ly, εbg, T0, T1, P0, dt0, gs0, dT_ref0, dτ_crit, rad_x, rad_y, x_off, y_off, dmin, restartName
end

# nondimensionalizes material parameters
function nonDimMatParam(η_reg, η_max, Adif, Edif, Vdif, Adis, Edis, Vdis, ALTP, ELTP, VLTP, LTP_σL, LTP_K, LTP_σb, βL, βb, P_Href, R, G0, ρ0, Cp, λ, α, CD)
    # flow law parameter
    η_reg   = nondimensionalize(η_reg,   CD)
    η_max   = nondimensionalize(η_max,   CD)
    Adif    = nondimensionalize(Adif,    CD)
    Edif    = nondimensionalize(Edif,    CD)
    Vdif    = nondimensionalize(Vdif,    CD)
    Adis    = nondimensionalize(Adis,    CD)
    Edis    = nondimensionalize(Edis,    CD)
    Vdis    = nondimensionalize(Vdis,    CD)
    ALTP    = nondimensionalize(ALTP,    CD)
    ELTP    = nondimensionalize(ELTP,    CD)
    VLTP    = nondimensionalize(VLTP,    CD)
    LTP_σL  = nondimensionalize(LTP_σL,  CD)
    LTP_K   = nondimensionalize(LTP_K,   CD)
    LTP_σb  = nondimensionalize(LTP_σb,  CD)
    βL      = nondimensionalize(βL,      CD)
    βb      = nondimensionalize(βb,      CD)
    P_Href  = nondimensionalize(P_Href,  CD)
        # other material parameters
    R       = nondimensionalize(R,       CD)
    G0      = nondimensionalize(G0,      CD)
    ρ0      = nondimensionalize(ρ0,      CD)
    Cp      = nondimensionalize(Cp,      CD)
    λ       = nondimensionalize(λ,       CD)
    α       = nondimensionalize(α,       CD)

    return η_reg, η_max, Adif, Edif, Vdif, Adis, Edis, Vdis, ALTP, ELTP, VLTP, LTP_σL, LTP_K, LTP_σb, βL, βb, P_Href, R, G0, ρ0, Cp, λ, α
end

# preallocates arrays containing velocity
function preAllocVel(nx, ny)
    Vx_o      = @zeros(nx+1, ny+2)
    Vy_o      = @zeros(nx+2, ny+1)
    ∇V        = @zeros(nx,   ny)
    ResVx     = @zeros(nx+1, ny)
    ResVy     = @zeros(nx,   ny-1)
    dψVx      = @zeros(nx+1, ny)
    dψVy      = @zeros(nx  , ny-1)
    VxUp      = @zeros(nx+1, ny)
    VyUp      = @zeros(nx  , ny-1)
    Vx_cen    = @zeros(nx,   ny)
    Vy_cen    = @zeros(nx,   ny)
    return Vx_o, Vy_o, ∇V, ResVx, ResVy, dψVx, dψVy, VxUp, VyUp, Vx_cen, Vy_cen
end

# preallocates arrays containing strain rate
function preAllocEps(nx, ny)
    εxx       = @zeros(nx,   ny)
    εyy       = @zeros(nx,   ny)
    εxy       = @zeros(nx+1, ny+1)
    εxy_cen   = @zeros(nx,   ny)
    εII       = @zeros(nx,   ny)
    εII_v     = @zeros(nx,   ny)
    εxx_f     = @zeros(nx,   ny)
    εyy_f     = @zeros(nx,   ny)
    εxy_f     = @zeros(nx+1, ny+1)
    εII_ela   = @zeros(nx,   ny)
    εII_dif   = @zeros(nx,   ny)
    εII_dis   = @zeros(nx,   ny)
    εII_dis_g = @zeros(nx,   ny)
    εII_LTP   = @zeros(nx,   ny)
    εII_LTP_g = @zeros(nx,   ny)
    εII_nl    = @zeros(nx,   ny)
    εxx_elaOld= @zeros(nx,   ny)
    εyy_elaOld= @zeros(nx,   ny)
    εxy_elaOld= @zeros(nx+1, ny+1)
    dom       = @zeros(nx,   ny)
    return εxx, εyy, εxy, εxy_cen, εII, εII_v, εxx_f, εyy_f, εxy_f, εII_ela, εII_dif, εII_dis, εII_dis_g, εII_LTP, εII_LTP_g, εII_nl, εxx_elaOld, εyy_elaOld, εxy_elaOld, dom 
end

# preallocates arrays containing stress
function preAllocTau(nx, ny)
    ResP      = @zeros(nx,   ny)
    dψP       = @zeros(nx,   ny)
    PUp       = @zeros(nx,   ny)
    τxx       = @zeros(nx+2, ny)
    τyy       = @zeros(nx,   ny)
    τxy       = @zeros(nx+1, ny+1)
    τxy_cen   = @zeros(nx,   ny)
    τII       = @zeros(nx,   ny)
    τII_reg   = @zeros(nx,   ny)
    τII_vtrue = @zeros(nx,   ny)
    τxx_o     = @zeros(nx+2, ny)
    τyy_o     = @zeros(nx,   ny)
    τxy_o     = @zeros(nx+1, ny+1)
    τII_o     = @zeros(nx,   ny)
    τII_oo    = @zeros(nx,   ny)
    τII_p     = @zeros(nx,   ny)
    P_o       = @zeros(nx+2, ny)
    P_oo      = @zeros(nx+2, ny)
    P_p       = @zeros(nx+2, ny)
    ρ_o       = @zeros(nx,   ny)
    ResP, dψP, PUp, τxx, τyy, τxy, τxy_cen, τII, τII_reg, τII_vtrue, τxx_o, τyy_o, τxy_o, τII_o, τII_oo, τII_p, P_o, P_oo, P_p, ρ_o 
end

# preallocates arrays containing viscosity
function preAllocEta(nx, ny)
    η_ml      = @zeros(nx,   ny)
    η_dx      = @zeros(nx+1, ny)
    η_dy      = @zeros(nx,   ny-1)
    η_nodes   = @zeros(nx+1, ny+1)
    η_e_nodes = @zeros(nx+1, ny+1)
    η_dif_new = @zeros(nx,   ny)
    η_dis_new = @zeros(nx,   ny)    
    η_LTP_new = @zeros(nx,   ny)
    η_o       = @zeros(nx,   ny)
    η_dif_o   = @zeros(nx,   ny)
    η_dis_o   = @zeros(nx,   ny)
    η_LTP_o   = @zeros(nx,   ny)
    return η_ml, η_dx, η_dy, η_nodes, η_e_nodes, η_dif_new, η_dis_new, η_LTP_new, η_o, η_dif_o, η_dis_o, η_LTP_o
end

# preallocates arrays containing temperature
function preAllocTemp(nx, ny)
    H         = @zeros(nx,   ny)
    H_dif     = @zeros(nx,   ny)
    H_dis     = @zeros(nx,   ny)
    H_LTP     = @zeros(nx,   ny)
    H_reg     = @zeros(nx,   ny)
    ResT      = @zeros(nx,   ny)
    TUp       = @zeros(nx,   ny)
    dT_diff   = @zeros(nx,   ny)
    qx        = @zeros(nx+1, ny)
    qy        = @zeros(nx,   ny+1)
    T_o       = @zeros(nx+2, ny)
    T_oo      = @zeros(nx+2, ny)
    T_p       = @zeros(nx+2, ny)
    return H, H_dif, H_dis, H_LTP, H_reg, ResT, TUp, dT_diff, qx, qy, T_o, T_oo, T_p
end

# preallocates arrays containing error
function preAllocErr(nx, ny)
    AbsVx     = @zeros(nx-1, ny)
    AbsVy     = @zeros(nx,   ny-1)
    RelVx     = @zeros(nx-1, ny)
    RelVy     = @zeros(nx,   ny-1)
    ErrVx     = @zeros(nx-1, ny)
    ErrVy     = @zeros(nx,   ny-1)
    AbsT      = @zeros(nx,   ny)
    RelT      = @zeros(nx,   ny)
    ErrT      = @zeros(nx,   ny)
    AbsP      = @zeros(nx,   ny)
    RelP      = @zeros(nx,   ny)
    ErrP      = @zeros(nx,   ny)
    return AbsVx, AbsVy, RelVx, RelVy, ErrVx, ErrVy, AbsT, RelT, ErrT, AbsP, RelP, ErrP
end

# preallocates other arrays
function preAllocRest(nx, ny, dt0, tol0, CD)
    dt_CFL    = @zeros(nx,   ny)
    Time      = 0.0
    iter_dt   = 0
    dt        = copy(dt0)
    dt_dim_s  = ustrip(dimensionalize(dt, s, CD))
    dt_o      = 0.0
    dψT       = 0.0
    tol       = copy(tol0)
    return dt_CFL, Time, iter_dt, dt, dt_dim_s, dt_o, dψT, tol
end

# converts solution arrays to DataArrays
function Sol2PS(P, T, Vx, Vy)
    P         = Data.Array(P)
    P_o       = copy(P)
    T         = Data.Array(T)
    T_o       = copy(T)
    Vx        = Data.Array(Vx)
    Vy        = Data.Array(Vy)
    return P, P_o, T, T_o, Vx, Vy
end

# converts rheology arrays to DataArrays
function Rheo2PS(η, η_e, η_v, η_dif, η_dis, η_LTP, ωI, ωIb, ε_part)
    η         = Data.Array(η)
    η_e       = Data.Array(η_e)
    η_v       = Data.Array(η_v)
    η_dif     = Data.Array(η_dif)
    η_dis     = Data.Array(η_dis)
    η_LTP     = Data.Array(η_LTP)
    ωI        = Data.Array(ωI)
    ωIb       = Data.Array(ωIb)
    ε_part    = Data.Array(ε_part)
    return η, η_e, η_v, η_dif, η_dis, η_LTP, ωI, ωIb, ε_part
end

# converts material property arrays to DataArrays
function Mat2PS(G, G_nodes, Kb, κ, ρ0, ρCp)
    G         = Data.Array(G)
    G_nodes   = Data.Array(G_nodes)
    Kb        = Data.Array(Kb)
    κ         = Data.Array(κ)
    ρ         = Data.Array(ρ0)
    ρ_o       = copy(ρ)
    ρCp       = Data.Array(ρCp)
    return G, G_nodes, Kb, κ, ρ, ρ_o, ρCp
end

# converts coordinate arrays to DataArrays
function Coord2PS(dxn, dxc, dxn_v, dyn, dyc, dyn_v)
    dxn       = Data.Array(dxn)
    dxc       = Data.Array(dxc)
    dxn_v     = Data.Array(dxn_v)
    dyn       = Data.Array(dyn)
    dyc       = Data.Array(dyc)   
    dyn_v     = Data.Array(dyn_v)
    return dxn, dxc, dxn_v, dyn, dyc, dyn_v
end

# interpolates values by half a cell
function inter(X)
    Y = zeros(size(X, 1) - 1, size(X, 2) - 1)
    for ix = 1 : size(X, 1) - 1
        for iy = 1 : size(X, 2) - 1
            Y[ix, iy] = (X[ix,iy] + X[ix+1,iy] + X[ix,iy+1] + X[ix+1,iy+1])/4.0
        end
    end
    return Y
end

# extrapolates values by one cell
function extra!(X, Y, nx, ny)
    Y[2:end-1,2:end-1] = inter(X)
    for iy = 1 : ny+1
        Y[1, iy]   = Y[2, iy]
        Y[end, iy] = Y[end-1, iy]
    end
    for ix = 1 : nx+1
        Y[ix, 1]   = Y[ix, 2]
        Y[ix, end] = Y[ix, end-1]
    end
end

# interpolate from cell centers to vertices
@parallel_indices (ix, iy) function center2vertex!(vert, cen)
    vert[ix, iy] = (cen[ix, iy] + cen[ix, iy+1] + cen[ix+1, iy] + cen[ix+1, iy+1]) * 0.25
    return nothing
end

# interpolate from cell centers to vertices if there is extra centers in x-direction
@parallel_indices (ix, iy) function center2vertex_x!(vert, cen)
    vert[ix, iy] = (cen[ix, iy-1] + cen[ix, iy] + cen[ix+1, iy-1] + cen[ix+1, iy]) * 0.25
    return nothing
end

# interpolate from vertices to cell centers if there is extra centers in x-direction
@parallel_indices (ix, iy) function vertex2center_x!(cen, vert)
    cen[ix, iy] = (vert[ix-1,iy] + vert[ix,iy] + vert[ix-1,iy+1] + vert[ix,iy+1]) * 0.25
    nothing
end