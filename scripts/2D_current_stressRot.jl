const USE_GPU = true

@static if USE_GPU
    using CUDA
end

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(0)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, GeoParams, JLD2
using JustPIC, JustPIC._2D
const backend_JP = @static if USE_GPU
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using DEDLoc.CUDA_2D

@views function ElaDifDisLTP_2D()
###############
#### Input ####
###############

    # flags
    elaFlag     = 1.0       # elasticity flag (Float)
    diffFlag    = 1.0       # diffusion activation flag (Float)
    heatFlag    = 1.0       # shear heating flag (Float)
    dampFlag    = 1.0       # dampening flag (Float)
    compFlag    = 1.0       # compressibility flag (Float)
    plotFlag    = true      # plot flag (bool)
    saveFlag    = true      # save Flag (bool)
    saveName    = "AV1"
    bcType      = 1         # boundary condition type
    TempType    = 1         # initial temperature type
    nRestart    = 5         # save restart database every n steps
    restartFlag = false     # load restart database (bool)
    restartName = ""        # name of specific restart file
    saveAllFlag = true      # save some full fields to make pictures
    saveAllOut  = 100       # output frequency of saveAll
    outDirName  = "Out"     # name of directory for pictures

    # setup
    maxStrain   = 1         # final strain
    t_end       = 20e3yr    # model time
    Lx          = 60km      # length of profile
    Ly          = Lx/6.0    # height of profile
    εbg         = 2e-13/s   # strain rate
    T0          = 650C      # reference temperature
    T1          = T0        # temperature outside
    P0          = 20e3MPa   # background pressure
    gs0         = 1e-3mm    # initial grain size
    gridType    = "RefineY" # type of grid ["Regular", "RefineY"]

    # solver
    dt0         = 2.5e1yr   # reference timestep
    nx          = 384       # number of cells x
    ny          = 64        # number of cells y
    dT_ref0     = 50K       # change of temperature for timestep scaling
    dτ_crit     = 100MPa    # critical change of stress for timestep scaling
    VdmpX0      = 12.0      # dampening parameter for velocity update
    VdmpY0      = 3.0       # dampening parameter for velocity update
    Vfac        = 5.0       # velocity pseudo step is divided by this
    Tfac        = 2.0       # temperature pseudo step is divided by this (increse when increasing resolution and no)
    Tdmp        = 16.0      # dampening parameter for temperature update
    Pdmp        = 0.0       # NOT IN USE ANYMORE. dampening parameter for pressure update 
    tol0        = 1e-6      # tolerance
    nt          = 500e3     # maximum number of iterations
    nt_min      = 5e3       # minimum number of iterations
    nout        = 1e3       # output frequency
    η_rel       = 0.01      # viscosity relaxation
    CFL         = 0.5       # CFL-criterion

    # anomaly
    anomType    = 1         # [0: load from file, 1: center, 2: boundary, 3: boundary and inside]
    anomFile    = ""
    rad_x       = Lx/40    # major semi-axis for ellipsoidal anomaly
    rad_y       = Lx/120    # minor semi-axis for ellipsoidal anomaly
    x_off       = 0.0km     # horizontal offset
    y_off       = 0.0km     # vertical offset
    θ           = 0         # rotation of ellipse
    ω0          = 100.0     # factor in prefactor
    σbFlag      = true      # also apply weakening to sigma b


    ## Material parameters
    AVFlag      = 1                    # activation volume flag [0: none, 1: background pressure, 2: full pressure]
    η_reg       = 1e12Pas              # minimum viscosity
    σL_hard     = false                # add pressure dependence for σL
    σb_hard     = false                # add pressure dependence for σb

    # diffusion creep
    mdif        = 3.0                  # grain size exponent
    Adif        = 1.5e9*MPa^-1*μm^mdif*s^-1 # prefactor
    Edif        = 375e3J/mol           # activation energy
    Vdif        = 6e-6m^3/mol          # activation volume

    # dislocation creep 
    ndis        = 3.5                  # powerlaw exponent
    Adis        = 1.1e5*MPa^-ndis/s    # prefactor
    Edis        = 530e3J/mol           # activation energy
    Vdis        = 15e-6m^3/mol         # activation volume
    η_max       = 1e40Pas              # maximum viscosity 

    # low-temperature plasticity (Hansen 2019)
    ALTP        = 5e20/s               # prefactor
    ELTP        = Edis                 # activation energy
    VLTP        = Vdis                 # activation volume
    LTP_σL      = 3.1e3MPa             # constant
    LTP_K       = 3.2e3MPa*µm^0.5      # constant
    LTP_σb      = 1.8e3MPa             # back stress
    βL          = 0.09 / (1000*MPa)    # pressure dependence for σL
    βb          = 0.02 / (1000*MPa)    # pressure dependence for σb
    P_Href      = 6500*MPa             # reference pressure from experiments

    # conversion between differential and deviatoric
    FE          = 2.0 / sqrt(3)
    FT          = sqrt(3)

    # other
    R           = 8.314J/mol/K         # gas constant
    G0          = 100e9Pa              # shear modulus
    ν           = 0.25                 # poisson ratio
    ρ0          = 3300kg/m^3           # density
    Cp          = 750J/kg/K            # heat capacity
    λ           = 3J/s/m/K             # thermal conductivity
    α           = 0e-5/K               # thermal expansivity



##########################
#### Input Processing ####
##########################

    # save input
    if saveFlag
        name = saveName*"_input.jld2";
        jldsave(name; Lx, Ly, nx, ny, εbg, T0, P0, dt0, dT_ref0, dτ_crit, VdmpX0, VdmpY0, Vfac, Pdmp, Tdmp, G0, LTP_σb, λ, ω0)
    end

    # model time
    #t_end   = maxStrain / εbg

    # nondimensionalize
    CD, t_end, Lx, Ly, εbg, T0, T1, P0, dt0, gs0, dT_ref0, dτ_crit, rad_x, rad_y, x_off, y_off, restartName = nonDimInput(t_end, Lx, Ly, εbg, T0, T1, P0, dt0, gs0, dT_ref0, dτ_crit, rad_x, rad_y, x_off, y_off, restartName, restartFlag, saveName)
    η_reg, η_max, Adif, Edif, Vdif, Adis, Edis, Vdis, ALTP, ELTP, VLTP, LTP_σL, LTP_K, LTP_σb, βL, βb, P_Href, R, G0, ρ0, Cp, λ, α = nonDimMatParam(η_reg, η_max, Adif, Edif, Vdif, Adis, Edis, Vdis, ALTP, ELTP, VLTP, LTP_σL, LTP_K, LTP_σb, βL, βb, P_Href, R, G0, ρ0, Cp, λ, α, CD)

    # apply hardening to LTP
    if σL_hard && AV > 0 
        LTP_σL  = LTP_σL * (1.0 + P0 * βL)/(1.0 + P_Href * βL)
    end
    if σb_hard && AV > 0 
        LTP_σb  = LTP_σb * (1.0 + P0 * βb)/(1.0 + P_Href * βb)
    end     

    # derived properties
    ind_cen = Int64(ceil(ny/2))
    xn, xc, dxn, dxc, dxn_v, yn, yc, dyn, dyc, dyn_v, dxn_g, dxn_I, dxc_I, dyn_I, dyc_I = makeCoordsNew(Lx, Ly, nx, ny, gridType; d0=dmin)    # grid
    max_Lxy = max(Lx, Ly)                                 # maximum model dimension
    min_dxy = minimum([dxn[:]; dxc[:]; dyn[:]; dyc[:]])   # minimum spacing
    max_n   = max(nx, ny)                                 # maximum number of nodes in any direction
    dampVx  = dampFlag * (1.0-VdmpX0/nx)                  # damping for velocity in x
    dampVy  = dampFlag * (1.0-VdmpY0/ny)                  # damping for velocity in y
    dampT   = dampFlag * (1.0-Tdmp/max_n)                 # damping for temperature
    dampP   = 0.0                                         # damping for pressure
    K0      = (2*G0*(1+ν))/(3*(1-2*ν))                    # bulk modulus
    κ0      = λ/(ρ0*Cp)                                   # reference diffusivity for temperature pseudo step
    ρ0      = ones(nx,ny) .* ρ0 * exp(P0/K0)              # density field
    ρCp     = ρ0.*Cp                                      # heat capacity term
    κ       = λ./(ρCp)                                    # heat diffusivity
    dT_ref  = dT_ref0                                     # change of temperature for timestep scaling
    maxFac  = 5.0 * nondimensionalize(1e-12/s, CD) / εbg  # factor that plays into the maximum timestep
    jldsave(saveName*"_coords.jld2", xn=ustrip(dimensionalize(xn, m, CD)), 
                                     xc=ustrip(dimensionalize(xc, m, CD)), 
                                     yn=ustrip(dimensionalize(yn, m, CD)),
                                     yc=ustrip(dimensionalize(yc, m, CD)))

#########################
#### Setup particles ####
#########################
    xvi       = xn[:, 1], yn[1, :]
    xci       = xc[2:end-1, 1], yc[1, 2:end-1]
    # staggered velocity grids    
    grid_vxi  = (
        (xvi[1], yc[1, :]),
        (xc[:, 1], xvi[2]),
    )
    # move grid to the device
    xvi_device = Data.Array.(xvi)
    xci_device = Data.Array.(xci)
    grid_vxi_device = (
        Data.Array.(grid_vxi[1]),
        Data.Array.(grid_vxi[2]),
    )
    dxc_I_device, dyc_I_device = Data.Array(dxc_I), Data.Array(dyc_I)

    di        = xn[2,1]-xn[1,1], yn[1,2]-yn[1,1]
    nxcell    = 20
    max_xcell = 30
    min_xcell = 10
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi_device...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # material phase & temperature
    # particle_args = pT, = init_cell_arrays(particles, Val(1))
    pPhases, pT = init_cell_arrays(particles, Val(2))

    subgrid_arrays = SubgridDiffusionCellArrays(particles)

    dt₀ = @zeros(nx, ny)
    Tn  = @zeros(nx+1, ny+1)
    ΔT  = @zeros(nx+2, ny)
    ΔTn = @zeros(nx+1, ny+1)
    
    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

####################
#### Initialize ####
####################
    
    ## preallocate
    Vx_o, Vy_o, ∇V, ResVx, ResVy, dψVx, dψVy, VxUp, VyUp, Vx_cen, Vy_cen                                                                                               = preAllocVel(nx, ny)
    εxx, εyy, εxy, εxy_cen, εII, εII_v, εxx_f, εyy_f, εxy_f, εII_ela, εII_dif, εII_dis, εII_dis_g, εII_LTP, εII_LTP_g, εII_nl, εxx_elaOld, εyy_elaOld, εxy_elaOld, dom = preAllocEps(nx, ny)
    ResP, dψP, PUp, τxx, τyy, τxy, τxy_cen, τII, τII_reg, τII_vtrue, τxx_o, τyy_o, τxy_o, τII_o, τII_oo, τII_p, P_o, P_oo, P_p, ρ_o                                    = preAllocTau(nx, ny)
    η_ml, η_dx, η_dy, η_nodes, η_e_nodes, η_dif_new, η_dis_new, η_LTP_new, η_o, η_dif_o, η_dis_o, η_LTP_o                                                              = preAllocEta(nx, ny)
    H, H_dif, H_dis, H_LTP, H_reg, ResT, TUp, dT_diff, qx, qy, T_o, T_oo, T_p                                                                                          = preAllocTemp(nx, ny)
    AbsVx, AbsVy, RelVx, RelVy, ErrVx, ErrVy, AbsT, RelT, ErrT, AbsP, RelP, ErrP                                                                                       = preAllocErr(nx, ny)
    dt_CFL, Time, iter_dt, dt, dt_dim_s, dt_o, dψT, tol                                                                                                                = preAllocRest(nx, ny, dt0, tol0, CD)
    
    # xy component of vorticity tensor
    ω_xy           = @zeros(nx+1, ny+1)
    τxx_noghost    = @zeros(nx, ny)
    τxx_o_noghost  = @zeros(nx, ny)
    τxx_v          = @zeros(nx + 1, ny + 1)
    τyy_v          = @zeros(nx + 1, ny + 1)

    # initial conditions
    ωI        =  ones(nx,   ny)
    ωIb       =  ones(nx,   ny)
    P         =  ones(nx+2, ny)   .* P0
    T         =  ones(nx+2, ny)   .* T0
    G         =  ones(nx,   ny)   .* G0
    Kb        =  ones(nx,   ny)   .* K0
    Vx        = zeros(nx+1, ny+2)
    Vy        = zeros(nx+2, ny+1)
    G_nodes   = zeros(nx+1, ny+1)
    extra!(G, G_nodes, nx, ny)

    # set up temperature field
    InitTemp!(T, TempType, T0, T1, nx, ny, Lx, Ly, xc, yc)     
    
    # boundary conditions and initial guess
    VxBot, VxTop, VxLeft, VxRight, VyBot, VyTop, VyLeft, VyRight, V0 = InitBC!(Vx, Vy, Lx, Ly, nx, ny, yc, xn, yn, εbg, bcType)

    # anomaly
    setAnoms!(ωI, ωIb, anomType, ω0, σbFlag, Lx, Ly, xc, yc, rad_x, rad_y, x_off, y_off, θ, anomFile)

    # viscosity
    DIF       = DifCreep(AVFlag, Edif, Vdif, P0, R, η_rel)
    DIS       = DisCreep(AVFlag, ndis, Edis, Vdis, P0, R, η_rel, η_max)
    LTP       = LTPCreep(AVFlag, LTP_σL, LTP_K, LTP_σb, gs0, ELTP, VLTP, P0, R, η_rel, η_max)
    η, η_v, η_e, η_dif, η_dis, η_LTP, ε_part, Pre_dif, Pre_dis = initalVisc(P0, T, εbg, gs0, R, AVFlag, dt0, G, Adif, mdif, Edif, Vdif, Adis, ndis, Edis, Vdis, ALTP, ELTP, VLTP, LTP_σL, LTP_K, LTP_σb, ωI, ωIb, FE, FT)

    # tracking
    t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo = initEvo(T, τII, Vx, η_v, εII)

    # convert to ParallelStencil arrays
    P, P_o, T, T_o, Vx, Vy                            = Sol2PS(P, T, Vx, Vy)
    η, η_e, η_v, η_dif, η_dis, η_LTP, ωI, ωIb, ε_part = Rheo2PS(η, η_e, η_v, η_dif, η_dis, η_LTP, ωI, ωIb, ε_part)
    G, G_nodes, Kb, κ, ρ, ρ_o, ρCp                    = Mat2PS(G, G_nodes, Kb, κ, ρ0, ρCp)
    dxn, dxc, dxn_v, dyn, dyc, dyn_v, dxn_g           = Coord2PS(dxn, dxc, dxn_v, dyn, dyc, dyn_v, dxn_g)
    
    # initialize T at the particles
    centroid2particle!(pT, xci_device, Data.Array(T[2:end-1, :]), particles)

    # visualisation
    ENV["GKSwstype"]="nul"
    if isdir(outDirName)==false mkdir(outDirName) end

    # load
    if restartFlag
        Time, iter_dt, dt, Vx, Vy, T, T_o, P, P_o, ρ, η, η_dif, η_dis, η_LTP, τxx, τyy, τxy, τII, τII_o, VxUp, VyUp, PUp, TUp, err_evo, its_evo, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD = loadRestart(restartName)
    end

    # numerical stuff
    err = 1
    iter, mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps, dampReset, ndampRe, nRaise= 0, 0, 0, 0, 0, 0, 0, 0, 0
    err_evo, its_evo, ErrVx_evo, ErrVy_evo, ErrT_evo, ErrP_evo = (Array{Float64}(undef, 0) for _ = 1:6)
    VdmpX, VdmpY = VdmpX0, VdmpY0
    dampVx, dampVy = dampFlag * (1.0 - VdmpX/nx), dampFlag * (1.0 - VdmpY/ny)

###################
#### Time Loop ####
###################
    nanFlag = false
    while Time < t_end && !nanFlag
        iter_dt += 1
        @printf("------------------\nStep %d (t=%.2f kyr)\n------------------\n", iter_dt, ustrip(dimensionalize(Time, yr, CD))*1e-3)

        # save info
        dt_o  = copy(dt)
        @parallel save_oldold!(τII_o, T_o, P_o, τII_oo, T_oo, P_oo)
        @parallel save_old!(τxx, τyy, τxy, τII, T, P, η, η_dif, η_dis, η_LTP, Vx, Vy, ρ, τxx_o, τyy_o, τxy_o, τII_o, T_o, P_o, η_o, η_dif_o, η_dis_o, η_LTP_o, Vx_o, Vy_o, ρ_o)
 
##################################
#### Initialize new time step ####
##################################
        # reset
        err      = 1
        ReFlag   = false
        ConvFlag = false
        iter     = 0
        iter_re  = 0

        while ReFlag || (err > tol && iter < nt && !ConvFlag)
            # reset solution variables in case of restart
            if ReFlag
                @parallel resetVals!(τxx, τyy, τxy, τII, T, P, η, η_dif, η_dis, η_LTP, Vx, Vy, ρ, τxx_o, τyy_o, τxy_o, τII_o, T_o, P_o, η_o, η_dif_o, η_dis_o, η_LTP_o, Vx_o, Vy_o, ρ_o)
                iter_re += iter
            end

            # update timestep
            if iter_dt > 1
                dt       = updateTimestep(dt, dt0, T, T_oo, τII, τII_oo, dT_ref, dτ_crit, ReFlag, maxFac)                
                ReFlag   = false
                dt_dim_s = ustrip(dimensionalize(dt, s, CD))
                printNewTimestep(dt_dim_s)
            end

            # pseudo step for diffusion
            dψT = min(min_dxy^2 / κ0 / (4.1 * Tfac), dt / 2.0)

            # elastic viscosity
            @parallel update_η_e!(η_e, η_e_nodes, G, G_nodes, dt)
            @parallel old_elastic_strain_rate!(εxx_elaOld, εyy_elaOld, εxy_elaOld, τxx_o, τyy_o, τxy_o, η_e, η_e_nodes, elaFlag)

            # check if scaling is still ok
            if dt < 1e-8
                CD_new, dt, dt_o, dt0, Time, t_end, VxTop, VxBot, VxLeft, VxRight, VyTop, VyBot, VyLeft, VyRight, V0, Pre_dif, Pre_dis, ALTP, λ, Cp, η_reg, η_max = rescaleTime!(dt, dt_o, dt0, Time, t_end, Vx, Vx_o, Vy, Vy_o, VxTop, VxBot, VxLeft, VxRight, VyTop, VyBot, VyLeft, VyRight, V0, ρ, ρ_o, η, η_o, η_e, η_v, η_ml, η_dif, η_dif_o, η_dif_new, η_dis, η_dis_o, η_dis_new, η_LTP, η_LTP_o, η_LTP_new, εxx, εyy, εxy, εxy_cen, εxx_f, εyy_f, εxy_f, εII, εII_v, εII_nl, εII_dif, εII_dis, εII_dis_g, εII_LTP, εII_LTP_g, εxx_elaOld, εyy_elaOld, εxy_elaOld, κ, qx, qy, dT_diff, H, Pre_dif, Pre_dis, ndis, ALTP, λ, Cp, η_reg, η_max, CD, 10.0)
                rescaleEvoTime!(t_evo, dt_evo, Vxmax_evo, ηvmin_evo, εmax_evo, 10.0)
                CD = CD_new
                @printf("########\n Changed scaling\n########\n")
            end

            # adjust tolerance
            tol = findTol(dt_dim_s, tol0)

            # make predictions based on last timestep
            #τII_p = makePrediction(τII, τII_oo, τII_p, dt, dt_o, true)
            #T_p   = makePrediction(T,   T_oo,   T_p,   dt, dt_o, true)
            #P_p   = makePrediction(P,   P_oo,   P_p,   dt, dt_o, false)

            # set to predictions
            #@parallel setPrediction!(τII, T, P, τII_p, T_p, P_p)

            # reset
            err = 1
            iter, mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps, dampReset, ndampRe, nRaise = 0, 0, 0, 0, 0, 0, 0, 0, 0
            err_evo, its_evo, ErrVx_evo, ErrVy_evo, ErrT_evo, ErrP_evo = (Array{Float64}(undef, 0) for _ = 1:6)
            VdmpX, VdmpY = VdmpX0, VdmpY0
            dampVx, dampVy = dampFlag * (1.0 - VdmpX/nx), dampFlag * (1.0 - VdmpY/ny)

########################
#### Iteration Loop ####
########################
            while !ReFlag && ((err > tol || iter < nt_min) && iter < nt && !nanFlag) && !ConvFlag
                iter += 1
                
                # divergence of velocity
                @parallel divVel!(∇V, Vx, Vy, dxn, dyn)

                # update density
                @parallel update_ρ!(ρ, ρ_o, ∇V, κ, ρCp, dt, λ, Cp, compFlag)

                # full strain rate
                @parallel full_strain_rate!(εxx_f, εyy_f, εxy_f, Vx, Vy, ∇V, dxn, dyn, dxc, dyc)

                # effective strain rate
                @parallel eff_strain_rate!(εxx, εyy, εxy, εxx_f, εyy_f, εxy_f, εxx_elaOld, εyy_elaOld, εxy_elaOld)
                @parallel interp_strain_rate!(εxy, εxy_cen)
                @parallel second_strain_rate!(εxx, εyy, εxy_cen, εII)

                # diffusion creep viscosity
                comp_η(DIF, η_dif, η_dif_new, Pre_dif, T, P, ωI)

                # partition strain rate between different mechanisms
                @parallel part_ε_new!(εII, τII, εII_v, εII_dif, εII_nl, εII_dis_g, εII_LTP_g, τII_reg, τII_vtrue, ε_part, η_e, η_dif, η_dis, η_LTP, η_reg)

                # dislocation creep viscosity
                ε_min = minimum(εII)/1e12
                comp_η(DIS, η_dis, η_dis_new, εII_dis_g, Pre_dis, T, P, ωI, ε_min)

                # low temperature plasticity viscosity
                comp_η(LTP, η_LTP, η_LTP_new, εII_LTP_g, ALTP, T, P, ωIb, ε_min, FE, FT)

                # effective viscosity
                @parallel eff_η!(η, η_v, η_e, η_dif, η_dis, η_LTP, η_reg, elaFlag)
                @parallel interp_η!(η, η_nodes, η_ml)
                @parallel (2:ny)   bc_expand_x!(η_nodes)
                @parallel (1:nx+1) bc_expand_y!(η_nodes)
                @parallel (2:ny-1) bc_expand_x!(η_ml)
                @parallel (1:nx)   bc_expand_y!(η_ml)
                @parallel interp_η_dxdy!(η_ml, η_dx, η_dy)
                @parallel (1:ny)   bc_expand_x!(η_dx)
                applyEtaBC!(η_nodes, η_dx, bcType, ny)

                # stress
                @parallel update_τ!(τxx, τyy, τxy, τxy_cen, τII, η, η_nodes, εxx, εyy, εxy, εxy_cen)
                applyTauBC!(τxx, τxy, bcType, nx, ny)

                # true partitioning
                @parallel true_part_ε!(τII, τII_reg, τII_vtrue, εII, εII_v, εII_dif, εII_dis, εII_LTP, η_e, η_dif, η_dis, η_LTP, η_reg)

                # dissipative energy
                @parallel diss_energy!(H, τxx, τyy, τxy_cen, εxx, εyy, εxy_cen, η_e, heatFlag, elaFlag)

                # diffusion
                @parallel fluxes!(qx, qy, T, dxc, dyc, λ)
                applyFluxBC!(qx, qy, bcType, nx, ny)
                @parallel diffusion!(dT_diff, qx, qy, dxn, dyn, diffFlag)

                # residuals
                @parallel residuals!(ResVx, ResVy, ResP, ResT, τxx, τyy, τxy, ∇V, P, P_o, T, T_o, Kb, dT_diff, H, ρCp, dxc, dyc, dxn_v, dyn_v, dt, α, compFlag)

                # new pseudo step
                @parallel new_step!(dψVx, dψVy, dψP, η_ml, η_dx, η_dy, G, Kb, dxc, dyc, max_Lxy, min_dxy, Vfac, max_n)

                # update
                @parallel comp_update!(VxUp, VyUp, PUp, TUp, ResVx, ResVy, ResP, ResT, dampVx, dampVy, dampP, dampT)
                @parallel apply_update!(Vx, Vy, P, T, VxUp, VyUp, PUp, TUp, dψVx, dψVy, dψP, dψT)

                # enforce boundary conditions
                applyVelBC!(Vx, Vy, bcType, VxTop, VxBot, VxLeft, VxRight, VyTop, VyBot, nx, ny)
                applyPTBC!(P, T, bcType, ny)
                
                # check if timestep is ok
                if iter_dt > 1
                    ReFlag = checkTimestep(dt, dt_CFL, dT_ref, dτ_crit, CFL, T, T_o, τII, τII_o, Vx, Vy, Vx_cen, Vy_cen, dxn, dyn, iter, CD)
                end
                
                # print
                if (iter == 1 || mod(iter, nout) == 0) && !ReFlag
                    # check error
                    @parallel AbsErr!(AbsVx, AbsVy, AbsT, AbsP, ResVx, ResVy, ResT, ResP, V0, T0, P0)
                    @parallel RelErr!(RelVx, RelVy, RelT, RelP, ResVx, ResVy, ResT, ResP, Vx, Vy, T, P, V0, T0, P0)
                    @parallel MinErr!(ErrVx, ErrVy, ErrT, ErrP, AbsVx, AbsVy, AbsT, AbsP, RelVx, RelVy, RelT, RelP)
                    mean_ResVx       = mean(ErrVx)
                    mean_ResVy       = mean(ErrVy)
                    mean_ResT        = mean(ErrT)
                    mean_ResP        = mean(ErrP)
                    mean_ResEps      = mean(abs.(εII_dis .- εII_dis_g) ./ εII)
                    err              = max(mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps)
                    push!(ErrVx_evo, mean_ResVx)
                    push!(ErrVy_evo, mean_ResVy) 
                    push!(ErrT_evo,  mean_ResT)
                    push!(ErrP_evo,  mean_ResP)
                    push!(err_evo, err)
                    push!(its_evo, iter)
                    @printf("Its = %d, err = %1.3e [mean_ResVx=%1.3e, mean_ResVy=%1.3e, mean_dT=%1.3e, mean_dP=%1.3e, mean_Resε=%1.3e] \n", iter, err, mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps)
                    nanFlag          = isnan(err) || isinf(err)
                    tol, nRaise      = raiseTol2(tol, nRaise, iter)
                    ReFlag, ConvFlag = checkError(err_evo, nout, ReFlag, iter, nt_min)
                    #dampVx, dampVy, dampT, VdmpX, VdmpY, Tdmp, dampReset, tol, ndampRe = AdjustDamp(dampVx, dampVy, dampT, VdmpX, VdmpY, Tdmp, VdmpX0, VdmpY0, nx, ny, err_evo, dampReset, tol, ndampRe, dampFlag)
                end
            end
        end

        # tracking
        @parallel diss_energy_full!(εII_dif, εII_dis, εII_LTP, εII_v, H, H_dif, H_dis, H_LTP, H_reg, τII_vtrue, τII_reg, τxx, τyy, τxy_cen, εxx, εyy, εxy_cen, η_e, heatFlag, elaFlag)
        Time = trackProperties!(Time, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, dt, T, τII, Vx, η_v, εII, H, iter, iter_re, err, nx)


        # compute vorticity 
        @parallel (1:nx+1, 1:ny+1) compute_vorticity!(ω_xy, Vx, Vy, dxc_I_device, dyc_I_device)

        @. ΔT = T - T_o
        @parallel subgrid_characteristic_time!(dt₀, ρCp, λ, di)
        centroid2particle!(subgrid_arrays.dt₀, xci_device, dt₀, particles)

        @parallel (1:nx+1, 2:ny) center2vertex_x!(Tn, T)
        @parallel (1:nx+1, 2:ny) center2vertex_x!(ΔTn, ΔT)
        @parallel (1:nx+1)       bc_expand_y!(Tn)
        @parallel (1:nx+1)       bc_expand_y!(ΔTn)
        
        subgrid_diffusion!(pT, Tn, ΔTn, subgrid_arrays, particles, xvi_device, dt)

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), (Vx, Vy), grid_vxi_device, dt)

        # rotate stresses
        @views τxx_o_noghost .= τxx_o[2:end-1, :]
        rotate_stress!(pτ, τxx_o_noghost, τyy_o, τxy_o, ω_xy, particles, xci_device, xvi_device, dt)

        # interpolate stress back to the grid
        stress2grid!(τxx_o_noghost, τyy_o, τxy_o, pτ, xvi_device, xci_device, particles)
        @views τxx_o[2:end-1, :] .= τxx_noghost
        applyTauBC!(τxx_o, 1, bcType, nx, ny)

        # advect particles in memory
        move_particles!(particles, xvi_device, particle_args)
        # check if we need to inject particles
        # need stresses on the vertices for injection purposes
        @parallel (1:nx+1, 2:ny) center2vertex_x!(τxx_v, τxx) #center2vertex!(τxx_v, τxx)
        @parallel (1:nx+1)       bc_expand_y!(τxx_v)
        @parallel (2:nx, 2:ny)   center2vertex!(τyy_v, τyy) #center2vertex!(τyy_v, τyy)
        @parallel (2:ny)         bc_expand_x!(τyy_v)
        @parallel (1:nx+1)       bc_expand_y!(τyy_v)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T, τxx_v, τyy_v, τxy, ω_xy),
            xvi_device
        )
        
        particle2grid!(Tn, pT, xvi_device, particles)

        @parallel (2:nx+1, 1:ny) vertex2center_x!(T, Tn)
        @parallel (1:ny)         bc_expand_x!(T)


        # plotting
        if (plotFlag && mod(iter_dt, 1) == 0)
            @parallel ε_ela!(εII_ela, τII, τII_o, η_e)
            @parallel (1:nx, 1:ny) domMech!(dom, εII_ela, εII_dif, εII_dis, εII_LTP)
            SH_2D_plot(xn, yn, xc, yc, dt, Vx, Vy, T, P, τII, t_evo, CD, iter_dt, its_evo, ErrVx_evo, ErrVy_evo, ErrT_evo, ErrP_evo, dom, outDirName)
        end

        # save full fields
        saveFullField(saveAllFlag, saveAllOut, saveName, iter_dt, Time, τII, εII, Vx, Vy, η_v, T, P, ρ, H, H_dif, H_dis, H_LTP, CD)

        # save evolution
        saveEvo(saveFlag, nanFlag, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD, saveName, xc, ind_cen)

        # save restart database
        saveRestart(iter_dt, nRestart, nanFlag, saveName, Time, dt, Vx, Vy, T, T_o, P, P_o, ρ, η, η_dif, η_dis, η_LTP, τxx, τyy, τxy, τII, τII_o, VxUp, VyUp, PUp, TUp, err_evo, its_evo, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD)
    end
    return nothing
end

@time ElaDifDisLTP_2D()
