const USE_GPU = true # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(0)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, GeoParams, JLD2
import ParallelStencil: INDICES
ix, iy = INDICES[1], INDICES[2]

include("Modules/Support_InitBC.jl")
include("Modules/Support_Geom.jl")
include("Modules/Support_Timestep.jl")
include("Modules/Support_SaveLoadPlot.jl")
include("Modules/Support_Rheology.jl")

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
    nx          = 1536      # number of cells x
    ny          = 256       # number of cells y
    dT_ref0     = 50K       # change of temperature for timestep scaling
    dτ_crit     = 100MPa    # critical change of stress for timestep scaling
    VdmpX0      = 12.0      # dampening parameter for velocity update
    VdmpY0      = 3.0       # dampening parameter for velocity update
    Vfac        = 5.0       # velocity pseudo step is divided by this
    Tfac        = 2.0       # temperature pseudo step is divided by this (increse when increasing resolution and no)
    Tdmp        = 16.0      # dampening parameter for temperature update
    Pdmp        = 0.0       # NOT IN USE ANYMORE. dampening parameter for pressure update 
    tol0        = 1e-6      # tolerance
    Vtol        = 1e-2      # tolerance for viscosity convergence
    nt          = 500e3     # maximum number of iterations
    nout        = 1e3       # output frequency
    η_rel       = 0.01      # viscosity relaxation
    CFL         = 0.5       # CFL-criterion

    # anomaly
    anomType    = 1         # [0: load from file, 1: center, 2: boundary, 3: boundary and inside]
    anomFile    = ""
    rad_x       = Lx/160    # major semi-axis for ellipsoidal anomaly
    rad_y       = Lx/480    # minor semi-axis for ellipsoidal anomaly
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
    xn, xc, dxn, dxc, dxn_v, yn, yc, dyn, dyc, dyn_v = makeCoordsNew(Lx, Ly, nx, ny, gridType)    # grid
    dxn, dxc, dxn_v, dyn, dyc, dyn_v                 = Coord2PS(dxn, dxc, dxn_v, dyn, dyc, dyn_v) # turn PS arrays
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

    # visualisation
    ENV["GKSwstype"]="nul"
    if isdir(outDirName)==false mkdir(outDirName) end

    # load
    if restartFlag
        Time, iter_dt, dt, Vx, Vy, T, T_o, P, P_o, ρ, η, η_dif, η_dis, η_LTP, τxx, τyy, τxy, τII, τII_o, VxUp, VyUp, PUp, TUp, err_evo, its_evo, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD = loadRestart(restartName)
    end

    # numerical stuff
    err = 1
    iter, mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps, mean_ResVisc, dampReset, ndampRe, nRaise= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    err_evo, its_evo, ErrVx_evo, ErrVy_evo, ErrT_evo, ErrP_evo, Errη_evo = (Array{Float64}(undef, 0) for _ = 1:7)
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
            τII_p = makePrediction(τII, τII_oo, τII_p, dt, dt_o, true)
            T_p   = makePrediction(T,   T_oo,   T_p,   dt, dt_o, true)
            P_p   = makePrediction(P,   P_oo,   P_p,   dt, dt_o, false)

            # set to predictions
            @parallel setPrediction!(τII, T, P, τII_p, T_p, P_p)

            # reset
            err = 1
            iter, mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps, mean_ResVisc, dampReset, ndampRe, nRaise = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            err_evo, its_evo, ErrVx_evo, ErrVy_evo, ErrT_evo, ErrP_evo, Errη_evo = (Array{Float64}(undef, 0) for _ = 1:7)
            VdmpX, VdmpY = VdmpX0, VdmpY0
            dampVx, dampVy = dampFlag * (1.0 - VdmpX/nx), dampFlag * (1.0 - VdmpY/ny)

########################
#### Iteration Loop ####
########################
            while !ReFlag && (err > tol && iter < nt && !nanFlag) && !ConvFlag
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
                    @parallel getResVisc!(Errη, η_v, η_new, η_dif_new, η_dis_new, η_LTP_new, η_reg)
                    mean_ResVx       = mean(ErrVx)
                    mean_ResVy       = mean(ErrVy)
                    mean_ResT        = mean(ErrT)
                    mean_ResP        = mean(ErrP)
                    mean_ResEps      = mean(abs.(εII_dis .- εII_dis_g) ./ εII)
                    mean_ResVisc     = mean(Errη) * tol / Vtol
                    err              = max(mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps, mean_ResVisc)
                    push!(ErrVx_evo, mean_ResVx)
                    push!(ErrVy_evo, mean_ResVy) 
                    push!(ErrT_evo,  mean_ResT)
                    push!(ErrP_evo,  mean_ResP)
                    push!(Errη_evo,  mean_ResVisc)
                    push!(err_evo, err)
                    push!(its_evo, iter)
                    @printf("Its = %d, err = %1.3e [mean_ResVx=%1.3e, mean_ResVy=%1.3e, mean_dT=%1.3e, mean_dP=%1.3e, mean_Resε=%1.3e, mean_Resη=%1.3e] \n", iter, err, mean_ResVx, mean_ResVy, mean_ResT, mean_ResP, mean_ResEps, mean_ResVisc)
                    nanFlag          = isnan(err) || isinf(err)
                    tol, nRaise      = raiseTol2(tol, nRaise, iter)
                    ReFlag, ConvFlag = checkError(err_evo, nout, ReFlag, iter)
                    #dampVx, dampVy, dampT, VdmpX, VdmpY, Tdmp, dampReset, tol, ndampRe = AdjustDamp(dampVx, dampVy, dampT, VdmpX, VdmpY, Tdmp, VdmpX0, VdmpY0, nx, ny, err_evo, dampReset, tol, ndampRe, dampFlag)
                end
            end
        end

        # tracking
        @parallel diss_energy_full!(εII_dif, εII_dis, εII_LTP, εII_v, H, H_dif, H_dis, H_LTP, H_reg, τII_vtrue, τII_reg, τxx, τyy, τxy_cen, εxx, εyy, εxy_cen, η_e, heatFlag, elaFlag)
        Time = trackProperties!(Time, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, dt, T, τII, Vx, η_v, εII, H, iter, iter_re, err, nx)

        # plotting
        if (plotFlag && mod(iter_dt, 1) == 0)
            @parallel ε_ela!(εII_ela, τII, τII_o, η_e)
            @parallel (1:nx, 1:ny) domMech!(dom, εII_ela, εII_dif, εII_dis, εII_LTP)
            SH_2D_plot(xn, yn, xc, yc, dt, Vx, Vy, T, P, τII, t_evo, CD, iter_dt, its_evo, ErrVx_evo, ErrVy_evo, ErrT_evo, ErrP_evo, Errη_evo, dom, outDirName)
        end

        # save full fields
        saveFullField(saveAllFlag, saveAllOut, saveName, iter_dt, Time, τII, εII, Vx, Vy, η_v, T, P, ρ, H, H_dif, H_dis, H_LTP, CD)

        # save evolution
        saveEvo(saveFlag, nanFlag, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD, saveName, xc, ind_cen)

        # save restart database
        saveRestart(iter_dt, nRestart, nanFlag, saveName, Time, dt, Vx, Vy, T, T_o, P, P_o, ρ, η, η_dif, η_dis, η_LTP, τxx, τyy, τxy, τII, τII_o, VxUp, VyUp, PUp, TUp, err_evo, its_evo, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD)
    end
end

function nonDimInput(t_end, Lx, Ly, εbg, T0, T1, P0, dt0, gs0, dT_ref0, dτ_crit, rad_x, rad_y, x_off, y_off, restartName, restartFlag, saveName)
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

    return CD, t_end, Lx, Ly, εbg, T0, T1, P0, dt0, gs0, dT_ref0, dτ_crit, rad_x, rad_y, x_off, y_off, restartName
end

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

function Sol2PS(P, T, Vx, Vy)
    P         = Data.Array(P)
    P_o       = copy(P)
    T         = Data.Array(T)
    T_o       = copy(T)
    Vx        = Data.Array(Vx)
    Vy        = Data.Array(Vy)
    return P, P_o, T, T_o, Vx, Vy
end

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


function inter(X)
    Y = zeros(size(X, 1) - 1, size(X, 2) - 1);
    for ix = 1 : size(X, 1) - 1
        for iy = 1 : size(X, 2) - 1
            Y[ix, iy] = (X[ix,iy] + X[ix+1,iy] + X[ix,iy+1] + X[ix+1,iy+1])/4.0;
        end
    end
    return Y;
end

function extra!(X, Y, nx, ny)
    Y[2:end-1,2:end-1] = inter(X);
    for iy = 1 : ny+1
        Y[1, iy]   = Y[2, iy];
        Y[end, iy] = Y[end-1, iy];
    end
    for ix = 1 : nx+1
        Y[ix, 1]   = Y[ix, 2];
        Y[ix, end] = Y[ix, end-1];
    end
end

# iteration loop
@parallel function update_ρ!(ρ::A, ρ_o::A, ∇V::A, κ::A, ρCp::A, dt::N, λ::N, Cp::N, compFlag::N) where {A<:Data.Array, N<:Number}
    @all(ρ)   = @all(ρ_o) * exp(-@all(∇V) * dt * compFlag)
    @all(ρCp) = @all(ρ) * Cp
    @all(κ)   = λ / @all(ρCp)
    return
end

@parallel function divVel!(∇V::A, Vx::A, Vy::A, dxn::A, dyn::A) where {A<:Data.Array}
    @all(∇V) = @d_xi(Vx) / @all(dxn) + @d_yi(Vy) / @all(dyn)
    return   
end

@parallel function update_P!(P::A, P_o::A, ∇V::A, Kb::A, η::A, ResP::A, dψP::A, PUp::A, dt::N, max_nxy::N, dampP::N, compFlag::N) where {A<:Data.Array, N<:Number}
    @all(ResP) = @all(∇V)  + compFlag  * (@all(P) - @all(P_o)) / (dt * @all(Kb))
    @all(dψP)  = 10.0      * @all(η)   / max_nxy
    @all(PUp)  = dampP     * @all(PUp) + @all(ResP)
    @inn_x(P)  = @inn_x(P) - @all(PUp) * @all(dψP)
    return    
end

@parallel function diss_energy_full!(εII_dif::A, εII_dis::A, εII_pei::A, εII_v::A, H::A, H_dif::A, H_dis::A, H_pei::A, H_reg::A, τII_vtrue::A, τII_reg::A, τxx::A, τyy::A, τxy_cen::A, εxx::A, εyy::A, εxy_cen::A, η_e::A, heatFlag::N, elaFlag::N) where {A<:Data.Array, N<:Number}
    @all(H_dif)   = @all(τII_vtrue) * @all(εII_dif) * 2.0 * heatFlag;
    @all(H_dis)   = @all(τII_vtrue) * @all(εII_dis) * 2.0 * heatFlag;
    @all(H_pei)   = @all(τII_vtrue) * @all(εII_pei) * 2.0 * heatFlag;
    @all(H_reg)   = @all(τII_reg)   * @all(εII_v)   * 2.0 * heatFlag;
    @all(H)       = (@inn_x(τxx)    * (@all(εxx)     - elaFlag * @inn_x(τxx)   / (2.0 * @all(η_e))) +
                     @all(τyy)      * (@all(εyy)     - elaFlag * @all(τyy)     / (2.0 * @all(η_e))) +
                     @all(τxy_cen)  * (@all(εxy_cen) - elaFlag * @all(τxy_cen) / (2.0 * @all(η_e))) * 2) * heatFlag;
    return    
end

@parallel function diss_energy!(H::A, τxx::A, τyy::A, τxy_cen::A, εxx::A, εyy::A, εxy_cen::A, η_e::A, heatFlag::N, elaFlag::N) where {A<:Data.Array, N<:Number}
    @all(H)       = (@inn_x(τxx)    * (@all(εxx)     - elaFlag * @inn_x(τxx)   / (2.0 * @all(η_e))) +
                     @all(τyy)      * (@all(εyy)     - elaFlag * @all(τyy)     / (2.0 * @all(η_e))) +
                     @all(τxy_cen)  * (@all(εxy_cen) - elaFlag * @all(τxy_cen) / (2.0 * @all(η_e))) * 2) * heatFlag;
    return    
end

@parallel function fluxes!(qx::A, qy::A, T::A, dxc::A, dyc::A, λ::N) where {A<:Data.Array, N<:Number}
    @all(qx)      = - λ * @d_xa(T) / @all(dxc)
    @inn_y(qy)    = - λ * @d_yi(T) / @inn_y(dyc)
    return    
end

@parallel function diffusion!(dT_diff::A, qx::A, qy::A, dxn::A, dyn::A, diffFlag::N) where {A<:Data.Array, N<:Number}
    @all(dT_diff) = - (@d_xa(qx) / @all(dxn) + @d_ya(qy) / @all(dyn)) * diffFlag
    return
end

@parallel function residuals!(ResVx::A, ResVy::A, ResP::A, ResT::A, τxx::A, τyy::A, τxy::A, ∇V::A, P::A, P_o::A, T::A, T_o::A, Kb::A, dT_diff::A, H::A, ρCp::A, dxc::A, dyc::A, dxn_v::A, dyn_v::A, dt::N, α::N, compFlag::N) where {A<:Data.Array, N<:Number}
    @all(ResVx) = (@d_xa(τxx) - @d_xa(P)) / @all(dxc) + @d_ya(τxy) / @all(dyn_v)
    @all(ResVy) = (@d_ya(τyy) - @d_yi(P)) / @inn_y(dyc) + @d_xi(τxy) / @all(dxn_v)
    @all(ResP)  = @all(∇V) + compFlag * (@inn_x(P) - @inn_x(P_o)) / (dt * @all(Kb)) - α * (@inn_x(T) - @inn_x(T_o)) / dt
    @all(ResT)  = (@all(dT_diff) + @all(H)) / @all(ρCp) - (@inn_x(T) - @inn_x(T_o)) / dt
    return    
end

@parallel function new_step!(dψVx::A, dψVy::A, dψP::A, η::A, η_dx::A, η_dy::A, G::A, Kb::A, dxc::A, dyc::A, max_Lxy::N, min_dxy::N, Vfac::N, max_nxy::I) where {A<:Data.Array, N<:Number, I<:Int64}
    @all(dψVx) = @all(dxc)^2   / @all(η_dx) / (Vfac)
    @all(dψVy) = @inn_y(dyc)^2 / @all(η_dy) / (Vfac)
    @all(dψP)  = 10.0 * @all(η) / max_nxy
    #@all(dψP)  = min_dxy * @all(η) / (max_Lxy * @all(Kb) / @all(G))
    return    
end

@parallel function comp_update!(VxUp::A, VyUp::A, PUp::A, TUp::A, ResVx::A, ResVy::A, ResP::A, ResT::A, dampVx::N, dampVy::N, dampP::N, dampT::N) where {A<:Data.Array, N<:Number}
    @all(VxUp) = dampVx * @all(VxUp) + @all(ResVx)
    @all(VyUp) = dampVy * @all(VyUp) + @all(ResVy)
    @all(PUp)  = dampP  * @all(PUp)  + @all(ResP)
    @all(TUp)  = dampT  * @all(TUp)  + @all(ResT)
    return    
end

@parallel function apply_update!(Vx::A, Vy::A, P::A, T::A, VxUp::A, VyUp::A, PUp::A, TUp::A, dψVx::A, dψVy::A, dψP::A, dψT::N) where {A<:Data.Array, N<:Number}
    @inn_y(Vx) = @inn_y(Vx) + @all(VxUp)  * @all(dψVx)
    @inn(Vy)   = @inn(Vy)   + @all(VyUp)  * @all(dψVy)
    @inn_x(P)  = @inn_x(P)  - @all(PUp)   * @all(dψP)
    @inn_x(T)  = @inn_x(T)  + @all(TUp)   * dψT
    return  
end

function checkError(err, nout, ReFlag, iter)
    if ReFlag == true
        return true, false
    end

    # converged based on stall
    n = max(Int64(round(10000 / nout)), 5)
    if length(err) > n && err[end] > err[end-n] * 0.5 && err[end] < 1e-3
        isOK = true
        for i = 0 : n-1
            if err[end-i] > err[end-i-1]
                isOK = false
                break
            end
        end
        if isOK
            @printf("Iter: %d, Converged because of stalling residual. \n", iter)
            return false, true
        end
    end

    return false, false
end

@parallel function AbsErr!(AbsVx::A, AbsVy::A, AbsT::A, AbsP::A, ResVx::A, ResVy::A, ResT::A, ResP::A, V0::N, T0::N, P0::N) where {A<:Data.Array, N<:Number}
    @all(AbsVx) = abs(@inn_x(ResVx)) / V0
    @all(AbsVy) = abs(@all(ResVy))   / V0
    @all(AbsT)  = abs(@all(ResT))    / T0
    @all(AbsP)  = abs(@all(ResP))    / P0
    return
end

@parallel function RelErr!(RelVx::A, RelVy::A, RelT::A, RelP::A, ResVx::A, ResVy::A, ResT::A, ResP::A, Vx::A, Vy::A, T::A, P::A, V0::N, T0::N, P0::N) where {A<:Data.Array, N<:Number}
    @all(RelVx) = abs(@inn_x(ResVx) / (@inn(Vx)  + V0*1e-15))
    @all(RelVy) = abs(@all(ResVy)   / (@inn(Vy)  + V0*1e-15))
    @all(RelT)  = abs(@all(ResT)    / (@inn_x(T) + T0*1e-15))
    @all(RelP)  = abs(@all(ResP)    / (@inn_x(P) + P0*1e-15))
    return
end

@parallel function MinErr!(ErrVx::A, ErrVy::A, ErrT::A, ErrP::A, AbsVx::A, AbsVy::A, AbsT::A, AbsP::A, RelVx::A, RelVy::A, RelT::A, RelP::A) where {A<:Data.Array}
    @all(ErrVx) = min(@all(AbsVx), @all(RelVx));
    @all(ErrVy) = min(@all(AbsVy), @all(RelVy));
    @all(ErrT)  = min(@all(AbsT),  @all(RelT));
    @all(ErrP)  = min(@all(AbsP),  @all(RelP));
    return
end

@parallel function getResVisc!(Errη::Data.Array, η::Data.Array, η_new::Data.Array, η_dif_new::Data.Array, η_dis_new::Data.Array, η_LTP_new::Data.Array, η_reg::Number)
    @all(η_new) = (1.0/@all(η_dif_new) + 1.0/@all(η_dis_new) + 1.0/@all(η_LTP_new)) ^ (-1.0) + η_reg
    @all(Errη)  = abs.((@all(η) - @all(η_new)) / @all(η_new))
    return
end

@time ElaDifDisLTP_2D()
