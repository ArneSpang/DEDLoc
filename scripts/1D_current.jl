using Printf, Statistics, Plots, GeoParams, JLD2

using DEDLoc.CPU_1D

function ElaDisDifLTP_1D(epsVal, Gval, sigB, L, λval, Tbg, ωmax, hL, reg, reso, saveName; anomType=2, regType=1)
    # epsVal:   background strainrate [1/s]
    # Gval:     Shear modulus [GPa]
    # sigB:     back stress [GPa]
    # L:        domain length [km]
    # λval:     thermal conductivity [J/(s*m*K)]
    # Tbg:      background temperature [C]
    # ωmax:     weakening factor
    # hL:       size of anomaly compared to model
    # reg:      regularization viscosity [Pas] or numerical diffusion length scale [m] (depending on regType)
    # reso:     number of cells
    # saveName: name of saved files
    # anomType: type of anomaly (1: step-like, 2: gaussian)
    # regType:  type of regularization (1: viscosity, 2: gradient) 
    
###############
#### Input ####
###############
    
    # flags
    ela         = 1.0       # elasticity flag (Float)
    diffFlag    = 1.0       # diffusion activation flag (Float)
    plotFlag    = false     # plot flag (bool)
    saveFlag    = true      # save Flag (bool)
    nRestart    = 10        # save restart database every n steps
    restartFlag = false     # load restart database (bool)
    restartName = ""        # name of specific restart file

    # Setup
    maxStrain   = 1.0       # maximum strain (controls model time)
    Lx          = L*km      # length of profile
    ε           = epsVal/s  # strain rate
    T0          = Tbg*C     # reference temperatrue
    P0          = 10e3MPa   # reference pressure

    # solver
    dt0         = 2.5e1yr   # reference time step
    dt_min      = 1s        # minimum time step (currently deactivated)
    nx          = reso      # number of cells
    dT_ref0     = 10K       # change of temperature for timestep scaling
    dτ_crit     = 50MPa     # critical change of stress for timestep reduction and dampening adjustment
    Vdmp0       = 2.0       # dampening parameter for velocity update
    Tdmp0       = 8.0       # dampening parameter for temperature update
    nt          = 1000e3    # maximum number of pt iterations
    nt_min      = 5e3       # minimum number of iterations
    nout        = 2e3       # output frequency
    η_rel       = 0.1       # viscosity relaxation parameter
    tol0        = 1e-6      # tolerance
    
    # anomaly
    an_x        = 0.0km     # position of anomaly
    an_l        = hL*Lx/2.0 # radius of anomaly


    ## Material parameters
    # dislocation creep
    ndis        = 3.5;                 # powerlaw exponent
    Adis0       = 1.1e5MPa^-ndis/s;    # prefactor
    Qdis        = 530e3J/mol;          # activation energy
    R           = 8.314J/mol/K;        # gas constant
    η_max       = 1e40Pas;             # maximum viscosity
    
    # diffusion creep
    mdif        = 3.0;                      # grain size exponent
    Adif0       = 1.5e9MPa^-1*μm^mdif*s^-1; # prefactor
    Qdif        = 375e3J/mol;               # activation energy
    gs0         = 1e-1mm;                   # initial grain size

    # low temperature plasticity (Hansen 2019)
    LTP_A0      = 5e20/s;                   # prefactor
    LTP_En      = 550kJ/mol;                # activation energy
    LTP_σL      = 3.1e3MPa;
    LTP_K       = 3.2e3MPa*µm^0.5;
    LTP_σb      = sigB*1e3MPa;              # back stress

    # other
    ρ0          = 3300kg/m^3    # density
    Cp          = 1000J/kg/K    # heat capacity
    G           = Gval*1e9Pa    # shear modulus
    ν           = 0.25          # poisson ratio
    λ           = λval*J/s/m/K  # thermal conductivity

##########################
#### Input Processing ####
##########################
    t_end   = maxStrain / ε
    
    # nondimensionalize
    if restartFlag
        if length(restartName) == 0
            restartName = saveName*"_restart.jld2";
        end
        CD = load(restartName, "CD");
    else
        CD = GEO_units(length=10km, temperature=600C, stress=100MPa, viscosity=1e20Pas);
    end
    Lx      = nondimensionalize(Lx,      CD)
    an_x    = nondimensionalize(an_x,    CD)
    an_l    = nondimensionalize(an_l,    CD)
    ε       = nondimensionalize(ε,       CD)
    dt0     = nondimensionalize(dt0,     CD)
    dt_min  = nondimensionalize(dt_min,  CD)
    t_end   = nondimensionalize(t_end,   CD)
    dT_ref0 = nondimensionalize(dT_ref0, CD)
    dτ_crit = nondimensionalize(dτ_crit, CD)
    T0      = nondimensionalize(T0,      CD)
    P0      = nondimensionalize(P0,      CD)
    ρ0      = nondimensionalize(ρ0,      CD)
    Cp      = nondimensionalize(Cp,      CD)
    Adis0   = nondimensionalize(Adis0,   CD)
    Qdis    = nondimensionalize(Qdis,    CD)
    Adif0   = nondimensionalize(Adif0,   CD)
    Qdif    = nondimensionalize(Qdif,    CD)
    LTP_A0  = nondimensionalize(LTP_A0,  CD)
    LTP_En  = nondimensionalize(LTP_En,  CD)
    LTP_σL  = nondimensionalize(LTP_σL,  CD)
    LTP_K   = nondimensionalize(LTP_K,   CD)
    LTP_σb  = nondimensionalize(LTP_σb,  CD)
    R       = nondimensionalize(R,       CD)
    η_max   = nondimensionalize(η_max,   CD)
    gs0     = nondimensionalize(gs0,     CD)
    G       = nondimensionalize(G,       CD)
    λ       = nondimensionalize(λ,       CD)

    # derived properties
    xn      = uSpace(-Lx/2.0, Lx/2.0, nx)
    xc      = 0.5*(xn[1:end-1] + xn[2:end])
    dxn     = diff(xn)
    dxc     = diff(xc)
    mindx   = minimum([dxn; dxc])
    Vshear  = 2.0 * Lx * ε
    Kb      = (2*G*(1+ν))/(3*(1-2*ν))
    dampVy  = 1.0-Vdmp0/nx
    dampT   = 1.0-Tdmp0/nx
    dT_ref  = dT_ref0
    maxFac  = 5.0 * nondimensionalize(1e-12/s, CD) / ε

    # regularization
    if regType == 1
        η_reg   = nondimensionalize(reg * Pas, CD)
        λ_num2  = nondimensionalize(0*m,       CD)
    elseif regType == 2
        η_reg   = nondimensionalize(0 * Pas,   CD)
        λ_num2  = nondimensionalize(reg * m,   CD) ^ 2
    else
        error("regType should be 1 (viscous) or 2 (gradient).")
    end

####################
#### Initialize ####
####################

    # preallocate
    Vy        = zeros(nx+1)
    Vy_o      = zeros(nx+1)
    Vy_oo     = zeros(nx+1)
    τxy       = zeros(nx)
    τxy_o     = zeros(nx)
    τxy_oo    = zeros(nx)
    τII       = zeros(nx)
    τII_o     = zeros(nx)
    τII_oo    = zeros(nx)
    τII_reg   = zeros(nx)
    τII_vtrue = zeros(nx)
    ResVy     = zeros(nx-1)
    dψ        = zeros(nx-1)
    η_vert    = zeros(nx-1)
    VyUp      = zeros(nx-1)
    εxy       = zeros(nx)
    εII       = zeros(nx)
    εII_v     = zeros(nx)
    εII_dif   = zeros(nx)
    εII_nl    = zeros(nx)
    εII_dis   = zeros(nx)
    εII_LTP   = zeros(nx)
    εII_dis_g = zeros(nx)
    εII_LTP_g = zeros(nx)
    d2γ       = zeros(nx-2)
    nl_part   = zeros(nx)
    η_dis_new = zeros(nx)
    η_dif_new = zeros(nx)
    η_LTP_new = zeros(nx)
    η_dis_o   = zeros(nx)
    η_dif_o   = zeros(nx)
    η_LTP_o   = zeros(nx)
    H_dis     = zeros(nx)
    H_dif     = zeros(nx)
    H_LTP     = zeros(nx)
    H_num     = zeros(nx)
    H         = zeros(nx)
    T         = zeros(nx)
    T_o       = zeros(nx)
    T_oo      = zeros(nx)
    ResT      = zeros(nx)
    dψT       = 0.0
    TUp       = zeros(nx)
    dT_diff   = zeros(nx)
    qx        = zeros(nx+1)
    P         = zeros(nx)

    # initial guess and boundary conditions
    Vy     .= (xn .+ Lx/2) ./ Lx .* Vshear
    Vy[end] = Vshear
    V0      = maximum(Vy) - minimum(Vy)

    # set up temperature field
    T      .= ones(nx) .* T0
    T_o    .= ones(nx) .* T0

    # set up pressure field
    P      .= ones(nx) .* P0

    # set up prefactor
    Adif    = Adif0  * ones(nx)
    Adis    = Adis0  * ones(nx)
    LTP_A   = LTP_A0 * ones(nx)

    # set up grain size field
    gs      = ones(nx) .* gs0

    # set up density field
    ρ       = ones(nx) .* ρ0 .* exp(P0/Kb)
    ρ_nodes = (ρ[1:end-1] .+ ρ[2:end]) ./ 2
    ρCp     = ρ        .* Cp
    κ       = mean(λ        ./ ρCp)

    # anomaly
    # step-like
    if anomType == 1
        ind         = abs.(xc .- an_x) .< an_l
        Adif[ind] .*= ωmax
        Adis[ind] .*= ωmax
        #LTP_A[ind].*= ωmax
    # gaussian
    elseif anomType == 2
        ω           = 1.0 .+ PDF(xc, 0, 2*an_l, ωmax-1.0)
        Adif      .*= ω
        Adis      .*= ω
        #LTP_A     .*= ω
    end
    
    

    # set up viscosity fields
    LTP_Σ   = LTP_σL .+ LTP_K .* gs.^-0.5;
    η_dif   = 0.5 .* Adif.^(-1.0)      .* gs.^mdif             .* exp.(Qdif ./ (T .* R))
    η_dis   = 0.5 .* Adis.^(-1.0/ndis) .* (ε/3)^(1.0/ndis-1.0) .* exp.(Qdis ./ (T .* R .* ndis))
    η_LTP   =(0.5 .* T .* R .* LTP_Σ ./ LTP_En .* asinh.(ε/3 ./ LTP_A .* exp.(LTP_En ./ (R .* T))) .+ LTP_σb) ./ (ε/3);  
    η_v     = (1.0 ./ η_dif .+ 1.0 ./ η_dis .+ 1.0 ./ η_LTP) .^ -1.0
    η_e     = G * dt0
    η       = (1.0 ./ η_v .+ 1.0 ./ η_e) .^ -1.0

    # tracking
    Time       = 0.0
    Time_evo   = [0.0]
    dt_evo     = Array{Float64}(undef, 0)
    τ_evo      = [mean(τxy)]
    T_evo      = [maximum(T)]
    T2_evo     = [mean(T)]
    Vy_evo     = [maximum(Vy)]
    η_evo      = [minimum(η)]
    η2_evo     = [mean(η)]
    ηv_evo     = [minimum(η_v)]
    ηv2_evo    = [mean(η_v)]
    H_evo      = [maximum(H)]
    H2_evo     = [mean(H)]
    H_dif_evo  = [maximum(H_dif)]
    H_dif2_evo = [mean(H_dif)]
    H_dis_evo  = [maximum(H_dis)]
    H_dis2_evo = [mean(H_dis)]
    H_LTP_evo  = [maximum(H_LTP)]
    H_LTP2_evo = [mean(H_LTP)]
    ε_evo      = [εxy[Int(round(nx/2))]]
    εv_evo     = [εxy[Int(round(nx/2))]]
    iter_evo   = [0]
    ResV_evo   = [0.0]
    ResT_evo   = [0.0]
    iter_dt    = 0
    VyProf_evo = [copy(Vy)]
    TProf_evo  = [copy(T)]
    dt         = copy(dt0)
    dt_o       = 0.0
    tol        = copy(tol0)

    # visualization
    ENV["GKSwstype"]="nul";
    if isdir("Outdir")==false mkdir("Outdir") end; 
    loadpath = "./Outdir/"; 
    anim = Animation(loadpath,String[]);
    println("Animation directory: $(anim.dir)");

    # load restart
    if restartFlag
        @load restartName CD Time iter_dt dt dT_ref Vy T T_o T ρ η η_vert η_e η_v η_dif η_dif_new η_dis η_dis_new η_LTP η_LTP_new τxy τxy_o τII τII_o εxy εII_nl εII_dis εII_dis_g εII_LTP_g nl_part H_dif H_dis H_LTP qx dT_diff ResVy ResT dψ dψT VyUp TUp Time_evo dt_evo τ_evo T_evo T2_evo Vy_evo η_evo η2_evo ηv_evo ηv2_evo H_evo H2_evo H_dif_evo H_dif2_evo H_dis_evo H_dis2_evo H_LTP_evo H_LTP2_evo ε_evo εv_evo iter_evo ResV_evo ResT_evo VyProf_evo TProf_evo
    end

    err_evo, errV_evo, errT_evo, its_evo = Array{Float64}(undef, 0), Array{Float64}(undef, 0), Array{Float64}(undef, 0), Array{Float64}(undef, 0) 
    iter = 0; err = 1
    mean_ResVy = 0; mean_ResT = 0; mean_ResEps = 0; mean_ResEps2 = 0
    Vdmp = Vdmp0; Tdmp = Tdmp0; dampReset = 0
    dampVy = 1.0-Vdmp/nx; dampT = 1.0-Tdmp/nx

###################
#### Time Loop ####
###################
    nanFlag = false
    while Time < t_end
        iter_dt += 1
        @printf("------------------\nStep %d (t=%.2f kyr)\n------------------\n", iter_dt, ustrip(dimensionalize(Time, yr, CD))*1e-3);

        # save info
        τII_oo = copy(τII_o); τxy_oo = copy(τxy_o); T_oo = copy(T_o); Vy_oo = copy(Vy_o);
        τII_o = copy(τII); τxy_o = copy(τxy); T_o = copy(T); Vy_o = copy(Vy); dt_o = copy(dt);
        η_dif_o = copy(η_dif); η_dis_o = copy(η_dis); η_LTP_o = copy(η_LTP);

##################################
#### Initialize new time step ####
##################################
        # reset       
        err    = 1;
        iter   = 0;
        ReFlag = false;

        while ReFlag || (err > tol && iter < nt)

        # reset solution variables in case of restart
        if ReFlag
            τII = copy(τII_o); τxy = copy(τxy_o); T = copy(T_o); Vy = copy(Vy_o);
            η_dif = copy(η_dif_o); η_dis = copy(η_dis_o); η_LTP = copy(η_LTP_o);
        end

        # update timestep
        if iter_dt > 1
            dt     = updateTimestep(dt, dt0, T, T_oo, dT_ref, τxy, τxy_oo, dτ_crit, ReFlag, maxFac, dt_min)
            ReFlag = false;
            dt_dim = ustrip(dimensionalize(dt, yr, CD))
            if dt_dim > 1
                @printf("Reset timestep to %1.3e yr.\n", dt_dim)
            elseif dt_dim > 1 / 365.25
                @printf("Reset timestep to %1.3e days.\n", dt_dim*365.25)
            elseif dt_dim > 1 / (365.25 * 24)
                @printf("Reset timestep to %1.3e hours.\n", dt_dim*365.25*24)
            else
                @printf("Reset timestep to %1.3e seconds.\n", dt_dim*365.25*24*3600)
            end
        end

        # elastic viscosity
        η_e               = G * dt

        # check if scaling is still ok
        if dt < 1e-9
            dt, dt_o, dt0, Time, t_end, Vy, T, T_o, T_oo, dT_ref, dT_ref0, P, τxy, τxy_o, ρ, η, η_e, η_v, η_vert, τII, τII_o, τII_oo, τII_reg, τII_vtrue, dτ_crit, εxy, εII, εII_v, εII_nl, εII_dif, εII_dis_g, εII_LTP_g, η_dif, η_dif_new, η_dis, η_dis_new, η_LTP, η_LTP_new, ρ_nodes, ρCp, κ, qx, dT_diff, H, R, gs, Adif, mdif, Qdif, Adis, ndis, Qdis, LTP_A, LTP_En, LTP_Σ, LTP_σb, G, Kb, λ, Cp, dxc, dxn, mindx, η_reg, η_max, λ_num2, CD_new = rescaleDimensions(10, dt, dt_o, dt0, Time, t_end, Vy, T, T_o, T_oo, dT_ref, dT_ref0, P, τxy, τxy_o, ρ, η, η_e, η_v, η_vert, τII, τII_o, τII_oo, τII_reg, τII_vtrue, dτ_crit, εxy, εII, εII_v, εII_nl, εII_dif, εII_dis_g, εII_LTP_g, η_dif, η_dif_new, η_dis, η_dis_new, η_LTP, η_LTP_new, ρ_nodes, ρCp, κ, qx, dT_diff, H, R, gs, Adif, mdif, Qdif, Adis, ndis, Qdis, LTP_A, LTP_En, LTP_Σ, LTP_σb, G, Kb, λ, Cp, dxc, dxn, mindx, η_reg, η_max, λ_num2, CD);
            Time_evo, dt_evo, τ_evo, T_evo, T2_evo, Vy_evo, VyProf_evo, TProf_evo, η_evo, η2_evo, ηv_evo, ηv2_evo, H_evo, H2_evo, H_dif_evo, H_dif2_evo, H_dis_evo, H_dis2_evo, H_LTP_evo, H_LTP2_evo, ε_evo, εv_evo = rescaleEvo(Time_evo, dt_evo, τ_evo, T_evo, T2_evo, Vy_evo, VyProf_evo, TProf_evo, η_evo, η2_evo, ηv_evo, ηv2_evo, H_evo, H2_evo, H_dif_evo, H_dif2_evo, H_dis_evo, H_dis2_evo, H_LTP_evo, H_LTP2_evo, ε_evo, εv_evo, CD, CD_new);
            CD = CD_new
            @printf("########\n Changed scaling\n########\n")
        end

        # adjust tolerance
        tol   = findTol(ustrip(dimensionalize(dt, s, CD)), tol0);

        # make prediction based on last timestep
        τII_p = makePrediction(τII_o, τII_oo, dt, dt_o, true);
        T_p   = makePrediction(T_o,   T_oo,   dt, dt_o, true);  

        # set to prediction
        τII   = copy(τII_p); T = copy(T_p);

        # reset
        err_evo, errV_evo, errT_evo, its_evo = Array{Float64}(undef, 0), Array{Float64}(undef, 0), Array{Float64}(undef, 0), Array{Float64}(undef, 0) 
        iter = 0; err = 1
        mean_ResVy = 0; mean_ResT = 0; mean_ResEps = 0; mean_ResEps2 = 0; VyUp .= 0.0; TUp .= 0.0
        Vdmp = Vdmp0; Tdmp = Tdmp0; dampReset = 0; ndampRe = 0
        dampVy = 1.0-Vdmp/nx; dampT = 1.0-Tdmp/nx

########################
#### Iteration Loop ####
########################
        while !ReFlag && (err > tol || iter < nt_min) && iter < nt && !nanFlag
            iter += 1

            # effective strain rate
            εxy              .= 0.5 * diff(Vy) ./ dxn .+ ela .* τxy_o ./ (2.0 * η_e)
            εII              .= abs.(εxy)

            # diffusion creep viscosity
            η_dif_new        .= 0.5 .* Adif.^(-1.0) .* gs.^mdif .* exp.(Qdif ./ (T .* R))
            η_dif            .= exp.((1-η_rel) * log.(η_dif) .+ η_rel * log.(η_dif_new))

            # guess nonlinear part of strain rate
            min_ε             = minimum(εII)/1e12
            εII_v            .= εII       .- τII       ./ (2.0 * η_e)
            τII_reg          .= 2.0       .* η_reg .* εII_v
            τII_vtrue        .= τII       .- τII_reg
            εII_dif          .= τII_vtrue ./ (2.0 .* η_dif)
            εII_nl           .= εII_v     .- εII_dif
            
            # guess nonlinear partitioning
            nl_part          .= η_dis ./ η_LTP
            εII_dis_g        .= εII_nl .*       1 ./ (1 .+ nl_part)
            εII_LTP_g        .= εII_nl .* nl_part ./ (1 .+ nl_part)
            εII_dis_g[εII_dis_g .< min_ε] .= min_ε
            εII_LTP_g[εII_LTP_g .< min_ε] .= min_ε

            # dislocation creep viscosity
            η_dis_new        .= 0.5 .* Adis.^(-1.0/ndis) .* εII_dis_g.^(1.0/ndis-1.0) .* exp.(Qdis ./ (T .* R .* ndis))
            η_dis_new[η_dis_new .> η_max] .= η_max
            η_dis            .= exp.((1-η_rel) * log.(η_dis) .+ η_rel * log.(η_dis_new))

            # low temperature plasticity
            η_LTP_new        .= (T .* R .* LTP_Σ ./ LTP_En .* asinh.(εII_LTP_g ./ LTP_A .* exp.(LTP_En ./ (R .* T))) .+ LTP_σb) ./ (2 .* εII_LTP_g)
            η_LTP_new[η_LTP_new .> η_max] .= η_max
            η_LTP            .= exp.((1-η_rel) * log.(η_LTP) .+ η_rel * log.(η_LTP_new))

            # effective viscosity
            η_v              .= (1.0 ./ η_dif .+ 1.0 ./ η_dis .+ 1.0 ./ η_LTP) .^ -1.0
            η_v             .+= η_reg
            η                .= (1.0 ./ η_v .+ ela ./ η_e) .^ -1.0
            η_vert           .= (η[1:end-1] .+ η[2:end]) ./ 2.0

            # stress
            τxy              .= 2.0 .* η .* εxy
            τII              .= abs.(τxy)

            # true strain rate partitioning
            εII_v            .= εII       .- τII       ./ (2.0 * η_e)
            τII_reg          .= 2.0       .* η_reg .* εII_v
            τII_vtrue        .= τII       .- τII_reg
            εII_dif          .= τII_vtrue ./ (2.0 .* η_dif)
            εII_dis          .= τII_vtrue ./ (2.0 .* η_dis)
            εII_LTP          .= τII_vtrue ./ (2.0 .* η_LTP)

            # dissipative energy
            H_dif            .= τII_vtrue .* εII_dif
            H_dis            .= τII_vtrue .* εII_dis
            H_LTP            .= τII_vtrue .* εII_LTP
            H                .= τII .* τII ./ (2.0 * η_v)

            # strain rate diffusion (gradient regularization)
            if regType == 2
                d2γ              .= diff(diff(εII_v) ./ dxc) ./ dxn[2:end-1] .* diffFlag
                H_num[2:end-1]   .= τII[2:end-1] .* λ_num2 .* d2γ
                H_num[1]          = H_num[2]
                H_num[end]        = H_num[end-1]
            end

            # diffusion
            qx[2:end-1]      .= - λ .* diff(T) ./ dxc
            dT_diff          .= - diff(qx) ./ dxn .* diffFlag

            # residuals
            ResVy            .= diff(τxy) ./ dxc
            ResT             .= (dT_diff .+ H .+ H_num) ./ ρCp .- (T .- T_o) ./ dt

            # new pseudo step
            dψ               .= dxc.^2 ./ η_vert ./ 2.1 ./ 1.2
            dψT               = min(mindx^2 / κ / 2.1 * 0.5, 0.99 * dt)

            # update
            VyUp             .= dampVy .* VyUp .+ ResVy
            Vy[2:end-1]     .+= VyUp .* dψ
            TUp              .= dampT .* TUp .+ ResT
            T               .+= TUp .* dψT
            
            # check if timestep is ok
            if iter_dt > 1
                max_dτ            = maximum(abs.(τxy_o .- τxy))
                dT                = T .- T_o
                max_dT            = maximum(abs.(dT))
                if abs(maximum(dT)) > abs(minimum(dT))
                    dt_heat   = dt * dT_ref/max_dT
                else
                    dt_heat   = dt * dT_ref/max_dT * 10
                end
                if dt > dt_heat
                    ReFlag = true
                    @printf("Iter: %d, Restarting time step because of heat production: %1.3e K \n", iter, ustrip(dimensionalize(max_dT, K, CD)))
                elseif max_dτ > dτ_crit
                    ReFlag = true
                    @printf("Iter: %d, Restarting time step because of stress change: %1.3e MPa \n", iter, ustrip(dimensionalize(max_dτ, MPa, CD)))
                end
            end

            # check error
            mean_ResVy  = maximum(abs.(ResVy)) / V0
            mean_ResT   = mean(abs.(ResT)) / T0
            mean_ResEps = mean(abs.((εII_dis .- εII_dis_g) ./ (εII)))
            mean_ResEps2= mean(abs.((εII_LTP .- εII_LTP_g) ./ (εII)))
            err         = max(mean_ResVy, mean_ResT, mean_ResEps, mean_ResEps2)
            nanFlag     = isnan(err);

            # print
            if mod(iter, nout) == 0
                push!(err_evo,  err)
                push!(errV_evo, mean_ResVy)
                push!(errT_evo, mean_ResT)
                push!(its_evo,  iter)
                @printf("Its = %d, err = %1.3e [mean_ResVy=%1.3e, mean_dT=%1.3e, mean_Resε=%1.3e, mean_Resε2=%1.3e] \n", iter, err, mean_ResVy, mean_ResT, mean_ResEps, mean_ResEps2)
                dampVy, dampT, Vdmp, Tdmp, dampReset, tol, ndampRe = AdjustDamp(dampVy, dampT, Vdmp, Tdmp, nx, err_evo, dampReset, tol, ndampRe)
            end
        end
        end

########################
### End of time loop ###
########################
        @printf("Its = %d, err = %1.3e [mean_ResVy=%1.3e, mean_dT=%1.3e, mean_Resε=%1.3e, mean_Resε2=%1.3e] \n", iter, err, mean_ResVy, mean_ResT, mean_ResEps, mean_ResEps2)
        # tracking
        Time   += dt
        trackStuff!(err_evo, errV_evo, errT_evo, its_evo, Time_evo, dt_evo, τ_evo, T_evo, T2_evo, Vy_evo, VyProf_evo, TProf_evo, η_evo, η2_evo, ηv_evo, ηv2_evo, H_evo, H2_evo, H_dif_evo, H_dif2_evo, H_dis_evo, H_dis2_evo, H_LTP_evo, H_LTP2_evo, ε_evo, εv_evo, iter_evo, ResV_evo, ResT_evo, err, iter, Time, τxy, T, Vy, η, η_v, H, H_dif, H_dis, H_LTP, εxy, εII_v, nx, mean_ResVy, mean_ResT, dt, T_o)

        # plot
        if plotFlag || Time > 0.9*t_end
            SH_1D_plot(xn, xc, Vy, its_evo, errV_evo, errT_evo, T, η_e, η_dif, η_dis, η_LTP, CD, iter_dt, Time_evo, dt, τ_evo, T_evo, plotFlag, saveFlag, saveName)
        end

        # save
        if saveFlag && !nanFlag
        jldsave(saveName*".jld2", xc=xc, xn=xn, t=Time_evo, dt=dt_evo, τ_mean=τ_evo, T_max=T_evo, T_mean=T2_evo, Vy_max=Vy_evo, Vy_prof=VyProf_evo, η_min=η_evo, η_mean=η2_evo, ηv_min=ηv_evo, ηv_mean=ηv2_evo, 
                                  H_max=H_evo, H_mean=H2_evo, H_dif_max=H_dif_evo, H_dif_mean=H_dif2_evo, H_dis_max=H_dis_evo, H_dis_mean=H_dis2_evo, H_LTP_max=H_LTP_evo, H_LTP_mean=H_LTP2_evo, ε_cen=ε_evo, εv_cen=εv_evo, iter=iter_evo, ResV=ResV_evo, ResT=ResT_evo, CD=CD)
        end

        # save restart database
        if mod(iter_dt, nRestart) == 0 && !nanFlag
            jldsave(saveName*"_restart.jld2";
                    CD, Time, iter_dt, dt, dT_ref,
                    Vy, T, T_o, P, ρ, 
                    η, η_vert, η_e, η_v, η_dif, η_dif_new, η_dis, η_dis_new, η_LTP, η_LTP_new,
                    τxy, τxy_o, τII, τII_o, εxy, εII_nl, εII_dis, εII_dis_g, εII_LTP_g, nl_part,
                    H_dif, H_dis, H_LTP, qx, dT_diff, 
                    ResVy, ResT, dψ, dψT, VyUp, TUp,
                    Time_evo, dt_evo, τ_evo, T_evo, T2_evo, Vy_evo, 
                    η_evo, η2_evo, ηv_evo, ηv2_evo, H_evo, H2_evo, 
                    H_dif_evo, H_dif2_evo, H_dis_evo, H_dis2_evo, H_LTP_evo, H_LTP2_evo, ε_evo, εv_evo,
                    iter_evo, ResV_evo, ResT_evo, VyProf_evo, TProf_evo);
            @printf("Saved restart database.\n")
        end
    end
end

#@time ElaDisDifLTP_1D(5e-13, 80.0, 1.8, 10.0, 3.0, 650.0, 2.0, 0.02, 1e12, 127, "Ref")