using Printf, Statistics, Plots, GeoParams, JLD2

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

function updateTimestep(dt, dt0, T, T_o, dT_ref, τxy, τxy_o, dτ_crit, ReFlag, maxFac, dt_min)
    if ReFlag
        dt_new        = 0.5 * dt
    else
        max_dτ        = maximum(abs.(τxy .- τxy_o))
        dT            = T .- T_o
        max_dT        = maximum(abs.(dT))
        if abs(maximum(dT)) > abs(minimum(dT))
            dt_heat   = dt * dT_ref/max_dT
        else
            dt_heat   = dt * dT_ref/max_dT * 10
        end
        dt_stress     = dt * dτ_crit/max_dτ
        dt_new        = min(dt_heat, dt_stress, maxFac*dt0)
        # be hesitant to increase timestep
        if dt_new > dt
            dt_new = (dt_new + dt) / 2.0
        end
    end

    # employ minimum time step
    #dt_new = max(dt_new, dt_min)

    return dt_new
end

function trackStuff!(err_evo, errV_evo, errT_evo, its_evo, Time_evo, dt_evo, τ_evo, T_evo, T2_evo, Vy_evo, VyProf_evo, TProf_evo, η_evo, η2_evo, ηv_evo, ηv2_evo, H_evo, H2_evo, H_dif_evo, H_dif2_evo, H_dis_evo, H_dis2_evo, H_LTP_evo, H_LTP2_evo, ε_evo, εv_evo, iter_evo, ResV_evo, ResT_evo, err, iter, Time, τxy, T, Vy, η, η_v, H, H_dif, H_dis, H_LTP, εxy, εII_v, nx, mean_ResVy, mean_ResT, dt, T_o)
    push!(err_evo,    err)
    push!(errV_evo,   mean_ResVy)
    push!(errT_evo,   mean_ResT)
    push!(its_evo,    iter)
    push!(Time_evo,   Time)
    push!(dt_evo,     dt)
    push!(τ_evo,      mean(τxy))
    push!(T_evo,      maximum(T))
    push!(T2_evo,     mean(T))
    push!(Vy_evo,     maximum(abs.(Vy)))
    push!(VyProf_evo, copy(Vy))
    push!(TProf_evo,  copy(T))
    push!(η_evo,      minimum(η))
    push!(η2_evo,     mean(η))
    push!(ηv_evo,     minimum(η_v))
    push!(ηv2_evo,    mean(η_v))
    push!(H_evo,      maximum(H))
    push!(H2_evo,     mean(H))
    push!(H_dif_evo,  maximum(H_dif))
    push!(H_dif2_evo, mean(H_dif))
    push!(H_dis_evo,  maximum(H_dis))
    push!(H_dis2_evo, mean(H_dis))
    push!(H_LTP_evo,  maximum(H_LTP))
    push!(H_LTP2_evo, mean(H_LTP))
    push!(ε_evo,      εxy[Int(round(nx/2))])
    push!(εv_evo,     εII_v[Int(round(nx/2))])
    push!(iter_evo,   iter)
    push!(ResV_evo,   mean_ResVy)
    push!(ResT_evo,   mean_ResT)
    @printf("Maximum T change: %1.2e \n", maximum(abs.(T .- T_o)))
end

function AdjustDamp(dampVy, dampT, Vdmp, Tdmp, nx, err, dampReset, tol, ndampRe)
    # don't do anything for the first few cycles
    if length(err) < 10
        return dampVy, dampT, Vdmp, Tdmp, dampReset, tol, ndampRe
    end

    # if convergence is working but slow, speed it up
    if err[end] < err[end-1] < err[end-2] < err[end-3] && err[end] > err[end-1]*0.70
        Vdmp = Vdmp/2; Tdmp = Tdmp/2;
        @printf("Adjusted Vdmp/Tdmp to %.2f/%.2f \n", Vdmp, Tdmp)
    # speed up if there is stagnation
    elseif abs(err[end] - err[end-9]) < 0.3*err[end]
        Vdmp = Vdmp/2; Tdmp = Tdmp/2;
        @printf("Adjusted Vdmp/Tdmp to %.2f/%.2f \n", Vdmp, Tdmp)
    end

    # if residual is caught in oscillations
    if err[end] > 1.5*err[end-5] && dampReset < 1
        Vdmp = nx/4.0; Tdmp = nx/2.0;
        dampReset = 10
        @printf("Reset Vdmp/Tdmp to %.2f/%.2f \n", Vdmp, Tdmp)
        ndampRe += 1
        tol = raiseTol(tol, ndampRe)
    else
        dampReset -= 1;
    end
    
    dampVy = 1.0-Vdmp/nx; dampT = 1.0-Tdmp/nx;
    return dampVy, dampT, Vdmp, Tdmp, dampReset, tol, ndampRe
end

function raiseTol(tol, nRe)
    if 1 < nRe < 8
        return tol * 2.0
    else
        return tol
    end
end

function SH_1D_plot(xn, xc, Vy, its_evo, errV_evo, errT_evo, T, η_e, η_dif, η_dis, η_LTP, CD, step, time_evo, dt, τ_evo, T_evo, plotFlag, saveFlag, saveName)
    # redimensionalize
    xn_p    = dimensionalize(xn,       km,      CD)
    xc_p    = dimensionalize(xc,       km,      CD)
    Vy_p    = dimensionalize(Vy,       cm/yr,   CD)
    T_p     = dimensionalize(T,        C,       CD)
    η_e_p   = dimensionalize(η_e,      Pa*s,    CD)
    η_dif_p = dimensionalize(η_dif,    Pa*s,    CD)
    η_dis_p = dimensionalize(η_dis,    Pa*s,    CD)
    η_LTP_p = dimensionalize(η_LTP,    Pa*s,    CD)
    t_p     = dimensionalize(time_evo, yr,      CD)
    if ustrip(dimensionalize(dt,       yr,      CD)) > 1e-3
        dt_p    = dimensionalize(dt,       yr,      CD)
        dtval   = @sprintf("dt = %1.6f yr", dt_p.val)
    else
        dt_p    = dimensionalize(dt,       s,      CD)
        dtval   = @sprintf("dt = %1.6f s",  dt_p.val)
    end
    τ_p     = dimensionalize(τ_evo,    MPa,     CD)
    Tmax_p  = dimensionalize(T_evo,    C,       CD)

    nx      = length(xc_p) 
    η_e_p   = η_e_p .* ones(nx)

    minT    = minimum(T_p)
    maxT    = maximum(T_p)
    Tticks  = [minT, maxT]
    Tlabels = [@sprintf("%.2f", T.val) for T in Tticks]

    # Plotting
    p1         = plot(Vy_p,        xn_p,    xlabel="Vy",        ylabel="x",        legend=:none, tickfontsize = 6, title="Step $(step)")
    p2         = plot(t_p,         τ_p,     xlabel="Time",      ylabel="mean τ",   legend=:none, tickfontsize = 6, title=dtval)
    p3         = plot(its_evo,     errV_evo, xlabel="Iteration", ylabel="Residual", label="V",   tickfontsize = 6, yaxis=:log)
    plot!(its_evo,     errT_evo,                                                    label="T")
    p4         = plot(t_p,         Tmax_p,  xlabel="Time",      ylabel="Tmax",     legend=:none, tickfontsize = 6)
    p5         = plot(T_p,         xc_p,    xlabel="T",         ylabel="x",        legend=:none, tickfontsize = 6, xticks=(Tticks, Tlabels))
    p6         = plot(η_e_p,       xc_p,    xlabel="η",         ylabel="x",        label=:none,  tickfontsize = 6, xaxis=:log)
    plot!(p6,         η_dif_p,     xc_p,    label="Dif")
    plot!(p6,         η_dis_p,     xc_p,    label="Dis")
    plot!(p6,         η_LTP_p,     xc_p,    label="LTP")
    pn         = plot(p1, p2, p3, p4, p5, p6, layout=(2,3));
    if plotFlag 
        display(pn)
    end
    if saveFlag
        savefig(pn, "Outdir/"*saveName*".png")
    end
    @printf("Mean Stress = %.2f MPa\n", τ_p[end].val)
    @printf("Max Temperature = %.2f C\n", Tmax_p[end].val)

    return
end

function uSpace(x1, x2, n)
    # x1:    starting coordinate
    # x2:    ending coordinate
    # n:     number of cells

    # inner region will always be around one quarter of domain
    # innermost grid cells will be domain size divided by 1e4 for 127/128 cells
    L         = x2 - x1
    if mod(n,2) == 0
        n_c   = Int64(ceil(n/8)*2)
        dxcen = L/1e4*128/n
    else
        n_c   = Int64(ceil(n/8)*2-1)
        dxcen = L/1e4*128/(n+1)
    end

    x      = zeros(n+1)
    x[1]   = x1
    L_c    = dxcen*n_c
    L_l    = (L-L_c)/2.0 + dxcen
    n_l    = Int64((n-n_c)/2) + 1
    ind_cl = Int64((n - n_c) / 2 + 1)
    ind_cr = Int64(ind_cl + n_c)

    # left section
    avg    = L_l / n_l
    dxout  = 2*avg  - dxcen
    inc    = (dxout-dxcen) / (n_l-1)
    dx     = dxout
    for i = 2 : ind_cl
        x[i] = x[i-1] + dx
        dx  -= inc
    end

    # make central section
    x[ind_cl] = (x1+x2)/2.0 - n_c/2*dxcen
    for i = ind_cl+1 : ind_cr
        x[i] = x[i-1] + dxcen
    end

    # right section
    for i = ind_cr+1 : n+1
        dx  += inc
        x[i] = x[i-1] + dx
    end
    if x[end] < x2 - 1e-12*L || x[end] > x2 + 1e-12*L
        error("Problem in grid creation: Coordinates don't add up.\n")
    else
        x[end] = x2
    end

    return x
end

function makePrediction(x, x_o, dt, dt_o, limitFlag)
    x_p  = -1.0 .* ones(length(x))
    den  = 0
    if limitFlag
        while minimum(x_p) <= 0 && den < 10
            x_p .= x .+ 1/2^den .* (x .- x_o) .* dt/dt_o
            den += 1
        end
        if den == 10
            return x
        else
            return x_p
        end
    else
        return x .+ (x .- x_o) .* dt/dt_o
    end
end

function findTol(dt, tol0)
    return tol0 * 10 ^ (-2 / (1 + exp(-2 * (log10(dt) - 1))) + 2)
end

function rescaleDimensions(fac, dt, dt_o, dt0, Time, t_end, Vy, T, T_o, T_oo, dT_ref, dT_ref0, P, τxy, τxy_o, ρ, η, η_e, η_v, η_vert, τII, τII_o, τII_oo, τII_reg, τII_vtrue, dτ_crit, εxy, εII, εII_v, εII_nl, εII_dif, εII_dis_g, εII_LTP_g, η_dif, η_dif_new, η_dis, η_dis_new, η_LTP, η_LTP_new, ρ_nodes, ρCp, κ, qx, dT_diff, H, R, gs, Adif, mdif, Qdif, Adis, ndis, Qdis, LTP_A, LTP_En, LTP_Σ, LTP_σb, G, Kb, λ, Cp, dxc, dxn, mindx, η_reg, η_max, λ_num2, CD)
    dt          = dimensionalize(dt,          s,                   CD)
    dt_o        = dimensionalize(dt_o,        s,                   CD)
    dt0         = dimensionalize(dt0,         s,                   CD)
    Time        = dimensionalize(Time,        s,                   CD)
    t_end       = dimensionalize(t_end,       s,                   CD)
    Vy          = dimensionalize(Vy,          cm/yr,               CD)
    T           = dimensionalize(T,           K,                   CD)
    T_o         = dimensionalize(T_o,         K,                   CD)
    T_oo        = dimensionalize(T_oo,        K,                   CD)
    dT_ref      = dimensionalize(dT_ref,      K,                   CD)
    dT_ref0     = dimensionalize(dT_ref0,     K,                   CD)
    P           = dimensionalize(P,           Pa,                  CD)
    τxy         = dimensionalize(τxy,         Pa,                  CD)
    τxy_o       = dimensionalize(τxy_o,       Pa,                  CD)
    ρ           = dimensionalize(ρ,           kg/m^3,              CD)
    η           = dimensionalize(η,           Pas,                 CD)
    η_e         = dimensionalize(η_e,         Pas,                 CD)
    η_v         = dimensionalize(η_v,         Pas,                 CD)
    η_vert      = dimensionalize(η_vert,      Pas,                 CD)
    τII         = dimensionalize(τII,         Pa,                  CD)
    τII_o       = dimensionalize(τII_o,       Pa,                  CD)
    τII_oo      = dimensionalize(τII_oo,      Pa,                  CD)
    τII_reg     = dimensionalize(τII_reg,     Pa,                  CD)
    τII_vtrue   = dimensionalize(τII_vtrue,   Pa,                  CD)
    dτ_crit     = dimensionalize(dτ_crit,     Pa,                  CD)
    εxy         = dimensionalize(εxy,         s^-1,                CD)
    εII         = dimensionalize(εII,         s^-1,                CD)
    εII_v       = dimensionalize(εII_v,       s^-1,                CD)
    εII_nl      = dimensionalize(εII_nl,      s^-1,                CD)
    εII_dif     = dimensionalize(εII_dif,     s^-1,                CD)
    εII_dis_g   = dimensionalize(εII_dis_g,   s^-1,                CD)
    εII_LTP_g   = dimensionalize(εII_LTP_g,   s^-1,                CD)
    η_dif       = dimensionalize(η_dif,       Pas,                 CD)
    η_dif_new   = dimensionalize(η_dif_new,   Pas,                 CD)
    η_dis       = dimensionalize(η_dis,       Pas,                 CD)
    η_dis_new   = dimensionalize(η_dis_new,   Pas,                 CD)
    η_LTP       = dimensionalize(η_LTP,       Pas,                 CD)
    η_LTP_new   = dimensionalize(η_LTP_new,   Pas,                 CD)
    ρ_nodes     = dimensionalize(ρ_nodes,     kg/m^3,              CD)
    ρCp         = dimensionalize(ρCp,         kg/(m*K*s^2),        CD)
    κ           = dimensionalize(κ,           m^2/s,               CD)
    qx          = dimensionalize(qx,          kg/s^3,              CD)
    dT_diff     = dimensionalize(dT_diff,     kg/(s^3*m),          CD)
    H           = dimensionalize(H,           Pa/s,                CD)
    R           = dimensionalize(R,           J/(mol*K),           CD)
    gs          = dimensionalize(gs,          μm,                  CD)
    Adif        = dimensionalize(Adif,        MPa^-1*μm^mdif*s^-1, CD)
    Qdif        = dimensionalize(Qdif,        J/mol,               CD)
    Adis        = dimensionalize(Adis,        MPa^-ndis/s,         CD)
    Qdis        = dimensionalize(Qdis,        J/mol,               CD)
    LTP_A       = dimensionalize(LTP_A,       s^-1,                CD)
    LTP_En      = dimensionalize(LTP_En,      J/mol,               CD)
    LTP_Σ       = dimensionalize(LTP_Σ,       MPa,                 CD)
    LTP_σb      = dimensionalize(LTP_σb,      MPa,                 CD)
    G           = dimensionalize(G,           Pa,                  CD)
    Kb          = dimensionalize(Kb,          Pa,                  CD)
    λ           = dimensionalize(λ,           J/(s*m*K),           CD)
    Cp          = dimensionalize(Cp,          J/(kg*K),            CD)
    dxc         = dimensionalize(dxc,         m,                   CD)
    dxn         = dimensionalize(dxn,         m,                   CD)
    mindx       = dimensionalize(mindx,       m,                   CD)
    η_reg       = dimensionalize(η_reg,       Pas,                 CD)
    η_max       = dimensionalize(η_max,       Pas,                 CD)
    λ_num2      = dimensionalize(λ_num2,      m^2,                 CD)
    
    CD2         = SI_units(length=CD.Length, temperature=CD.temperature, stress=CD.stress, viscosity=CD.viscosity/fac);
    
    dt          = nondimensionalize(dt,        CD2)
    dt_o        = nondimensionalize(dt_o,      CD2)
    dt0         = nondimensionalize(dt0,       CD2)
    Time        = nondimensionalize(Time,      CD2)
    t_end       = nondimensionalize(t_end,     CD2)
    Vy          = nondimensionalize(Vy,        CD2)
    T           = nondimensionalize(T,         CD2)
    T_o         = nondimensionalize(T_o,       CD2)
    T_oo        = nondimensionalize(T_oo,      CD2)
    dT_ref      = nondimensionalize(dT_ref,    CD2)
    dT_ref0     = nondimensionalize(dT_ref0,   CD2)
    P           = nondimensionalize(P,         CD2)
    τxy         = nondimensionalize(τxy,       CD2)
    τxy_o       = nondimensionalize(τxy_o,     CD2)
    ρ           = nondimensionalize(ρ,         CD2)
    η           = nondimensionalize(η,         CD2)
    η_e         = nondimensionalize(η_e,       CD2)
    η_v         = nondimensionalize(η_v,       CD2)
    η_vert      = nondimensionalize(η_vert,    CD2)
    τII         = nondimensionalize(τII,       CD2)
    τII_o       = nondimensionalize(τII_o,     CD2)
    τII_oo      = nondimensionalize(τII_oo,    CD2)
    τII_reg     = nondimensionalize(τII_reg,   CD2)
    τII_vtrue   = nondimensionalize(τII_vtrue, CD2)
    dτ_crit     = nondimensionalize(dτ_crit,   CD2)
    εxy         = nondimensionalize(εxy,       CD2)
    εII         = nondimensionalize(εII,       CD2)
    εII_v       = nondimensionalize(εII_v,     CD2)
    εII_nl      = nondimensionalize(εII_nl,    CD2)
    εII_dif     = nondimensionalize(εII_dif,   CD2)
    εII_dis_g   = nondimensionalize(εII_dis_g, CD2)
    εII_LTP_g   = nondimensionalize(εII_LTP_g, CD2)
    η_dif       = nondimensionalize(η_dif,     CD2)
    η_dif_new   = nondimensionalize(η_dif_new, CD2)
    η_dis       = nondimensionalize(η_dis,     CD2)
    η_dis_new   = nondimensionalize(η_dis_new, CD2)
    η_LTP       = nondimensionalize(η_LTP,     CD2)
    η_LTP_new   = nondimensionalize(η_LTP_new, CD2)
    ρ_nodes     = nondimensionalize(ρ_nodes,   CD2)
    ρCp         = nondimensionalize(ρCp,       CD2)
    κ           = nondimensionalize(κ,         CD2)
    qx          = nondimensionalize(qx,        CD2)
    dT_diff     = nondimensionalize(dT_diff,   CD2)
    H           = nondimensionalize(H,         CD2)
    R           = nondimensionalize(R,         CD2)
    gs          = nondimensionalize(gs,        CD2)
    Adif        = nondimensionalize(Adif,      CD2)
    Qdif        = nondimensionalize(Qdif,      CD2)
    Adis        = nondimensionalize(Adis,      CD2)
    Qdis        = nondimensionalize(Qdis,      CD2)
    LTP_A       = nondimensionalize(LTP_A,     CD2)
    LTP_En      = nondimensionalize(LTP_En,    CD2)
    LTP_Σ       = nondimensionalize(LTP_Σ,     CD2)
    LTP_σb      = nondimensionalize(LTP_σb,    CD2)
    G           = nondimensionalize(G,         CD2)
    Kb          = nondimensionalize(Kb,        CD2)
    λ           = nondimensionalize(λ,         CD2)
    Cp          = nondimensionalize(Cp,        CD2)
    dxc         = nondimensionalize(dxc,       CD2)
    dxn         = nondimensionalize(dxn,       CD2)
    mindx       = nondimensionalize(mindx,     CD2)
    η_reg       = nondimensionalize(η_reg,     CD2)
    η_max       = nondimensionalize(η_max,     CD2)
    λ_num2      = nondimensionalize(λ_num2,    CD2)

    return dt, dt_o, dt0, Time, t_end, Vy, T, T_o, T_oo, dT_ref, dT_ref0, P, τxy, τxy_o, ρ, η, η_e, η_v, η_vert, τII, τII_o, τII_oo, τII_reg, τII_vtrue, dτ_crit, εxy, εII, εII_v, εII_nl, εII_dif, εII_dis_g, εII_LTP_g, η_dif, η_dif_new, η_dis, η_dis_new, η_LTP, η_LTP_new, ρ_nodes, ρCp, κ, qx, dT_diff, H, R, gs, Adif, mdif, Qdif, Adis, ndis, Qdis, LTP_A, LTP_En, LTP_Σ, LTP_σb, G, Kb, λ, Cp, dxc, dxn, mindx, η_reg, η_max, λ_num2, CD2
end

function rescaleEvo(Time, dt, τ, T, T2, Vy, VyProf, TProf, η, η2, ηv, ηv2, H, H2, H_dif, H_dif2, H_dis, H_dis2, H_LTP, H_LTP2, ε, εv, CD, CD_new)
    Time       = nondimensionalize(dimensionalize(Time,       s,       CD), CD_new)
    dt         = nondimensionalize(dimensionalize(dt,         s,       CD), CD_new)
    τ          = nondimensionalize(dimensionalize(τ,          MPa,     CD), CD_new)
    T          = nondimensionalize(dimensionalize(T,          K,       CD), CD_new)
    T2         = nondimensionalize(dimensionalize(T2,         K,       CD), CD_new)
    Vy         = nondimensionalize(dimensionalize(Vy,         cm/yr,   CD), CD_new)
    η          = nondimensionalize(dimensionalize(η,          Pas,     CD), CD_new)
    η2         = nondimensionalize(dimensionalize(η2,         Pas,     CD), CD_new)
    ηv         = nondimensionalize(dimensionalize(ηv,         Pas,     CD), CD_new)
    ηv2        = nondimensionalize(dimensionalize(ηv2,        Pas,     CD), CD_new)
    H          = nondimensionalize(dimensionalize(H,          Pa/s,    CD), CD_new)
    H2         = nondimensionalize(dimensionalize(H2,         Pa/s,    CD), CD_new)
    H_dif      = nondimensionalize(dimensionalize(H_dif,      Pa/s,    CD), CD_new)
    H_dif2     = nondimensionalize(dimensionalize(H_dif2,     Pa/s,    CD), CD_new)
    H_dis      = nondimensionalize(dimensionalize(H_dis,      Pa/s,    CD), CD_new)
    H_dis2     = nondimensionalize(dimensionalize(H_dis2,     Pa/s,    CD), CD_new)
    H_LTP      = nondimensionalize(dimensionalize(H_LTP,      Pa/s,    CD), CD_new)
    H_LTP2     = nondimensionalize(dimensionalize(H_LTP2,     Pa/s,    CD), CD_new)
    ε          = nondimensionalize(dimensionalize(ε,          s^-1,    CD), CD_new)
    εv         = nondimensionalize(dimensionalize(εv,         s^-1,    CD), CD_new)
    for i = eachindex(VyProf)
        VyProf[i] = nondimensionalize(dimensionalize(VyProf[i],     cm/yr,   CD), CD_new)
        TProf[i]  = nondimensionalize(dimensionalize(TProf[i],      K,       CD), CD_new)
    end
    return Time, dt, τ, T, T2, Vy, VyProf, TProf, η, η2, ηv, ηv2, H, H2, H_dif, H_dif2, H_dis, H_dis2, H_LTP, H_LTP2, ε, εv
end

function PDF(x, x0, FWHM, ymax) 
    σ = FWHM/(2*sqrt(2*log(2))) 
    return ymax * exp.(-0.5*((x .- x0)./σ).^2)
end