export updateTimestep, trackStuff!, AdjustDamp, raiseTol, SH_1D_plot, uSpace, makePrediction, findTol, rescaleDimensions, rescaleEvo, PDF

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