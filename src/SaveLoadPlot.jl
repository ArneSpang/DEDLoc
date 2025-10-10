export initEvo, trackProperties!, saveEvo, saveFullField, saveRestart, loadRestart
export SH_2D_plot

# initialize vectors used for tracking
function initEvo(T, τII, Vx, η_v, εII)
    return [0.0], Array{Float64}(undef, 0), [maximum(T)], [mean(T)], [maximum(τII)], [mean(τII)], [maximum(abs.(Vx))], [minimum(η_v)], [maximum(εII)], 
           Array{Int64}(undef, 0), Array{Int64}(undef, 0), Array{Float64}(undef, 0), Array{Int64}(undef, 0), Array{Int64}(undef, 0)
end

# track properties for evolution file
function trackProperties!(Time, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, dt, T, τII, Vx, η_v, εII, H, iter, iter_re, err, nx)
    Time   += dt
    n       = Int64(nx/2)
    push!(t_evo,     Time)
    push!(dt_evo,    dt)
    push!(Tmax_evo,  maximum(T))
    push!(Tmean_evo, mean(T))
    push!(τmax_evo,  maximum(τII))
    push!(τ_evo,     mean(τII))
    push!(Vxmax_evo, maximum(abs.(Vx)))
    push!(ηvmin_evo, minimum(η_v))
    push!(εmax_evo,  maximum(εII))
    push!(iter_evo,  iter)
    push!(fiter_evo, iter + iter_re)
    push!(res_evo,   err)
    push!(ltip_evo,  argmax(H[1:n,:])[1])
    push!(rtip_evo,  argmax(H[n+1:end,:])[1] + n)

    return Time
end


# save evolution file
function saveEvo(saveFlag, nanFlag, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD, saveName, xc, ind_cen)
    if saveFlag && !nanFlag
        t_evo_s      = ustrip(dimensionalize(t_evo,       yr,    CD))
        dt_evo_s     = ustrip(dimensionalize(dt_evo,      s,     CD))
        Tmax_evo_s   = ustrip(dimensionalize(Tmax_evo,    C,     CD))
        Tmean_evo_s  = ustrip(dimensionalize(Tmean_evo,   C,     CD))
        τmax_evo_s   = ustrip(dimensionalize(τmax_evo,    MPa,   CD))
        τ_evo_s      = ustrip(dimensionalize(τ_evo,       MPa,   CD))
        Vxmax_evo_s  = ustrip(dimensionalize(Vxmax_evo,   cm/yr, CD))
        ηvmin_evo_s  = ustrip(dimensionalize(ηvmin_evo,   Pas,   CD))
        εmax_evo_s   = ustrip(dimensionalize(εmax_evo,    s^-1,  CD))
        
        jldsave(saveName*".jld2", x_vec=xc[:,ind_cen], t=t_evo_s, dt=dt_evo_s, T_max=Tmax_evo_s, T_mean=Tmean_evo_s, τ_max=τmax_evo_s, τ_mean=τ_evo_s, Vx_max=Vxmax_evo_s, ηv_min=ηvmin_evo_s, ε_max=εmax_evo_s, iter=iter_evo, iterA=fiter_evo, err=res_evo, ltip=ltip_evo, rtip=rtip_evo)
    end
    return nothing
end

# save a selection of full fields
function saveFullField(saveAllFlag, nout, saveName, iter_dt, Time, τII, εII, Vx, Vy, η_v, T, P, ρ, H, H_dif, H_dis, H_LTP, CD)
    if (saveAllFlag && mod(iter_dt, nout) == 0)
        jldsave(saveName*@sprintf("_%04d", iter_dt)*".jld2", 
                t     = ustrip(dimensionalize(Time,         yr,     CD)), 
                τ     = ustrip(dimensionalize(Array(τII),   Pa,     CD)), 
                ε     = ustrip(dimensionalize(Array(εII),   s^-1,   CD)), 
                Vx    = ustrip(dimensionalize(Array(Vx),    m/s,    CD)), 
                Vy    = ustrip(dimensionalize(Array(Vy),    m/s,    CD)), 
                ηv    = ustrip(dimensionalize(Array(η_v),   Pas,    CD)),
                T     = ustrip(dimensionalize(Array(T),     C,      CD)),
                P     = ustrip(dimensionalize(Array(P),     Pa,     CD)),
                ρ     = ustrip(dimensionalize(Array(ρ),     kg/m^3, CD)),
                H     = ustrip(dimensionalize(Array(H),     Pa/s,   CD)),
                H_dif = ustrip(dimensionalize(Array(H_dif), Pa/s,   CD)), 
                H_dis = ustrip(dimensionalize(Array(H_dis), Pa/s,   CD)),
                H_LTP = ustrip(dimensionalize(Array(H_LTP), Pa/s,   CD)))
    end
    return nothing
end

# save a restart database
function saveRestart(iter_dt, nRestart, nanFlag, saveName, Time, dt, Vx, Vy, T, T_o, P, P_o, ρ, η, η_dif, η_dis, η_LTP, τxx, τyy, τxy, τII, τII_o, VxUp, VyUp, PUp, TUp, err_evo, its_evo, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD)
    if mod(iter_dt, nRestart) == 0 && !nanFlag
        jldsave(saveName*"_restart.jld2";
                Time = Time, iter_dt = iter_dt, dt = dt,
                Vx = Array(Vx), Vy = Array(Vy),
                T = Array(T), T_o = Array(T_o), P = Array(P), P_o = Array(P_o), ρ = Array(ρ),
                η = Array(η), η_dif = Array(η_dif), η_dis = Array(η_dis), η_LTP = Array(η_LTP), 
                τxx = Array(τxx), τyy = Array(τyy), τxy = Array(τxy), τII = Array(τII), τII_o = Array(τII_o), 
                VxUp = Array(VxUp), VyUp = Array(VyUp), PUp = Array(PUp), TUp = Array(TUp),
                err_evo = err_evo, its_evo = its_evo, t_evo = t_evo, dt_evo = dt_evo, Tmax_evo = Tmax_evo, Tmean_evo = Tmean_evo, 
                τmax_evo = τmax_evo, τ_evo = τ_evo, Vxmax_evo = Vxmax_evo, ηvmin_evo = ηvmin_evo, εmax_evo = εmax_evo, 
                iter_evo = iter_evo, fiter_evo = fiter_evo, res_evo = res_evo, ltip_evo = ltip_evo, rtip_evo = rtip_evo, CD = CD)
        @printf("Saved restart database.\n")
    end
    return nothing
end

# load a restart database and return everything as Cuda Arrays
function loadRestart(name)
    @load name Time iter_dt dt  Vx Vy T T_o P P_o ρ η η_dif η_dis η_LTP τxx τyy τxy τII τII_o VxUp VyUp PUp TUp err_evo dt_evo its_evo t_evo Tmax_evo Tmean_evo τmax_evo τ_evo Vxmax_evo ηvmin_evo εmax_evo iter_evo fiter_evo res_evo ltip_evo rtip_evo CD

    return Time, iter_dt, dt, 
           Data.Array(Vx), Data.Array(Vy),
           Data.Array(T), Data.Array(T_o), Data.Array(P), Data.Array(P_o), Data.Array(ρ),
           Data.Array(η), Data.Array(η_dif), Data.Array(η_dis), Data.Array(η_LTP),
           Data.Array(τxx), Data.Array(τyy), Data.Array(τxy), Data.Array(τII), Data.Array(τII_o), 
           Data.Array(VxUp), Data.Array(VyUp), Data.Array(PUp), Data.Array(TUp),
           err_evo, its_evo, t_evo, dt_evo, Tmax_evo, Tmean_evo, τmax_evo, τ_evo, Vxmax_evo, ηvmin_evo, εmax_evo, iter_evo, fiter_evo, res_evo, ltip_evo, rtip_evo, CD
end

# checks which deformation mechanism is dominant
@parallel_indices (ix,iy) function domMech!(dom::Data.Array, ε_ela::Data.Array, ε_dif::Data.Array, ε_dis::Data.Array, ε_LTP::Data.Array)
    if ε_ela[ix,iy] > ε_dif[ix,iy] && ε_ela[ix,iy] > ε_dis[ix,iy] && ε_ela[ix,iy] > ε_LTP[ix,iy]
        dom[ix,iy] = 1
    elseif ε_dif[ix,iy] > ε_ela[ix,iy] && ε_dif[ix,iy] > ε_dis[ix,iy] && ε_dif[ix,iy] > ε_LTP[ix,iy]
        dom[ix,iy] = 2
    elseif ε_dis[ix,iy] > ε_ela[ix,iy] && ε_dis[ix,iy] > ε_dif[ix,iy] && ε_dis[ix,iy] > ε_LTP[ix,iy]
        dom[ix,iy] = 3
    elseif ε_LTP[ix,iy] > ε_ela[ix,iy] && ε_LTP[ix,iy] > ε_dif[ix,iy] && ε_LTP[ix,iy] > ε_dis[ix,iy]
        dom[ix,iy] = 4
    else
        dom[ix,iy] = 0
    end
    return nothing
end

# makes plot
function SH_2D_plot(xn, yn, xc, yc, dt, Vx, Vy, T, P, τII, t_evo, CD, step, its_evo, ErrVx_evo, ErrVy_evo, ErrT_evo, ErrP_evo, dom, outDirName)
    
    # redimensionalize
    xn_p    = ustrip(dimensionalize(xn,                   km,      CD))
    yn_p    = ustrip(dimensionalize(yn,                   km,      CD))
    xc_p    = ustrip(dimensionalize(xc[2:end-1,:],        km,      CD))
    yc_p    = ustrip(dimensionalize(yc[:,2:end-1],        km,      CD))
    dt_p    = ustrip(dimensionalize(dt,                   s,       CD))
    Vx_p    = ustrip(dimensionalize(Array(Vx)[:,2:end-1], cm/yr,   CD))
    Vy_p    = ustrip(dimensionalize(Array(Vy)[2:end-1,:], cm/yr,   CD))
    T_p     = ustrip(dimensionalize(Array(T),             C,       CD))
    P_p     = ustrip(dimensionalize(Array(P),             MPa,     CD))
    τ_p     = ustrip(dimensionalize(Array(τII),           MPa,     CD))
    t_e_p   = ustrip(dimensionalize(t_evo,                yr,      CD))

    Tmax_p  = maximum(T_p)

    elapsed = @sprintf("Time: %.3f kyr", t_e_p[end] / 1000)

    ErrP_evo .= max.(ErrP_evo, 1e-15)

    p1      = heatmap(xn_p[:,1], yc_p[1,:], Vx_p', xlabel="X [km]", ylabel="Y [km]", title="Vx [cm/yr] - Step $(step)", framestyle=:box)
    p2      = heatmap(xc_p[:,1], yc_p[1,:], τ_p',  xlabel="X [km]", ylabel="Y[km]", clim=(min(0, minimum(τ_p)), max(2000, maximum(τ_p))), title="τII [MPa] - "*displayTimestep(dt_p), framestyle=:box)
    p3      = plot(xlabel="Iteration", ylabel="Residual", title=elapsed, yaxis=:log, legend_position=:bottomleft, framestyle=:box)
    plot!(p3, its_evo, ErrVx_evo, linestyle=:dash, label="Vx")
    plot!(p3, its_evo, ErrVy_evo, linestyle=:dash, label="Vy")
    plot!(p3, its_evo, ErrT_evo,  linestyle=:dash, label="T")
    plot!(p3, its_evo, ErrP_evo,  linestyle=:dash, label="P")
    p4      = heatmap(xc_p[:,1], yc_p[1,:], Array(dom)',  xlabel="X [km]", ylabel="Y[km]", title="Dominant mechanism", clim=(0,4))
    savefig(plot(p1, p2, p3, p4, layout=(2,2)), @sprintf("%s/%06d.png", outDirName, step))

    @printf("Mean Stress = %.2f MPa\n",   mean(τ_p))
    @printf("Max Temperature = %.2f C\n", Tmax_p)

    return nothing
end