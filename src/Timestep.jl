# displays time step in appropriate unit (for plotting)
function displayTimestep(dt_dim)
    if dt_dim > 3600*24*365.25
        return @sprintf("dt = %1.3f y", dt_dim/(3600*24*365.25))
    elseif dt_dim > 3600*24
        return @sprintf("dt = %1.3f d", dt_dim/(3600*24))
    elseif dt_dim > 3600
        return @sprintf("dt = %1.3f h", dt_dim/3600)
    else
        return @sprintf("dt = %1.3f s", dt_dim)
    end
end

# rescale quantities that contain time in their units
function rescaleTime!(dt, dt_o, dt0, Time, t_end, Vx, Vx_o, Vy, Vy_o, VxTop, VxBot, VxLeft, VxRight, VyTop, VyBot, VyLeft, VyRight, V0, ρ, ρ_o, η, η_o, η_e, η_v, η_ml, η_dif, η_dif_o, η_dif_new, η_dis, η_dis_o, η_dis_new, η_LTP, η_LTP_o, η_LTP_new, εxx, εyy, εxy, εxy_cen, εxx_f, εyy_f, εxy_f, εII, εII_v, εII_nl, εII_dif, εII_dis, εII_dis_g, εII_LTP, εII_LTP_g, εxx_elaOld, εyy_elaOld, εxy_elaOld, κ, qx, qy, dT_diff, H, Pre_dif, Pre_dis, ndis, ALTP, λ, Cp, η_reg, η_max, CD, fac)
    dt           *= fac           
    dt_o         *= fac           
    dt0          *= fac           
    Time         *= fac           
    t_end        *= fac           
    Vx          ./= fac 
    Vx_o        ./= fac       
    Vy          ./= fac
    Vy_o        ./= fac
    VxTop        /= fac
    VxBot        /= fac
    VxLeft       /= fac
    VxRight      /= fac
    VyTop        /= fac
    VyBot        /= fac
    VyLeft       /= fac
    VyRight      /= fac
    V0           /= fac
    ρ           .*= fac^2
    ρ_o         .*= fac^2       
    η           .*= fac
    η_o         .*= fac       
    η_e         .*= fac       
    η_v         .*= fac       
    η_ml        .*= fac       
    η_dif       .*= fac
    η_dif_o     .*= fac       
    η_dif_new   .*= fac        
    η_dis       .*= fac
    η_dis_o     .*= fac       
    η_dis_new   .*= fac        
    η_LTP       .*= fac
    η_LTP_o     .*= fac       
    η_LTP_new   .*= fac        
    εxx         ./= fac       
    εyy         ./= fac       
    εxy         ./= fac       
    εxy_cen     ./= fac       
    εxx_f       ./= fac        
    εyy_f       ./= fac        
    εxy_f       ./= fac        
    εII         ./= fac        
    εII_v       ./= fac        
    εII_nl      ./= fac        
    εII_dif     ./= fac        
    εII_dis     ./= fac        
    εII_dis_g   ./= fac        
    εII_LTP     ./= fac        
    εII_LTP_g   ./= fac        
    εxx_elaOld  ./= fac        
    εyy_elaOld  ./= fac        
    εxy_elaOld  ./= fac        
    κ           ./= fac       
    qx          ./= fac      
    qy          ./= fac      
    dT_diff     ./= fac  
    H           ./= fac        
    Pre_dif      *= fac
    Pre_dis      *= fac^(1/ndis)
    ALTP         /= fac      
    λ            /= fac   
    Cp           /= fac^2    
    η_reg        *= fac         
    η_max        *= fac         

    CD2           = SI_units(length=CD.Length, temperature=CD.temperature, stress=CD.stress, viscosity=CD.viscosity/fac)
    
    return CD2, dt, dt_o, dt0, Time, t_end, VxTop, VxBot, VxLeft, VxRight, VyTop, VyBot, VyLeft, VyRight, V0, Pre_dif, Pre_dis, ALTP, λ, Cp, η_reg, η_max
end

# rescale quantities that contain time in their units (outdated bulky version)
function rescaleTimeLong(fac, dt, dt_o, dt0, Time, t_end, Vx, Vy, Vx_o, Vy_o, VxTop, VxBot, V0, ρ, ρ_o, η, η_o, η_e, η_v, η_ml, η_dif, η_dif_o, η_dif_new, η_dis, η_dis_o, η_dis_new, η_LTP, η_LTP_o, η_LTP_new, εxx, εyy, εxy, εxy_cen, εxx_f, εyy_f, εxy_f, εII, εII_v, εII_nl, εII_dif, εII_dis, εII_dis_g, εII_LTP, εII_LTP_g, εxx_elaOld, εyy_elaOld, εxy_elaOld, κ, qx, qy, dT_diff, H, Pre_dif, mdif, Pre_dis, ndis, A_LTP, λ, Cp, η_reg, η_max, CD)
    _dt          = dimensionalize(dt,                 s,                   CD)
    _dt_o        = dimensionalize(dt_o,               s,                   CD)
    _dt0         = dimensionalize(dt0,                s,                   CD)
    _Time        = dimensionalize(Time,               s,                   CD)
    _t_end       = dimensionalize(t_end,              s,                   CD)
    _Vx          = dimensionalize(Array(Vx),          cm/yr,               CD)
    _Vx_o        = dimensionalize(Array(Vx_o),        cm/yr,               CD)
    _Vy          = dimensionalize(Array(Vy),          cm/yr,               CD)
    _Vy_o        = dimensionalize(Array(Vy_o),        cm/yr,               CD)
    _VxTop       = dimensionalize(VxTop,              cm/yr,               CD)
    _VxBot       = dimensionalize(VxBot,              cm/yr,               CD)
    _V0          = dimensionalize(V0,                 cm/yr,               CD)
    _ρ           = dimensionalize(Array(ρ),           kg/m^3,              CD)
    _ρ_o         = dimensionalize(Array(ρ_o),         kg/m^3,              CD)
    _η           = dimensionalize(Array(η),           Pas,                 CD)
    _η_o         = dimensionalize(Array(η_o),         Pas,                 CD)
    _η_e         = dimensionalize(Array(η_e),         Pas,                 CD)
    _η_v         = dimensionalize(Array(η_v),         Pas,                 CD)
    _η_ml        = dimensionalize(Array(η_ml),        Pas,                 CD)
    _η_dif       = dimensionalize(Array(η_dif),       Pas,                 CD)
    _η_dif_o     = dimensionalize(Array(η_dif_o),     Pas,                 CD)
    _η_dif_new   = dimensionalize(Array(η_dif_new),   Pas,                 CD)
    _η_dis       = dimensionalize(Array(η_dis),       Pas,                 CD)
    _η_dis_o     = dimensionalize(Array(η_dis_o),     Pas,                 CD)
    _η_dis_new   = dimensionalize(Array(η_dis_new),   Pas,                 CD)
    _η_LTP       = dimensionalize(Array(η_LTP),       Pas,                 CD)
    _η_LTP_o     = dimensionalize(Array(η_LTP_o),     Pas,                 CD)
    _η_LTP_new   = dimensionalize(Array(η_LTP_new),   Pas,                 CD)
    _εxx         = dimensionalize(Array(εxx),         s^-1,                CD)
    _εyy         = dimensionalize(Array(εyy),         s^-1,                CD)
    _εxy         = dimensionalize(Array(εxy),         s^-1,                CD)
    _εxy_cen     = dimensionalize(Array(εxy_cen),     s^-1,                CD)
    _εxx_f       = dimensionalize(Array(εxx_f),       s^-1,                CD)
    _εyy_f       = dimensionalize(Array(εyy_f),       s^-1,                CD)
    _εxy_f       = dimensionalize(Array(εxy_f),       s^-1,                CD)
    _εII         = dimensionalize(Array(εII),         s^-1,                CD)
    _εII_v       = dimensionalize(Array(εII_v),       s^-1,                CD)
    _εII_nl      = dimensionalize(Array(εII_nl),      s^-1,                CD)
    _εII_dif     = dimensionalize(Array(εII_dif),     s^-1,                CD)
    _εII_dis     = dimensionalize(Array(εII_dis),     s^-1,                CD)
    _εII_dis_g   = dimensionalize(Array(εII_dis_g),   s^-1,                CD)
    _εII_LTP     = dimensionalize(Array(εII_LTP),     s^-1,                CD)
    _εII_LTP_g   = dimensionalize(Array(εII_LTP_g),   s^-1,                CD)
    _εxx_elaOld  = dimensionalize(Array(εxx_elaOld),  s^-1,                CD)
    _εyy_elaOld  = dimensionalize(Array(εyy_elaOld),  s^-1,                CD)
    _εxy_elaOld  = dimensionalize(Array(εxy_elaOld),  s^-1,                CD)
    _κ           = dimensionalize(Array(κ),           m^2/s,               CD)
    _qx          = dimensionalize(Array(qx),          kg/s^3,              CD)
    _qy          = dimensionalize(Array(qy),          kg/s^3,              CD)
    _dT_diff     = dimensionalize(Array(dT_diff),     kg/(s^3*m),          CD)
    _H           = dimensionalize(Array(H),           Pa/s,                CD)
    _Pre_dif     = dimensionalize(Array(Pre_dif),     MPa*μm^(-mdif)*s,    CD)
    _Pre_dis     = dimensionalize(Array(Pre_dis),     MPa*s^(1/ndis),      CD)
    _A_LTP       = dimensionalize(Array(A_LTP),       s^-1,                CD)
    _λ           = dimensionalize(λ,                  J/(s*m*K),           CD)
    _Cp          = dimensionalize(Cp,                 J/(kg*K),            CD)
    _η_reg       = dimensionalize(η_reg,              Pas,                 CD)
    _η_max       = dimensionalize(η_max,              Pas,                 CD)

    CD2          = SI_units(length=CD.Length, temperature=CD.temperature, stress=CD.stress, viscosity=CD.viscosity/fac);

    dt           =            nondimensionalize(_dt,          CD2)
    dt_o         =            nondimensionalize(_dt_o,        CD2)
    dt0          =            nondimensionalize(_dt0,         CD2)
    Time         =            nondimensionalize(_Time,        CD2)
    t_end        =            nondimensionalize(_t_end,       CD2)
    Vx           = Data.Array(nondimensionalize(_Vx,          CD2))
    Vx_o         = Data.Array(nondimensionalize(_Vx_o,        CD2))
    Vy           = Data.Array(nondimensionalize(_Vy,          CD2))
    Vy_o         = Data.Array(nondimensionalize(_Vy_o,        CD2))
    VxTop        =            nondimensionalize(_VxTop,       CD2)
    VxBot        =            nondimensionalize(_VxBot,       CD2)
    V0           =            nondimensionalize(_V0,          CD2)
    ρ            = Data.Array(nondimensionalize(_ρ,           CD2))
    ρ_o          = Data.Array(nondimensionalize(_ρ_o,         CD2))
    η            = Data.Array(nondimensionalize(_η,           CD2))
    η_o          = Data.Array(nondimensionalize(_η_o,         CD2))
    η_e          = Data.Array(nondimensionalize(_η_e,         CD2))
    η_v          = Data.Array(nondimensionalize(_η_v,         CD2))
    η_ml         = Data.Array(nondimensionalize(_η_ml,        CD2))
    η_dif        = Data.Array(nondimensionalize(_η_dif,       CD2))
    η_dif_o      = Data.Array(nondimensionalize(_η_dif_o,     CD2))
    η_dif_new    = Data.Array(nondimensionalize(_η_dif_new,   CD2))
    η_dis        = Data.Array(nondimensionalize(_η_dis,       CD2))
    η_dis_o      = Data.Array(nondimensionalize(_η_dis_o,     CD2))
    η_dis_new    = Data.Array(nondimensionalize(_η_dis_new,   CD2))
    η_LTP        = Data.Array(nondimensionalize(_η_LTP,       CD2))
    η_LTP_o      = Data.Array(nondimensionalize(_η_LTP_o,     CD2))
    η_LTP_new    = Data.Array(nondimensionalize(_η_LTP_new,   CD2))
    εxx          = Data.Array(nondimensionalize(_εxx,         CD2))
    εyy          = Data.Array(nondimensionalize(_εyy,         CD2))
    εxy          = Data.Array(nondimensionalize(_εxy,         CD2))
    εxy_cen      = Data.Array(nondimensionalize(_εxy_cen,     CD2))
    εxx_f        = Data.Array(nondimensionalize(_εxx_f,       CD2))
    εyy_f        = Data.Array(nondimensionalize(_εyy_f,       CD2))
    εxy_f        = Data.Array(nondimensionalize(_εxy_f,       CD2))
    εII          = Data.Array(nondimensionalize(_εII,         CD2))
    εII_v        = Data.Array(nondimensionalize(_εII_v,       CD2))
    εII_nl       = Data.Array(nondimensionalize(_εII_nl,      CD2))
    εII_dif      = Data.Array(nondimensionalize(_εII_dif,     CD2))
    εII_dis      = Data.Array(nondimensionalize(_εII_dis,     CD2))
    εII_dis_g    = Data.Array(nondimensionalize(_εII_dis_g,   CD2))
    εII_LTP      = Data.Array(nondimensionalize(_εII_LTP,     CD2))
    εII_LTP_g    = Data.Array(nondimensionalize(_εII_LTP_g,   CD2))
    εxx_elaOld   = Data.Array(nondimensionalize(_εxx_elaOld,  CD2))
    εyy_elaOld   = Data.Array(nondimensionalize(_εyy_elaOld,  CD2))
    εxy_elaOld   = Data.Array(nondimensionalize(_εxy_elaOld,  CD2))
    κ            = Data.Array(nondimensionalize(_κ,           CD2))
    qx           = Data.Array(nondimensionalize(_qx,          CD2))
    qy           = Data.Array(nondimensionalize(_qy,          CD2))
    dT_diff      = Data.Array(nondimensionalize(_dT_diff,     CD2))
    H            = Data.Array(nondimensionalize(_H,           CD2))
    Pre_dif      = Data.Array(nondimensionalize(_Pre_dif,     CD2))
    Pre_dis      = Data.Array(nondimensionalize(_Pre_dis,     CD2))
    A_LTP        = Data.Array(nondimensionalize(_A_LTP,       CD2))
    λ            =            nondimensionalize(_λ,           CD2)
    Cp           =            nondimensionalize(_Cp,          CD2)
    η_reg        =            nondimensionalize(_η_reg,       CD2)
    η_max        =            nondimensionalize(_η_max,       CD2)
    
    return dt, dt_o, dt0, Time, t_end, Vx, Vx_o, Vy, Vy_o, VxTop, VxBot, V0, ρ, ρ_o, η, η_o, η_e, η_v, η_ml, η_dif, η_dif_o, η_dif_new, η_dis, η_dis_o, η_dis_new, η_LTP, η_LTP_o, η_LTP_new, εxx, εyy, εxy, εxy_cen, εxx_f, εyy_f, εxy_f, εII, εII_v, εII_nl, εII_dif, εII_dis, εII_dis_g, εII_LTP, εII_LTP_g, εxx_elaOld, εyy_elaOld, εxy_elaOld, κ, qx, qy, dT_diff, H, Pre_dif, Pre_dis, A_LTP, λ, Cp, η_reg, η_max, CD2
end

# rescale tracking vectors that contain time in their units
function rescaleEvoTime!(t_evo, dt_evo, Vxmax_evo, ηvmin_evo, εmax_evo, fac)
    t_evo     .*= fac
    dt_evo    .*= fac
    Vxmax_evo ./= fac
    ηvmin_evo .*= fac
    εmax_evo  ./= fac
end

# save current solution
@parallel function save_old!(τxx::A, τyy::A, τxy::A, τII::A, T::A, P::A, η::A, η_dif::A, η_dis::A, η_LTP::A, Vx::A, Vy::A, ρ::A, τxx_o::A, τyy_o::A, τxy_o::A, τII_o::A, T_o::A, P_o::A, η_o::A, η_dif_o::A, η_dis_o::A, η_LTP_o::A, Vx_o::A, Vy_o::A, ρ_o::A) where {A<:Data.Array}
    @all(τxx_o)    = @all(τxx)
    @all(τyy_o)    = @all(τyy)
    @all(τxy_o)    = @all(τxy)
    @all(τII_o)    = @all(τII)
    @all(T_o)      = @all(T)
    @all(P_o)      = @all(P)
    @all(η_o)      = @all(η)
    @all(η_dif_o)  = @all(η_dif)
    @all(η_dis_o)  = @all(η_dis)
    @all(η_LTP_o)  = @all(η_LTP)
    @all(Vx_o)     = @all(Vx)
    @all(Vy_o)     = @all(Vy)
    @all(ρ_o)      = @all(ρ)
    return
end

# save previous solution for prediction
@parallel function save_oldold!(τII_o::A, T_o::A, P_o::A, τII_oo::A, T_oo::A, P_oo::A) where {A<:Data.Array}
    @all(τII_oo)   = @all(τII_o)
    @all(T_oo)     = @all(T_o)
    @all(P_oo)     = @all(P_o)
    return
end

# reset fields when time step is restarted
@parallel function resetVals!(τxx::A, τyy::A, τxy::A, τII::A, T::A, P::A, η::A, η_dif::A, η_dis::A, η_LTP::A, Vx::A, Vy::A, ρ::A, τxx_o::A, τyy_o::A, τxy_o::A, τII_o::A, T_o::A, P_o::A, η_o::A, η_dif_o::A, η_dis_o::A, η_LTP_o::A, Vx_o::A, Vy_o::A, ρ_o::A) where {A<:Data.Array}
    @all(τxx)   = @all(τxx_o)
    @all(τyy)   = @all(τyy_o)
    @all(τxy)   = @all(τxy_o)
    @all(τII)   = @all(τII_o)
    @all(T)     = @all(T_o)     
    @all(P)     = @all(P_o)     
    @all(η)     = @all(η_o)     
    @all(η_dif) = @all(η_dif_o) 
    @all(η_dis) = @all(η_dis_o) 
    @all(η_LTP) = @all(η_LTP_o)
    @all(Vx)    = @all(Vx_o)    
    @all(Vy)    = @all(Vy_o)   
    @all(ρ)     = @all(ρ_o)     
    return
end

# update physical timestep
function updateTimestep(dt, dt0, T, T_o, τ, τ_o, dT_ref, dτ_crit, ReFlag, maxFac)
    if ReFlag
        dt_new       = 0.5 * dt
    else
        # check if LTP is active anywhere
        max_dT       = maximum(abs.(T .- T_o))
        mean_dτ      = mean(abs.(τ .- τ_o))
        dt_heat      = dt * dT_ref/max_dT
        dt_stress    = dt * dτ_crit/mean_dτ
        dt_new       = min(dt_heat, dt_stress, maxFac*dt0)
    end
    # be hesitant to increase the timestep
    dt_new = min(dt_new, 1.5*dt)

    return dt_new
end

# prints time step after it has been updated
function printNewTimestep(dt_dim)
    if dt_dim > 3600*24*365.25
        @printf("Reset timestep to %1.3e yr.\n", dt_dim/(3600*24*365.25))
    elseif dt_dim > 3600*24
        @printf("Reset timestep to %1.3e days.\n", dt_dim/(3600*24))
    elseif dt_dim > 3600
        @printf("Reset timestep to %1.3e hours.\n", dt_dim/3600)
    else
        @printf("Reset timestep to %1.3e seconds.\n", dt_dim)
    end
    return
end

# used in makePrediction
@parallel function predict!(x::A, x_o::A, x_p::A, dt::N, dt_o::N, den::I) where {A<:Data.Array, N<:Number, I<:Int64}
    @all(x_p) = @all(x) + 1/2^den * (@all(x) - @all(x_o)) * dt/dt_o
    return
end

# find a good prediction for next solution
function makePrediction(x::A, x_o::A, x_p::A, dt::N, dt_o::N, lim::B) where {A<:Data.Array, N<:Number, B<:Bool}
    den  = 0
    x_p .= -@ones(size(x_p))
    if lim
        while minimum(x_p) <= 0 && den < 10
            @parallel predict!(x, x_o, x_p, dt, dt_o, den)
            den += 1
        end
        if den == 10
            return copy(x)
        else
            return copy(x_p)
        end
    else
        @parallel predict!(x, x_o, x_p, dt, dt_o, den)
        return copy(x_p)
    end
end

# set fields to predicted values
@parallel function setPrediction!(τII::A, T::A, P::A, τII_p::A, T_p::A, P_p::A) where {A<:Data.Array}
    @all(τII)      = @all(τII_p)
    @all(T)        = @all(T_p)
    @all(P)        = @all(P_p)
    return
end

# used in checkTimestep
@parallel function interp_V!(Vx_cen::A, Vy_cen::A, Vx::A, Vy::A) where {A<:Data.Array}
    @all(Vx_cen) = @av_xi(Vx)
    @all(Vy_cen) = @av_yi(Vy)
    return
end

# used in checkTimestep
@parallel function find_CFL_step!(max_dt::A, Vx_cen::A, Vy_cen::A, dxn::A, dyn::A, CFL::N) where {A<:Data.Array, N<:Number}
    @all(max_dt) = CFL / (@all(Vx_cen)/@all(dxn) + @all(Vy_cen)/@all(dyn))
    return
end

# check if timestep is not too big
function checkTimestep(dt, dt_CFL, dT_ref, dτ_crit, CFL, T, T_o, τII, τII_o, Vx, Vy, Vx_cen, Vy_cen, dxn, dyn, iter, CD)
    @parallel interp_V!(Vx_cen, Vy_cen, Vx, Vy)
    @parallel find_CFL_step!(dt_CFL, Vx_cen, Vy_cen, dxn, dyn, CFL)
    dt_vel       = minimum(abs.(dt_CFL))
    max_dT       = maximum(abs.(T .- T_o))
    mean_dτ      = mean(abs.(τII .- τII_o))

    if max_dT > dT_ref
        @printf("Iter: %d, Restarting time step because of heat production: %1.3e K \n", iter, ustrip(dimensionalize(max_dT, K, CD)))
        return true
    elseif mean_dτ > dτ_crit
        @printf("Iter: %d, Restarting time step because of stress change: %1.3e MPa \n", iter, ustrip(dimensionalize(mean_dτ, MPa, CD)))
        return true
    elseif dt > dt_vel
        @printf("Iter: %d, Restarting time step because of CFL\n", iter)
        return true
    else
        return false
    end
end

# adjust tolerance based on time step (relaxed for very small steps)
function findTol(dt, tol0)
    return tol0 * 10 ^ (-2 / (1 + exp(-2 * (log10(dt) - 1))) + 2)
end

# used in AdjustDamp
function raiseTol(tol, nRe)
    if 1 < nRe < 8
        return tol * 2.0
    else
        return tol
    end
end

# adjust damping parameters (does not work, so currently unused)
function AdjustDamp(dampVx, dampVy, dampT, VdmpX, VdmpY, Tdmp, VdmpX0, VdmpY0, nx, ny, err, dampReset, tol, ndampRe, dampFlag)
    # don't do anything for the first few cycles
    if length(err) < 10
        return dampVx, dampVy, dampT, VdmpX, VdmpY, Tdmp, dampReset, tol, ndampRe
    end

    # if convergence is working but slow, speed it up
    if err[end] < err[end-1] < err[end-2] < err[end-3] && err[end] > err[end-1]*0.70
        VdmpX = VdmpX/2; VdmpY = VdmpY/2; #Tdmp = Tdmp/2;
        @printf("Adjusted VdmpX/VdmpY/Tdmp to %.2f/%.2f/%.2f \n", VdmpX, VdmpY, Tdmp)
    # speed up if there is stagnation
    elseif abs(err[end] - err[end-9]) < 0.3*err[end]
        VdmpX = VdmpX/2; VdmpY = VdmpY/2; #Tdmp = Tdmp/2;
        @printf("Adjusted VdmpX/VdmpY/Tdmp to %.2f/%.2f/%.2f \n", VdmpX, VdmpY, Tdmp)
    end

    # if residual is caught in oscillations
    if err[end] > 1.5*err[end-5] && dampReset < 1
        VdmpX = VdmpX0; VdmpY = VdmpY0; #Tdmp = nx/2.0;
        dampReset = 10
        @printf("Reset VdmpX/VdmpY/Tdmp to %.2f/%.2f/%.2f \n", VdmpX, VdmpY, Tdmp)
        ndampRe += 1
        tol = raiseTol(tol, ndampRe)
    else
        dampReset -= 1;
    end
    
    dampVx, dampVy = dampFlag * (1.0 - VdmpX/nx), dampFlag * (1.0 - VdmpY/ny)
    return dampVx, dampVy, dampT, VdmpX, VdmpY, Tdmp, dampReset, tol, ndampRe
end

# raise tolerance if convergence is slow
function raiseTol2(tol, nRe, iter)
    if nRe < 8
        if iter > 10e3 && mod(iter, 10e3) == 0
            return tol * 2.0, nRe + 1
        end
    end
    return tol, nRe
end