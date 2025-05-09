## viscosity structures
struct DifCreep{T, F}
    E::T
    V::Union{Nothing, T}
    R::T
    η_rel::T
    Fun::F

    function DifCreep(AVFlag, E::T, V::Union{Nothing, T}, P0::Union{Nothing, T}, R::T, η_rel::T) where T
        E, difCr = if AVFlag == 0
            Q = E / R
            Q, dif_creep_1!
        elseif AVFlag == 1
            Q = (E + P0*V) / R
            Q, dif_creep_1!
        else
            E, dif_creep_2!
        end

        return new{T, typeof(difCr)}(E, V, R, η_rel, difCr)
    end
end

struct DisCreep{T, F}
    n::T
    E::T
    V::Union{Nothing, T}
    R::T
    η_rel::T
    η_max::T
    Fun::F

    function DisCreep(AVFlag, n::T, E::T, V::Union{Nothing, T}, P0::Union{Nothing, T}, R::T, η_rel::T, η_max::T) where T
        E, disCr = if AVFlag == 0
            Q = E / (n * R)
            Q, dis_creep_1!
        elseif AVFlag == 1
            Q = (E + P0*V) / (n * R)
            Q, dis_creep_1!
        else
            E, dis_creep_2!
        end

        return new{T, typeof(disCr)}(n, E, V, R, η_rel, η_max, disCr)
    end
end

struct LTPCreep{T, F}
    Σ::T
    σb::T
    E::T
    V::Union{Nothing, T}
    R::T
    η_rel::T
    η_max::T
    Fun::F

    function LTPCreep(AVFlag, σL::T, K::T, σb::T, gs::T, E::T, V::Union{Nothing, T}, P0::Union{Nothing, T}, R::T, η_rel::T, η_max::T) where T
        E, LTPCr = if AVFlag == 0
            Q = E / R
            Q, LTP_creep_1!
        elseif AVFlag == 1
            Q = (E + P0*V) / R
            Q, LTP_creep_1!
        else
            E, LTP_creep_2!
        end

        Σ = σL + K * gs^(-0.5)

        return new{T, typeof(LTPCr)}(Σ, σb, E, V, R, η_rel, η_max, LTPCr)
    end
end

## initialize time step
# update elastic viscosity at the beginning of a timestep
@parallel function update_η_e!(η_e::A, η_e_nodes::A, G::A, G_nodes::A, dt::N) where {A<:Data.Array, N<:Number}
    @all(η_e)       = @all(G)       * dt
    @all(η_e_nodes) = @all(G_nodes) * dt
    return
end

# save elastic strain rate from previous timestep to be added to full strain rate to compute effective strain rate
@parallel function old_elastic_strain_rate!(εxx_elaOld::A, εyy_elaOld::A, εxy_elaOld::A, τxx_o::A, τyy_o::A, τxy_o::A, η_e::A, η_e_nodes::A, elaFlag::N) where {A<:Data.Array, N<:Number}
    @all(εxx_elaOld) = elaFlag * @inn_x(τxx_o) / (2.0 * @all(η_e))
    @all(εyy_elaOld) = elaFlag * @all(τyy_o)   / (2.0 * @all(η_e))
    @all(εxy_elaOld) = elaFlag * @all(τxy_o)   / (2.0 * @all(η_e_nodes))
    return    
end


## strain rates
# comptue full strain rate from velocity field
@parallel function full_strain_rate!(εxx_f::A, εyy_f::A, εxy_f::A, Vx::A, Vy::A, ∇V::A, dxn::A, dyn::A, dxc::A, dyc::A) where {A<:Data.Array}
    @all(εxx_f) = @d_xi(Vx) / @all(dxn) - 1.0/3.0 * @all(∇V)
    @all(εyy_f) = @d_yi(Vy) / @all(dyn) - 1.0/3.0 * @all(∇V)
    @all(εxy_f) = 0.5 * (@d_ya(Vx) / @all(dyc) + @d_xa(Vy) / @all(dxc))
    return    
end

# add old elastic strain rate to get effective strain rate
@parallel function eff_strain_rate!(εxx::A, εyy::A, εxy::A, εxx_f::A, εyy_f::A, εxy_f::A, εxx_elaOld::A, εyy_elaOld::A, εxy_elaOld::A) where {A<:Data.Array}
    @all(εxx)     = @all(εxx_f) + @all(εxx_elaOld)
    @all(εyy)     = @all(εyy_f) + @all(εyy_elaOld)
    @all(εxy)     = @all(εxy_f) + @all(εxy_elaOld)
    return    
end

# interpolate τxy to cell centers for second invariant computation
@parallel function interp_strain_rate!(εxy::A, εxy_cen::A) where {A<:Data.Array}
    @all(εxy_cen) = @av(εxy)
    return
end

# compute second invariant of strain rate
@parallel function second_strain_rate!(εxx::A, εyy::A, εxy_cen::A, εII::A) where {A<:Data.Array}
    @all(εII)     = sqrt(0.5 * (@all(εxx)^2 + @all(εyy)^2) + @all(εxy_cen)^2)
    return
end

# partition strain rate between deformation mechanisms
@parallel function part_ε_new!(εII::A, τII::A, εII_v::A, εII_dif::A, εII_nl::A, εII_dis_g::A, εII_LTP_g::A, τII_reg::A, τII_vtrue::A, ε_part::A, η_e::A, η_dif::A, η_dis::A, η_LTP::A, η_reg::N) where {A<:Data.Array, N<:Number}
    @all(εII_v)     = @all(εII) - @all(τII) / (2.0 * @all(η_e))
    @all(τII_reg)   = 2.0 * η_reg * @all(εII_v)
    @all(τII_vtrue) = @all(τII) - @all(τII_reg)
    @all(εII_dif)   = @all(τII_vtrue) / (2.0 * @all(η_dif))
    @all(εII_nl)    = @all(εII_v) - @all(εII_dif)
    @all(ε_part)    = @all(η_dis) / @all(η_LTP);
    @all(εII_dis_g) = @all(εII_nl) * (1           /(1 + @all(ε_part)))
    @all(εII_LTP_g) = @all(εII_nl) * (@all(ε_part)/(1 + @all(ε_part)))
    return
end

# compute true strain rate partitioning after viscosity and stress update
@parallel function true_part_ε!(τII::A, τII_reg::A, τII_vtrue::A, εII::A, εII_v::A, εII_dif::A, εII_dis::A, εII_LTP::A, η_e::A, η_dif::A, η_dis::A, η_LTP::A, η_reg::N) where {A<:Data.Array, N<:Number}
    @all(εII_v)     = @all(εII) - @all(τII) / (2.0 * @all(η_e))
    @all(τII_reg)   = 2.0 * η_reg * @all(εII_v)
    @all(τII_vtrue) = @all(τII) - @all(τII_reg)
    @all(εII_dif)   = @all(τII_vtrue) / (2.0 * @all(η_dif))
    @all(εII_dis)   = @all(τII_vtrue) / (2.0 * @all(η_dis))
    @all(εII_LTP)   = @all(τII_vtrue) / (2.0 * @all(η_LTP))
    return
end


## viscosity
# set initial visc guess based on equal partitioning
function initalVisc(P0, T, εbg, gs, R, AVFlag, dt0, G, Adif, mdif, Edif, Vdif, Adis, ndis, Edis, Vdis, ALTP, ELTP, VLTP, LTP_σL, LTP_K, LTP_σb, ωI, ωIb, FE, FT)
    Pre_dif   = 0.5 * Adif ^ (-1.0)      .* gs.^(mdif) * FE / FT
    Pre_dis   = 0.5 * Adis ^ (-1.0/ndis)               * FE^(1.0/ndis) / FT
    LTP_Σ     = LTP_σL + LTP_K * gs^(-0.5)
    ε_g       = εbg/3

    if AVFlag == 0
        Q_dif = Edif / R 
        Q_dis = Edis / (R * ndis)
        Q_LTP = ELTP / R
    else
        Q_dif = (Edif + P0 * Vdif) / R
        Q_dis = (Edis + P0 * Vdis) / (R * ndis)
        Q_LTP = (ELTP + P0 * VLTP) / R
    end

    η_dif     = ωI .*  Pre_dif                                                 .* exp.(Q_dif ./ T[2:end-1,:])
    η_dis     = ωI .*  Pre_dis .* ε_g  ^ (1.0/ndis-1.0)                        .* exp.(Q_dis ./ T[2:end-1,:])
    η_LTP     =        T[2:end-1,:] .* LTP_Σ ./ Q_LTP .* asinh.(FE*ε_g ./ ALTP .* exp.(Q_LTP ./ T[2:end-1,:]) .+ ωIb .* LTP_σb) ./ (2*ε_g*FT)

    η_v       = (1.0 ./ η_dis .+ 1.0 ./ η_dif) .^ (-1.0)
    η_e       = G * dt0
    η         = (1.0 ./ η_v .+ 1.0 ./ η_e) .^ (-1.0)
    ε_part    = η_dis ./ η_LTP

    return η, η_v, η_e, η_dif, η_dis, η_LTP, ε_part, Pre_dif, Pre_dis
end

# update diffusion creep viscosity with constant activation enthalpy
@parallel function dif_creep_1!(η::A, η_new::A, T::A, P::A, ωI::A, B::N, Q::N, V::N, R::N, η_rel::N) where {A<:Data.Array, N<:Number}
    @all(η_new) = @all(ωI) * B * exp(Q / @inn_x(T))
    @all(η)     = exp((1.0-η_rel) * log(@all(η)) .+ η_rel * log(@all(η_new)))
    return    
end

# update diffusion creep viscosity with variable activation enthalpy
@parallel function dif_creep_2!(η::A, η_new::A, T::A, P::A, ωI::A, B::N, E::N, V::N, R::N, η_rel::N) where {A<:Data.Array, N<:Number}
    @all(η_new) = @all(ωI) * B * exp((E + @inn_x(P) * V) / (R * @inn_x(T)))
    @all(η)     = exp((1.0-η_rel) * log(@all(η)) .+ η_rel * log(@all(η_new)))
    return    
end

# update diffusion creep viscosity (calls dif_creep_1! or dif_creep_2!)
comp_η(ηStruc::DifCreep, η, η_new, A, T, P, ωI) = @parallel ηStruc.Fun(η, η_new, T, P, ωI, A, ηStruc.E, ηStruc.V, ηStruc.R, ηStruc.η_rel)

# update dislocation creep viscosity with constant activation enthalpy
@parallel function dis_creep_1!(η::A, η_new::A, εII_g::A, T::A, P::A, ωI::A, B::N, n::N, Q::N, V::N, R::N, η_rel::N, η_max::N, ε_min::N) where {A<:Data.Array, N<:Number}
    @all(εII_g) = ifelse(@all(εII_g) < ε_min, ε_min, @all(εII_g))
    @all(η_new) = @all(ωI) * B * @all(εII_g)^(1.0/n-1.0) * exp(Q / @inn_x(T))
    @all(η_new) = ifelse(@all(η_new) > η_max, η_max, @all(η_new))
    @all(η)     = exp((1.0-η_rel) * log(@all(η)) + η_rel * log(@all(η_new)))
    return
end

# update dislocation creep viscosity with variable activation enthalpy
@parallel function dis_creep_2!(η::A, η_new::A, εII_g::A, T::A, P::A, ωI::A, B::N, n::N, E::N, V::N, R::N, η_rel::N, η_max::N, ε_min::N) where {A<:Data.Array, N<:Number}
    @all(εII_g) = ifelse(@all(εII_g) < ε_min, ε_min, @all(εII_g))
    @all(η_new) = @all(ωI) * B * @all(εII_g)^(1.0/n-1.0) * exp((E + @inn_x(P) * V) / (n * R * @inn_x(T)))
    @all(η_new) = ifelse(@all(η_new) > η_max, η_max, @all(η_new))
    @all(η)     = exp((1.0-η_rel) * log(@all(η)) + η_rel * log(@all(η_new)))
    return
end

# update dislocation creep viscosity (calls dis_creep_1! or dis_creep_2!)
comp_η(ηStruc::DisCreep, η, η_new, εII_g, A, T, P, ωI, ε_min) = @parallel ηStruc.Fun(η, η_new, εII_g, T, P, ωI, A, ηStruc.n, ηStruc.E, ηStruc.V, ηStruc.R, ηStruc.η_rel, ηStruc.η_max, ε_min)

# update LTP viscosity with constant activation enthalpy (Hansen)
@parallel function LTP_creep_1!(η::Ar, η_new::Ar, εII_g::Ar, T::Ar, P::Ar, ωI::Ar, A::N, Σ::N, σb::N, Q::N, V::N, R::N, η_rel::N, η_max::N, ε_min::N, FE::N, FT::N) where {Ar<:Data.Array, N<:Number}
    @all(εII_g) = ifelse(@all(εII_g) < ε_min, ε_min, @all(εII_g))
    @all(η_new) = (@inn_x(T) * Σ/Q * asinh(FE*@all(εII_g)/A * exp(Q/@inn_x(T))) + @all(ωI) * σb) / (2.0*@all(εII_g)*FT)
    @all(η_new) = ifelse(@all(η_new) > η_max, η_max, @all(η_new))
    @all(η)     = exp((1.0-η_rel) * log(@all(η)) + η_rel * log(@all(η_new)))
    return    
end

# update LTP viscosity with variable activation enthalpy (Hansen)
@parallel function LTP_creep_2!(η::Ar, η_new::Ar, εII_g::Ar, T::Ar, P::Ar, ωI::Ar, A::N, Σ::N, σb::N, E::N, V::N, R::N, η_rel::N, η_max::N, ε_min::N, FE::N, FT::N) where {Ar<:Data.Array, N<:Number}
    @all(εII_g) = ifelse(@all(εII_g) < ε_min, ε_min, @all(εII_g))
    @all(η_new) = (@inn_x(T) * Σ * R/(E + @inn_x(P)*V) * asinh(FE*@all(εII_g)/A * exp((E + @inn_x(P)*V)/(R * @inn_x(T)))) + @all(ωI) * σb) / (2.0*@all(εII_g)*FT)
    @all(η_new) = ifelse(@all(η_new) > η_max, η_max, @all(η_new))
    @all(η)     = exp((1.0-η_rel) * log(@all(η)) + η_rel * log(@all(η_new)))
    return    
end

# update LTP creep viscosity (calls LTP_creep_1! or LTP_creep_2!)
comp_η(ηStruc::LTPCreep, η, η_new, εII_g, A, T, P, ωI, ε_min, FE, FT) = @parallel ηStruc.Fun(η, η_new, εII_g, T, P, ωI, A, ηStruc.Σ, ηStruc.σb, ηStruc.E, ηStruc.V, ηStruc.R, ηStruc.η_rel, ηStruc.η_max, ε_min, FE, FT)

# update LTP viscosity (Kameyama)
@parallel function Kam_creep!(η::Ar, η_new::Ar, εII_g::Ar, T::Ar, S::Ar, A::N, E::N, σp::N, q::N, γ::N, R::N, η_rel::N, η_max::N, ε_min::N) where {Ar<:Data.Array, N<:Number}
    @all(εII_g) = ifelse(@all(εII_g) < ε_min, ε_min, @all(εII_g))
    @all(S)     = E / (R * @inn_x(T)) * (1.0-γ)^(q-1.0) * q * γ
    @all(η_new) = 0.5 * A^(-1/@all(S)) * @all(εII_g)^(1.0/@all(S) - 1.0) * γ * σp * exp(E/(@all(S) * @inn_x(T) * R) * (1.0-γ)^q)
    @all(η_new) = ifelse(@all(η_new) > η_max, η_max, @all(η_new))
    @all(η)     = exp((1.0-η_rel) * log(@all(η)) + η_rel * log(@all(η_new)))
    return
end

# compute effective viscosity
@parallel function eff_η!(η::A, η_v::A, η_e::A, η_dif::A, η_dis::A, η_LTP::A, η_reg::N, elaFlag::N) where {A<:Data.Array, N<:Number}
    @all(η_v)     = 1.0 / (1.0 / @all(η_dif) + 1.0     / @all(η_dis) + 1.0 / @all(η_LTP)) + η_reg
    @all(η)       = 1.0 / (1.0 / @all(η_v)   + elaFlag / @all(η_e))
    return
end

# interpolate viscosity to nodes
@parallel function interp_η!(η::A, η_nodes::A, η_ml::A) where {A<:Data.Array}
    @inn(η_nodes) = @av(η)
    @inn(η_ml)    = @maxloc(η)
    return
end

# interpolate viscosity to edges for psuedo time steps
@parallel function interp_η_dxdy!(η::A, η_dx::A, η_dy::A) where {A<:Data.Array}
    @inn_x(η_dx)  = @av_xa(η)
    @all(η_dy)    = @av_ya(η)
    return
end


## stress
# update stress
@parallel function update_τ!(τxx::A, τyy::A, τxy::A, τxy_cen::A, τII::A, η::A, η_nodes::A, εxx::A, εyy::A, εxy::A, εxy_cen::A) where {A<:Data.Array}
    @inn_x(τxx)   = 2.0 * @all(η)       * @all(εxx)
    @all(τyy)     = 2.0 * @all(η)       * @all(εyy)
    @all(τxy)     = 2.0 * @all(η_nodes) * @all(εxy)
    @all(τxy_cen) = 2.0 * @all(η)       * @all(εxy_cen)
    @all(τII)     = sqrt(0.5 * (@inn_x(τxx)^2 + @all(τyy)^2) + @all(τxy_cen)^2)
    return    
end

## printing
# compute second invariant of elastic strain rate for identifying dominant mechanism
@parallel function ε_ela!(εII_ela::A, τII::A, τII_o::A, η_e::A) where {A<:Data.Array}
    @all(εII_ela) = abs((@all(τII) - @all(τII_o)) / (2 * @all(η_e)))
    return
end