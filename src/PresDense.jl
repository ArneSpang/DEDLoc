export update_ρ!, divVel!, update_P!

# update density
@parallel function update_ρ!(ρ::A, ρ_o::A, ∇V::A, κ::A, ρCp::A, dt::N, λ::N, Cp::N, compFlag::N) where {A<:Data.Array, N<:Number}
    @all(ρ)   = @all(ρ_o) * exp(-@all(∇V) * dt * compFlag)
    @all(ρCp) = @all(ρ) * Cp
    @all(κ)   = λ / @all(ρCp)
    return
end

# compute divergence of velocity
@parallel function divVel!(∇V::A, Vx::A, Vy::A, dxn::A, dyn::A) where {A<:Data.Array}
    @all(∇V) = @d_xi(Vx) / @all(dxn) + @d_yi(Vy) / @all(dyn)
    return   
end

# all pressure updates in one place (currently not used)
@parallel function update_P!(P::A, P_o::A, ∇V::A, Kb::A, η::A, ResP::A, dψP::A, PUp::A, dt::N, max_nxy::N, dampP::N, compFlag::N) where {A<:Data.Array, N<:Number}
    @all(ResP) = @all(∇V)  + compFlag  * (@all(P) - @all(P_o)) / (dt * @all(Kb))
    @all(dψP)  = 10.0      * @all(η)   / max_nxy
    @all(PUp)  = dampP     * @all(PUp) + @all(ResP)
    @inn_x(P)  = @inn_x(P) - @all(PUp) * @all(dψP)
    return    
end