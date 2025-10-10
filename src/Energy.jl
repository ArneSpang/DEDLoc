# compute dissipation for all mechanisms
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

# compute dissipation
@parallel function diss_energy!(H::A, τxx::A, τyy::A, τxy_cen::A, εxx::A, εyy::A, εxy_cen::A, η_e::A, heatFlag::N, elaFlag::N) where {A<:Data.Array, N<:Number}
    @all(H)       = (@inn_x(τxx)    * (@all(εxx)     - elaFlag * @inn_x(τxx)   / (2.0 * @all(η_e))) +
                     @all(τyy)      * (@all(εyy)     - elaFlag * @all(τyy)     / (2.0 * @all(η_e))) +
                     @all(τxy_cen)  * (@all(εxy_cen) - elaFlag * @all(τxy_cen) / (2.0 * @all(η_e))) * 2) * heatFlag;
    return    
end

# compute fluxes
@parallel function fluxes!(qx::A, qy::A, T::A, dxc::A, dyc::A, λ::N) where {A<:Data.Array, N<:Number}
    @all(qx)      = - λ * @d_xa(T) / @all(dxc)
    @inn_y(qy)    = - λ * @d_yi(T) / @inn_y(dyc)
    return    
end

# compute temperature diffusion
@parallel function diffusion!(dT_diff::A, qx::A, qy::A, dxn::A, dyn::A, diffFlag::N) where {A<:Data.Array, N<:Number}
    @all(dT_diff) = - (@d_xa(qx) / @all(dxn) + @d_ya(qy) / @all(dyn)) * diffFlag
    return
end