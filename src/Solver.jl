# compute residuals
@parallel function residuals!(ResVx::A, ResVy::A, ResP::A, ResT::A, τxx::A, τyy::A, τxy::A, ∇V::A, P::A, P_o::A, T::A, T_o::A, Kb::A, dT_diff::A, H::A, ρCp::A, dxc::A, dyc::A, dxn_v::A, dyn_v::A, dt::N, α::N, compFlag::N) where {A<:Data.Array, N<:Number}
    @all(ResVx) = (@d_xa(τxx) - @d_xa(P)) / @all(dxc) + @d_ya(τxy) / @all(dyn_v)
    @all(ResVy) = (@d_ya(τyy) - @d_yi(P)) / @inn_y(dyc) + @d_xi(τxy) / @all(dxn_v)
    @all(ResP)  = @all(∇V) + compFlag * (@inn_x(P) - @inn_x(P_o)) / (dt * @all(Kb)) - α * (@inn_x(T) - @inn_x(T_o)) / dt
    @all(ResT)  = (@all(dT_diff) + @all(H)) / @all(ρCp) - (@inn_x(T) - @inn_x(T_o)) / dt
    return    
end

# compute new PT step
@parallel function new_step!(dψVx::A, dψVy::A, dψP::A, η::A, η_dx::A, η_dy::A, G::A, Kb::A, dxc::A, dyc::A, max_Lxy::N, min_dxy::N, Vfac::N, max_nxy::I) where {A<:Data.Array, N<:Number, I<:Int64}
    @all(dψVx) = @all(dxc)^2   / @all(η_dx) / (Vfac)
    @all(dψVy) = @inn_y(dyc)^2 / @all(η_dy) / (Vfac)
    @all(dψP)  = 10.0 * @all(η) / max_nxy
    #@all(dψP)  = min_dxy * @all(η) / (max_Lxy * @all(Kb) / @all(G))
    return    
end

# compute update
@parallel function comp_update!(VxUp::A, VyUp::A, PUp::A, TUp::A, ResVx::A, ResVy::A, ResP::A, ResT::A, dampVx::N, dampVy::N, dampP::N, dampT::N) where {A<:Data.Array, N<:Number}
    @all(VxUp) = dampVx * @all(VxUp) + @all(ResVx)
    @all(VyUp) = dampVy * @all(VyUp) + @all(ResVy)
    @all(PUp)  = dampP  * @all(PUp)  + @all(ResP)
    @all(TUp)  = dampT  * @all(TUp)  + @all(ResT)
    return    
end

# apply update
@parallel function apply_update!(Vx::A, Vy::A, P::A, T::A, VxUp::A, VyUp::A, PUp::A, TUp::A, dψVx::A, dψVy::A, dψP::A, dψT::N) where {A<:Data.Array, N<:Number}
    @inn_y(Vx) = @inn_y(Vx) + @all(VxUp)  * @all(dψVx)
    @inn(Vy)   = @inn(Vy)   + @all(VyUp)  * @all(dψVy)
    @inn_x(P)  = @inn_x(P)  - @all(PUp)   * @all(dψP)
    @inn_x(T)  = @inn_x(T)  + @all(TUp)   * dψT
    return  
end

# check if error is stalling and can be considered converged
function checkError(err, nout, ReFlag, iter, nt_min)
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
            @printf("Iter: %d, Converged because of asymptotic convergence. \n", iter)
            return false, true
        end
    end

    # converged based on stall, including up and down
    n = max(Int64(round(5000 / nout)), 5)
    if length(err) > n && err[end] < 1e-3
        vals = err[end-n:end]
        expo = log10.(vals)
        if std(expo) < 0.1
            @printf("Iter: %d, Converged because of stalling residual. \n", iter)
            return false, true
        end
    end

    return false, false
end

# compute absolute error
@parallel function AbsErr!(AbsVx::A, AbsVy::A, AbsT::A, AbsP::A, ResVx::A, ResVy::A, ResT::A, ResP::A, V0::N, T0::N, P0::N) where {A<:Data.Array, N<:Number}
    @all(AbsVx) = abs(@inn_x(ResVx)) / V0
    @all(AbsVy) = abs(@all(ResVy))   / V0
    @all(AbsT)  = abs(@all(ResT))    / T0
    @all(AbsP)  = abs(@all(ResP))    / P0
    return
end

# compute relative error
@parallel function RelErr!(RelVx::A, RelVy::A, RelT::A, RelP::A, ResVx::A, ResVy::A, ResT::A, ResP::A, Vx::A, Vy::A, T::A, P::A, V0::N, T0::N, P0::N) where {A<:Data.Array, N<:Number}
    @all(RelVx) = abs(@inn_x(ResVx) / (@inn(Vx)  + V0*1e-15))
    @all(RelVy) = abs(@all(ResVy)   / (@inn(Vy)  + V0*1e-15))
    @all(RelT)  = abs(@all(ResT)    / (@inn_x(T) + T0*1e-15))
    @all(RelP)  = abs(@all(ResP)    / (@inn_x(P) + P0*1e-15))
    return
end

# select minimum error
@parallel function MinErr!(ErrVx::A, ErrVy::A, ErrT::A, ErrP::A, AbsVx::A, AbsVy::A, AbsT::A, AbsP::A, RelVx::A, RelVy::A, RelT::A, RelP::A) where {A<:Data.Array}
    @all(ErrVx) = min(@all(AbsVx), @all(RelVx));
    @all(ErrVy) = min(@all(AbsVy), @all(RelVy));
    @all(ErrT)  = min(@all(AbsT),  @all(RelT));
    @all(ErrP)  = min(@all(AbsP),  @all(RelP));
    return
end