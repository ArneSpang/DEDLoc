export makeCoordsNew, Coord2PS, InitTemp!, setAnoms!

# creates 1D vector of coordinates with refinement in the center (used by makeCoordsNew)
function vSpace(x1, x2, n, r; i=false)
    # x1: starting coordinate
    # x2: ending coordinate
    # n: number of cells
    # r: ratio between largest and smallest cell
    # i: invert ratio

    x      = zeros(n+1)
    x[1]   = x1
    avg    = (x2 - x1) / n
    r      = ifelse(i, 1/r, r)
    
    if mod(n,2) == 0
        mid   = Int64(n/2+1)
        small = 2 / (1+r) * avg
        big   = r * small
        inc   = (big-small)/(n/2-1)
        dx    = big
        for i = 2 : mid
            x[i] = x[i-1] + dx
            dx  -= inc
        end
        for i = mid+1 : n+1
            dx  += inc
            x[i] = x[i-1] + dx 
        end
    else
        NR    = (r+1)*(n+1)
        small = 2 * (x2 - x1) / (NR - 2)
        big   = r * small
        mid   = Int64((n+1)/2)+1
        inc   = (big-small)/((n-1)/2)
        dx    = big
        for i = 2 : mid
            x[i] = x[i-1] + dx
            dx  -= inc
        end
        dx  += inc
        for i = mid+1 : n+1
            dx  += inc
            x[i] = x[i-1] + dx 
        end
    end
    x[end] = x2

    return x
end

# creates coordinates
function makeCoordsNew(Lx, Ly, nx, ny, mode)
    if mode=="Regular"
        dx     = Lx / nx
        xn     = collect(-Lx/2.0:dx:Lx/2.0)
        xc     = collect(-Lx/2.0-dx/2.0:dx:Lx/2.0+dx/2.0)
        dy     = Ly / ny
        yn     = collect(-Ly/2.0:dy:Ly/2.0)
        yc     = collect(-Ly/2.0-dy/2.0:dy:Ly/2.0+dy/2.0)
    elseif mode=="RefineY"
        dx     = Lx / nx
        xn     = collect(-Lx/2.0:dx:Lx/2.0)
        xc     = collect(-Lx/2.0-dx/2.0:dx:Lx/2.0+dx/2.0)
        yn     = vSpace(-Ly/2, Ly/2, ny, 2.0)
        yc     = (yn[1:end-1] .+ yn[2:end]) / 2.0
        yc     = [2*yn[1]-yc[1]; yc; 2*yn[end]-yc[end]]
    else
        print("Coordinate mode not understood! Using regular grid spacing!\n")
        dx     = Lx / nx
        xn     = collect(-Lx/2.0:dx:Lx/2.0)
        xc     = collect(-Lx/2.0-dx/2.0:dx:Lx/2.0+dx/2.0)
        dy     = Ly / ny
        yn     = collect(-Ly/2.0:dy:Ly/2.0)
        yc     = collect(-Ly/2.0-dy/2.0:dy:Ly/2.0+dy/2.0)        
    end

    dxn    = diff(xn)
    dxc    = diff(xc)
    dyn    = diff(yn)
    dyc    = diff(yc)

    mxn    = [xn[ix] for ix=1:nx+1, iy=1:ny+1]
    mxc    = [xc[ix] for ix=1:nx+2,   iy=1:ny]
    myn    = [yn[iy] for ix=1:nx+1, iy=1:ny+1]
    myc    = [yc[iy] for ix=1:nx,   iy=1:ny+2]

    mdxn   = [ dxn[ix] for ix=1:nx,   iy=1:ny]
    mdyn   = [ dyn[iy] for ix=1:nx,   iy=1:ny]
    mdxc   = [ dxc[ix] for ix=1:nx+1, iy=1:ny+1]
    mdyc   = [ dyc[iy] for ix=1:nx+1, iy=1:ny+1]
    mdxn_v = [ dxn[ix] for ix=1:nx,   iy=1:ny-1]
    mdyn_v = [ dyn[iy] for ix=1:nx+1, iy=1:ny]

    return mxn, mxc, mdxn, mdxc, mdxn_v, myn, myc, mdyn, mdyc, mdyn_v
end

# initialize temperature field
function InitTemp!(T, TempType, T0, T1, nx, ny, Lx, Ly, xc, yc)
    # 1: constant T0
    # 2: center = T0, linear increase towards T1 in x-direction
    # 3: center = T0, linear increase towards T1 in y-direction
    # 4: center = T0, linear increase towards T1 in circular pattern

    if TempType == 1
        T .= T0
    elseif TempType == 2
        Tlin = (T1-T0) .* abs.(xc[:,1] ./ (Lx/2.0)) .+ T0
        for i = 1 : ny
            T[:,i] .= Tlin
        end
    elseif TempType == 3
        Tlin = (T1-T0) .* abs.(yc[1,2:end-1] ./ (Ly/2.0)) .+ T0
        for i = 1 : nx+2
            T[i,:] .= Tlin
        end
    elseif TempType == 4
        dist = (xc[2:end-1,:] .^ 2.0 .+ yc[:,2:end-1] .^ 2.0) .^ 0.5
        T .= (T1-T0) .* dist ./ (max(Lx, Ly) ./ 2.0) .+ T0
    end
end

# identifies indices that are within an ellipse (used by setAnoms)
function findEllipse(x, y, xe, ye, rx, ry, θ)
    return (((x .- xe).*cosd(θ) .+ (y .- ye).*sind(θ)) ./ rx).^2 .+ (((x .- xe) .*sind(θ) .- (y .- ye).*cosd(θ)) ./ ry).^2 .< 1
end

# sets rheological anomalies
function setAnoms!(ω, ωb, anomType, ω0, σbFlag, Lx, Ly, xc, yc, rad_x, rad_y, x_off, y_off, θ, fName)
    # 0: load field from fName
    # 1: center
    # 2: outsides
    # 3: outsides and right of center
    #11: center (Gaussian shape)
    #12: outsides (Gaussian shape)
    #13: outsides and right of center (Gaussian shape)

    x     = xc[2:end-1,:]
    y     = yc[:,2:end-1]

    # box-cut anomaly
    if anomType < 10
        if anomType == 0
            inds   = BitMatrix(load(fName, "inds"))
        elseif anomType == 1
            inds   = findEllipse(x, y, x_off, y_off, rad_x, rad_y, θ)
        elseif anomType == 2
            inds   = findEllipse(x, y, Lx/2.0, 0.0, rad_x, rad_y, θ) .|| findEllipse(x, y, -Lx/2.0, 0.0, rad_x, rad_y, θ)
        elseif anomType == 3
            inds   = findEllipse(x, y, Lx/2.0, 0.0, rad_x, rad_y, θ) .|| findEllipse(x, y, -Lx/2.0, 0.0, rad_x, rad_y, θ) .|| findEllipse(x, y, Lx/6.0, 0.0, rad_x, rad_y, θ)
        end
        
        ω[inds] .= ω0
    # Gaussian shape
    else
        if anomType == 11
            ω    .+= PDF2D(x, y, 0.0, 0.0, 2*rad_x, 2*rad_y, ω0-1.0)
        elseif anomType == 12
            ω    .+= PDF2D(x, y, Lx/2.0, 0.0, 2*rad_x, 2*rad_y, ω0-1.0) .+ PDF2D(x, y, -Lx/2.0, 0.0, 2*rad_x, 2*rad_y, ω0-1.0)
        elseif anomType == 13
            ω    .+= PDF2D(x, y, Lx/2.0, 0.0, 2*rad_x, 2*rad_y, ω0-1.0) .+ PDF2D(x, y, -Lx/2.0, 0.0, 2*rad_x, 2*rad_y, ω0-1.0) .+ PDF2D(x, y, Lx/6.0, 0.0, 2*rad_x, 2*rad_y, ω0-1.0)
        end
    end

    ω .= 1.0 ./ ω
    if σbFlag
        ωb .= ω
    end

    return nothing
end

# sets Gaussian profile
function PDF2D(x, y, x0, y0, hx, hy, zmax)
    return zmax * exp.(-4*log(2)*(((x .- x0)./hx).^2 .+ ((y .- y0)./hy).^2))
end