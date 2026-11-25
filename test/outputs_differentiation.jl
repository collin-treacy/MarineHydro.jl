using Test
using MarineHydro
using PyCall
using LinearAlgebra
using DifferentiationInterface 
import ForwardDiff 
import Zygote


@testset "Integrate Pressure " begin
    cpt = pyimport("capytaine")
    radius = 1.0
    resolution = (4, 4) #smaller data just to test
    cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution)
    cptmesh.keep_immersed_part(inplace=true)
    mesh = Mesh(cptmesh)
    @testset "Re(Forces) and Im(Forces) test" begin
        pressure = rand(size(mesh.areas))   # Random pressure values
        dof = [0.0,0.0,1.0]
        # Differentiability check
        real_grad_zygote, = Zygote.gradient(mesh -> real.(integrate_pressure(mesh, pressure, dof)), mesh) #with respect to mesh
        @test real_grad_zygote !== nothing

        imag_grad_zygote, = Zygote.gradient(mesh -> imag.(integrate_pressure(mesh, pressure, dof)), mesh) #with respect to mesh
        @test imag_grad_zygote !== nothing

        grad_areas = real_grad_zygote.areas #integration depends on normal and area only
        @test grad_areas !== nothing
        grad_normals = real_grad_zygote.normals
        @test typeof(grad_normals) == Matrix{Float64}
        @test grad_normals !== nothing

        # Test the gradient shape (should be the same shape as mesh panels)
        @test size(grad_areas) == size(mesh.areas)
        @test size(grad_normals) == size(mesh.normals)
    end
end

@testset "Gradient accuracy check with Finite diff [w.r.t omega]" begin
    using FiniteDifferences
    cpt = pyimport("capytaine")
    radius = 1.0
    resolution = (4, 4) #smaller data just to test
    cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution).keep_immersed_part()
    mesh = Mesh(cptmesh)
    omega = 1.12
    dof = [0.0,0.0,1.0]
    green_functions = (
        Rankine(),
        RankineReflected(),
        GFWu(),
    )

    function check_added_mass(ω,mesh,dof)
        A = calculate_radiation_forces(mesh,dof,ω)[1]
        return A
    end

    @testset "A(w) partials with Wu " begin
        A(w) = check_added_mass(w, mesh, dof)
        A_w_grad, = Zygote.gradient(A,omega)
        #checking accuracy with finite FiniteDifferences
    # Central difference with order 5
        fd_grad = FiniteDifferences.central_fdm(5, 1)(A, omega)
        @test A_w_grad !== nothing
        @test typeof(A_w_grad) == Float64
        @test A_w_grad ≈  fd_grad atol=1e-6 rtol=1e-6
    end

    @testset "A(w) with w using exact Delhommeau" begin
        green_functions = (
            Rankine(),
            RankineReflected(),
            ExactGuevelDelhommeau(),
        )
        A(w) = check_added_mass(w, mesh, dof)
        A_w_grad1, = Zygote.gradient(A,omega)
        #checking accuracy with FiniteDifferences
    # Central difference with order 5
        fd_grad1 = FiniteDifferences.central_fdm(5, 1)(A, omega)
        @test A_w_grad1 !== nothing
        @test typeof(A_w_grad1) == Float64
        @test A_w_grad1 ≈  fd_grad1 atol=1e-6 rtol=1e-6
    end
end

@testset "Gradient accuracy check with Finite diff [w.r.t omega] for MDOF cylnder" begin
    
    # Description of problem
    omegas = [1.0, 1.5] # frequencies [rad/s]
    beta = 0 # incident wave angle [rad]
    t_DOFs = ["Heave"] # translational DOFs
    r_DOFs = ["Pitch"] # rotational DOFs
    DOFs = [t_DOFs; r_DOFs] # all DOFs

    # Create Mesh object
    radius = 1.5  
    center = (0.0,0.0,0.0) 
    len = 2.5
    faces_max_radius = 0.7
    cpt = pyimport("capytaine")
    cptmesh = cpt.meshes.predefined.mesh_horizontal_cylinder(
                radius=radius,
                center=center, 
                length=len, 
                faces_max_radius = faces_max_radius
                ).keep_immersed_part(inplace=true)

    # Get MarineHydro values
    mesh = Mesh(cptmesh)
    rigid_dof_list = DOFs
    rotation_center = [1.0, 1.0, 0.0] # off set for nonzero off-diagoinal elements
    floatingbody = FloatingBody(mesh, rigid_dof_list, rotation_center, "Horizontal_Cylinder")

    # Radiation solve functions
    function A_and_B_vec(w)
        parameters = (wave_frequencies=w,
            radiating_dofs=collect(keys(floatingbody.dofs)),
            influenced_dofs=collect(keys(floatingbody.dofs)))
        data = hydrodynamic_coefficients(floatingbody, parameters, DirectBEM())
        return vcat(vec(data.added_mass), vec(data.radiation_damping))
    end

    function F_ex_vec(w)
        parameters = (wave_frequencies=[w],
            wave_directions=[beta],
            influenced_dofs=collect(keys(floatingbody.dofs)))
        data = hydrodynamic_coefficients(floatingbody, parameters, DirectBEM())
        return vcat(real.(vec(data.excitation_force)),imag.(vec(data.excitation_force)))
    end

    backend = AutoForwardDiff()



    for omega in omegas
        @testset "Verify sensitivity of added mass and damping wrt Omega: $omega" begin
            A_and_B_FD = FiniteDifferences.central_fdm(5, 1)(A_and_B_vec, omega)
            A_and_B_AD = derivative(A_and_B_vec, backend, omega)
            @test A_and_B_AD !== nothing
            @test typeof(A_and_B_AD) == Vector{Float64}
            @test A_and_B_AD ≈ A_and_B_FD atol=1e-6 rtol=1e-6
        end
        @testset "Verify sensitivity of excitation force wrt Omega: $omega" begin
            F_ex_FD = FiniteDifferences.central_fdm(5, 1)(F_ex_vec, omega)
            F_ex_AD = derivative(F_ex_vec, backend, omega)
            @test F_ex_AD !== nothing
            @test typeof(F_ex_AD) == Vector{Float64}
            @test F_ex_AD ≈ F_ex_FD atol=1e-6 rtol=1e-6
        end
    end  
end



# ---------------------------------------------------------------------------
# Sensitivities of all hydrodynamic outputs: AD (via DifferentiationInterface)
# vs finite differences, over the combinations of
#   Rankine pair       DelhommeauRankine + DelhommeauRankineReflected | Rankine + RankineReflected
#   wave Green's fn    GFWu | ExactGuevelDelhommeau
#   BEM formulation    Direct (forward speed 0 only) | Indirect (forward speeds 0 and 2)
#   AD backend         ForwardDiff | Enzyme (reverse)
#   shape              cylinder | frustum
#   input              mesh (radial panel position) | frequency | wave direction | forward speed
# Every case is its own nested testset, so a failure prints the exact
# combination and the output entry / input that disagree.
#
# By default only a representative subset runs (every option appears at
# least once; see `in_default_tier`). Set MH_EXHAUSTIVE_AD=true to run every
# combination: Enzyme compiles once per Green's function tuple (minutes) and
# ExactGuevelDelhommeau takes seconds per solve, so the full set takes hours.
#
# Note, for differentiating wrt mesh (radial panel position),
# GZ within rankine.jl has zero value but nonzero partials
# (stacked cylinder panels are coplanar). `_delhommeau_atan` zeroes
# the solid-angle atan for |GZ| < 1e-4 * panel radius, so the primal
# is flat in that band while the AD derivative (through `_abs_ad` /
# `_sign_ad`) is the one of the smooth underlying integral. The finite
# differences below therefore use a fixed step large enough to leave
# that band instead of an adaptive step that can stay inside it.
# On the cylinder, exactly stacked panels also sit on the cone of the GFWu
# approximation (nonzero dGF/dhh at hh = 0), where AD returns a one-sided
# derivative; this is why the relative tolerance is 5%.
# ---------------------------------------------------------------------------

import FiniteDifferences

const SENS_HAS_ENZYME = try
    @eval import Enzyme
    true
catch
    false
end
const SENS_EXHAUSTIVE = lowercase(get(ENV, "MH_EXHAUSTIVE_AD", "false")) in ("1", "true", "yes")

const SENS_OMEGAS = [1.0, 1.5]       # frequencies [rad/s]
const SENS_BETAS = [0.0, pi/4]       # incident wave angles [rad]
const SENS_SPEEDS = [0.0, 2.0]       # forward speeds in x [m/s] (Indirect only)
const SENS_DOFS = ["Heave", "Pitch"]
const SENS_ROTATION_CENTER = [1.0, 1.0, 0.0]  # offset for nonzero off-diagonal terms
const SENS_FD_STEP = 1e-3
const SENS_RTOL = 5e-2
const SENS_ATOL = 1e-6

const SENS_RANKINE_PAIRS = [
    ("DelhommeauRankine", (DelhommeauRankine(), DelhommeauRankineReflected())),
    ("Rankine", (Rankine(), RankineReflected())),
]
const SENS_WAVE_GFS = [("GFWu", GFWu()), ("ExactGuevelDelhommeau", ExactGuevelDelhommeau())]
const SENS_BEM = [("Direct", DirectBEM, [0.0]), ("Indirect", IndirectBEM, SENS_SPEEDS)]
const SENS_AD = ["ForwardDiff", "Enzyme"]
const SENS_SHAPES = ["cylinder", "frustum"]
const SENS_INPUTS = [:mesh, :frequency, :direction, :speed]
const SENS_INPUT_NAMES = Dict(:mesh => "mesh (radial panel position)", :frequency => "frequency",
    :direction => "wave direction", :speed => "forward speed")

sens_backend(ad) = ad == "ForwardDiff" ? AutoForwardDiff() :
    AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)

# Explicit linear interpolation instead of `range`: Base's Float64 ranges use
# TwicePrecision bit-splitting, which Enzyme differentiates incorrectly
# (d/da of collect(range(a, 0, length=4)) comes out wrong), while ForwardDiff
# Duals take a generic path. So differentiated quantities never go through `range`.
function sens_mesh(x::AbstractVector)
    r_side = x[1:end-1]
    draft = x[end]
    n_side = length(r_side)
    z_side = [-1e-3 + (1e-3 - draft) * (i - 1) / (n_side - 1) for i in 1:n_side]
    n_bottom = 3
    z_bottom = ones(n_bottom) * (-draft)
    r_bottom = [r_side[end] * (1 - i / n_bottom) for i in 1:n_bottom]
    return axisymmetric_mesh(vcat(r_side, r_bottom), vcat(z_side, z_bottom), 10, false)
end

sens_design(shape) = vcat(shape == "cylinder" ? fill(0.5, 3) : collect(range(0.5, 0.1, length=3)), 1.5)

function sens_problem(x, ω, β, U)
    floatingbody = FloatingBody(sens_mesh(x), SENS_DOFS, SENS_ROTATION_CENTER, "floating body")
    parameters = (wave_frequencies=ω, wave_directions=β,
        radiating_dofs=collect(keys(floatingbody.dofs)),
        influenced_dofs=collect(keys(floatingbody.dofs)), forward_speeds=U)
    return floatingbody, parameters
end

# Excitation force is FK + diffraction, both checked here, so it is left out.
sens_flatten(d) = vcat(vec(d.added_mass), vec(d.radiation_damping),
    real.(vec(d.Froude_Krylov_force)), imag.(vec(d.Froude_Krylov_force)),
    real.(vec(d.diffraction_force)), imag.(vec(d.diffraction_force)))

# All outputs as a function of one input vector `v`; the other inputs come from `base`.
# `input` and `base` are DI `Constant` contexts, so Enzyme compiles once per formulation.
function sens_outputs(v, input, base)
    x = input === :mesh ? v : base.x
    ω = input === :frequency ? v : base.ω
    β = input === :direction ? v : base.β
    U = input === :speed ? v : base.U
    floatingbody, parameters = sens_problem(x, ω, β, U)
    return sens_flatten(hydrodynamic_coefficients(floatingbody, parameters, base.formulation))
end
sens_output_entry(v, i, input, base) = sens_outputs(v, input, base)[i]

# Human-readable name of every entry of `sens_outputs`, e.g.
# "radiation_damping[influenced=Pitch, radiating=Heave, ω=1.0, U=0.0, β=0.0]".
# Built from the raw array layout of `hydrodynamic_coefficients`:
#   radiation (influenced, radiating, ω, U, β), forces (influenced, ω, β, U).
# (`label_hydrodynamic_coefficients` assumes 4-D radiation arrays when the
# forward speed is 0, but the raw arrays are always 5-D, so it is not used.)
function sens_output_labels(base)
    floatingbody, parameters = sens_problem(base.x, base.ω, base.β, base.U)
    data = hydrodynamic_coefficients(floatingbody, parameters, base.formulation)
    dofs = collect(keys(floatingbody.dofs))
    radiation_axes = [("influenced", dofs), ("radiating", dofs), ("ω", base.ω), ("U", base.U), ("β", base.β)]
    force_axes = [("influenced", dofs), ("ω", base.ω), ("β", base.β), ("U", base.U)]
    labels = String[]
    for (field, part, axes_) in ((:added_mass, "", radiation_axes), (:radiation_damping, "", radiation_axes),
            (:Froude_Krylov_force, "Re ", force_axes), (:Froude_Krylov_force, "Im ", force_axes),
            (:diffraction_force, "Re ", force_axes), (:diffraction_force, "Im ", force_axes))
        A = getproperty(data, field)
        @assert size(A) == Tuple(length(vals) for (_, vals) in axes_) "unexpected layout of $field: $(size(A))"
        for ci in CartesianIndices(A)
            coords = join(["$name=$(vals[ci[k]] isa Real ? round(vals[ci[k]], digits=4) : vals[ci[k]])"
                           for (k, (name, vals)) in enumerate(axes_)], ", ")
            push!(labels, "$(part)$(field)[$coords]")
        end
    end
    return labels
end

function sens_input_labels(input, base)
    input === :mesh && return vcat(["r_side[$i]" for i in 1:length(base.x)-1], "draft")
    input === :frequency && return ["ω=$ω" for ω in base.ω]
    input === :direction && return ["β=$(round(β, digits=4))" for β in base.β]
    return ["U=$U" for U in base.U]
end

function sens_jacobian_fd(input, base, v0)
    method = FiniteDifferences.central_fdm(5, 1)
    return reduce(hcat, [
        method(t -> sens_outputs(v0 .+ t .* ((1:length(v0)) .== j), input, base), 0.0, SENS_FD_STEP)
        for j in eachindex(v0)])
end

function sens_jacobian_ad(ad, input, base, v0)
    backend = sens_backend(ad)
    if ad == "ForwardDiff"
        return DifferentiationInterface.jacobian(sens_outputs, backend, v0,
            DifferentiationInterface.Constant(input), DifferentiationInterface.Constant(base))
    end
    # Enzyme reverse on this vector-valued pipeline fails ("Conversion of boxed
    # type Vector{Float64}"), but scalar gradients work: build J one row at a time.
    m = length(sens_outputs(v0, input, base))
    rows = [DifferentiationInterface.gradient(sens_output_entry, backend, copy(v0),
                DifferentiationInterface.Constant(i), DifferentiationInterface.Constant(input),
                DifferentiationInterface.Constant(base))' for i in 1:m]
    return reduce(vcat, rows)
end

# Representative subset: every Green's function, BEM method, AD backend, shape
# and input is exercised at least once, while keeping the run to minutes.
function in_default_tier(ad, rankine, wave, bem, shape, input)
    if ad == "ForwardDiff"
        wave == "GFWu" && return true
        # ExactGuevelDelhommeau takes seconds per solve: one case per BEM method.
        return shape == "frustum" && ((rankine == "DelhommeauRankine" && bem == "Indirect" && input === :frequency) ||
                                      (rankine == "Rankine" && bem == "Direct" && input === :mesh))
    end
    # Enzyme: one compile per Green's function tuple, so two formulations only.
    wave == "GFWu" || return false
    shape == "frustum" || return false
    rankine == "Rankine" && bem == "Direct" && input in (:mesh, :frequency) && return true
    rankine == "DelhommeauRankine" && bem == "Indirect" && input === :mesh && return true
    return false
end

@testset "AD vs finite differences of hydrodynamic outputs ($(SENS_EXHAUSTIVE ? "exhaustive" : "default subset"))" begin
    for ad in SENS_AD, (bem, BEM, speeds) in SENS_BEM,
            (rankine, rankine_gfs) in SENS_RANKINE_PAIRS, (wave, wave_gf) in SENS_WAVE_GFS,
            shape in SENS_SHAPES, input in SENS_INPUTS

        input === :speed && bem == "Direct" && continue  # Direct BEM has no forward speed
        SENS_EXHAUSTIVE || in_default_tier(ad, rankine, wave, bem, shape, input) || continue
        case = "$ad | $bem BEM | $rankine + $wave | $shape | d/d $(SENS_INPUT_NAMES[input])"

        @testset "$case" begin
            if ad == "Enzyme" && !SENS_HAS_ENZYME
                @test_skip "Enzyme not installed"
            else
                base = (x=sens_design(shape), ω=SENS_OMEGAS, β=SENS_BETAS, U=speeds,
                    formulation=BEM((rankine_gfs..., wave_gf)))
                v0 = input === :mesh ? base.x : input === :frequency ? base.ω :
                     input === :direction ? base.β : base.U

                J_FD = sens_jacobian_fd(input, base, v0)
                J_AD = sens_jacobian_ad(ad, input, base, v0)
                out_labels = sens_output_labels(base)
                in_labels = sens_input_labels(input, base)

                @test size(J_AD) == size(J_FD) == (length(out_labels), length(in_labels))
                @test all(isfinite, J_AD)
                for j in axes(J_AD, 2), i in axes(J_AD, 1)
                    @testset "∂ $(out_labels[i]) / ∂ $(in_labels[j])" begin
                        @test J_AD[i, j] ≈ J_FD[i, j] atol=SENS_ATOL rtol=SENS_RTOL
                    end
                end
            end
        end
    end
end
