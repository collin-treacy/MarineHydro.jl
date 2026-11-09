using MarineHydro
using PyCall
using StaticArrays
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Printf

mesh_from_cpt = false
static_arrays_mesh = false

if mesh_from_cpt
    cpt = pyimport("capytaine")
    function mesh_r(r)
        Mesh(cpt.mesh_sphere(
            name="sphere", radius=r, center=(0.0, 0.0, 0.0),
            resolution=(6, 6)).immersed_part())
    end
    fd_mesh_rules!(mesh_r)
else  
    function mesh_r(r)
        d1=0.16
        d2=0.37
        # r1=0.88
        r2=0.35
        wavebot_mesh(r,r2,d1,d2,(3, 5, 3),10)
    end
end

if static_arrays_mesh
    mesh_r_(r) = StaticArraysMesh(mesh_r(r))
else
    mesh_r_(r) = mesh_r(r)
end

body(r) = FloatingBody(mesh_r_(r), [:Heave], "floatingbody")



const formulation = DirectBEM()                # default: direct + Wu
const r0 = 1.0
const ω0 = 1.03
const β0 = 0.0
const ωs = [0.8, 1.03, 1.4]
const fwd = AutoForwardDiff()
const rev = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))


radiation_solve(r, ω, form=formulation) = solve(RadiationProblem(body(r), ω), form)

# sol = radiation_solve(r0, ω0)
# @printf("A₃₃ = %.6f\nB₃₃ = %.6f\n", added_mass(sol).Heave, radiation_damping(sol).Heave)

added_mass_r(r) = added_mass(radiation_solve(r, ω0)).Heave
added_mass_ω(ω) = added_mass(radiation_solve(r0, ω)).Heave

dA_dr  = DI.derivative(added_mass_r, fwd, r0)
# dA_dr′ = DI.derivative(added_mass_r, rev, r0)   # first Enzyme call compiles
# dA_dω  = DI.derivative(added_mass_ω, fwd, ω0)
# dA_dω′ = DI.derivative(added_mass_ω, rev, ω0)

println("dA_dr: $dA_dr")

# @printf("dA₃₃/dr   ForwardDiff %.6f   Enzyme %.6f\n", dA_dr, dA_dr′)
# @printf("dA₃₃/dω   ForwardDiff %.6f   Enzyme %.6f\n", dA_dω, dA_dω′)

using LinearAlgebra
using ForwardDiff

# 1. Differentiating wrt b: f(b) = sum(A \ b)
function solve_b(b)
    A = [1.0 1.0; 
         1.0 3.0]
    x = A \ b
    return sum(x)
end

# 2. Differentiating wrt A: f(A_vec) = sum(A \ b)
function solve_A(A_flat)
    A = reshape(A_flat, 2, 2)
    b = [1.0, 2.0]
    x = A \ b
    return sum(x)
end

b0 = [2.0, 3.0]
A0_flat = [3.0, 1.0, 1.0, 3.0]

grad_b = DI.gradient(solve_b, fwd, b0)
grad_A = DI.gradient(solve_A, fwd, A0_flat)

println("Gradient wrt b: ", grad_b)
println("Gradient wrt A: ", grad_A)