using MarineHydro
using DifferentiationInterface 
import ForwardDiff 


# Hydrostatics
function hydrostatic_program(x)
    r1 = x[1]
    r2 = x[2]
    d1 = x[3]
    d2 = x[4]
    rho_w = 1025.0 # density of fluid [kg/m^3]
    g = 9.81 # acceleration due to gravity [m/s^2]
    K = rho_w * g * pi * r1^2 # hydrostatic stiffness [kg/s^2]
    d3 = (r2/(r1-r2))*d2
    dt = d2+d3
    Vt = pi * r1^2 * d1 + pi * r1^2 * dt/3 - pi * r2^2 * d3/3
    M = rho_w * Vt # mass of body [kg]
    display("Mass: $M")
    return [M, K]
end

# Radiation
function radiation_program(mesh, omega, dof) 
    A, B = calculate_radiation_forces(mesh,dof,omega)
    # display("Damping: $B")
    return [A, B]
end

# Diffraction
function diffraction_program(mesh, omega, dof) 
    F_D = DiffractionForce(mesh,omega,dof)
    return [real(F_D),imag(F_D)]
end

function incident_program(mesh, omega, dof) 
    F_FK = FroudeKrylovForce(mesh,omega,dof)
    return [real(F_FK),imag(F_FK)]
end

# Everything
dof = [0.0,0.0,1.0]

function compute_for_omega(mesh, omega)
    # r1 = x[1]
    # r2 = x[2]
    # d1 = x[3]
    # d2 = x[4]

    # M, K = hydrostatic_program(r1, r2, d1, d2)
    A, B = radiation_program(mesh, omega, dof)
    F_D_real, F_D_imag = diffraction_program(mesh, omega, dof) 
    F_FK_real, F_FK_imag = incident_program(mesh, omega, dof)
    # add negatives to imag since wot uses +i omega t instead of -i omega t
    return [A, B, F_D_real, -F_D_imag, F_FK_real, -F_FK_imag]
end

# function compute_for_omega(x, omega)
#     # set_rho!(1025.0)
#     mesh = wavebot_mesh(x[1],x[2],x[3],x[4],(3,5,3),20)
#     val = all_programs(mesh, x, omega)
#     return val
# end

backend = AutoForwardDiff()

function WavebotMesh(x_val)
    return wavebot_mesh(x_val[1],x_val[2],x_val[3],x_val[4],(3,3,3),10)
end




function compute_vals(x, omegas)
    set_rho!(1025.0)
    mesh = WavebotMesh(x)
    M_val, K_val = hydrostatic_program(x)
    results_per_omega = [compute_for_omega(mesh, omega) for omega in omegas]
    vec_reduced = reduce(vcat, results_per_omega)
    
    return [M_val; K_val; vec_reduced]
end

function compute_all(x_val, omegas)
    return value_and_jacobian(x -> compute_vals(x, omegas), backend, x_val)
end

function surface_area(x)
    mesh = WavebotMesh(x)
    return sum(mesh.areas)
end

function compute_SA_and_grad(x_val)
    return value_and_gradient(x -> surface_area(x), backend, x_val)
end

function draft_fun(x)
    mesh = WavebotMesh(x)
    draft = abs(minimum(mesh.vertices[:,3]))
    return draft
end

function compute_draft_and_grad(x_val)
    return value_and_gradient(x -> draft_fun(x), backend, x_val)
end





# # Hydrostatics
# function hydrostatic_program(r1, r2, d1, d2)
#     rho_w = 1025.0 # density of fluid [kg/m^3]
#     g = 9.81 # acceleration due to gravity [m/s^2]
#     K = rho_w * g * pi * r1^2 # hydrostatic stiffness [kg/s^2]
#     d3 = (r2/(r1-r2))*d2
#     dt = d2+d3
#     Vt = pi * r1^2 * d1 + pi * r1^2 * dt/3 - pi * r2^2 * d3/3
#     M = rho_w * Vt # mass of body [kg]
#     # display("Mass: $M")
#     return [M, K]
# end

# # Radiation
# function radiation_program(mesh, omega, dof) 
#     A, B = calculate_radiation_forces(mesh,dof,omega)
#     # display("Damping: $B")
#     return [A, B]
# end

# # Diffraction
# function diffraction_program(mesh, omega, dof) 
#     F_D = DiffractionForce(mesh,omega,dof)
#     return [real(F_D),imag(F_D)]
# end

# function incident_program(mesh, omega, dof) 
#     F_FK = FroudeKrylovForce(mesh,omega,dof)
#     return [real(F_FK),imag(F_FK)]
# end

# # Everything
# dof = [0.0,0.0,1.0]

# function all_programs(mesh, x, omega)
#     r1 = x[1]
#     r2 = x[2]
#     d1 = x[3]
#     d2 = x[4]

#     M, K = hydrostatic_program(r1, r2, d1, d2)
#     A, B = radiation_program(mesh, omega, dof)
#     F_D_real, F_D_imag = diffraction_program(mesh, omega, dof) 
#     F_FK_real, F_FK_imag = incident_program(mesh, omega, dof)
#     F_ex_real = F_D_real + F_FK_real
#     F_ex_imag = F_D_imag + F_FK_imag
#     # add negatives to imag since wot uses +i omega t instead of -i omega t
#     return [M, K, A, B, F_D_real, -F_D_imag, F_FK_real, -F_FK_imag, F_ex_real, -F_ex_imag]
# end

# function compute_for_omega(x, omega)
#     set_rho!(1025.0)
#     mesh = wavebot_mesh(x[1],x[2],x[3],x[4],(3,3,3),10)
#     val = all_programs(mesh, x, omega)
#     return val
# end

# backend = AutoForwardDiff()

# function compute_all(x_val, omegas)
#     AD_grads = [value_and_jacobian(x -> compute_for_omega(x, omega), backend, x_val) for omega in omegas]
#     return AD_grads
# end


# x_val = [0.88, 0.35, 0.17, 0.37]*0.1
# hydrostatic_program(x_val)
# omegas = [0.3, 2*0.3]*2*pi
# AD_grads = compute_all(x_val, omegas)
# display(AD_grads)

# SA_and_grad = compute_SA_and_grad(x_val)
# display(SA_and_grad)

# draft_and_grad = compute_draft_and_grad(x_val)
# display(draft_and_grad)