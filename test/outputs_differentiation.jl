using Test
using MarineHydro
using PyCall
using LinearAlgebra
using DifferentiationInterface 
import ForwardDiff 
import Zygote
using FiniteDifferences

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
        data = compute_hydrodynamic_coefficients(parameters, floatingbody)
        return vcat(vec(data.added_mass), vec(data.radiation_damping)) 
    end

    # Incident + diffraction solve function
    function F_ex_vec(w)
        parameters = (wave_frequencies=[w],
            wave_directions=[beta],
            influenced_dofs=collect(keys(floatingbody.dofs)))
        data = compute_hydrodynamic_coefficients(parameters, floatingbody)
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



# Tests sensitivities wrt all inputs

# Note, for differentiating wrt mesh (radial panel position), 
# GZ within rankine.jl has zero value but nonzero partials.
# In this case, abs(GZ=0) and sign(GZ=0) are not
# differentiatiable, but we used smooth versions of these
# functions that are.

omegas_vals = [1.0, 1.5] # frequencies [rad/s]
betas_vals = [0.0, pi/4] # incident wave angle [rad]
forward_speeds_vals = [0.0, 2.0] # forward speed in x direction [m/s]
t_DOFs = ["Heave"] # translational DOFs
r_DOFs = ["Pitch"] # rotational DOFs
DOFs = [t_DOFs; r_DOFs] # all DOFs
r_initial = 0.5
n_side = 3 # number of design vars-1 (draft is also a design var)
@testset "Sensitivity wrt $ind_var_name for $shape" for shape in ["cylinder", "frustrum"], ind_var_name in ["mesh (radial panel position)", "frequency", "wave direction", "forward speed"]

    if shape == "cylinder"
        r_side = ones(n_side) * r_initial
    elseif shape == "frustrum"
        r_side = collect(range(r_initial, 0.1, length=n_side))
    end

    draft = 1.5
    x_val = vcat(r_side,draft)

    function arbitrary_axisymmetric_mesh(x::Vector, show_plot)
        r_side = x[1:end-1]
        draft = x[end]
        n_side = length(r_side)
        z_side = collect(range(-1e-3, -draft, length=n_side))
        panel_z_len = abs(z_side[1]-z_side[2])
        last_z = z_side[end]
        n_bottom = 3
        z_bottom = ones(n_bottom) * (-draft)
        r_bottom = collect(range(r_side[end], 0.0, length=n_bottom+1))[2:end]
        z_vals = vcat(z_side, z_bottom)
        r_vals = vcat(r_side, r_bottom)
        n_theta = 10
        MH_mesh = axisymmetric_mesh(r_vals,z_vals,n_theta,show_plot)
        return MH_mesh
    end

    function compute_vals(x, omegas, betas, forward_speeds)
        mesh = arbitrary_axisymmetric_mesh(x, false)
        rigid_dof_list = DOFs
        rotation_center = [1.0, 1.0, 0.0] # off set for nonzero off-diagoinal elements
        floatingbody = FloatingBody(mesh, rigid_dof_list, rotation_center, "floating body")
        parameters = (wave_frequencies=omegas,
            wave_directions=betas,
            radiating_dofs=collect(keys(floatingbody.dofs)),
            influenced_dofs=collect(keys(floatingbody.dofs)),
            forward_speeds=forward_speeds)
        data = compute_hydrodynamic_coefficients(parameters, floatingbody, direct=false)
        return vcat(
            vec(data.added_mass),
            vec(data.radiation_damping),
            real.(vec(data.Froude_Krylov_force)),
            imag.(vec(data.Froude_Krylov_force)),
            real.(vec(data.diffraction_force)),
            imag.(vec(data.diffraction_force)),
            real.(vec(data.excitation_force)),
            imag.(vec(data.excitation_force))) 
    end

    function compute_wrt_ind_var(ind_var)
        if ind_var_name=="mesh (radial panel position)"
            return compute_vals(ind_var, omegas_vals, betas_vals, forward_speeds_vals)
        elseif ind_var_name=="frequency"
            return compute_vals(x_val, ind_var, betas_vals, forward_speeds_vals)
        elseif ind_var_name=="wave direction"
            return compute_vals(x_val, omegas_vals, ind_var, forward_speeds_vals)
        elseif ind_var_name=="forward speed"
            return compute_vals(x_val, omegas_vals, betas_vals, ind_var)
        end
    end

    if ind_var_name=="mesh (radial panel position)"
        ind_var_val = x_val
    elseif ind_var_name=="frequency"
        ind_var_val = omegas_vals
    elseif ind_var_name=="wave direction"
        ind_var_val = betas_vals
    elseif ind_var_name=="forward speed"
        ind_var_val = forward_speeds_vals
    end
    
    method = central_fdm(5, 1; max_range=1e-4)
    J_FD = FiniteDifferences.jacobian(method, ind_var -> compute_wrt_ind_var(ind_var), ind_var_val)[1]

    backend = AutoForwardDiff()
    J_AD = value_and_jacobian(ind_var -> compute_wrt_ind_var(ind_var), backend, ind_var_val)[2]

    @testset "Verify jacobian matrix with finite difference" begin
        n_row = length(J_AD[:,1])
        n_col = length(J_AD[1,:])
        for i in 1:n_row
            for j in 1:n_col
                @testset "Row $i column $j" begin            
                    @test  J_AD[i,j] ≈ J_FD[i,j] atol=1e-6 rtol = 1e-4
                end
            end
        end
    end
end
 

