#!/usr/bin/python3
import numpy as np

def forward_mode_ratio(n_columns):
    return 1.25 + n_columns
def reverse_mode_ratio(n_columns):
    return 1.25 + 2.25*n_columns

# Flops to evaluate matrix-matrix product cost
# Matrix C = AB, A \in m x n, B \in n x l
def matrix_matrix_flops(m,n,l):
    return 2*m*n*l - m*l
# Flops to evaluate matrix-matrix product cost
# Matrix v = Ab, A \in m x n, b \in n x 1
def matrix_vector_flops(m,n):
    return matrix_matrix_flops(m,n,1)


def n_basis(dim,poly):
    n_basis_1d = poly+1
    return pow(n_basis_1d,dim)

def n_dofs_cell(dim,poly,n_state):
    nb = n_basis(dim,poly)
    return nb*n_state

def interpolate(dim, poly, n_state):
    nb = n_basis(dim,poly)
    return matrix_vector_flops(n_state,nb)

def interpolate_grad(dim, poly, n_state):
    return dim*interpolate(dim,poly,n_state)

def cofactor(dim):
    return pow(dim,dim)
def analytical_flux(dim):
    # Euler flux
    return 8*dim + 7
def numerical_flux(dim):
    # Roe flux with entropy fix
    return 46*dim+130

def residual_volume_flops(dim,poly):
    nb = n_basis(dim,poly)
    n_state = dim+2

    interpolate_solution_value = interpolate(dim, poly, n_state)
    interpolate_metric_gradient = interpolate_grad(dim, poly, dim)
    cofactor_matrix = cofactor(dim)

    flux = analytical_flux(dim)
    cofactor_flux = matrix_matrix_flops(dim,dim,n_state)
    diff_cofactor_flux = 2*dim*n_state*nb
    quad_weight = 1

    flops_vol_quadrature = interpolate_solution_value \
                          + interpolate_metric_gradient \
                          + cofactor_matrix \
                          + flux \
                          + cofactor_flux \
                          + diff_cofactor_flux \
                          + quad_weight
    n_quad = n_basis(dim,poly)

    flops_vol_quadrature = flops_vol_quadrature * n_quad + n_quad

    return flops_vol_quadrature

def residual_face_flops(dim,poly):
    nb = n_basis(dim,poly)
    n_state = dim+2

    interpolate_solution_value = interpolate(dim, poly, n_state)
    interpolate_metric_gradient = interpolate_grad(dim, poly, dim)
    cofactor_matrix = cofactor(dim)

    cofactor_normal = matrix_vector_flops(dim,dim)
    num_flux = numerical_flux(dim)
    #flux_normal = matrix_matrix_flops(n_state, dim, dim)
    # Basis functions from each cell side
    basis_flux_normal = 2*2*n_state*nb
    #add_to_res = 2*nb
    quad_weight = 2

    flops_vol_quadrature = interpolate_solution_value \
                          + interpolate_metric_gradient \
                          + cofactor_matrix \
                          + cofactor_normal \
                          + num_flux \
                          + basis_flux_normal \
                          + quad_weight
    n_quad = n_basis(dim-1,poly)

    flops_vol_quadrature = flops_vol_quadrature * n_quad + n_quad

    return flops_vol_quadrature

def residual_total_flops(dim,poly):
    cost = residual_volume_flops(dim,poly)
    cost = cost + dim*residual_face_flops(dim,poly)

    return cost

def dRdW_vmult_flops_per_cell(dim,poly):
    nb = n_basis(dim,poly)
    n_state = dim+2
    n_stencil = 1 + 2*dim
    n_dofs_per_cell = n_dofs_cell(dim,poly,n_state)

    cost = 2 * nb * n_state * n_stencil * n_dofs_per_cell

    return cost

def dRdX_vmult_flops_per_cell(dim,poly):
    nb = n_basis(dim,poly)
    n_state = dim+2
    n_stencil = 1
    n_dofs_per_cell = n_dofs_cell(dim,poly,dim)

    cost = 2 * nb * n_state * n_stencil * n_dofs_per_cell

    return cost

def dRdW_vmult_cost(dim,poly):
    flops = dRdW_vmult_flops_per_cell(dim,poly)
    r_flops = residual_total_flops(dim,poly)
    return flops / r_flops

def dRdX_vmult_cost(dim,poly):
    flops = dRdX_vmult_flops_per_cell(dim,poly)
    r_flops = residual_total_flops(dim,poly)
    return flops / r_flops
def d2RdWdW_vmult_cost(dim,poly):
    return dRdW_vmult_cost(dim,poly)
def d2RdWdX_vmult_cost(dim,poly):
    return dRdX_vmult_cost(dim,poly)
def d2RdXdW_vmult_cost(dim,poly):
    return dRdX_vmult_cost(dim,poly)
def d2RdXdX_vmult_cost(dim,poly):
    n_state = dim + 2
    cost = dRdX_vmult_cost(dim,poly)
    cost = cost * dim / n_state
    return cost

def AD_dRdX_vector_cost():
    return forward_mode_ratio(1)
def AD_vector_dRdX_cost():
    return reverse_mode_ratio(1)
def AD_Hessian_vector_cost():
    return forward_mode_ratio(1) * reverse_mode_ratio(1)

def form_dRdW_AD_cost(dim,poly):
    n_state = dim+2
    n_dofs_per_cell = n_dofs_cell(dim,poly,n_state)

    v_flops = residual_volume_flops(dim,poly)
    f_flops = residual_face_flops(dim,poly)
    r_flops = residual_total_flops(dim,poly)

    form_dRdW_volume_flops = forward_mode_ratio(n_dofs_per_cell) * v_flops
    form_dRdW_face_flops   = forward_mode_ratio(2*n_dofs_per_cell) * f_flops
    form_dRdW_flops        = form_dRdW_volume_flops + dim * form_dRdW_face_flops
    form_dRdW_cost         = form_dRdW_flops / r_flops

    return form_dRdW_cost

def form_dRdX_AD_cost(dim,poly):
    nx_dofs_per_cell = n_dofs_cell(dim,poly,dim)

    v_flops = residual_volume_flops(dim,poly)
    f_flops = residual_face_flops(dim,poly)
    r_flops = residual_total_flops(dim,poly)

    form_dRdX_volume_flops = forward_mode_ratio(nx_dofs_per_cell) * v_flops
    form_dRdX_face_flops   = forward_mode_ratio(nx_dofs_per_cell) * f_flops
    form_dRdX_flops        = form_dRdX_volume_flops + dim * form_dRdX_face_flops
    form_dRdX_cost         = form_dRdX_flops / r_flops

    return form_dRdX_cost
    
    

residual_table = ''
dRdW_vmult_table = ''
dRdX_vmult_table = ''
d2RdXdX_vmult_table = ''
dRdW_form_table = ''
dRdX_form_table = ''
d2RddW_form_table = ''
d2RddX_form_table = ''

worth_form_dRdX = ''
worth_form_d2RdWdW = ''
worth_form_d2RdWdX = ''
worth_form_d2RdXdX = ''
for poly in range(1,5):
    residual_table = residual_table + ('%d ' % poly)
    dRdW_vmult_table = dRdW_vmult_table + ('%d ' % poly)
    dRdX_vmult_table = dRdX_vmult_table + ('%d ' % poly)
    d2RdXdX_vmult_table = d2RdXdX_vmult_table + ('%d ' % poly)
    dRdW_form_table = dRdW_form_table + ('%d ' % poly)
    dRdX_form_table = dRdX_form_table + ('%d ' % poly)
    d2RddW_form_table = d2RddW_form_table + ('%d ' % poly)
    d2RddX_form_table = d2RddX_form_table + ('%d ' % poly)

    worth_form_dRdX = worth_form_dRdX + ('%d ' % poly)
    worth_form_d2RdWdW = worth_form_d2RdWdW + ('%d ' % poly)
    worth_form_d2RdWdX = worth_form_d2RdWdX + ('%d ' % poly)
    worth_form_d2RdXdX = worth_form_d2RdXdX + ('%d ' % poly)
    for dim in range(2,4):
        n_state = dim + 2
        v_flops = residual_volume_flops(dim,poly)
        f_flops = dim*residual_face_flops(dim,poly)
        r_flops = residual_total_flops(dim,poly)
        print ('dim: %d poly: %d' % (dim,poly))
        print ('volume residual flops: %d' % v_flops)
        print ('face residual flops: %d' % f_flops)
        print ('total residual flops: %d' % r_flops)
        residual_table = residual_table + (
            ' & %d  &  %d  &  %d  ' %
                ( residual_volume_flops(dim,poly)
                , residual_face_flops(dim,poly)
                , residual_total_flops(dim,poly))
                )

        dRdW_vmult_flops = dRdW_vmult_flops_per_cell(dim,poly)
        dRdW_vmult_rcost = dRdW_vmult_cost(dim,poly)
        dRdW_vmult_table = dRdW_vmult_table + ('& %d ' % dRdW_vmult_flops)
        dRdW_vmult_table = dRdW_vmult_table + ('& %4.1f ' % dRdW_vmult_rcost)
        print ('dRdW vmult flops: %d' % dRdW_vmult_flops)

        dRdX_vmult_flops = dRdX_vmult_flops_per_cell(dim,poly)
        dRdX_vmult_rcost = dRdX_vmult_cost(dim,poly)
        dRdX_vmult_table = dRdX_vmult_table + ('& %d ' % dRdX_vmult_flops)
        dRdX_vmult_table = dRdX_vmult_table + ('& %4.2f ' % dRdX_vmult_rcost)
        print ('dRdX vmult flops: %d' % dRdX_vmult_flops)

        d2RdXdX_vmult_flops = dRdX_vmult_flops * dim/n_state
        d2RdXdX_vmult_rcost = d2RdXdX_vmult_cost(dim,poly)
        d2RdXdX_vmult_table = d2RdXdX_vmult_table + ('& %d ' % d2RdXdX_vmult_flops)
        d2RdXdX_vmult_table = d2RdXdX_vmult_table + ('& %4.2f ' % d2RdXdX_vmult_rcost)

        form_dRdW_cost = form_dRdW_AD_cost(dim,poly)
        dRdW_form_table = dRdW_form_table + ('& %4.1f ' % form_dRdW_cost)

        form_dRdX_cost = form_dRdX_AD_cost(dim,poly)
        dRdX_form_table = dRdX_form_table + ('& %4.1f ' % form_dRdX_cost)

        d2RddW_ratio = form_dRdW_cost * reverse_mode_ratio(1)
        d2RddX_ratio = form_dRdX_cost * reverse_mode_ratio(1)
        d2RddW_form_table = d2RddW_form_table + ('& %4.1f ' % d2RddW_ratio)
        d2RddX_form_table = d2RddX_form_table + ('& %4.1f ' % d2RddX_ratio)

        # Dividing by 2 its formationg since it can be used for the transpose too
        iterations_dRdX       = (form_dRdX_cost/2) / ( AD_dRdX_vector_cost() - dRdX_vmult_rcost)
        worth_form_dRdX       = worth_form_dRdX + ('& %4.0f ' % iterations_dRdX)

        rel_cost_form_d2RdWdW = d2RddW_ratio
        rel_cost_vmult_d2RdWdW_v = dRdW_vmult_flops/r_flops
        rel_cost_AD_d2RdWdW_v = AD_Hessian_vector_cost()
        iterations_d2RdWdW = rel_cost_form_d2RdWdW / ( rel_cost_AD_d2RdWdW_v - rel_cost_vmult_d2RdWdW_v)
        worth_form_d2RdWdW = worth_form_d2RdWdW + ('& %4.0f ' % iterations_d2RdWdW)

        rel_cost_form_d2RdWdX = d2RddX_ratio / 2 # Since we use it for 2 different matrices.
        rel_cost_vmult_d2RdWdX_v = dRdX_vmult_flops/r_flops
        rel_cost_AD_d2RdWdX_v = AD_Hessian_vector_cost()
        iterations_d2RdWdX = rel_cost_form_d2RdWdX / ( rel_cost_AD_d2RdWdX_v - rel_cost_vmult_d2RdWdX_v)
        worth_form_d2RdWdX = worth_form_d2RdWdX + ('& %4.0f ' % iterations_d2RdWdX)

        rel_cost_form_d2RdXdX = d2RddX_ratio # Since we use it for 2 different matrices.
        rel_cost_vmult_d2RdXdX_v = dRdX_vmult_flops/r_flops*dim/(n_state)
        rel_cost_AD_d2RdXdX_v = AD_Hessian_vector_cost()
        iterations_d2RdXdX = rel_cost_form_d2RdXdX / ( rel_cost_AD_d2RdXdX_v - rel_cost_vmult_d2RdXdX_v)
        worth_form_d2RdXdX = worth_form_d2RdXdX + ('& %4.0f ' % iterations_d2RdXdX)
        print ('\n')

    residual_table = residual_table + '\\\\ \n'
    dRdW_vmult_table = dRdW_vmult_table + '\\\\ \n'
    dRdX_vmult_table = dRdX_vmult_table + '\\\\ \n'
    d2RdXdX_vmult_table = d2RdXdX_vmult_table + '\\\\ \n'
    dRdW_form_table = dRdW_form_table + '\\\\ \n'
    dRdX_form_table = dRdX_form_table + '\\\\ \n'
    worth_form_dRdX = worth_form_dRdX + '\\\\ \n'
    worth_form_d2RdWdW = worth_form_d2RdWdW + '\\\\ \n'
    worth_form_d2RdWdX = worth_form_d2RdWdX + '\\\\ \n'
    worth_form_d2RdXdX = worth_form_d2RdXdX + '\\\\ \n'
    d2RddW_form_table = d2RddW_form_table + '\\\\ \n'
    d2RddX_form_table = d2RddX_form_table + '\\\\ \n'

print('Residual assembly table')
print(residual_table)

print('dRdW vmult assembly table')
print(dRdW_vmult_table)
print('dRdX vmult assembly table')
print(dRdX_vmult_table)

print('d2RdXdX vmult assembly table')
print(d2RdXdX_vmult_table)

print('dRdW assembly table')
print(dRdW_form_table)
print('dRdX assembly table')
print(dRdX_form_table)

print('Iterations worth forming dRdX assembly table')
print(worth_form_dRdX)
print('Iterations worth forming d2RdWdW assembly table')
print(worth_form_d2RdWdW)
print('Iterations worth forming d2RdWdX assembly table')
print(worth_form_d2RdWdX)
print('Iterations worth forming d2RdXdX assembly table')
print(worth_form_d2RdXdX)

print('d2RddW assembly table')
print(d2RddW_form_table)
print('d2RddX assembly table')
print(d2RddX_form_table)
