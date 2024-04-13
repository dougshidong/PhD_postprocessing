#!/usr/bin/python3
import re
from io import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

pdfName = './optimization_results.pdf'
pp=PdfPages(pdfName)

n_average_iterations_per_linear_solves = 83.66
n_average_nonlinear_iterations_per_ode_solves = 4.05
n_linear_solves_per_bfgs = (n_average_nonlinear_iterations_per_ode_solves+1)
# times 2 because we count the preconditioner application as a vmult
n_vmult_per_bfgs = 2*n_linear_solves_per_bfgs*n_average_iterations_per_linear_solves

n_vmult_per_subkkt_p2a = 12 + 2; # 12 vmult and 2 precon
n_vmult_per_subkkt_p2 = 12 + 2*n_average_iterations_per_linear_solves; # 12 vmult and 2 inverses

n_vmult_per_subkkt_p4a = 12 + 4
n_vmult_per_subkkt_p4 = 12 + 4*n_average_iterations_per_linear_solves; # 12 vmult and 2 inverses

n_dim = 2
n_state = n_dim + 2
def n_vmult_to_form_AD_KKT_operator(poly):
    n_hessian_matrices = 8 # objective and residual
    n_dofs_1D = poly+1
    n_dofs_cell = pow(n_dofs_1D, n_dim) * n_state
    n_nonzero = pow(n_dofs_cell,2) #+ (n_dim*n_dim*n_dofs_1D*n_state) * n_dofs_cell # Mostly just cells with respect to themselves
    return n_hessian_matrices*n_nonzero; # 12 vmult and 2 precon
def n_vmult_to_apply_AD_KKT_operator(poly):
    n_hessian_matrices = 8 # objective and residual
    n_dofs_1D = poly+1
    n_dofs_cell = pow(n_dofs_1D, n_dim) * n_state
    n_residual_ops = n_dofs_cell # relative number of (residual cost / vmult cost)
    n_ad_ops = 5*n_residual_ops
    return n_hessian_matrices*n_ad_ops;

# Reduced Space BFGS ************************************************************************************
n_cells_list = ["50X5"]
poly_list = [1,2,3]
n_dofs_list = [4000,9000,16000]
n_design_list = [20,40,60,80,100]
opttype = 'optimization_reduced_space_bfgs'
delim = (8,) + (15,)*3 + (10,)*4
plt.figure(figsize=(8,6))
for n_cells in n_cells_list:
    for poly in poly_list:
        plt.figure(opttype+str(poly),figsize=(8,6))
        n_iterations_list = []
        n_vmult_list = []
        time_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, snorm, nfval, ngrad, ls_nfval, ls_ngrad \
                = np.genfromtxt(fname, skip_header=15, skip_footer=2,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                time = (re.findall("\d+\.\d+", last_line))[0]
                time = int(float(time))

            plt.title('Reduced space BFGS Gradient vs Iterations, P='+str(poly)+', nDoFs='+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=6,
                    label='n_ctl = ' + str(n_design)
                    +'  time='+str(time))

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(itera[-1]);
            n_vmult_list.append(itera[-1]*n_vmult_per_bfgs);
            time_list.append(time);

        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        plt.figure("Iterations_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_iterations_list, '-o', ms=6,
                label='Reduced space BFGS P='+str(poly))

        plt.figure("nvmult_vs_nctl2",figsize=(8,6))
        plt.plot(n_design_list, n_vmult_list, '-o', ms=6,
                label='Reduced space BFGS P='+str(poly))

#        plt.figure("Time_vs_nctl_RQNSQP",figsize=(8,6))
#        plt.plot(n_design_list, time_list, '-o', ms=6,
#                label='Full Space RQNSQP P='+str(poly))

        plt.figure("Time_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, time_list, '-o', ms=6,
                label='Reduced space BFGS P='+str(poly))

opttype = 'optimization_reduced_space_newton'
delim = (8,) + (15,)*3 + (10,)*6
plt.figure(figsize=(8,6))
n_vmult_apply_hessian_list = [[],[],[]]
for n_cells in n_cells_list:
    for poly in poly_list:
        plt.figure(opttype+str(poly),figsize=(8,6))
        n_iterations_list = []
        n_vmult_form_hessian_list = []
        time_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, snorm, nfval, ngrad, iterCG, flagCG, ls_nfval, ls_ngrad \
                = np.genfromtxt(fname, skip_header=18, skip_footer=2,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);
            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                time = (re.findall("\d+\.\d+", last_line))[0]
                time = int(float(time))

            plt.title('Reduced space Newton Gradient vs Iterations, P='+str(poly)+', nDoFs='+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=6,
                    label='n_ctl = ' + str(n_design)
                    +'  time='+str(time))

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations = itera[-1]
            n_cg_it = sum(iterCG)
            n_iterations_list.append(n_iterations)
            total_n_vmult = n_iterations*n_vmult_per_bfgs \
                            + n_cg_it*n_average_iterations_per_linear_solves*2 
            n_vmult_form_hessian_list.append(total_n_vmult + n_iterations * n_vmult_to_form_AD_KKT_operator(poly))
            n_vmult_apply_hessian_list[poly-1].append(total_n_vmult + n_cg_it * n_vmult_to_apply_AD_KKT_operator(poly))
            print(n_vmult_apply_hessian_list[poly-1])
            time_list.append(time);

        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

#        plt.figure("Iterations_vs_nctl",figsize=(8,6))
#        plt.plot(n_design_list, n_iterations_list, '-.o', ms=6,
#                label='Reduced space Newton P='+str(poly))

        plt.figure("nvmult_vs_nctl2",figsize=(8,6))
        print(n_design_list)
        print(n_vmult_list)
        plt.plot(n_design_list, n_vmult_form_hessian_list, '-.o', ms=6,
                label='Reduced space Newton P='+str(poly)+ ' form AD operators')
#        plt.figure("Time_vs_nctl_RNSQP",figsize=(8,6))
#        plt.plot(n_design_list, time_list, '-.o', ms=6,
#                label='Full Space RNSQP P='+str(poly))

        plt.figure("Time_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, time_list, '-.o', ms=6,
                label='Reduced space Newton P='+str(poly))


for n_cells in n_cells_list:
    for poly in poly_list:
        plt.figure("nvmult_vs_nctl2",figsize=(8,6))
        plt.plot(n_design_list, n_vmult_apply_hessian_list[poly-1], ':^', ms=6,
                label='Reduced space Newton P='+str(poly)+ ' apply AD operators')


opttype = 'optimization_full_space_p2a'
delim = (18,)*8
#plt.figure(figsize=(8,6))
for n_cells in n_cells_list:
    for poly in poly_list:
        plt.figure(opttype+str(poly),figsize=(8,6))
        n_iterations_list = []
        n_vmult_list = []
        time_list = []

        n_vmult_per_kkt = 8*pow(pow(poly+1,2)*4,2); # 12 vmult and 2 precon
        print (n_vmult_per_kkt);

        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, nkkt, ls_nfval, ls_ngrad \
                = np.genfromtxt(fname, skip_header=6, skip_footer=2,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                time = (re.findall("\d+\.\d+", last_line))[0]
                time = int(float(time))

            plt.title('Full Space with P2A Gradient vs Iterations, P='+str(poly) +', nDoFs='+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=6,
                    label='n_ctl = ' + str(n_design)
                    +'  time='+str(time))

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(itera[-1]);
            total_n_subkkt = sum(nkkt)
            total_vmult = total_n_subkkt*n_vmult_per_subkkt_p2a
            total_vmult_without_hess_assembly = total_vmult
            total_vmult_with_hess_assembly = total_vmult + n_vmult_per_kkt*itera[-1]
            n_vmult_list.append(total_vmult_with_hess_assembly)
            time_list.append(time);

        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        plt.figure("Iterations_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_iterations_list, '--o', ms=6,
                label='Full Space P2A P='+str(poly))

        plt.figure("nvmult_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_vmult_list, '--o', ms=6,
                label='Full Space P2A P='+str(poly))

        plt.figure("Time_vs_nctl_P2A",figsize=(8,6))
        plt.plot(n_design_list, time_list, '--o', ms=6,
                label='Full Space P2A P='+str(poly))


opttype = 'optimization_full_space_p2'
delim = (18,)*8
#plt.figure(figsize=(8,6))
for n_cells in n_cells_list:
    for poly in poly_list:
        plt.figure(opttype+str(poly),figsize=(8,6))
        n_iterations_list = []
        n_vmult_list = []
        time_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, nkkt, ls_nfval, ls_ngrad \
                = np.genfromtxt(fname, skip_header=6, skip_footer=2,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                time = (re.findall("\d+\.\d+", last_line))[0]
                time = int(float(time))

            plt.title('Full Space with P2 Gradient vs Iterations, P='+str(poly) +', nDoFs='+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=6,
                    label='n_ctl = ' + str(n_design)
                    +'  time='+str(time))

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(itera[-1]);
            total_n_subkkt = sum(nkkt)
            n_vmult_list.append(total_n_subkkt*n_vmult_per_subkkt_p2)
            time_list.append(time);

        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        plt.figure("Iterations_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_iterations_list, '-.o', ms=6,
                label='Full Space P2 P='+str(poly))

        plt.figure("nvmult_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_vmult_list, '-.o', ms=6,
                label='Full Space P2 P='+str(poly))

        plt.figure("Time_vs_nctl_P2",figsize=(8,6))
        plt.plot(n_design_list, time_list, '-.o', ms=6,
                label='Full Space P2 P='+str(poly))

opttype = 'optimization_full_space_p4'
delim = (18,)*8
#plt.figure(figsize=(8,6))
for n_cells in n_cells_list:
    for poly in poly_list:
        plt.figure(opttype+str(poly),figsize=(8,6))
        n_iterations_list = []
        n_vmult_list = []
        time_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, nkkt, ls_nfval, ls_ngrad \
                = np.genfromtxt(fname, skip_header=6, skip_footer=2,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                time = (re.findall("\d+\.\d+", last_line))[0]
                time = int(float(time))

            plt.title('Full Space with P4 Gradient vs Iterations, P='+str(poly) +', nDoFs='+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=6,
                    label='n_ctl = ' + str(n_design)
                    +'  time='+str(time))

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(itera[-1]);
            total_n_subkkt = sum(nkkt)
            n_vmult_list.append(total_n_subkkt*n_vmult_per_subkkt_p4)
            time_list.append(time);

        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        plt.figure("Iterations_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_iterations_list, '-.o', ms=6,
                label='Full Space P4 P='+str(poly))

        plt.figure("nvmult_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_vmult_list, '-.o', ms=6,
                label='Full Space P4 P='+str(poly))

        plt.figure("Time_vs_nctl_P4",figsize=(8,6))
        plt.plot(n_design_list, time_list, '-.o', ms=6,
                label='Full Space P4 P='+str(poly))

opttype = 'optimization_full_space_p4a'
delim = (18,)*8
#plt.figure(figsize=(8,6))
for n_cells in n_cells_list:
    for poly in poly_list:
        plt.figure(opttype+str(poly),figsize=(8,6))
        n_iterations_list = []
        n_vmult_list = []
        time_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, nkkt, ls_nfval, ls_ngrad \
                = np.genfromtxt(fname, skip_header=6, skip_footer=2,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                time = (re.findall("\d+\.\d+", last_line))[0]
                time = int(float(time))

            plt.title('Full Space with P4A Gradient vs Iterations, P='+str(poly) +', nDoFs='+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=6,
                    label='n_ctl = ' + str(n_design)
                    +'  time='+str(time))

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(itera[-1]);
            total_n_subkkt = sum(nkkt)
            n_vmult_list.append(total_n_subkkt*n_vmult_per_subkkt_p4a)
            time_list.append(time);

        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        plt.figure("Iterations_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_iterations_list, '-.o', ms=6,
                label='Full Space P4A P='+str(poly))

        plt.figure("nvmult_vs_nctl",figsize=(8,6))
        plt.plot(n_design_list, n_vmult_list, '-.o', ms=6,
                label='Full Space P4A P='+str(poly))

        plt.figure("Time_vs_nctl_P4A",figsize=(8,6))
        plt.plot(n_design_list, time_list, '-.o', ms=6,
                label='Full Space P4A P='+str(poly))

plt.figure("Iterations_vs_nctl",figsize=(8,6))
plt.title('Iterations vs Design Variables')
plt.xlabel(r'Number of Design Variables')
plt.ylabel(r'Number of Iterations to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("nvmult_vs_nctl",figsize=(8,6))
plt.title('Number of matrix-vector vmult vs Design Variables')
plt.xlabel(r'Number of Design Variables')
plt.ylabel(r'Number of vmult to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("nvmult_vs_nctl2",figsize=(8,6))
plt.title('Number of matrix-vector vmult vs Design Variables')
plt.xlabel(r'Number of Design Variables')
plt.ylabel(r'Number of vmult to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("Time_vs_nctl")
plt.title('Time vs Design Variables')
plt.xlabel(r'Number of Design Variables')
plt.ylabel(r'Time to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("Time_vs_nctl_RQNSQP")
plt.title('Time vs Design Variables')
plt.xlabel(r'Number of Design Variables')
plt.ylabel(r'Time to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("Time_vs_nctl_P2A")
plt.title('Time vs Design Variables')
plt.xlabel(r'Number of Design Variables')
plt.ylabel(r'Time to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("Time_vs_nctl_P2")
plt.title('Time vs Design Variables')
plt.xlabel(r'Number of Design Variables')
plt.ylabel(r'Time to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

pp.close()


