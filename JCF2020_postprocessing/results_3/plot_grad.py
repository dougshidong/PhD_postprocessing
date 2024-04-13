#!/usr/bin/python3
import re
from io import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

pdfName = './optimization_results_more_design.pdf'
pdfName = './optimization_results.pdf'

pp=PdfPages(pdfName)

n_average_iterations_per_linear_solves = 83.66
n_average_nonlinear_iterations_per_ode_solves = 4.05

marker_str = [ 'o', 'v', 's', 'X', 'D', '>','*','+']
linestyle_str = [ '-', ':', '--', '-.','-',':','--','-.']
color_str = [ 'tab:blue', 'tab:green', 'tab:red', 'black', 'tab:cyan', 'tab:purple', 'tab:olive']
marker_size = 6

square_figsize=(8,6)
wide_figsize=(14,6)
big_figsize=(14,12)
figsize = wide_figsize

table_print = ''

col_names = ['$n$', '$m$', 'Method', 'Cycles', 'Subits', 'Subits/Cycle', 'Work', 'Work/Cycle']
df = pd.DataFrame(columns = col_names)

n_dim = 2
n_state = n_dim + 2

n_cells_list = ["50X5"]
poly_list = [1,2,3]
n_dofs_list = [4000,9000,16000]
n_design_list = [20,40,60,80,100]

n_design_list_more = [20,40,60,80,100,160,320,640]
n_design_list_more = n_design_list

GradientvsSubiterations_design_list = [20,60,100]
ReducedGradientvsSubiterations_design_list = [20,40,60,80,100]
WorkvsNDOFS_design_list = n_design_list_more#[20,40,60,80,100]

full_space_delim = (18,)*12

fig_FS_subiter, axs_FS_subiter = plt.subplots(3,figsize=big_figsize,sharex=True)
fig_RS_subiter, axs_RS_subiter = plt.subplots(3,figsize=big_figsize,sharex=True)

fix_Hessian_cost = True

# Reduced Space BFGS ************************************************************************************
opttype = 'optimization_reduced_space_bfgs'
delim = (8,) + (15,)*3 + (10,)*5
plt.figure(figsize=figsize)
for n_cells in n_cells_list:
    pwork_list = []
    for poly in poly_list:
        plt.figure(opttype+str(poly),figsize=figsize)
        n_iterations_list = []
        work_list = []
        nkkt_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, snorm, nfval, ngrad, ls_nfval, ls_ngrad, work \
                = np.genfromtxt(fname, skip_header=17, skip_footer=3,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);
            total_subit = 0
            total_work  = work[-1]
            if fix_Hessian_cost:
                total_work  = total_work + total_subit * 0
                
            cycles = itera[-1]

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                time_line = lines[-2]
                time = (re.findall("\d+\.\d+", time_line))[0]
                time = int(float(time))

            plt.title(r'Reduced-space BFGS Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                    label=r'$n$ = ' + str(n_design)
                    )#+'  time='+str(time))

            table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
                n_design, n_dofs_list[poly-1],
                opttype, cycles,
                total_subit, total_subit/cycles,
                total_work, total_work / cycles
            )
            table_print = table_print + table_line + '\n'

            row = pd.DataFrame([{
                                'Method':  'Reduced-space BFGS',
                                '$n$':     n_design, 
                                '$m$':     n_dofs_list[poly-1],
                                'Cycles':  cycles,
                                'Subits':  0,
                                'Work':    total_work
                                }]
                                , index = ['RS_QN_'+str(n_design)+'_P'+str(poly)])
            df = df.append(row)


            if n_design in ReducedGradientvsSubiterations_design_list:
                ax = axs_RS_subiter[poly-1]
                ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
                ax.semilogy(itera, gnorm,
                        linestyle = linestyle_str[0],
                        color=color_str[ms_index],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'BFGS, $n$ = ' + str(n_design))

            plt.legend()
            plt.xlabel(r'Design Cycles')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(cycles);
            work_list.append(total_work)

        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        pwork_list.append(work_list)

        plt.figure("Iterations_vs_nctl_ReducedSpace",figsize=figsize)
        plt.plot(n_design_list, n_iterations_list, '-o', ms=marker_size,
                label=r'Reduced-space BFGS, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("work_vs_nctl_2",figsize=figsize)
        plt.plot(n_design_list,
                work_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[4],
                marker=marker_str[4],
                ms=marker_size,
                label=r'Reduced-space BFGS, $m$ = '+str(n_dofs_list[poly-1]))

opttype = 'optimization_reduced_space_newton'
delim = (8,) + (15,)*3 + (10,)*7
plt.figure(figsize=figsize)
n_vmult_apply_hessian_list = [[],[],[]]
for n_cells in n_cells_list:
    pwork_list = []
    for poly in poly_list:
        n_iterations_list = []
        n_vmult_form_hessian_list = []
        work_list = []
        nkkt_list = []
        for n_design in n_design_list_more:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, snorm, nfval, ngrad, iterCG, flagCG, ls_nfval, ls_ngrad, work \
                = np.genfromtxt(fname, skip_header=19, skip_footer=3,comments='&&&', \
                        filling_values="0", \
                        delimiter=delim,unpack=True);
            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                time_line = lines[-2]
                time = (re.findall("\d+\.\d+", time_line))[0]
                time = int(float(time))

            plt.figure(opttype+str(poly),figsize=figsize)
            plt.title(r'Reduced-space Newton Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                    label=r'$n$ = ' + str(n_design)
                    )#+'  time='+str(time))

            total_subit = sum(iterCG)
            total_work  = work[-1]
            if fix_Hessian_cost:
                total_work  = total_work + total_subit * 4*7
            cycles = itera[-1]

            table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
                n_design, n_dofs_list[poly-1],
                opttype, cycles,
                total_subit, total_subit/cycles,
                total_work, total_work / cycles
            )
            table_print = table_print + table_line + '\n'
            row = pd.DataFrame([{
                                'Method':  'Reduced-space Newton',
                                '$n$':     n_design, 
                                '$m$':     n_dofs_list[poly-1],
                                'Cycles':  cycles,
                                'Subits':  sum(iterCG),
                                'Work':    total_work
                                }]
                                , index = ['RS_N_'+str(n_design)+'_P'+str(poly)])
            df = df.append(row)


            if n_design in ReducedGradientvsSubiterations_design_list:
                ax = axs_RS_subiter[poly-1]
                ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
                ax.semilogy(itera, gnorm,
                        linestyle = linestyle_str[1],
                        color=color_str[5],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'Newton, $n$ = ' + str(n_design))

            plt.legend()
            plt.xlabel(r'Design Cycles')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            work_list.append(total_work)

        plt.figure(opttype+str(poly),figsize=figsize)
        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        pwork_list.append(work_list)
#        plt.figure("Iterations_vs_nctl_ReducedSpace",figsize=figsize)
#        plt.plot(n_design_list_more, n_iterations_list, '-.o', ms=marker_size,
#                label=r'Reduced-space Newton, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("work_vs_nctl_2",figsize=figsize)
        plt.plot(n_design_list_more, work_list, '-.o',
                linestyle = linestyle_str[poly-1],
                color=color_str[5],
                marker=marker_str[5],
                ms=marker_size,
                label=r'Reduced-space Newton, $m$ = '+str(n_dofs_list[poly-1]))
        plt.figure("work_vs_nctl_3",figsize=figsize)
        plt.plot(n_design_list_more, work_list, '-.o',
                linestyle = linestyle_str[poly-1],
                color=color_str[5],
                marker=marker_str[5],
                ms=marker_size,
                label=r'Reduced-space Newton, $m$ = '+str(n_dofs_list[poly-1]))
#                label=r'Full-space RNSQP, $m$ = '+str(n_dofs_list[poly-1]))

    for i,n_design in enumerate(WorkvsNDOFS_design_list):
        j = n_design_list_more.index(n_design)
        x = []
        y = []
        for poly in poly_list:
            x.append(n_dofs_list[poly-1])
            y.append(pwork_list[poly-1][j])

        plt.figure("work_vs_nstate",figsize=figsize)
        plt.plot(x, y,
                linestyle = linestyle_str[i],
                color=color_str[5],
                marker=marker_str[i],
                ms=marker_size,
                label=r'Reduced-space Newton, $n$ = '+str(n_design))


#for n_cells in n_cells_list:
#    for poly in poly_list:
#        plt.figure("work_vs_nctl2",figsize=figsize)
#        plt.plot(n_design_list, n_vmult_apply_hessian_list[poly-1], ':^', ms=marker_size,
#                label=r'Reduced-space Newton, $m$ = '+str(n_dofs_list[poly-1])) ' apply AD operators')

opttype = 'optimization_full_space_p4'
#plt.figure(figsize=figsize)
for n_cells in n_cells_list:
    pwork_list = []
    for poly in poly_list:
        n_iterations_list = []
        work_list = []
        nkkt_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, sctl, ssim, sadj, nkkt, ls_nfval, ls_ngrad, work\
                = np.genfromtxt(fname, skip_header=6, skip_footer=3,comments='&&&', \
                        filling_values="0", \
                        delimiter=full_space_delim,unpack=True);

            total_subit = sum(nkkt)
            total_work  = work[-1]
            if fix_Hessian_cost:
                total_work  = total_work + total_subit * (4*7*2)
            cycles = itera[-1]

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                time_line = lines[-2]
                time = (re.findall("\d+\.\d+", time_line))[0]
                time = int(float(time))

            plt.figure(opttype+str(poly),figsize=figsize)
            plt.title(r'Full-space with $\mathbf{P}_4$ Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                    label=r'$n$ = ' + str(n_design)
                    )#+'  time='+str(time))
            table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
                n_design, n_dofs_list[poly-1],
                opttype, cycles,
                total_subit, total_subit/cycles,
                total_work, total_work / cycles
            )
            table_print = table_print + table_line + '\n'

            row = pd.DataFrame([{
                                'Method':  'Full-space $\\PFour$',
                                '$n$':     n_design, 
                                '$m$':     n_dofs_list[poly-1],
                                'Cycles':  cycles,
                                'Subits':  total_subit,
                                'Work':    total_work
                                }]
                                , index = ['FS_P4_'+str(n_design)+'_P'+str(poly)])
            df = df.append(row)

            plt.legend()
            plt.xlabel(r'Design Cycles')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(cycles);
            work_list.append(total_work)
            nkkt_list.append(total_subit)

            if n_design in GradientvsSubiterations_design_list:
                ax = axs_FS_subiter[poly-1]
                ms_index = GradientvsSubiterations_design_list.index(n_design)
                ax.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[0],
                        color=color_str[0],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\mathbf{P}_4$, $n$ = ' + str(n_design))

                plt.figure("FullspaceGradientvsSubiterations"+str(poly),figsize=figsize)
                plt.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[0],
                        color=color_str[0],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\mathbf{P}_4$, $n$ = ' + str(n_design))

        plt.figure(opttype+str(poly),figsize=figsize)
        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        pwork_list.append(work_list)

        plt.figure("Iterations_vs_nctl_FullSpace",figsize=figsize)
        plt.plot(n_design_list, n_iterations_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[0],
                marker=marker_str[0],
                ms=marker_size+2,
                label=r'Full-space $\mathbf{P}_4$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("work_vs_nctl",figsize=figsize)
        plt.plot(n_design_list,
                work_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[0],
                marker=marker_str[0],
                ms=marker_size,
                label=r'Full-space $\mathbf{P}_4$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("nkkt_vs_nctl",figsize=figsize)
        plt.plot(n_design_list,
                nkkt_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[0],
                marker=marker_str[0],
                ms=marker_size,
                label=r'Full-space $\mathbf{P}_4$, $m$ = '+str(n_dofs_list[poly-1]))




opttype = 'optimization_full_space_p2'
#plt.figure(figsize=figsize)
for n_cells in n_cells_list:
    pwork_list = []
    for poly in poly_list:
        n_iterations_list = []
        work_list = []
        nkkt_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, sctl, ssim, sadj, nkkt, ls_nfval, ls_ngrad, work\
                = np.genfromtxt(fname, skip_header=6, skip_footer=3,comments='&&&', \
                        filling_values="0", \
                        delimiter=full_space_delim,unpack=True);
            total_subit = sum(nkkt)
            total_work  = work[-1]
            if fix_Hessian_cost:
                total_work  = total_work + total_subit * (4*7)
            cycles = itera[-1]

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                time_line = lines[-2]
                time = (re.findall("\d+\.\d+", time_line))[0]
                time = int(float(time))

            plt.figure(opttype+str(poly),figsize=figsize)
            plt.title(r'Full-space with $\mathbf{P}_2$ Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                    label=r'$n$ = ' + str(n_design)
                    )#+'  time='+str(time))

            table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
                n_design, n_dofs_list[poly-1],
                opttype, cycles,
                total_subit, total_subit/cycles,
                total_work, total_work / cycles
            )
            table_print = table_print + table_line + '\n'
            row = pd.DataFrame([{
                                'Method':  'Full-space $\\PTwo$',
                                '$n$':     n_design, 
                                '$m$':     n_dofs_list[poly-1],
                                'Cycles':  cycles,
                                'Subits':  total_subit,
                                'Work':    total_work
                                }]
                                , index = ['FS_P2_'+str(n_design)+'_P'+str(poly)])
            df = df.append(row)

            plt.legend()
            plt.xlabel(r'Design Cycles')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(cycles);
            work_list.append(total_work)
            nkkt_list.append(total_subit)

            if n_design in GradientvsSubiterations_design_list:
                ax = axs_FS_subiter[poly-1]
                ms_index = GradientvsSubiterations_design_list.index(n_design)
                ax.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[1],
                        color=color_str[1],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\mathbf{P}_2$, $n$ = ' + str(n_design))

                plt.figure("FullspaceGradientvsSubiterations"+str(poly),figsize=figsize)
                ms_index = GradientvsSubiterations_design_list.index(n_design)
                plt.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[1],
                        color=color_str[1],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\mathbf{P}_2$, $n$ = ' + str(n_design))

        plt.figure(opttype+str(poly),figsize=figsize)
        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        pwork_list.append(work_list)

        plt.figure("Iterations_vs_nctl_FullSpace",figsize=figsize)
        plt.plot(n_design_list, n_iterations_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[1],
                marker=marker_str[1],
                ms=marker_size+1,
                label=r'Full-space $\mathbf{P}_2$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("work_vs_nctl",figsize=figsize)
        plt.plot(n_design_list,
                work_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[1],
                marker=marker_str[1],
                ms=marker_size,
                label=r'Full-space $\mathbf{P}_2$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("nkkt_vs_nctl",figsize=figsize)
        plt.plot(n_design_list,
                nkkt_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[1],
                marker=marker_str[1],
                ms=marker_size,
                label=r'Full-space $\mathbf{P}_2$, $m$ = '+str(n_dofs_list[poly-1]))

opttype = 'optimization_full_space_p4a'
#plt.figure(figsize=figsize)
for n_cells in n_cells_list:
    pwork_list = []
    for poly in poly_list:
        n_iterations_list = []
        work_list = []
        nkkt_list = []
        for n_design in n_design_list:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, sctl, ssim, sadj, nkkt, ls_nfval, ls_ngrad, work\
                = np.genfromtxt(fname, skip_header=6, skip_footer=3,comments='&&&', \
                        filling_values="0", \
                        delimiter=full_space_delim,unpack=True);
            total_subit = sum(nkkt)
            total_work  = work[-1]
            if fix_Hessian_cost:
                #total_work  = total_work + total_subit * (4*7*2)
                n_matrices_to_form = 3
                Nnz = (1+pow(n_dim,2))*pow(poly+1,2)*(n_state)
                n_hess_vec = 4 * 2 # kkt and precond
                reduction_from_matvec_form = -total_subit * n_hess_vec * 7
                addition_from_hess_form = (cycles * 6*(1+Nnz)*n_matrices_to_form)
                addition_from_vmult     = total_subit * n_hess_vec
                total_work  = total_work + reduction_from_matvec_form + addition_from_hess_form  + addition_from_vmult
            cycles = itera[-1]

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                time_line = lines[-2]
                time = (re.findall("\d+\.\d+", time_line))[0]
                time = int(float(time))

            plt.figure(opttype+str(poly),figsize=figsize)
            plt.title(r'Full-space with $\tilde{\mathbf{P}}_4$ Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                    label=r'$n$ = ' + str(n_design)
                    )#+'  time='+str(time))
            table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
                n_design, n_dofs_list[poly-1],
                opttype, cycles,
                total_subit, total_subit/cycles,
                total_work, total_work / cycles
            )
            table_print = table_print + table_line + '\n'
            row = pd.DataFrame([{
                                'Method':  'Full-space $\\PFourA$',
                                '$n$':     n_design, 
                                '$m$':     n_dofs_list[poly-1],
                                'Cycles':  cycles,
                                'Subits':  total_subit,
                                'Work':    total_work
                                }]
                                , index = ['FS_P4A_'+str(n_design)+'_P'+str(poly)])
            df = df.append(row)

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(cycles);
            work_list.append(total_work)
            nkkt_list.append(total_subit)

            if n_design in GradientvsSubiterations_design_list:
                ax = axs_FS_subiter[poly-1]
                ms_index = GradientvsSubiterations_design_list.index(n_design)
                ax.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[2],
                        color=color_str[2],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\tilde{\mathbf{P}}_4$, $n$ = ' + str(n_design))

                plt.figure("FullspaceGradientvsSubiterations"+str(poly),figsize=figsize)
                ms_index = GradientvsSubiterations_design_list.index(n_design)
                plt.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[2],
                        color=color_str[2],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\tilde{\mathbf{P}}_4$, $n$ = ' + str(n_design))

        plt.figure(opttype+str(poly),figsize=figsize)
        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        pwork_list.append(work_list)

        plt.figure("Iterations_vs_nctl_FullSpace",figsize=figsize)
        plt.plot(n_design_list, n_iterations_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[2],
                marker=marker_str[2],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_4$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("work_vs_nctl",figsize=figsize)
        plt.plot(n_design_list,
                work_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[2],
                marker=marker_str[2],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_4$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("nkkt_vs_nctl",figsize=figsize)
        plt.plot(n_design_list,
                nkkt_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[2],
                marker=marker_str[2],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_4$, $m$ = '+str(n_dofs_list[poly-1]))


opttype = 'optimization_full_space_p2a'
#plt.figure(figsize=figsize)
for n_cells in n_cells_list:
    pwork_list = []
    for poly in poly_list:
        n_iterations_list = []
        work_list = []
        nkkt_list = []

        for n_design in n_design_list_more:
            fname = opttype \
                +'_'+n_cells \
                +'_'+"P"+str(poly) \
                +'_'+str(n_design)+'.log'

            print("Reading from " + fname)
            itera, value, gnorm, cnorm, snorm, sctl, ssim, sadj, nkkt, ls_nfval, ls_ngrad, work\
                = np.genfromtxt(fname, skip_header=6, skip_footer=3,comments='&&&', \
                        filling_values="0", \
                        delimiter=full_space_delim,unpack=True);
            total_subit = sum(nkkt)
            total_work  = work[-1]
            cycles = itera[-1]

            if fix_Hessian_cost:
                Nnz = (1+pow(n_dim,2))*pow(poly+1,2)*(n_state)
                print ("poly = %d   Nnz = %d" % (poly,Nnz))

                jacobian_form_cost = 6*Nnz/n_state
                lagrangian_hess_vec = n_hess_vec * 7
                apply_kkt = 7 * 2 + 1 * 2 + lagrangian_hess_vec # dRdX, dRdXT, dRdW, dRdWT, Hess-vec
                apply_p2inv = 7 * 2 + 2 * 2 # dRdX, dRdXT, \tildedRdW, \tildedRdWT

                my_estimated_cost = cycles * jacobian_form_cost + total_subit*(apply_kkt + apply_p2inv)

                print ("Actual cost = %d \t Estimated Cost = %d" % (total_work, my_estimated_cost))

                #total_work  = total_work + total_subit * (4*7) 
                n_matrices_to_form = 3
                n_hess_vec = 4 # kkt only and no hessians in precond

                reduction_from_matvec_form = -total_subit * n_hess_vec * 7
                addition_from_hess_form = (cycles * 6*(1+Nnz)*n_matrices_to_form)
                addition_from_vmult     = total_subit * n_hess_vec
                net_cost_change  = reduction_from_matvec_form + addition_from_hess_form + addition_from_vmult
                print ( ' reduction_from_matvec_form %d   addition_from_hess_form %d     addition_from_vmult %d' % (reduction_from_matvec_form,addition_from_hess_form,addition_from_vmult))
                print ( ' net_cost_change = %d' % net_cost_change )
                total_work  = total_work + net_cost_change

            time = 0
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                time_line = lines[-2] 
                time = (re.findall("\d+\.\d+", time_line))[0]
                time = int(float(time))

            plt.figure(opttype+str(poly),figsize=figsize)
            plt.title(r'Full-space with $\tilde{\mathbf{P}}_2$ Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly-1]))
            plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                    label=r'$n$ = ' + str(n_design)
                    )#+'  time='+str(time))

            table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
                n_design, n_dofs_list[poly-1],
                opttype, cycles,
                total_subit, total_subit/cycles,
                total_work, total_work / cycles
            )
            table_print = table_print + table_line + '\n'
            row = pd.DataFrame([{
                                'Method':  'Full-space $\\PTwoA$',
                                '$n$':     n_design, 
                                '$m$':     n_dofs_list[poly-1],
                                'Cycles':  cycles,
                                'Subits':  total_subit,
                                'Work':    total_work
                                }]
                                , index = ['FS_P2A_'+str(n_design)+'_P'+str(poly)])
            df = df.append(row)

            plt.legend()
            plt.xlabel(r'Iterations')
            plt.ylabel(r'Gradient Norm')
            plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
            plt.tight_layout()

            n_iterations_list.append(cycles);
            work_list.append(total_work)
            nkkt_list.append(total_subit)

            if n_design in GradientvsSubiterations_design_list:
                ax = axs_FS_subiter[poly-1]
                ms_index = GradientvsSubiterations_design_list.index(n_design)
                ax.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[3],
                        color=color_str[3],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

                plt.figure("FullspaceGradientvsSubiterations"+str(poly),figsize=figsize)
                ms_index = GradientvsSubiterations_design_list.index(n_design)
                plt.semilogy(np.cumsum(nkkt), gnorm,
                        linestyle = linestyle_str[3],
                        color=color_str[3],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'$\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

        plt.figure(opttype+str(poly),figsize=figsize)
        pp.savefig(bbx_inches='tight')
        plt.close(opttype+str(poly))

        pwork_list.append(work_list)

        plt.figure("Iterations_vs_nctl_FullSpace",figsize=figsize)
        plt.plot(n_design_list_more, n_iterations_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[3],
                marker=marker_str[3],
                ms=marker_size-1,
                label=r'Full-space $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("work_vs_nctl",figsize=figsize)
        plt.plot(n_design_list_more,
                work_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[3],
                marker=marker_str[3],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("nkkt_vs_nctl",figsize=figsize)
        plt.plot(n_design_list_more,
                nkkt_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[3],
                marker=marker_str[3],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly-1]))

        plt.figure("work_vs_nctl_3",figsize=figsize)
        plt.plot(n_design_list_more,
                work_list,
                linestyle = linestyle_str[poly-1],
                color=color_str[3],
                marker=marker_str[3],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly-1]))

    for i,n_design in enumerate(WorkvsNDOFS_design_list):
        j = n_design_list_more.index(n_design)
        x = []
        y = []
        for poly in poly_list:
            x.append(n_dofs_list[poly-1])
            y.append(pwork_list[poly-1][j])

        plt.figure("work_vs_nstate",figsize=figsize)
        plt.plot(x, y,
                linestyle = linestyle_str[i],
                color=color_str[3],
                marker=marker_str[i],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_2$, $n$ = '+str(n_design))

for poly in poly_list:
    plt.figure('FullspaceGradientvsSubiterations'+str(poly),figsize=figsize)
    plt.title(r'Full-space Gradient vs Subiterations, $m$ = '+str(n_dofs_list[poly-1]))
    plt.xlabel(r'Number of Subiterations')
    plt.ylabel(r'Gradient Norm')
    plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
    plt.legend()
    plt.tight_layout()
    pp.savefig(bbx_inches='tight')
    plt.close()

#plt.figure('FullspaceGradientvsSubiterations',figsize=figsize)
axs_FS_subiter[0].legend()
for i,ax in enumerate(axs_FS_subiter):
    ax.set_title(r'$m$ = ' + str(n_dofs_list[i]))
    #ax.set(xlabel='Number of subiterations', ylabel='Gradient Norm')
    ax.set(ylabel='Gradient Norm')
    ax.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
    #ax.legend()
    #ax.tight_layout()
axs_FS_subiter[2].set(xlabel='Number of subiterations')
#plt.xlabel(r'Number of Subiterations')
#plt.ylabel(r'Gradient Norm')
#plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
#plt.legend()
fig_FS_subiter.suptitle(r'Full-space Gradient vs Subiterations')
fig_FS_subiter.tight_layout()
fig_FS_subiter.subplots_adjust(top=0.92)
pp.savefig(fig_FS_subiter,bbx_inches='tight')
#fig_FS_subiter.close()

plt.figure("Iterations_vs_nctl_FullSpace",figsize=figsize)
plt.title(r'Full-space Design Cycles vs Control Variables')
plt.ylabel(r'Number of Design Cycles to Convergence')
plt.xlabel(r'Number of Control Variables $n$')
plt.yticks(np.arange(3, 6, 1.0))
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("work_vs_nctl",figsize=figsize)
plt.title(r'Work Units vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Work Units to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("nkkt_vs_nctl",figsize=figsize)
plt.title(r'Total Subiterations vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Total Subiterations to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend(loc=2)
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("work_vs_nctl_2",figsize=figsize)
plt.title(r'Work Units vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Work units to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("Iterations_vs_nctl_ReducedSpace",figsize=figsize)
plt.title(r'Reduced-space Design Cycles vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Number of Design Cycles to Convergence')
#plt.yticks(np.arange(3, 6, 1.0))
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

axs_RS_subiter[0].legend()
for i,ax in enumerate(axs_RS_subiter):
    ax.set_title(r'$m$ = ' + str(n_dofs_list[i]))
    #ax.set(xlabel='Number of design cycles', ylabel='Gradient Norm')
    ax.set(ylabel='Gradient Norm')
    ax.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
#axs_RS_subiter[2].set(xlabel='Number of design cycles or Newton subiterations')
axs_RS_subiter[2].set(xlabel='Design Cycles')
fig_RS_subiter.suptitle(r'Reduced-space Gradient vs Design Cycles')
fig_RS_subiter.tight_layout()
fig_RS_subiter.subplots_adjust(top=0.92)
pp.savefig(fig_RS_subiter,bbx_inches='tight')
#fig_RS_subiter.close()

plt.figure("work_vs_nctl_3",figsize=figsize)
plt.title(r'Work Units vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Work Units to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

plt.figure("work_vs_nstate",figsize=figsize)
plt.title(r'Work Units vs State Variables')
plt.xlabel(r'Number of State Variables $m$')
plt.ylabel(r'Work Units to Convergence')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
plt.legend()
plt.tight_layout()
pp.savefig(bbx_inches='tight')
plt.close()

pp.close()

print(table_print)

#col_names = ['$n$', '$m$', 'Method', 'Cycles', 'Subits', 'Subits/Cycle', 'Work', 'Work/Cycle']
df['Subits/Cycle'] = df['Subits'] / df['Cycles']
df['Work/Cycle'] = df['Work'] / df['Cycles']

table_print = ''
print(df)
for method in df['Method'].unique():
    method_rows = df.loc[df.index[df['Method'] == method].tolist()]

    table_print = table_print+'\\midrule \n'
    table_print = table_print + '\multicolumn{16}{c}{\\textbf{' + method + '}} \\\\ \n'
    table_print = table_print+'\\midrule \n'
    for ndes in n_design_list_more:

        line = ''
        line = '%s %4d  &' % (line, ndes)

        row = method_rows[(method_rows['$n$'] == ndes)]
        if row.empty:
            break
        cycles = row['Cycles'].to_numpy()
        subits = row['Subits'].to_numpy()
        subitsc = row['Subits/Cycle'].to_numpy()
        work = row['Work'].to_numpy()
        workc = row['Work/Cycle'].to_numpy()

        line = line + ('{:4.0f} & {:4.0f} & {:4.0f} &'.format(cycles[0], cycles[1], cycles[2]))
        line = line + ('{:4.0f} & {:4.0f} & {:4.0f} &'.format(subits[0], subits[1], subits[2]))
        line = line + ('{:5.1f} & {:5.1f} & {:5.1f} &'.format(subitsc[0], subitsc[1], subitsc[2]))
        line = line + ('{:7.0f} & {:7.0f} & {:7.0f} &'.format(work[0], work[1], work[2]))
        line = line + ('{:7.0f} & {:7.0f} & {:7.0f} '.format(workc[0], workc[1], workc[2]))

        line = line + '\\\\ \n'
        table_print = table_print + line

print(table_print)

