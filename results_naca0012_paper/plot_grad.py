#!/usr/bin/python3
import re
from io import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import count_flops

pdfName = './naca_optimization_results_more_design.pdf'
pdfName = './naca_optimization_results.pdf'

pp=PdfPages(pdfName)

marker_str = [ 'o', 'v', 's', 'X', 'D', '>','*','+']
linestyle_str = [ '-', ':', '--', '-.','-',':','--','-.']
color_str = [ 'tab:blue', 'tab:green', 'tab:red', 'black', 'tab:cyan', 'tab:purple', 'tab:olive']
marker_size = 6

full_space_color = 3
full_space_linestyle = 3
bfgs_color = 4
bfgs_linestyle = 0
newton_color = 5
newton_linestyle = 1

square_figsize=(8,6)
wide_figsize=(14,6)
big_figsize=(14,12)
figsize = wide_figsize

table_print = ''

col_names = ['$n$', '$m$', 'Method', 'Cycles', 'Subits', 'Subits/Cycle', 'Work', 'Work/Cycle']
df = pd.DataFrame(columns = col_names)

n_dim = 2
n_state = n_dim + 2

#poly_list = [1,2,3]
#n_dofs_list = [4000,9000,16000]
poly_list = [0,1]
n_dofs_list = [8248,32992]
poly_list = [0,1,2]
n_dofs_list = [8248,32992,74232]

#n_design_list = [20,40,60,80,100]
n_design_list = [20,40,60]

n_design_list_more = n_design_list
#n_design_list_more = [20,40,60,80,100,160,320,640]

GradientvsSubiterations_design_list = [20,40,60]
ReducedGradientvsSubiterations_design_list = [20,40,60]

# GradientvsSubiterations_design_list = [20,60,100]
# ReducedGradientvsSubiterations_design_list = [20,40,60,80,100]
WorkvsNDOFS_design_list = n_design_list_more#[20,40,60,80,100]

full_space_delim = (18,)*12 + (18,)*4
rs_bfgs_delim = (8,) + (15,)*3 + (10,)*5 + (18,)*4
rs_newton_delim = (8,) + (15,)*3 + (10,)*7 + (18,)*4

fig_FS_subiter, axs_FS_subiter = plt.subplots(len(n_dofs_list),figsize=big_figsize,sharex=True)
fig_RS_subiter, axs_RS_subiter = plt.subplots(len(n_dofs_list),figsize=big_figsize,sharex=True)

fig_all_cycles, axs_all_cycles = plt.subplots(len(n_dofs_list),figsize=big_figsize,sharex=True)

plt.figure("gradnorm_vs_designcycles",figsize=figsize)
ax_gradvscycles = plt.gca()


fix_Hessian_cost = True

# Crops the first part of the file
# Used in bump optimization since it did 2 optimizations runs, one to fit the FFD
# the other to inverse design the FFD.
TAG = 'run'
def file_after_run(input_filename, output_filename):
    tag_found = True
    with open(input_filename) as in_file:
        with open(output_filename, 'w') as out_file:
            for line in in_file:
                if not tag_found:
                    if TAG in line:
                        tag_found = True
                else:
                    out_file.write(line)
                    print(line)

def cost_to_form_dRdW(poly):
    n_dofs_1D = poly+1
    n_dofs_cell = pow(n_dofs_1D, n_dim) * n_state
    n_nonzero = pow(n_dofs_cell, 2) #+ (n_dim*n_dim*n_dofs_1D*n_state) * n_dofs_cell # Mostly just cells with respect to themselves
    return n_hessian_matrices*n_nonzero; # 12 vmult and 2 precon

custom_cost_1 = True

# Reduced Space BFGS ************************************************************************************
opttype = 'optimization_reduced_space_bfgs'
plt.figure(figsize=figsize)
pwork_list = []
for poly in poly_list:
    plt.figure(opttype+str(poly),figsize=figsize)
    n_iterations_list = []
    work_list = []
    nkkt_list = []
    for n_design in n_design_list:
        input_fname = opttype \
            +'_'+"P"+str(poly) \
            +'_'+str(int(n_design/2))+'.log'

        print("Reading from " + input_fname)
        fname = "temp_processed_result.log"
        file_after_run(input_fname, fname)

        itera, value, gnorm, snorm, nfval, ngrad, ls_nfval, ls_ngrad, work, dRdW_form, dRdW_mult, dRdX_mult, d2R_mult \
            = np.genfromtxt(fname, skip_header=16, skip_footer=3,comments='&&&', \
                    filling_values="0", \
                    delimiter=rs_bfgs_delim,unpack=True);
        total_subit = 0
        total_work  = work[-1]
        cycles = itera[-1]

        if fix_Hessian_cost:
            total_work  = total_work + total_subit * 0
        if custom_cost_1:
            total_work = 0
            total_work = total_work + dRdW_form[-1] * count_flops.form_dRdW_AD_cost(n_dim, poly)
            total_work = total_work + dRdW_mult[-1] * count_flops.dRdW_vmult_cost(n_dim,poly)

            #total_work = total_work + cycles        * count_flops.form_dRdX_AD_cost(n_dim,poly)
            #total_work = total_work + dRdX_mult[-1] * count_flops.dRdX_vmult_cost(n_dim,poly)
            total_work = total_work + cycles        * 0
            total_work = total_work + dRdX_mult[-1] * count_flops.AD_vector_dRdX_cost()

            #total_work = total_work + cycles        * count_flops.form_d2R_AD_cost(n_dim,poly)
            #total_work = total_work + d2R_mult[-1] * count_flops.dRdX_vmult_cost(n_dim,poly):
            

        time = 0
        with open(fname, 'r') as f:
            lines = f.read().splitlines()
            time_line = lines[-2]
            time = (re.findall("\d+\.\d+", time_line))[0]
            time = int(float(time))

        plt.title(r'Reduced-space BFGS Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly]))
        plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                label=r'$n$ = ' + str(n_design)
                )#+'  time='+str(time))

        table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
            n_design, n_dofs_list[poly],
            opttype, cycles,
            total_subit, total_subit/cycles,
            total_work, total_work / cycles
        )
        table_print = table_print + table_line + '\n'

        row = pd.DataFrame([{
                            'Method':  'Reduced-space BFGS',
                            '$n$':     n_design, 
                            '$m$':     n_dofs_list[poly],
                            'Cycles':  cycles,
                            'Subits':  0,
                            'Work':    total_work
                            }]
                            , index = ['RS_QN_'+str(n_design)+'_P'+str(poly)])
        df = pd.concat([df, row], ignore_index=True)


        ax = axs_all_cycles[poly]
        ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
        ax.semilogy(itera, gnorm,
                linestyle = linestyle_str[bfgs_linestyle],
                color=color_str[bfgs_color],
                marker=marker_str[ms_index],
                ms=marker_size,
                label=r'BFGS, $n$ = ' + str(n_design))

        if n_design in ReducedGradientvsSubiterations_design_list:
            ax = axs_RS_subiter[poly]
            ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(itera, gnorm,
                    linestyle = linestyle_str[bfgs_linestyle],
                    color=color_str[ms_index],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'BFGS, $n$ = ' + str(n_design))

        if poly-1 == 0 and n_design in ReducedGradientvsSubiterations_design_list:
            ax = ax_gradvscycles
            ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(itera, gnorm,
                    linestyle = linestyle_str[bfgs_linestyle],
                    color=color_str[ms_index],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'RS BFGS, $n$ = ' + str(n_design))


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
            label=r'RS BFGS, $m$ = '+str(n_dofs_list[poly]))

    plt.figure("work_vs_nctl_2",figsize=figsize)
    plt.plot(n_design_list,
            work_list,
            linestyle = linestyle_str[poly],
            color=color_str[bfgs_color],
            marker=marker_str[poly],
            ms=marker_size,
            label=r'RS BFGS, $m$ = '+str(n_dofs_list[poly]))
            #label=r'RS BFGS, $p$ = '+str(poly))


for i,n_design in enumerate(WorkvsNDOFS_design_list):
    j = n_design_list_more.index(n_design)
    x = []
    y = []
    for poly in poly_list:
        x.append(n_dofs_list[poly])
        y.append(pwork_list[poly][j])

    plt.figure("work_vs_nstate",figsize=figsize)
    plt.plot(x, y,
            linestyle = linestyle_str[i],
            color=color_str[bfgs_color],
            marker=marker_str[i],
            ms=marker_size,
            label=r'RS BFGS, $n$ = '+str(n_design))

opttype = 'optimization_reduced_space_newton'
plt.figure(figsize=figsize)
n_vmult_apply_hessian_list = [[],[],[]]
pwork_list = []
for poly in poly_list:
    n_iterations_list = []
    n_vmult_form_hessian_list = []
    work_list = []
    nkkt_list = []
    for n_design in n_design_list_more:
        input_fname = opttype \
            +'_'+"P"+str(poly) \
            +'_'+str(int(n_design/2))+'.log'

        print("Reading from " + input_fname)
        fname = "temp_processed_result.log"
        file_after_run(input_fname, fname)

        itera, value, gnorm, snorm, nfval, ngrad, iterCG, flagCG, ls_nfval, ls_ngrad, work, dRdW_form, dRdW_mult, dRdX_mult, d2R_mult \
            = np.genfromtxt(fname, skip_header=17, skip_footer=3,comments='&&&', \
                    filling_values="0", \
                    delimiter=rs_newton_delim,unpack=True);
        time = 0
        with open(fname, 'r') as f:
            lines = f.read().splitlines()
            time_line = lines[-2]
            time = (re.findall("\d+\.\d+", time_line))[0]
            time = int(float(time))

        plt.figure(opttype+str(poly),figsize=figsize)
        plt.title(r'Reduced-space Newton Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly]))
        plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                label=r'$n$ = ' + str(n_design)
                )#+'  time='+str(time))

        total_subit = sum(iterCG)
        total_work  = work[-1]
        cycles = itera[-1]

        if fix_Hessian_cost:
            total_work  = total_work + total_subit * 4*7
        if custom_cost_1:
            total_work = 0
            total_work = total_work + dRdW_form[-1] * count_flops.form_dRdW_AD_cost(n_dim, poly)
            total_work = total_work + dRdW_mult[-1] * count_flops.dRdW_vmult_cost(n_dim,poly)

            total_work = total_work + cycles        * count_flops.form_dRdX_AD_cost(n_dim,poly)
            total_work = total_work + dRdX_mult[-1] * count_flops.dRdX_vmult_cost(n_dim,poly)
            #total_work = total_work + cycles        * 0
            #total_work = total_work + dRdX_mult[-1] * count_flops.AD_vector_dRdX_cost()

            total_work = total_work + cycles        * count_flops.form_d2R_AD_cost(n_dim,poly)
            total_work = total_work + d2R_mult[-1]*0.25 * count_flops.d2R_vmult_cost(n_dim,poly)

        table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
            n_design, n_dofs_list[poly],
            opttype, cycles,
            total_subit, total_subit/cycles,
            total_work, total_work / cycles
        )
        table_print = table_print + table_line + '\n'
        row = pd.DataFrame([{
                            'Method':  'Reduced-space Newton',
                            '$n$':     n_design, 
                            '$m$':     n_dofs_list[poly],
                            'Cycles':  cycles,
                            'Subits':  sum(iterCG),
                            'Work':    total_work
                            }]
                            , index = ['RS_N_'+str(n_design)+'_P'+str(poly)])
        df = pd.concat([df, row], ignore_index=True)

        ax = axs_all_cycles[poly]
        ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
        ax.semilogy(itera, gnorm,
                linestyle = linestyle_str[newton_linestyle],
                color=color_str[newton_color],
                marker=marker_str[ms_index],
                ms=marker_size,
                label=r'Newton, $n$ = ' + str(n_design))

        if n_design in ReducedGradientvsSubiterations_design_list:
            ax = axs_RS_subiter[poly]
            ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(itera, gnorm,
                    linestyle = linestyle_str[newton_linestyle],
                    color=color_str[newton_color],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'Newton, $n$ = ' + str(n_design))

        if poly-1 == 0 and n_design in ReducedGradientvsSubiterations_design_list:
            ax = ax_gradvscycles
            ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(itera, gnorm,
                    linestyle = linestyle_str[newton_linestyle],
                    color=color_str[newton_color],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'RS Newton, $n$ = ' + str(n_design))

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
#                label=r'RS Newton, $m$ = '+str(n_dofs_list[poly]))

    plt.figure("work_vs_nctl_2",figsize=figsize)
    plt.plot(n_design_list_more, work_list, '-.o',
            linestyle = linestyle_str[poly],
            color=color_str[newton_color],
            marker=marker_str[poly],
            ms=marker_size,
            label=r'RS Newton, $m$ = '+str(n_dofs_list[poly]))
            #label=r'RS Newton, $p$ = '+str(poly))
    plt.figure("work_vs_nctl_3",figsize=figsize)
    plt.plot(n_design_list_more, work_list, '-.o',
            linestyle = linestyle_str[poly],
            color=color_str[newton_color],
            marker=marker_str[newton_color],
            ms=marker_size,
            label=r'RS Newton, $m$ = '+str(n_dofs_list[poly]))
#                label=r'FS RNSQP, $m$ = '+str(n_dofs_list[poly]))

for i,n_design in enumerate(WorkvsNDOFS_design_list):
    j = n_design_list_more.index(n_design)
    x = []
    y = []
    for poly in poly_list:
        x.append(n_dofs_list[poly])
        y.append(pwork_list[poly][j])

    plt.figure("work_vs_nstate",figsize=figsize)
    plt.plot(x, y,
            linestyle = linestyle_str[i],
            color=color_str[newton_color],
            marker=marker_str[i],
            ms=marker_size,
            label=r'RS Newton, $n$ = '+str(n_design))


#for n_cells in n_cells_list:
#    for poly in poly_list:
#        plt.figure("work_vs_nctl2",figsize=figsize)
#        plt.plot(n_design_list, n_vmult_apply_hessian_list[poly], ':^', ms=marker_size,
#                label=r'RS Newton, $m$ = '+str(n_dofs_list[poly])) ' apply AD operators')

opttype = 'optimization_full_space_p2a'
#plt.figure(figsize=figsize)
pwork_list = []
for poly in poly_list:
    n_iterations_list = []
    work_list = []
    nkkt_list = []

    for n_design in n_design_list_more:
        input_fname = opttype \
            +'_'+"P"+str(poly) \
            +'_'+str(int(n_design/2))+'.log'

        print("Reading from " + input_fname)
        fname = "temp_processed_result.log"
        file_after_run(input_fname, fname)
        itera, value, gnorm, cnorm, snorm, sctl, ssim, sadj, nkkt, ls_nfval, ls_ngrad, work, dRdW_form, dRdW_mult, dRdX_mult, d2R_mult \
            = np.genfromtxt(fname, skip_header=6, skip_footer=3,comments='&&&', \
                    filling_values="0", \
                    delimiter=full_space_delim,unpack=True);
        total_subit = sum(nkkt)
        total_work  = work[-1]
        cycles = itera[-1]

        if custom_cost_1:
            total_work = 0
            total_work = total_work + dRdW_form[-1] * count_flops.form_dRdW_AD_cost(n_dim, poly)
            total_work = total_work + dRdW_mult[-1] * count_flops.dRdW_vmult_cost(n_dim,poly)

            total_work = total_work + cycles        * count_flops.form_dRdX_AD_cost(n_dim,poly)
            total_work = total_work + dRdX_mult[-1] * count_flops.dRdX_vmult_cost(n_dim,poly)
            #total_work = total_work + cycles        * 0
            #total_work = total_work + dRdX_mult[-1] * count_flops.AD_vector_dRdX_cost()

            total_work = total_work + cycles        * count_flops.form_d2R_AD_cost(n_dim,poly)
            total_work = total_work + d2R_mult[-1]*0.25 * count_flops.d2R_vmult_cost(n_dim,poly)

        time = 0
        with open(fname, 'r') as f:
            lines = f.read().splitlines()
            time_line = lines[-2] 
            time = (re.findall("\d+\.\d+", time_line))[0]
            time = int(float(time))

        plt.figure(opttype+str(poly),figsize=figsize)
        plt.title(r'Full-space with $\tilde{\mathbf{P}}_2$ Gradient vs Design Cycles, $m$ = '+str(n_dofs_list[poly]))
        plt.semilogy(itera, gnorm, '-o', ms=marker_size,
                label=r'$n$ = ' + str(n_design)
                )#+'  time='+str(time))

        table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
            n_design, n_dofs_list[poly],
            opttype, cycles,
            total_subit, total_subit/cycles,
            total_work, total_work / cycles
        )
        table_print = table_print + table_line + '\n'
        row = pd.DataFrame([{
                            'Method':  'Full-space $\\PTwoA$',
                            '$n$':     n_design, 
                            '$m$':     n_dofs_list[poly],
                            'Cycles':  cycles,
                            'Subits':  total_subit,
                            'Work':    total_work
                            }]
                            , index = ['FS_P2A_'+str(n_design)+'_P'+str(poly)])
        df = pd.concat([df, row], ignore_index=True)


        plt.legend()
        plt.xlabel(r'Iterations')
        plt.ylabel(r'Gradient Norm')
        plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
        plt.tight_layout()

        n_iterations_list.append(cycles);
        work_list.append(total_work)
        nkkt_list.append(total_subit)

        ax = axs_all_cycles[poly]
        ms_index = GradientvsSubiterations_design_list.index(n_design)
        ax.semilogy(itera, gnorm,
                linestyle = linestyle_str[full_space_linestyle],
                color=color_str[full_space_color],
                marker=marker_str[ms_index],
                ms=marker_size,
                label=r'Full-space $\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

        if n_design in GradientvsSubiterations_design_list:
            ax = axs_FS_subiter[poly]
            ms_index = GradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(np.cumsum(nkkt), gnorm,
                    linestyle = linestyle_str[full_space_linestyle],
                    color=color_str[full_space_color],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'$\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

            plt.figure("FullspaceGradientvsSubiterations"+str(poly),figsize=figsize)
            ms_index = GradientvsSubiterations_design_list.index(n_design)
            plt.semilogy(np.cumsum(nkkt), gnorm,
                    linestyle = linestyle_str[full_space_linestyle],
                    color=color_str[full_space_color],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'$\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

        if poly-1 == 0 and n_design in ReducedGradientvsSubiterations_design_list:
            ax = ax_gradvscycles
            ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(itera, gnorm,
                    linestyle = linestyle_str[full_space_linestyle],
                    color=color_str[full_space_color],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'FS $\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

    plt.figure(opttype+str(poly),figsize=figsize)
    pp.savefig(bbx_inches='tight')
    plt.close(opttype+str(poly))

    pwork_list.append(work_list)

    plt.figure("Iterations_vs_nctl_FullSpace",figsize=figsize)
    plt.plot(n_design_list_more, n_iterations_list,
            linestyle = linestyle_str[poly],
            color=color_str[full_space_color],
            marker=marker_str[3],
            ms=marker_size-1,
            label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly]))

    plt.figure("nkkt_vs_nctl",figsize=figsize)
    plt.plot(n_design_list_more,
            nkkt_list,
            linestyle = linestyle_str[poly],
            color=color_str[full_space_color],
            marker=marker_str[3],
            ms=marker_size,
            label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly]))


    plt.figure("work_vs_nctl",figsize=figsize)
    plt.plot(n_design_list_more,
            work_list,
            linestyle = linestyle_str[poly],
            color=color_str[full_space_color],
            marker=marker_str[3],
            ms=marker_size,
            label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly]))

    plt.figure("work_vs_nctl_2",figsize=figsize)
    plt.plot(n_design_list_more,
            work_list,
            linestyle = linestyle_str[poly],
            color=color_str[full_space_color],
            marker=marker_str[poly],
            ms=marker_size,
            label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly]))
            #label=r'FS $\tilde{\mathbf{P}}_2$, $p$ = '+str(poly))

    plt.figure("work_vs_nctl_3",figsize=figsize)
    plt.plot(n_design_list_more,
            work_list,
            linestyle = linestyle_str[poly],
            color=color_str[full_space_color],
            marker=marker_str[3],
            ms=marker_size,
            label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = '+str(n_dofs_list[poly]))

for i,n_design in enumerate(WorkvsNDOFS_design_list):
    j = n_design_list_more.index(n_design)
    x = []
    y = []
    for poly in poly_list:
        x.append(n_dofs_list[poly])
        y.append(pwork_list[poly][j])

    plt.figure("work_vs_nstate",figsize=figsize)
    plt.plot(x, y,
            linestyle = linestyle_str[i],
            color=color_str[full_space_color],
            marker=marker_str[i],
            ms=marker_size,
            label=r'FS $\tilde{\mathbf{P}}_2$, $n$ = '+str(n_design))

for poly in poly_list:
    plt.figure('FullspaceGradientvsSubiterations'+str(poly),figsize=figsize)
    plt.title(r'Full-space Gradient vs Subiterations, $m$ = '+str(n_dofs_list[poly]))
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
axs_FS_subiter[-1].set(xlabel='Number of subiterations')
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
plt.legend(loc=2)
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
#axs_RS_subiter[-1].set(xlabel='Number of design cycles or Newton subiterations')
axs_RS_subiter[-1].set(xlabel='Design Cycles')
fig_RS_subiter.suptitle(r'Reduced-space Gradient vs Design Cycles')
fig_RS_subiter.tight_layout()
fig_RS_subiter.subplots_adjust(top=0.92)
pp.savefig(fig_RS_subiter,bbx_inches='tight')
#fig_RS_subiter.close()

axs_all_cycles[0].legend()
for i,ax in enumerate(axs_all_cycles):
    ax.set_title(r'$m$ = ' + str(n_dofs_list[i]))
    #ax.set(xlabel='Number of design cycles', ylabel='Gradient Norm')
    ax.set(ylabel='Gradient Norm')
    ax.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
#axs_RS_subiter[-1].set(xlabel='Number of design cycles or Newton subiterations')
axs_all_cycles[-1].set(xlabel='Design Cycles')
fig_all_cycles.suptitle(r'Gradient vs Design Cycles')
fig_all_cycles.tight_layout()
fig_all_cycles.subplots_adjust(top=0.92)
pp.savefig(fig_all_cycles,bbx_inches='tight')
#fig_all_cycles.close()

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

plt.figure("gradnorm_vs_designcycles",figsize=figsize)
ax = ax_gradvscycles
ax.set_title(r'Gradient Norm vs Design Cycles m = ' + str(n_dofs_list[1]))
ax.set(ylabel=r'Gradient Norm')
ax.set(xlabel=r'Design Cycles')
ax.grid(b=True, which='major', color='black', linestyle='-',alpha=0.2)
ax.legend(loc=1)
plt.tight_layout()
pp.savefig(bbx_inches='tight')


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

