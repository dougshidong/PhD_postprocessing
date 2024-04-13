#!/usr/bin/python3
import re
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import count_flops

matplotlib.use('Agg')


def input_file_name_builder(optimization_type, polynomial_degree, n_design, path="./"):
    return (path + optimization_type
            + '_' + "P" + str(polynomial_degree)
            + '_' + str(int(n_design / 2))
            + '.log')


directory_of_log = './'
#directory_of_log = '/home/ddong/Codes/PHiLiP_backup/build_release/tests/integration_tests_control_files/euler_naca_optimization_constrained/'
#directory_of_log = '/home/ddong/Codes/PHiLiP_backup/build_release/tests/integration_tests_control_files/euler_naca_optimization_constrained/volume_constraint_only_p1_grid/'
pdfName = './naca_lift_target_optimization_results_more_design.pdf'
pdfName = './naca_drag_minimization_lift_volume_constrained_results_pref.pdf'
pdfName = './naca_drag_minimization_lift_volume_constrained_results_pref_largerfont.pdf'

pp = PdfPages(pdfName)

marker_str = ['o', 'v', 's', 'X', 'D', '>', '*', '+']
line_style_str = ['-', ':', '--', '-.',
                  '-', ':', '--', '-.', 
                  '-', ':', '--', '-.', ]
color_str = ['tab:blue', 'tab:green', 'tab:red', 'black', 'tab:cyan', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown']
marker_size = 6

full_space_color = 3
full_space_linestyle = 3
full_space_marker = 0
bfgs_color = 4
bfgs_linestyle = 0
bfgs_marker = 1
newton_color = 5
newton_linestyle = 1


#square_figure_size = (8, 6)
#wide_figure_size = (14, 6)
square_figsize=(8,6)
big_figsize=(14,12)
wide_figsize=(14,7)
figsize = wide_figsize
figsize = [x*0.65 for x  in wide_figsize]
big_figsize=[x*0.65 for x  in big_figsize]
wide_figsize= [x*0.65 for x  in wide_figsize]

table_print = ''

col_names = ['$n$', '$m$', 'Method', 'Cycles', 'Subits', 'Subits/Cycle', 'Work', 'Work/Cycle']
df = pd.DataFrame(columns=col_names)

n_dim = 2
n_state = n_dim + 2

max_design_cycles = 500
xlimit = False
objective_final_value = False

poly_list = [0, 1, 2]
n_dofs_list = [648*n_state*1**2, 648*n_state*2**2, 648*n_state*3**2]

# n_design_list = [20,40,60,80,100]
# n_design_list = [20,40,60,80]
#n_design_list = [20, 40, 80, 160, 320, 640]
n_design_list = [10, 20, 40, 80]
n_design_list = [10, 20, 30, 40, 50, 60, 70, 80]
n_design_list_more = n_design_list
ReducedGradientvsSubiterations_design_list = n_design_list
GradientvsSubiterations_design_list = n_design_list
#WorkvsNDOFS_design_list = [10, 20, 40, 80]
WorkvsNDOFS_design_list = n_design_list_more

full_space_delimiter = (18,) * 12 + (18,) * 4
rs_bfgs_delimiter = (8,) + (15,) * 3 + (10,) * 5 + (18,) * 4
rs_newton_delimiter = (8,) + (15,) * 3 + (10,) * 7 + (18,) * 4

#pdas_delimiter = (8,) + (15,) * 9 + (11,) * 8 + (18,) * 5
pdas_delimiter_rs = (6,) \
      + (15,) * 8 \
      + (11,) * 4 \
      + (15,) \
      + (11,) \
      + (10,) \
      + (11,) \
      + (18,) * 5
pdas_delimiter_fs = (6,) \
      + (15,) * 8 \
      + (15,) \
      + (11,) * 4 \
      + (15,) \
      + (11,) \
      + (10,) \
      + (11,) \
      + (18,) * 5

fig_FS_cycles, axs_FS_cycles = plt.subplots(len(n_dofs_list), figsize=big_figsize, sharex=True)
fig_RS_cycles, axs_RS_cycles = plt.subplots(len(n_dofs_list), figsize=big_figsize, sharex=True)

fig_all_cycles, axs_all_cycles = plt.subplots(len(n_dofs_list), figsize=big_figsize, sharex=True)

plt.figure("gradnorm_vs_designcycles", figsize=figsize)
ax_gradient_cycles = plt.gca()

fix_Hessian_cost = True

# Crops the first part of the file
# Used in bump optimization since it did 2 optimizations runs, one to fit the FFD
# the other to inverse design the FFD.
TAG = 'run'


def file_after_run(input_filename, output_filename):
    tag_found = True
    strings_to_avoid = ("iter", "run", "Newton", "Method", "optimization", "Optimization", "algorithm")
    with open(input_filename) as in_file:
        with open(output_filename, 'w') as out_file:
            for line in in_file:
                if not tag_found:
                    if TAG in line:
                        tag_found = True
                elif any(string in line for string in strings_to_avoid):
                    continue
                else:
                    out_file.write(line)
                    print(line)


custom_cost_1 = True

# Reduced Space BFGS ************************************************************************************
opttype = 'optimization_reduced_space_bfgs'
plt.figure(figsize=figsize)
pwork_list = []
for polyindex, poly in enumerate(poly_list):
    n_iterations_list = []
    work_list = []
    nkkt_list = []

    grad_vs_cycles_figure_name = opttype + str(poly) + "_grad_vs_cycles"
    value_vs_cycles_figure_name = opttype + str(poly) + "_value_vs_cycles"

    for n_design in n_design_list:
        input_fname = input_file_name_builder(opttype, poly, n_design, directory_of_log)

        print("Reading from " + input_fname)
        fname = "temp_processed_result.log"
        file_after_run(input_fname, fname)

        (iter, value, gnorm, cnorm, ecnorm,
         flowcnorm, identity, icnorm, snorm,
         linesear, iterPDAS, flagPDAS,
         iterGMRES, resiGMRES, flagGMRES, feasible, n_active_,
         n_vmult, dRdW_form, dRdW_mult, dRdX_mult, d2R_mult) \
            = np.genfromtxt(fname, comments='&&&',
                            skip_header=2,
                            filling_values="0",
                            delimiter=pdas_delimiter_rs, unpack=True)
        work = n_vmult
        total_subit = 0
        total_work = work[-1]
        cycles = iter[-1]

        if fix_Hessian_cost:
            total_work = total_work + total_subit * 0
        if custom_cost_1:
            total_work = 0
            total_work = total_work + dRdW_form[-1] * count_flops.form_dRdW_AD_cost(n_dim, poly)
            total_work = total_work + dRdW_mult[-1] * count_flops.dRdW_vmult_cost(n_dim, poly)

            # total_work = total_work + cycles        * count_flops.form_dRdX_AD_cost(n_dim,poly)
            # total_work = total_work + dRdX_mult[-1] * count_flops.dRdX_vmult_cost(n_dim,poly)
            total_work = total_work + cycles * 0
            total_work = total_work + dRdX_mult[-1] * count_flops.AD_vector_dRdX_cost()

            # total_work = total_work + cycles        * count_flops.form_d2R_AD_cost(n_dim,poly)
            # total_work = total_work + d2R_mult[-1] * count_flops.dRdX_vmult_cost(n_dim,poly):

        time = 0
        with open(fname, 'r') as f:
            lines = f.read().splitlines()
            time_line = lines[-2]
            time = (re.findall("\d+\.\d+", time_line))[0]
            time = int(float(time))

        plt.figure(grad_vs_cycles_figure_name, figsize=figsize)
        plt.semilogy(iter, gnorm, '-o', ms=marker_size,
                     label=r'$n$ = ' + str(n_design)
                     )  # +'  time='+str(time))

        plt.figure(value_vs_cycles_figure_name, figsize=figsize)
        final_value = 0
        if objective_final_value:
            final_value = value[-1]
        before_final_value = abs(value[-2]-final_value)
        plt.semilogy(iter, abs(value-final_value)+before_final_value, '-o', ms=marker_size,
                     label=r'$n$ = ' + str(n_design)
                     )  # +'  time='+str(time))

        #value = value[0:-2] - value[1:-1]
        #plt.semilogy(iter[1:-1], value, '-o', ms=marker_size,
        #             label=r'$n$ = ' + str(n_design))

        table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
            n_design, n_dofs_list[polyindex],
            opttype, cycles,
            total_subit, total_subit / cycles,
            total_work, total_work / cycles
        )
        table_print = table_print + table_line + '\n'

        row = pd.DataFrame([{
            'Method': 'Reduced-space BFGS',
            '$n$': n_design,
            '$m$': n_dofs_list[polyindex],
            'Cycles': cycles,
            'Subits': 0,
            'Work': total_work
        }]
            , index=['RS_QN_' + str(n_design) + '_P' + str(poly)])
        #df = df.append(row)
        df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

        if n_design in ReducedGradientvsSubiterations_design_list:
           ax = axs_RS_cycles[polyindex]
           ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
           ax.semilogy(iter, gnorm,
                   linestyle = line_style_str[0],
                   color=color_str[ms_index],
                   marker=marker_str[ms_index],
                   ms=marker_size,
                   label=r'RS-QN, $n$ = ' + str(n_design))

        ax = axs_all_cycles[polyindex]
        ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
        ax.semilogy(iter, gnorm,
               linestyle = line_style_str[bfgs_linestyle],
               color=color_str[ms_index],
               marker=marker_str[ms_index],
               ms=marker_size,
               label=r'RS-QN, $n$ = ' + str(n_design))

        if poly - 1 == 1 and n_design in ReducedGradientvsSubiterations_design_list:
        #if n_design in ReducedGradientvsSubiterations_design_list:
            ax = ax_gradient_cycles
            ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(iter, gnorm,
                        linestyle=line_style_str[0],
                        color=color_str[ms_index],
                        marker=marker_str[ms_index],
                        #ms=marker_size,
                        ms=2,
                        label=r'RS-QN, $n$ = ' + str(n_design))


        n_iterations_list.append(cycles);
        work_list.append(total_work)

    plt.figure(grad_vs_cycles_figure_name, figsize=figsize)
    plt.title(r'Reduced-space BFGS Gradient vs Design Cycles, $m$ = ' + str(n_dofs_list[polyindex]))
    if xlimit:
        plt.xlim([1, max_design_cycles])
    plt.xlabel(r'Design Cycles'); plt.ylabel(r'Gradient Norm')
    plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
    plt.legend(); plt.tight_layout(); pp.savefig(bbox_inches='tight')
    plt.close(grad_vs_cycles_figure_name)

    plt.figure(value_vs_cycles_figure_name, figsize=figsize)
    plt.title(r'Reduced-space BFGS Value vs Design Cycles, $m$ = ' + str(n_dofs_list[polyindex]))
    plt.xlabel(r'Design Cycles'); plt.ylabel(r'|Objective Value - Final Value|')
    if xlimit:
        plt.xlim([1, max_design_cycles])
    plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
    plt.legend(); plt.tight_layout(); pp.savefig(bbox_inches='tight')
    plt.close(value_vs_cycles_figure_name)

    pwork_list.append(work_list)

    plt.figure("Iterations_vs_nctl_ReducedSpace", figsize=figsize)
    plt.plot(n_design_list, n_iterations_list, '-o', ms=marker_size,
             label=r'RS-QN, $m$ = ' + str(n_dofs_list[polyindex]))

    plt.figure("Iterations_vs_nctl_ReducedandFullSpace", figsize=figsize)
    plt.plot(n_design_list_more, n_iterations_list,
             linestyle=line_style_str[bfgs_linestyle],
             color=color_str[polyindex],
             marker=marker_str[bfgs_marker],
             ms=marker_size - 1,
             label=r'RS-QN $m$ = ' + str(n_dofs_list[polyindex]))

    plt.figure("work_vs_nctl_2", figsize=figsize)
    plt.plot(n_design_list,
             work_list,
             linestyle=line_style_str[polyindex],
             color=color_str[4],
             marker=marker_str[polyindex],
             ms=marker_size,
             label=r'RS-QN, $m$ = ' + str(n_dofs_list[polyindex]))
    # label=r'RS-QN, $p$ = '+str(poly))

for i, n_design in enumerate(WorkvsNDOFS_design_list):
    j = n_design_list_more.index(n_design)
    x = []
    y = []
    for polyindex, poly in enumerate(poly_list):
        x.append(n_dofs_list[polyindex])
        y.append(pwork_list[polyindex][j])

    plt.figure("work_vs_nstate", figsize=figsize)
    plt.plot(x, y,
             linestyle=line_style_str[i],
             color=color_str[bfgs_color],
             marker=marker_str[i],
             ms=marker_size,
             label=r'RS-QN, $n$ = ' + str(n_design))

opttype = 'optimization_full_space_p4a'
# plt.figure(figsize=figsize)
pwork_list = []
for polyindex, poly in enumerate(poly_list):
    n_iterations_list = []
    work_list = []
    nkkt_list = []

    grad_vs_cycles_figure_name = opttype + str(poly) + "_grad_vs_cycles"

    for n_design in n_design_list_more:
        input_fname = input_file_name_builder(opttype, poly, n_design, directory_of_log)

        print("Reading from " + input_fname)
        fname = "temp_processed_result.log"
        file_after_run(input_fname, fname)
        (iter, value, gnorm, cnorm, ecnorm,
         flowcnorm, flow_cfl, identity, icnorm, snorm,
         linesear, iterPDAS, flagPDAS,
         iterGMRES, resiGMRES, flagGMRES, feasible, n_active_,
         n_vmult, dRdW_form, dRdW_mult, dRdX_mult, d2R_mult) \
            = np.genfromtxt(fname, skip_header=2, comments='&&&',
                            filling_values="0",
                            delimiter=pdas_delimiter_fs, unpack=True)
        work = n_vmult
        total_subit = sum(iterGMRES)
        total_work = work[-1]
        cycles = iter[-1]

        if custom_cost_1:
            total_work = 0
            #total_work = total_work + dRdW_form[-1] * count_flops.form_dRdW_AD_cost(n_dim, poly)
            total_work = total_work + (dRdW_form[-1] - cycles*2) * count_flops.form_dRdW_AD_cost(n_dim, poly)
            total_work = total_work + dRdW_mult[-1] * count_flops.dRdW_vmult_cost(n_dim, poly)

            total_work = total_work + cycles * count_flops.form_dRdX_AD_cost(n_dim, poly)
            total_work = total_work + dRdX_mult[-1] * count_flops.dRdX_vmult_cost(n_dim, poly)
            # total_work = total_work + cycles        * 0
            # total_work = total_work + dRdX_mult[-1] * count_flops.AD_vector_dRdX_cost()

            total_work = total_work + cycles * count_flops.form_d2R_AD_cost(n_dim, poly)
            total_work = total_work + d2R_mult[-1] * 0.25 * count_flops.d2R_vmult_cost(n_dim, poly)

        time = 0
        with open(fname, 'r') as f:
            lines = f.read().splitlines()
            time_line = lines[-2]
            time = (re.findall("\d+\.\d+", time_line))[0]
            time = int(float(time))

        plt.figure(grad_vs_cycles_figure_name, figsize=figsize)
        plt.semilogy(iter, gnorm, '-o', ms=marker_size,
                     label=r'$n$ = ' + str(n_design)
                     )  # +'  time='+str(time))

        plt.figure(value_vs_cycles_figure_name, figsize=figsize)
        final_value = 0
        if objective_final_value:
            final_value = value[-1]
        before_final_value = abs(value[-2]-final_value)
        plt.semilogy(iter, abs(value-final_value)+before_final_value, '-o', ms=marker_size,
                     label=r'$n$ = ' + str(n_design)
                     )  # +'  time='+str(time))

        table_line = '%10d & %10d & %16s & %10d & %10d & %10.1f & %10d & %10.1f \\\\' % (
            n_design, n_dofs_list[polyindex],
            opttype, cycles,
            total_subit, total_subit / cycles,
            total_work, total_work / cycles
        )
        table_print = table_print + table_line + '\n'
        row = pd.DataFrame([{
            'Method': 'Full-space $\\PTwoA$',
            '$n$': n_design,
            '$m$': n_dofs_list[polyindex],
            'Cycles': cycles,
            'Subits': total_subit,
            'Work': total_work
        }]
            , index=['FS_P2A_' + str(n_design) + '_P' + str(poly)])
        #df = df.append(row)
        df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

        plt.legend()
        plt.xlabel(r'Iterations')
        plt.ylabel(r'Gradient Norm')
        plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
        plt.tight_layout()

        n_iterations_list.append(cycles);
        work_list.append(total_work)
        nkkt_list.append(total_subit)

        if n_design in GradientvsSubiterations_design_list:
            ax = axs_FS_cycles[polyindex]
            ms_index = GradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(np.cumsum(iterPDAS), gnorm,
                        linestyle=line_style_str[0],
                        color=color_str[ms_index],
                        marker=marker_str[ms_index],
                        ms=marker_size,
                        label=r'FS $\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

            plt.figure("FullspaceGradientvsSubiterations" + str(poly), figsize=figsize)
            ms_index = GradientvsSubiterations_design_list.index(n_design)
            plt.semilogy(np.cumsum(iterPDAS), gnorm,
                         linestyle=line_style_str[0],
                         color=color_str[ms_index],
                         marker=marker_str[ms_index],
                         ms=marker_size,
                         label=r'FS $\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

        ax = axs_all_cycles[polyindex]
        ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
        ax.semilogy(iter, gnorm,
                    linestyle = line_style_str[full_space_linestyle],
                    color=color_str[ms_index],
                    marker=marker_str[ms_index],
                    ms=marker_size,
                    label=r'FS $\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

        if poly - 1 == 1 and n_design in ReducedGradientvsSubiterations_design_list:
        #if n_design in ReducedGradientvsSubiterations_design_list:
            ax = ax_gradient_cycles
            ms_index = ReducedGradientvsSubiterations_design_list.index(n_design)
            ax.semilogy(iter, gnorm,
                        linestyle=line_style_str[2],
                        color=color_str[ms_index],
                        marker=marker_str[ms_index],
                        #ms=marker_size,
                        ms=2,
                        label=r'FS $\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

    plt.figure(grad_vs_cycles_figure_name, figsize=figsize)
    plt.title(r'Full-space with $\tilde{\mathbf{P}}_2$ Gradient vs Design Cycles, $m$ = ' + str(n_dofs_list[polyindex]))
    if xlimit:
        plt.xlim([1, max_design_cycles])
    plt.xlabel(r'Design Cycles'); plt.ylabel(r'Gradient Norm')
    plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
    plt.legend(); plt.tight_layout(); pp.savefig(bbox_inches='tight')
    plt.close(grad_vs_cycles_figure_name)

    plt.figure(value_vs_cycles_figure_name, figsize=figsize)
    plt.title(r'Full-space with $\tilde{\mathbf{P}}_2$ Value vs Design Cycles, $m$ = ' + str(n_dofs_list[polyindex]))
    if xlimit:
        plt.xlim([1, max_design_cycles])
    plt.xlabel(r'Design Cycles'); plt.ylabel(r'|Objective Value - Final Value|')
    plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
    plt.legend(); plt.tight_layout(); pp.savefig(bbox_inches='tight')
    plt.close(value_vs_cycles_figure_name)

    pwork_list.append(work_list)

    plt.figure("Iterations_vs_nctl_FullSpace", figsize=figsize)
    plt.plot(n_design_list_more, n_iterations_list,
             linestyle=line_style_str[polyindex],
             color=color_str[full_space_color],
             marker=marker_str[3],
             ms=marker_size - 1,
             label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = ' + str(n_dofs_list[polyindex]))

    plt.figure("Iterations_vs_nctl_ReducedandFullSpace", figsize=figsize)
    plt.plot(n_design_list_more, n_iterations_list,
             linestyle=line_style_str[full_space_linestyle],
             color=color_str[polyindex],
             marker=marker_str[full_space_marker],
             ms=marker_size - 1,
             label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = ' + str(n_dofs_list[polyindex]))

    plt.figure("nkkt_vs_nctl", figsize=figsize)
    plt.plot(n_design_list_more,
             nkkt_list,
             linestyle=line_style_str[polyindex],
             color=color_str[full_space_color],
             marker=marker_str[3],
             ms=marker_size,
             label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = ' + str(n_dofs_list[polyindex]))

    plt.figure("work_vs_nctl", figsize=figsize)
    plt.plot(n_design_list_more,
             work_list,
             linestyle=line_style_str[polyindex],
             color=color_str[full_space_color],
             marker=marker_str[3],
             ms=marker_size,
             label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = ' + str(n_dofs_list[polyindex]))

    plt.figure("work_vs_nctl_2", figsize=figsize)
    plt.plot(n_design_list_more,
             work_list,
             linestyle=line_style_str[polyindex],
             color=color_str[full_space_color],
             marker=marker_str[polyindex],
             ms=marker_size,
             label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = ' + str(n_dofs_list[polyindex]))
    # label=r'FS $\tilde{\mathbf{P}}_2$, $p$ = '+str(poly))

    plt.figure("work_vs_nctl_3", figsize=figsize)
    plt.plot(n_design_list_more,
             work_list,
             linestyle=line_style_str[polyindex],
             color=color_str[full_space_color],
             marker=marker_str[3],
             ms=marker_size,
             label=r'FS $\tilde{\mathbf{P}}_2$, $m$ = ' + str(n_dofs_list[polyindex]))

for i, n_design in enumerate(WorkvsNDOFS_design_list):
    j = n_design_list_more.index(n_design)
    x = []
    y = []
    for polyindex, poly in enumerate(poly_list):
        x.append(n_dofs_list[polyindex])
        y.append(pwork_list[polyindex][j])

    plt.figure("work_vs_nstate", figsize=figsize)
    plt.plot(x, y,
             linestyle=line_style_str[i],
             color=color_str[full_space_color],
             marker=marker_str[i],
             ms=marker_size,
             label=r'FS $\tilde{\mathbf{P}}_2$, $n$ = ' + str(n_design))

for polyindex, poly in enumerate(poly_list):
    plt.figure('FullspaceGradientvsSubiterations' + str(poly), figsize=figsize)
    plt.title(r'Full-space Gradient vs Design Cyles, $m$ = ' + str(n_dofs_list[polyindex]))
    plt.xlabel(r'Number of Subiterations')
    plt.ylabel(r'Gradient Norm')
    plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
    plt.legend()
    plt.tight_layout()
# 15
    pp.savefig(bbox_inches='tight')
    plt.close()

plt.figure('FullspaceGradientvsSubiterations',figsize=figsize)
#axs_FS_cycles[0].legend()
for i, ax in enumerate(axs_FS_cycles):
    ax.set_title(r'State Variables, $m$ = ' + str(n_dofs_list[i]))
    # ax.set(xlabel='Number of cycles', ylabel='Gradient Norm')
    ax.set(ylabel='Gradient Norm')
    ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
    # ax.legend()
    # ax.tight_layout()
axs_FS_cycles[-1].set(xlabel='Design Cycles')
# plt.xlabel(r'Number of Subiterations')
# plt.ylabel(r'Gradient Norm')
# plt.grid(visible=True, which='major', color='black', linestyle='-',alpha=0.2)
# plt.legend()
#fig_FS_cycles.suptitle(r'Full-space Gradient vs Design Cycles')
fig_FS_cycles.tight_layout()
fig_FS_cycles.subplots_adjust(top=0.92)

handles, labels = axs_FS_cycles[0].get_legend_handles_labels()
handles = [matplotlib.lines.Line2D([0], [0], color='w', marker='o', markersize=0)] + handles
#labels = ['$m$ = # state var.'] + labels
labels = ['$n$ = # control var.'] + labels
fig_FS_cycles.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.075,), framealpha=1.0)
# 16
pp.savefig(fig_FS_cycles, bbox_inches='tight')
# fig_FS_subiter.close()

plt.figure("Iterations_vs_nctl_FullSpace", figsize=figsize)
plt.title(r'Full-space Design Cycles vs Control Variables')
plt.ylabel(r'Number of Design Cycles to Convergence')
plt.xlabel(r'Number of Control Variables $n$')
#plt.yticks(np.arange(3, 6, 1.0))
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
plt.legend()
plt.tight_layout()
# 17
pp.savefig(bbox_inches='tight')
plt.close()

plt.figure("Iterations_vs_nctl_ReducedandFullSpace", figsize=figsize)
#plt.title(r'Design Cycles to Convergence vs Control Variables')
plt.ylabel(r'Number of Design Cycles to Convergence')
plt.xlabel(r'Number of Control Variables $n$')
#plt.yticks(np.arange(3, 6, 1.0))
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
#plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
handles = [matplotlib.lines.Line2D([0], [0], color='w', marker='o', markersize=0)] + handles
labels = ['$m$ = # state var.'] + labels
#labels = ['$n$ = # control var.'] + labels
plt.legend(handles, labels)#, loc='lower right', bbox_to_anchor=(0.98, 0.075,), framealpha=1.0)
plt.tight_layout()
# 18
pp.savefig(bbox_inches='tight')
plt.close()

plt.figure("work_vs_nctl", figsize=figsize)
plt.title(r'Work Units vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Work Units to Convergence')
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
plt.legend()
plt.tight_layout()
# 19
pp.savefig(bbox_inches='tight')
plt.close()

plt.figure("nkkt_vs_nctl", figsize=figsize)
plt.title(r'Total Subiterations vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Total Subiterations to Convergence')
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
plt.legend(loc=2)
plt.tight_layout()
# 20
pp.savefig(bbox_inches='tight')
plt.close()

plt.figure("work_vs_nctl_2", figsize=figsize)
#plt.title(r'Work Units vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Work units to Convergence')
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
#plt.legend(loc=2)
handles, labels = plt.gca().get_legend_handles_labels()
handles = [matplotlib.lines.Line2D([0], [0], color='w', marker='o', markersize=0)] + handles
labels = ['$m$ = # state var.'] + labels
#labels = ['$n$ = # control var.'] + labels
#fig_RS_cycles.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(handles, labels)#, loc='lower right', bbox_to_anchor=(0.98, 0.075,), framealpha=1.0)
plt.tight_layout()
# 21
pp.savefig(bbox_inches='tight')
plt.close()

plt.figure("Iterations_vs_nctl_ReducedSpace", figsize=figsize)
plt.title(r'Reduced-space Design Cycles vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Number of Design Cycles to Convergence')
# plt.yticks(np.arange(3, 6, 1.0))
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
plt.legend()
plt.tight_layout()
# 22
pp.savefig(bbox_inches='tight')
plt.close()

for i, ax in enumerate(axs_RS_cycles):
    ax.set_title(r'State Variables, $m$ = ' + str(n_dofs_list[i]))
    # ax.set(xlabel='Number of design cycles', ylabel='Gradient Norm')
    ax.set(ylabel='Gradient Norm')
    ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
# axs_RS_cycles[-1].set(xlabel='Number of design cycles or Newton cycles')
axs_RS_cycles[-1].set(xlabel='Design Cycles')
#fig_RS_cycles.suptitle(r'Reduced-space Gradient vs Design Cycles')
fig_RS_cycles.tight_layout()
fig_RS_cycles.subplots_adjust(top=0.92)
#axs_RS_cycles[-1].legend()
# Legend outside on the center right
#fig_RS_cycles.legend(loc='center right', bbox_to_anchor=(1, 0.5))
handles, labels = axs_RS_cycles[0].get_legend_handles_labels()
handles = [matplotlib.lines.Line2D([0], [0], color='w', marker='o', markersize=0)] + handles
#labels = ['$m$ = # state var.'] + labels
labels = ['$n$ = # control var.'] + labels
#fig_RS_cycles.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
fig_RS_cycles.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.075,), framealpha=1.0)
# 23
pp.savefig(fig_RS_cycles, bbox_inches='tight')
# fig_RS_subiter.close()

axs_all_cycles[0].legend()
for i, ax in enumerate(axs_all_cycles):
    ax.set_title(r'$m$ = ' + str(n_dofs_list[i]))
    # ax.set(xlabel='Number of design cycles', ylabel='Gradient Norm')
    ax.set(ylabel='Gradient Norm')
    ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
axs_all_cycles[-1].set(xlabel='Design Cycles')
fig_all_cycles.suptitle(r'Full-space Gradient vs Design Cycles')
fig_all_cycles.tight_layout()
fig_all_cycles.subplots_adjust(top=0.92)
# 24
pp.savefig(fig_all_cycles, bbox_inches='tight')

plt.figure("work_vs_nctl_3", figsize=figsize)
plt.title(r'Work Units vs Control Variables')
plt.xlabel(r'Number of Control Variables $n$')
plt.ylabel(r'Work Units to Convergence')
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
plt.legend()
plt.tight_layout()
# 25
pp.savefig(bbox_inches='tight')
plt.close()

plt.figure("work_vs_nstate", figsize=figsize)
#plt.title(r'Work Units vs State Variables')
plt.xlabel(r'Number of State Variables $m$')
plt.ylabel(r'Work Units to Convergence')
plt.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
handles = [matplotlib.lines.Line2D([0], [0], color='w', marker='o', markersize=0)] + handles
#labels = ['$m$ = # state var.'] + labels
labels = ['$n$ = # control var.'] + labels
plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
# 26
pp.savefig(bbox_inches='tight')
plt.close()

plt.figure("gradnorm_vs_designcycles", figsize=figsize)
ax = ax_gradient_cycles
ax.set_title(r'Gradient Norm vs Design Cycles m = ' + str(n_dofs_list[1]))
ax.set(ylabel=r'Gradient Norm')
ax.set(xlabel=r'Design Cycles')
ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.2)
ax.legend(loc=1)
plt.tight_layout()
# 27
pp.savefig(bbox_inches='tight')

pp.close()

print(table_print)

# col_names = ['$n$', '$m$', 'Method', 'Cycles', 'Subits', 'Subits/Cycle', 'Work', 'Work/Cycle']
df['Subits/Cycle'] = df['Subits'] / df['Cycles']
df['Work/Cycle'] = df['Work'] / df['Cycles']

table_print = ''
print(df)
for method in df['Method'].unique():
    method_rows = df.loc[df.index[df['Method'] == method].tolist()]

    table_print = table_print + '\\midrule \n'
    table_print = table_print + '\multicolumn{16}{c}{\\textbf{' + method + '}} \\\\ \n'
    table_print = table_print + '\\midrule \n'
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

        #line = line + ('{:4.0f} & {:4.0f} &'.format(cycles[0], cycles[1]))
        #line = line + ('{:4.0f} & {:4.0f} &'.format(subits[0], subits[1]))
        #line = line + ('{:5.1f} & {:5.1f} &'.format(subitsc[0], subitsc[1]))
        #line = line + ('{:7.0f} & {:7.0f} &'.format(work[0], work[1]))
        #line = line + ('{:7.0f} & {:7.0f} '.format(workc[0], workc[1]))

        line = line + '\\\\ \n'
        table_print = table_print + line

print(table_print)
