#!/usr/bin/python3
import re
import numpy as np

fname = 'output_lifttarget_momentconstraint'

polynomial_ramping_degree = 1
number_of_times_ODESolver_appears_per_ODESolver = polynomial_ramping_degree + 2

sum_iterations = 0;
iterations_count = 0;
n_ode = 0
n_steps_count = 0
n_steps_sum = 0
with open(fname, 'r') as ifile:
    for line in ifile:
        if 'Total of' in line:
#            print (line)
#            print (int(re.findall(r'\d+', line)[1]))
            iterations_count = iterations_count + 1;
            it = int(re.findall(r'\d+', line)[1])
            sum_iterations = sum_iterations + it

        if 'ODESolver' in line:
#            print (line)
#            print (int(re.findall(r'\d+', line)[1]))
            n_ode = n_ode + 1;

        if 'stopped' in line:
            next_line = (next(ifile, '').strip())
            n_steps = int(re.findall(r'\d+', next_line)[1])
            if (n_steps is not 0):
                n_steps_count = n_steps_count + 1
                n_steps_sum = n_steps_sum + n_steps




avg = sum_iterations / iterations_count
print ('Average '+str(avg))

n_ode = n_ode / number_of_times_ODESolver_appears_per_ODESolver
print ('N ODE Solves '+str(n_ode))

avg_linear_solves_per_ode = n_steps_sum / n_steps_count
print ('N Linears Solves per ODE '+str(avg_linear_solves_per_ode))
