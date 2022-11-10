import numpy as np
import random

from util import check_bounds
from util import dump_results
from Cost_function import sphere
from Cost_function import objfRastrigin
from Cost_function import objfRosenbrock
from Cost_function import objfGriewank

class FOA:
    def __init__(self, cost_func=None, bounds=None, repeat_num=30, dimension=30, max_evaluations=30,
                 tol=1.0e-4, x0=None, cost=None, display=True, save_directory=None, save_results=False):

        parameter = ([0.2, 2, 1, -100, 100])
        alpha = parameter[0]
        beta = parameter[1]  # attractivite/luminosite
        gamma = parameter[2]  # coef d absorption
        lowerbound = parameter[3]
        upperbound = parameter[4]

        if bounds is None:

            bounds = []

            for i in range(dimension):
                    bounds.append(lowerbound)
                    bounds.append(upperbound)

        if x0 is None:

            x0 = []
            light = []

            # initialization population
            for i in range(repeat_num):
                for j in range(dimension):
                    x0.append(random.uniform(1, dimension) * (upperbound - lowerbound) + lowerbound)  # position luciole
                    light.append(beta)

        function_evaluations = 0

        # Evalutation
        if cost is None:
            cost = []
            cost_p = cost_func
            if save_directory is None:
                save_directory = 'results'

            if type(cost_p) is not float:

                function_evaluations = function_evaluations + 1
                if save_results:
                    dump_results(save_directory=save_directory, solution=x0, cost_dic=cost_p,
                                 current_evaluation=function_evaluations, initial_population=True)
            else:
                if save_results:
                    dump_results(save_directory=save_directory, solution=x0, cost_dic=cost_p,
                                 current_evaluation=function_evaluations, initial_population=True)
            cost.append(cost_p)

            if (min(cost) < tol) or (function_evaluations >= max_evaluations):
                return
            cost = np.array(cost)

        #
        x = x0.copy()  # copie du tableau de population
        fx = cost  # tableau
        f0 = fx  # best solution

        # Main LOOP: for each iteration
        for m in range(max_evaluations):
            for i in range(repeat_num):
                # generate new solution
                for j in range(len(x)):
                    if light[j] > light[i]:
                        # deplacement vers la luciole la plus lumineuse
                        r = np.sum(np.sqrt(x[i] - x[j]))  # distance entre deux lucioles
                        new_beta = beta * np.exp(-gamma * r)
                        x[i] += new_beta * (x[j] - x[i]) + alpha
                    else:
                        # deplacement aleatoire
                        x[i] += alpha

                trial_x = x.copy()

                # check_bounds
                trial_x = check_bounds(solution=trial_x, bounds=bounds)

                # evaluate new solution
                trial_cost = cost_func(trial_x)

                # update function_evaluations
                if type(trial_cost) is not float:

                    function_evaluations = function_evaluations + 1
                    if save_results:
                        dump_results(save_directory=save_directory, solution=trial_x, cost_dic=trial_cost,
                                     current_evaluation=function_evaluations, initial_population=False)
                else:
                    if save_results:
                        dump_results(save_directory=save_directory, solution=trial_x, cost_dic=trial_cost,
                                     current_evaluation=function_evaluations, initial_population=False)

                df = trial_cost - fx

                # accept the new solution with Metropolis equation
                if (df < 0) or (np.random.rand() < np.exp(-T * df / (np.abs(fx) + np.finfo(float).eps) / tol)):
                    x = trial_x.copy()
                    fx = trial_cost

                # update the best solution
                if fx < f0:
                    f0 = fx

                # stoping condition
                if (f0 < tol) or (function_evaluations >= max_evaluations):
                    return

                if display:
                    print(['The best solution after ', function_evaluations, 'evaluations is: ', f0])


if __name__=='__main__':
    execution = 2
    dimension=30
    bounds = []
    for i in range (dimension):
        bounds.append(-100)
        bounds.append(100)
    argument = dict(cost_func=objfRastrigin,repeat_num=30, dimension=len(bounds), bounds=bounds, max_evaluations=30, display=True,
            save_results=True, tol=-1.0e-4)
    for i in range (1,execution):
        FOA(**argument, save_directory='results'+str(i))

