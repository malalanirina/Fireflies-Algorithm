import numpy as np


def objfRastrigin(x):       #2D Shifted Rastrigin's function
    sum = 0
    for i in range (len(x)):
        for j in range (len(x[i])):
            sum += np.power(x[i][j],2)-10*np.cos(2*np.pi*x[i][j])+10

    return sum


def sphere(x):
    sum = 0
    for i in range (len(x)):
        for j in range(len(x[i])):
            sum += np.power(x[i][j], 2)

    return sum


def objfRosenbrock(x):      #2D Shifted Rosenbrock's function
    sum = 0
    for z in x:
        for j in range(len(x[i])):
            sum += np.power((np.power(x[z][j],2)-(x[z][j+1])),2)*100+np.power(x[z][j]-1,2)

    return sum


def objfGriewank(x):        #2D Shifted Griewank's function
    sum = 0
    product = 1
    result = 0
    for i in range(len(x[i])):
        for j in range(len(x)):
            product *= np.cos(x[i][j] / np.sqrt(i) + 1)
            sum += np.power(x[i][j], 2)

    return 1 + sum / 4000 - product


