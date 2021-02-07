#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
qap = [f for f in listdir("QA4QUBO/tests/") if isfile(join("QA4QUBO/tests/", f))]
npp = [f for f in listdir("QA4QUBO/npp/") if isfile(join("QA4QUBO/npp/", f))]
MAX = 1000

np.set_printoptions(threshold=sys.maxsize)

def getproblem():
    elements = list()
    i = 0
    for element in qap:
        elements.append(element)
        element = element[:-4]
        print(f"Write {i} for the problem {element}")
        i += 1
    
    problem = int(input("Which problem do you want to select? "))
    DIR = "QA4QUBO/tests/"+qap[problem]
    return DIR

def getS():
    elements = list()
    i = 0
    for element in npp:
        elements.append(element)
        element = element[:-4]
        print(f"Write {i} for the problem {element}")
        i += 1
    
    problem = int(input("Which problem do you want to select? "))
    DIR = "QA4QUBO/npp/"+npp[problem]
    return DIR

def generateS():
    S = list()
    DIR = getS()
    with open(DIR) as file:
        n = int(file.readline())
        for i in range(n):
            S.append(int(file.readline().rstrip("\n")))

    return S, n


def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

def main(_n):  
    nok = True
    i = 0
    max_range = 1000
    dir = "output_"+str(_n)+"_"+ str(max_range)
    while(nok):
        try:
            with open("output/"+dir+".txt", "r") as file:
                pass
            max_range = int(MAX/(2**i))
            if(not max_range):
                exit("File output terminati")
            dir = "output_"+str(_n)+"_"+ str(max_range)
            i += 1
        except IOError:
            nok = False
        
    _DIR = "output/"+dir+".txt"
    open(_DIR, 'a').close()
    
    """
    QAP = input("Do you want to use a QAP problem? (y/n) ")
    if(QAP in ['y', 'Y', 1, 's', 'S']):
        QAP = True
        DIR = getproblem()
        with open(DIR) as file:
            _Q = matrix.generate_QAP_problem(file)
            _n = len(_Q)
    else:
        QAP = False
        S, _n = generateS()
        _Q, c = matrix.generate_QUBO_problem(S)
    """
    QAP = False
    S = vector.generate_S(_n, max_range)
    #S = [25, 7, 13, 31, 42, 17, 21, 10]
    #S = [4,6,1,10,5,3,7,12,9,8]
    #S = [7, 2, 12, 1, 0, 6, 14, 5, 7, 8, 15, 4, 2, 7, 11, 10]
    _Q, c = matrix.generate_QUBO_problem(S)

    string = " ---------- Problem start ----------\n"
    print(string)
    write(_DIR, string)
    
    if not QAP:
        string = "\n S = "+str(S)+"\n"
        print(string)
        write(_DIR, string)
    
    
    z = solver.solve(d_min = 70, eta = 0.01, i_max = 1000, k = 3, lambda_zero = 1, n = _n, N = 10, N_max = 100, p_delta = 0.1, q = 0.2, topology = 'pegasus', Q = _Q, DIR = _DIR, sim = False)
    
    min_z = solver.function_f(_Q,np.atleast_2d(z).T).item()
    
    string = "So far we found:\n- z - \n"+str(np.atleast_2d(z).T)+"\nand has minimum = "+str(min_z)+"\n"
    diff2 = (c*c + 4*min_z)

    try:
        string += "c = "+str(c)+", c^2 = "+str(c**2)+", diff^2 = "+str(diff2)+","+str(np.sqrt(diff2))+"\n"
        print(string)
        write(_DIR, string)
    except:
        print(string)
        write(_DIR, string)



if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except:
        n = int(input("Insert n: "))
    main(n)