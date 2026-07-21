#see "Installation" doc for info on this. Define equations in the equation1, equation2 (etc.) functions at the top of the code, otherwise see the bottom of the code for variables.
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import time
    
def equation1(r0, r1, p0): #reactant 0, reactant 1, product 0 (counting from 0 since it's coding - search 'why do programmers count from 0'
    return (p0**2) / (r0 * r1) #e.g. ([NO]^2) / ([N2] * [O2])

def equation2(r0, r1, p0):
    return (p0**2) / ((r0 ** 2) * r1) #e.g. ([NO2]^2) / (([NO]^2) * [O2])

def solvingEquations1(vars, initialAmounts, stoichiometry, K):
    x = vars
    currentAmounts = [float((amount + (x * coeff))[0]) for amount, coeff in zip(initialAmounts, stoichiometry)]
    return equation1(currentAmounts[0], currentAmounts[1], currentAmounts[2]) - K

def solvingEquations2(vars, initialAmounts, stoichiometryEq1, stoichiometryEq2, K1, K2):
    x, y = vars
    currentAmounts = [(amount + (x * coeff)) for amount, coeff in zip(initialAmounts, stoichiometryEq1)] #for equation 1
    currentAmounts = [(amount + (y * coeff)) for amount, coeff in zip(currentAmounts, stoichiometryEq2)] #for equation 2
    return [equation1(currentAmounts[0], currentAmounts[1], currentAmounts[2]) - K1, equation2(currentAmounts[2], currentAmounts[1], currentAmounts[3]) - K2]

equilibriaNumber = 1  #number of equilibrium reactions, e.g. N2 + O2 -> 2NO, and 2NO + O2 -> 2NO2
substanceInfo = [{"Name": "N2", "Amount": 60, "CoeffEq1": -1, "CoeffEq2": 0},
    {"Name": "O2", "Amount": 68, "CoeffEq1": -1, "CoeffEq2": -1},
    {"Name": "NO", "Amount": 0,  "CoeffEq1": 2, "CoeffEq2": -2},
    {"Name": "NO2", "Amount": 0, "CoeffEq1": 0, "CoeffEq2": 2}] #information on the species involved. Amount = number of molecules. Coefficient as per the equation; the coefficient is negative if the species is being used up in the reaction.
initialAmounts = [substance["Amount"] for substance in substanceInfo] #collecting together the initial amounts of each chemical species
stoichiometryEq1 = [substance["CoeffEq1"] for substance in substanceInfo] #stoichiometry for equation 1
stoichiometryEq2 = [substance["CoeffEq2"] for substance in substanceInfo] #stoichiometry for equation 2
names = [substance["Name"] for substance in substanceInfo]

if equilibriaNumber == 1: #considering equation1 specifically (defined above)
    guessOfNumberReacted = 5 #guess of how many of the first reactant specified (N2 in this case) have reacted at equilibrium. Note the number of moles of N2 in the reaction here (1) - the 'guessOfNumberReacted' should be done for a one-mole species.
    K = 2 #equilibrium constant
    solution = least_squares(solvingEquations1, x0=guessOfNumberReacted, bounds=(0, substanceInfo[0]["Amount"]), args=(initialAmounts, stoichiometryEq1, K)) #upper bound is the starting amount of N2
    actualNumberReacted = solution.x
    equilibriumAmounts = [round(float((amount + (actualNumberReacted * coeff))[0])) for amount, coeff in zip(initialAmounts, stoichiometryEq1)]
    for name, amount in zip(names, equilibriumAmounts):
        print(str(name)+" has "+str(amount)+" molecules present at equilibrium")

if equilibriaNumber == 2: #considering both equation1 and equation2
    guessesOfNumberReacted = [31, 13] #guess of number of N2 molecules reacted in equation 1, and of NO molecules reacted in equation 2
    K1, K2 = 2, 0.06 #K values for equation 1 and equation 2
    solution = least_squares(solvingEquations2, x0 = (guessesOfNumberReacted[0], guessesOfNumberReacted[1]), bounds = ([0, 0], [substanceInfo[0]["Amount"], 80]), args = (initialAmounts, stoichiometryEq1, stoichiometryEq2, K1, K2))
    actualNumberReacted0, actualNumberReacted1 = solution.x
    equilibriumAmounts = [round(float((amount + (actualNumberReacted0 * coeff)))) for amount, coeff in zip(initialAmounts, stoichiometryEq1)]
    equilibriumAmounts = [round(float((amount + (actualNumberReacted1 * coeff)))) for amount, coeff in zip(equilibriumAmounts, stoichiometryEq2)]
    for name, amount in zip(names, equilibriumAmounts):
        print(str(name)+" has "+str(amount)+" molecules present at equilibrium")

        
