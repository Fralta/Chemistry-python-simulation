import pygame as pg
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import re
import copy
import pandas as pd
from scipy.constants import R

def fileWrite(outputList): 
    file = open("output.txt", "w")
    for i in range(0, len(outputList)):
        file.write(str(outputList[i])+"\n")
    file.close()    

def rotate(origin, point, angle): #angle in radians
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class substanceGen(pg.sprite.Sprite):  #class for chemical substance
    def __init__(self, substanceParams):   #sets up the atom / molecule
        # substanceParams = [[substanceType, atomNumber, x, y, xdir, ydir, substanceID, angleStartPos, rotationDirection, frameN, recentColl, frameWidth, productPair, velocityDisplayScale],
            #[name, colour, radius, mass], ...] #properties of (proposed) new atom / molecule
        pg.sprite.Sprite.__init__(self)  #functioning code
        screen = pg.display.get_surface() #get screen
        self.substanceType = substanceParams[0][0]  #F2, H2, N2, O2, Ne, NO..
        self.atomNumber = substanceParams[0][1]
        self.velocity = pg.math.Vector2(substanceParams[0][4], substanceParams[0][5])
        self.substanceID = substanceParams[0][6]
        velocityDisplayScale = substanceParams[0][13]
        self.mass = 0
        self.atomMasses = []
        self.colour = []
        self.atomRadii = []
        self.atomInfo = []
        self.pos = [pg.math.Vector2(substanceParams[0][2], substanceParams[0][3])]
        self.radius = 0
        self.iterations = 3 #iterations of substance position between current and next, to check for collisions in the intermediate positions between the current and next positions 
        for i in range(1, self.iterations + 2):
                self.pos.append(pg.math.Vector2(substanceParams[0][2] + i * substanceParams[0][4] * velocityDisplayScale / (self.iterations + 1), substanceParams[0][3] + i * substanceParams[0][5] * velocityDisplayScale / (self.iterations + 1)))
        self.reactingPos = self.pos[self.iterations+1] #specifies the exact next position of the substance (defaults on current pos + velocity, but reactingPos gives a greater position accuracy for collisions then just using this value)
        for i in range(1, self.atomNumber+1):
            self.atomInfo.append(substanceParams[i])
            self.colour.append(substanceParams[i][1])
            self.atomRadii.append(substanceParams[i][2])
            self.atomMasses.append(substanceParams[i][3])
            self.mass+=substanceParams[i][3]
            self.reactingAtomCentres = [] #centre of atom(s) at reacting position
            self.radius = max(self.radius, substanceParams[i][2]) #total molecule radius not including displacement from centre of mass
        if self.atomNumber == 1:
            self.radius = self.atomRadii[0]
            self.surfaceDimensions = pg.math.Vector2(self.radius * 3, self.radius *  3) #self.surfaceDimensions = 3x self.radius
            self.image = pg.Surface((self.surfaceDimensions), pg.SRCALPHA)
            pg.draw.circle(self.image, self.colour[0], self.surfaceDimensions * 1/2, self.radius)  #puts atom in position of the surface
            self.reactingAtomCentres = [self.reactingPos]
        else:
            if substanceParams[0][7] == "na": #na here and below indicates that these aren't being defined upon creation of the substance (7 is angle, 8 is rotation direction)
                startingAngle = random.randint(0, 12) * math.pi / 6 #fraction of 2pi radians to rotate by, gives full 2pi rotation after 12 frames
            else:
                startingAngle = substanceParams[0][7] * math.pi / 6
            if substanceParams[0][8] == "na": 
                self.angleDisp = [startingAngle, random.choice([-1, 1])] #second number is direction of rotation
            else:
                self.angleDisp = [startingAngle, substanceParams[0][8]]
            self.surfaceDimensions = pg.math.Vector2(self.radius * 10, self.radius * 10)
            self.image = pg.Surface((self.surfaceDimensions), pg.SRCALPHA)
            self.com = pg.math.Vector2(self.surfaceDimensions / 2) #rotation around 'centre of mass'
            if self.atomNumber == 2:
                l1 = (self.atomMasses[1] * (self.atomRadii[0] + self.atomRadii[1])) / (self.atomMasses[0] + self.atomMasses[1]) #l1 = r1 when m1r1 = m2r2
                self.imageAtomCentres = [self.com - (pg.math.Vector2(0, l1)) * 0.6, self.com + (pg.math.Vector2(0, self.atomRadii[0]
                                                                                + self.atomRadii[1] - l1)* 0.6)] #at 0º angle (no rotation), imageAtomCentres[0] is vertically above imageAtomCentres[1], 0.6 used to adjust atom separation to be more visually realistic
            if substanceParams[0][0] == "H2O":
                self.imageAtomCentres = [self.com - (pg.math.Vector2(0, 97/9)) * 0.04, self.com + pg.math.Vector2(80, 97-(97/9)) * 0.07, self.com + pg.math.Vector2(-80, 97-(97/9)) * 0.07]
            if substanceParams[0][0] == "NO2":
                self.imageAtomCentres = [self.com - (pg.math.Vector2(0, 97/9)) * 0.18, self.com + pg.math.Vector2(100, 137-(137/9)) * 0.09, self.com + pg.math.Vector2(-100, 137-(137/9)) * 0.09]
            for i2 in range(0, self.atomNumber):
                qx, qy = rotate(self.com, self.imageAtomCentres[i2], startingAngle) #origin, point, angle
                self.reactingAtomCentres.append(pg.math.Vector2(qx, qy) - self.com + self.reactingPos)
        self.rect = self.image.get_rect(center = (substanceParams[0][2], substanceParams[0][3]))
        self.willReact = 0
        self.currentColl = [] #only used to check current colls between non-reacting atoms (not used for reactions)
        self.recentColl = substanceParams[0][10]
        self.recentCollPrevious = [] #stores recentColl for the most recent 10 frames
        self.frameWidth = substanceParams[0][11]
        self.screenArea = (screen.get_rect()[2] - self.frameWidth, screen.get_rect()[3] - self.frameWidth)
        self.startingFrame = substanceParams[0][9] #frame when rotation begins
        self.product = ""
        self.reactingFrame = 0
        self.productPair = substanceParams[0][12] #substanceID of other molecule created during rxn

##class reactionSetup:
##    def __init__(self, reactionInfo):
##        self.reactants = reactionInfo[0]  # e.g., ['A', 'B']
##       # self.products = products    # e.g., ['C']
##        self.K = reactionInfo[1]  # equilibrium constant
##        self.k_forward = 1.0  # can normalize to 1
##        self.k_reverse = 1.0 / K
##
##    def get_probabilities(self):
##        return self.k_forward, self.k_reverse
##    
##    def matches(self, species):
##        return tuple(sorted(species)) == self.reactants

def collCheck(substance1, substance2, collDetected): #sub-function that does atom-atom collision checking

    
##    for i2 in range(0, substance1.atomNumber):
##        for i3 in range(0, substance2.atomNumber):
##            atomCollisionDistance = substance1.atomRadii[i2] + substance2.atomRadii[i3]
##            if substance1.reactingAtomCentres[i2].distance_to(substance2.reactingAtomCentres[i3]) < atomCollisionDistance:
##                collDetected = 1
    collisionDistance = substance1.radius + substance2.radius
    if substance1.reactingPos.distance_to(substance2.reactingPos) < collisionDistance:
        collDetected = 1
    return collDetected

def getSubstanceAtoms(substanceParams, atomInfo): #gets the constituent atoms and their counts from a substance formula
    substanceParams[0][1]=0 #resetting atom number to 0 in case same substanceParams is passed into this func again
    capitalLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    substanceFormula = substanceParams[0][0].lstrip("0123456789") #removes count of molecule in substanceFormula if present
    indices = [] #stores positions of all capital letters in the formula
    for capitalLetter in range(len(capitalLetters)):
        occurences = [m.start() for m in re.finditer(capitalLetters[capitalLetter], substanceFormula)] #finds positions of a given capital letter in the formula
        for occurence in occurences:
            indices.append(occurence)
    indices = list(filter((-1).__ne__, indices))
    indices.sort()
    atomList = []
    for i6 in range(len(indices) - 1): #splits up substanceFormula into each atom (including the number)
        atomList.append(substanceFormula[indices[i6]:indices[i6+1]])
    atomList.append(substanceFormula[indices[-1]:]) 
    for i7 in range(len(atomList)):
        atomList[i7] = list(filter(("").__ne__, re.split("([^a-zA-Z])", atomList[i7]))) #splits up string into each atom including their count
        if len(atomList[i7]) == 1:
            atomList[i7] = [str(atomList[i7][0]), 1] #adds count of 1 if no count is specified
    for i7 in range(len(atomList)): #need to update atomList before this for loop
        for i4 in range(0, len(atomInfo)): #for all possible atoms
            if atomInfo[i4][0] == atomList[i7][0]: #if atom type found at position i4 in atomInfo
                for i5 in range(int(atomList[i7][1])): #for each atom of this type in the molecule
                       substanceParams.append(atomInfo[i4]) #adds info about the atom to substanceParams
                       substanceParams[0][1]+=1
    if substanceParams[0][0] == "H2O": #need to rearrange atom order for H2O
        for i in range(1, 4):
            if substanceParams[i][0] == "O":
                substanceParams[1] = substanceParams[i]
            else:
                hParams = list(substanceParams[i])
        substanceParams[2] = substanceParams[3] = hParams
    return substanceParams

def substanceSetup(dimensions, atomInfo, frameWidth, substanceGenInfo,TStart,avgMr, velocityDisplayScale):
    substances = pg.sprite.Group(())
    substanceID = 1 #unique ID for each substance on screen. This variable stores the substanceID that will be used for the next substance that is generated.
    largestRadius = 0 #largest radius of the atoms in a substance
    avgSpeed = np.sqrt((2 * R * TStart) / (avgMr / 1000))  # m/s
    for i3 in range(0, len(atomInfo)):
            if atomInfo[i3][2] > largestRadius:
                largestRadius == atomInfo[i3][2]
    for i in range(0, len(substanceGenInfo)):  #for each substanceType
        for i2 in range(0, substanceGenInfo[i][1]):  #for each atom / molecule of the substance
            spawnTries = 0
            while spawnTries < 100: #give 100 tries to spawn, otherwise skip (could do this differently, can lead to an uneven number of each substance type on the screen)
                x = random.randint(largestRadius+frameWidth, dimensions-largestRadius-frameWidth) #random x coordinate
                y = random.randint(largestRadius+frameWidth, dimensions-largestRadius-frameWidth) #random y coordinate
                xdir, ydir = 0, 0
                while xdir == 0 and ydir == 0: #prevents substance from spawning with zero velocity (stationary)
                    angle = random.uniform(0, 2 * math.pi) #random angle of direction
                    xdir, ydir = avgSpeed * math.cos(angle), avgSpeed * math.sin(angle)  #random magnitude + direction of x and y travel
                substanceParams = [[substanceGenInfo[i][0], 0, x, y, xdir, ydir, substanceID, "na", "na", 0, [], frameWidth, 0, velocityDisplayScale]] #properties of (proposed) new atom / molecule
                substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
                substanceNew = substanceGen(substanceParams)
                collDetected = 0
                for substance2 in substances:
                    collDetected = collCheck(substanceNew, substance2, collDetected)
                if collDetected == 0:  #if new atom / molecule doesn't collide with any other substance currently present on screen
                    substances.add(substanceNew) #create the new atom / molecule
                    substanceID+=1
                    break
                spawnTries+=1
    return substances, substanceID

def angleStartFunc(angleStartArgs): #angleStartPos can be 0-12, corresponding to 12 starting rotation iterations (this compares the angle between the atoms at the current and future positions to determine angle start)
    angle = [0, 0]
    angleStartPos = "-"
    rotationDirection = 1
    positions = [[angleStartArgs[0][0], angleStartArgs[0][0] + angleStartArgs[0][1]], [angleStartArgs[1][0], angleStartArgs[1][0] + angleStartArgs[1][1]]]
    for i in range(0, 2):
        b, c = positions[0][i], positions[1][i]
        a = pg.math.Vector2(c[0], 0) #point at top of the screen that is vertically above c
        bc = c - b
        ac = c - a
        if i == 0: #(only considering positions at collision)
            try:
                if c[0] - b[0] == 0 and c[1] > b[1]: #if c vertically below b
                    angleStartPos = 0
                    if positions[0][1][0] > positions[1][1][0]: #if first atom's previous position is to the left of the second atom
                        rotationDirection = -1
                elif c[1] - b[1] == 0 and c[0] < b[0]: #if c horizontally to the left of b
                    angleStartPos = 3
                    if positions[0][1][1] > positions[1][1][1]:
                        rotationDirection = -1
                elif c[0] - b[0] == 0 and c[1] < b[1]: #if c vertically above b
                    angleStartPos = 6
                    if positions[0][1][0] < positions[1][1][0]:
                        rotationDirection = -1
                elif c[1] - b[1] == 0 and c[0] > b[0]: #if c horizontally to the right of b
                    angleStartPos = 9
                    if positions[0][1][1] < positions[1][1][1]:
                        rotationDirection = -1
                else:
                    cosine_angle = np.dot(bc, ac) / (np.linalg.norm(bc) * np.linalg.norm(ac))
                    angle[i] = np.arccos(cosine_angle)
            except Exception as e:
                print(str(e)+" "+str(positions)) #debugging
        else:
            cosine_angle = np.dot(bc, ac) / (np.linalg.norm(bc) * np.linalg.norm(ac))
            angle[i] = np.arccos(cosine_angle)
    if angleStartPos == "-": #non-vertical/horizontal cases
        if positions[0][0][0] < positions[1][0][0]: #if b is to the left of c at current positions
            angle[0] = (2 * math.pi) - angle[0]
        if positions[0][1][0] < positions[1][1][0]: #if b is to the left of c at next positions
            angle[1] = (2 * math.pi) - angle[1]
        if angle[1] < angle[0]: #gets correct rotation direction, + rounds up or down correctly
            angleStartPos = math.floor(angle[0] * 6 / math.pi) #using a multiplier of 6 due to the 6*2 = 12 rotation iterations
            rotationDirection = -1
        elif angle[1] > angle[0]:
            angleStartPos = math.ceil(angle[0] * 6 / math.pi)
        else: #special case where difference in current + next angles don't work (since they give the same angle)
            multiplier = (positions[0][0][0] - positions[1][0][0])/(angleStartArgs[1][1][0] - angleStartArgs[0][1][0])
            pos2 = [positions[0][0] + (multiplier * angleStartArgs[0][1]), positions[1][0] + (multiplier * angleStartArgs[1][1])] #positions at equal x coordinates when following current velocities
            if pos2[0][1] > pos2[1][1]: #if first atom is above second atom
                if angleStartArgs[0][1][0] > angleStartArgs[1][1][0]: #if first atom's x velocity is greater than that of second
                    angleStartPos = math.ceil(angle[0] * 6 / math.pi)
                else:
                    rotationDirection = -1
                    angleStartPos = math.floor(angle[0] * 6 / math.pi)
            else: #if second atom is above first atom
                if angleStartArgs[0][1][0] < angleStartArgs[1][1][0]: #if first atom's x velocity is smaller than that of the second
                    angleStartPos = math.ceil(angle[0] * 6 / math.pi)
                else:
                    rotationDirection = -1
                    angleStartPos = math.floor(angle[0] * 6 / math.pi)
    return angleStartPos, rotationDirection

def reactionProcessing(substances, frameN, substanceID, atomInfo, substance1, substance2, reactionSuccessful, threeSubstanceReaction, velocityDisplayScale):
    for substance3 in substances:
        if substances.has(substance3):
            if substance3.willReact == [substance1.substanceID,substance2.substanceID] and frameN == substance3.reactingFrame: #if three substance reaction was successful
                newx, newy = (substance1.pos[0][0]+substance2.pos[0][0]+substance3.pos[0][0])/3, (substance1.pos[0][1]+substance2.pos[0][1]+substance3.pos[0][1])/3
                totalmv = (substance1.mass * substance1.velocity) + (substance2.mass * substance2.velocity) + (substance3.mass * substance3.velocity)
                totalMass = substance1.mass + substance2.mass + substance3.mass
                reactionSuccessful = 1
                threeSubstanceReaction = 1
                thirdSubstance = substance3.substanceID #assigns substanceID of third substance for later

        if substance2.willReact == substance1.substanceID and substance2.reactingFrame == frameN: #if two substance reaction was successful
            newx, newy = (substance1.pos[0][0]+substance2.pos[0][0])/2, (substance1.pos[0][1]+substance2.pos[0][1])/2
            totalmv = (substance1.mass * substance1.velocity) + (substance2.mass * substance2.velocity)
            totalMass = substance1.mass + substance2.mass
            reactionSuccessful = 1

        if reactionSuccessful == 1:
            productNumber = 0
            products = [s.strip() for s in substance1.product.split('+') if s.strip()]
            try:
                productNumber = int(substance1.product[0]) #number of product molecules
            except:
                productNumber = len(products)
            newVelocity = totalmv / (productNumber * totalMass)
            newSubstanceRecentColl = []
            angleStartArgs = [[substance1.pos[0], substance1.velocity], [substance2.pos[0], substance2.velocity]]
            angleStartPos, rotationDirection = angleStartFunc(angleStartArgs)
            if productNumber == 1:
                substanceParams = [[substance1.product, 2, newx, newy, newVelocity[0], newVelocity[1], substanceID, angleStartPos, rotationDirection, frameN, [], substance1.frameWidth, 0, velocityDisplayScale],
                                   substance1.atomInfo[0], substance2.atomInfo[0]] #new substance properties
                substanceNew = substanceGen(substanceParams)
                substances.add(substanceNew)
                substanceID+=1
            if productNumber > 1:
                velocityMagnitude = np.linalg.norm(newVelocity)
                if velocityMagnitude < 1: #unrealistically low velocity for the product molecules
                    velocityMultiplier = 1 / velocityMagnitude
                    newVelocity = pg.math.Vector2(newVelocity[0] * velocityMultiplier, newVelocity[1] * velocityMultiplier) #boosts up velocity
            if productNumber == 2:
                if substance1.product == "H2O" or substance1.product == "NO2" or substance1.product == "N2+O2" :
                    newSubstancesSeparation, newSubstanceAtomNumber = [10, 0], 3 #separates the two new substances horizontally by this amount (see substanceParams[0][2] below)
                if substance1.product == "NO+NO":
                    substanceVectorSeparation = substance1.pos[0] - substance2.pos[0]
                    collisionDistance = substance1.radius + substance2.radius
                    newSubstancesSeparation, newSubstanceAtomNumber = [substanceVectorSeparation[1]/2, substanceVectorSeparation[0]/2], 2
                substanceParams = [[products[0], 0, newx-newSubstancesSeparation[0], newy-newSubstancesSeparation[1], newVelocity[0], newVelocity[1], substanceID, angleStartPos, rotationDirection,\
                                    frameN, [substanceID+1], substance1.frameWidth, substanceID+1, velocityDisplayScale]]
                substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
              #  print("product "+str(products)+" frame "+str(frameN))
                for i in range(0, 2): #for each new substance
                    if i == 1:
                        substanceParams[0][0] = products[1]
                        substanceParams[0][2]+=(2*newSubstancesSeparation[0])
                        substanceParams[0][3]+=(2*newSubstancesSeparation[1])
                        substanceParams[0][4]*=-1 #making this value (+ below value) opposite for each of the new products
                        substanceParams[0][5]*=-1
                        substanceParams[0][6] = substanceID
                        substanceParams[0][8] = rotationDirection * -1
                        substanceParams[0][10] = [substanceID-1]
                        substanceParams[0][12] = substanceID-1
                        substanceParams = getSubstanceAtoms(substanceParams[:1], atomInfo)
                    substanceNew = substanceGen(substanceParams)
                    substances.add(substanceNew)
                    substanceID+=1
            substances.remove(substance1)
            substances.remove(substance2) #other substances will still be iterated in the substance1 loop (as this loop has already started), but these can be ignored using substances.remove() and checking for substances.has
            if threeSubstanceReaction == 1:
                substances.remove(substance3)
                threeSubstanceReaction = 0
            reactionSuccessful = 0
            break #stops loop through substance3
    return substances, substanceID

def reactionChecking(substances, frameN, substance1, substance2, toggleReaction,reactionInfo, T, reactionCount):
    if substance1 != substance2:  #checks for reactions
            collDetected = 0
            for i in range(1, substance1.iterations+2):
                collDetected = collCheck(substance1, substance2, collDetected)
                if collDetected == 1 and toggleReaction == 1 and substance1.willReact == 0 and substance2.willReact == 0 and substance2.substanceID not in substance1.recentColl and substance1.substanceID not in substance2.recentColl:
                       # print(substance1.substanceID, substance2.recentColl)
                        nextReactionAllowed = 1 #assumes the reaction is allowed (although not certain - still needs to pass the probability check below)
                        if ((substance1.substanceType == "N" and substance2.substanceType == "O") or (substance1.substanceType == "O" and substance2.substanceType == "N")): #N + O -> NO
                            substance1.product = substance2.product = "NO"
                        elif ((substance1.substanceType == "N2" and substance2.substanceType == "O2")):# or (substance1.substanceType == "O2" and substance2.substanceType == "N2")): #N2 + O2 -> 2NO
                            substance1.product = substance2.product = "NO+NO"
                        #elif ((substance1.substanceType == "NO" and substance2.substanceType == "NO")): #2NO -> N2 + O2
                         #   substance1.product = substance2.product = "N2+O2"
                        #add more two substance reactions here
                        else:
                            nextReactionAllowed = 0
                        
                        if nextReactionAllowed == 1:
                            reactantList = sorted([str(substance1.substanceType), str(substance2.substanceType)])
                            reactants = "+".join(reactantList)
                            for r in reactionInfo:
                                if str(r[0]) == reactants or str(r[1]) == reactants:
                                    deltaG = -R * T * np.log(r[2])  # in J/mol
                                    EaF = 5000  # Forward activation energy in J/mol. Artificially very low, since otherwise the probability of successful rxn is vry low (can have unrealistically high temp instead)
                                    EaR = EaF - deltaG  # Reverse Ea
                                    Ea = EaF if str(r[0]) == reactants else EaR  
                            v1 = np.array([substance1.velocity[0], substance1.velocity[1]])
                            v2 = np.array([substance2.velocity[0], substance2.velocity[1]])
                            vRel = v1 - v2 #relative velocity
                            vRelMagSquared = np.dot(vRel, vRel)
                            m1 = (substance1.mass / 1000) / 6.022e23  # kg mass
                            m2 = (substance2.mass / 1000) / 6.022e23  # kg
                            mu = (m1 * m2) / (m1 + m2) #reduced mass
                            ERelMolecule = 0.5 * mu * vRelMagSquared  # Relative kinetic energy (in Joules)
                            ERel = ERelMolecule * (6.022e23) #converting to J mol-1
                               # randv = random.random()
                            if Ea < ERel: #reaction allowed according to probability
                                substance1.willReact = substance2.substanceID
                                substance2.willReact = substance1.substanceID
                                substance2.reactingFrame = frameN + 1
                                reactionCount+=1
                            else:
                                substances, breakLoop = collisionProcessing(substances, substance1, substance2, 0)
                        for substance3 in substances: #checking for three substance reactions
                            if substance3 != substance1 and substance3 != substance2:
                                collDetected, collDetected2 = 0, 0
                                collDetected = collCheck(substance1, substance3, collDetected)
                                collDetected2 = collCheck(substance2, substance3, collDetected) #checking substance3 colliding with either substance1 or substance2
                                if (collDetected == 1 or collDetected2 == 1) and substance3.willReact == 0:
                                    substanceTypes = [substance1.substanceType, substance2.substanceType, substance3.substanceType]
                                    nextReactionAllowed = 1
                                    if substanceTypes.count("H2") == 2 and substanceTypes.count("O2") == 1: #2H2 + O2 -> 2H2O
                                        substance1.product = substance2.product = substance3.product = "H2O+H2O"
                                    #elif substanceTypes.count("NO") == 2 and substanceTypes.count("O2") == 1: #2NO + O2 -> 2NO2
                                      # substance1.product = substance2.product = substance3.product = "NO2+NO2"
                                    #add more three substance reactions here (same indentation)
                                    else:
                                        nextReactionAllowed = 0  
                                    if nextReactionAllowed == 1:
                                        substance1.willReact = [substance2.substanceID,substance3.substanceID]
                                        substance2.willReact = [substance1.substanceID,substance3.substanceID]
                                        substance3.willReact = [substance2.substanceID,substance1.substanceID]
                                        substance3.reactingFrame = frameN + 1 #specifies that reaction will take place in next frame
    return substances, reactionCount

def collisionProcessing(substances, substance1, substance2, breakLoop):
        if substance1 != substance2 and substances.has(substance2):
            collDetected = 0
            for i in range(1, substance1.iterations+2):
                collDetected = collCheck(substance1, substance2, collDetected) 
                if collDetected == 1 and substance2.substanceID not in substance1.currentColl and\
                   substance2.substanceID not in substance1.recentColl and substance1.substanceID not in\
                   substance2.recentColl and substance1.willReact == 0: #if this collision hasn't been checked +substance1 hasn't reacted (need to check both recentColls here)
                    dV = substance1.reactingPos - substance2.reactingPos #dV = distance vector
                    v1, v2 = substance1.velocity, substance2.velocity
                    m1, m2 = substance1.mass, substance2.mass
                    if np.linalg.norm(dV) == 0: #if normal line between substance is exactly 0 (i.e. a head-on collision)
                        substance1.velocity = (((m1 - m2) / (m1 + m2)) * (v1)) + (((2 * m2) / (m1 + m2)) * (v2))
                        substance2.velocity = (((m2 - m1) / (m1 + m2)) * (v2)) + (((2 * m1) / (m1 + m2)) * (v1))
                    else:
                        substance1.velocity = v1 - round(dV * (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, dV) / (np.linalg.norm(dV) ** 2), 3) #velocity result rounded to 3dp (currently gives a sufficient level of accuracy)
                        substance2.velocity = v2 - round(-dV * (2 * m1 / (m1 + m2)) * np.dot(v2 - v1, -dV) / (np.linalg.norm(dV) ** 2), 3)
                    relPos = substance2.pos[0] - substance1.pos[0]
                    if (substance1.velocity[0] == 0 and substance1.velocity[1] == 0) or (substance2.velocity[0] == 0 and substance2.velocity == 0):
                        print("Error" +str(v1)+" "+str(substance1.velocity)+" "+str(v2)+" "+str(substance2.velocity))
                    substance1.currentColl.append(substance2.substanceID)
                    substance2.currentColl.append(substance1.substanceID)
                    substance1.recentColl.append(substance2.substanceID)
                    substance2.recentColl.append(substance1.substanceID)
                    breakLoop = 1
        return substances, breakLoop

def substanceUpdates(substances, frameN, velocityDisplayScale): #updates the properties (position, angle, recentColl) of each substance in a given frame
    for substance1 in substances: #need to update positions after all collisions from previous positions have been checked
        substance1.pos[0] = substance1.reactingPos #sets new position on screen to previous reactingPos
        delta = pg.math.Vector2(substance1.velocity[0] * velocityDisplayScale, substance1.velocity[1] * velocityDisplayScale) #vector change in substance1's position on the screen, scaled appropriately by the magnitude of velocityDisplayScale
        for i in range(1, substance1.iterations+2):
            step = i / (substance1.iterations + 1)
            substance1.pos[i] = substance1.pos[0] + delta * step
        substance1.reactingPos = substance1.pos[substance1.iterations+1] #specifies the exact next position of the substance (defaults on current pos + velocity, but reactingPos gives a greater position accuracy for collisions then just using this value)
        if substance1.atomNumber > 1:
            substance1.image = pg.Surface((substance1.surfaceDimensions), pg.SRCALPHA) #clear image
            for i2 in range(0, substance1.atomNumber):
                angle = substance1.angleDisp[0] + (frameN - substance1.startingFrame + 1) * substance1.angleDisp[1] * math.pi / 6 #current angle = original angle on creation + (frame count since substance created * rotation direction * unit of rotation (pi/6 radians, from 12 total rotation positions)
                qx, qy = rotate(substance1.com, substance1.imageAtomCentres[i2], angle) #origin, point, angle
                pg.draw.circle(substance1.image, substance1.colour[i2], (qx, qy), substance1.atomRadii[i2]) #makes a circle of the given atom at qx, qy
                substance1.reactingAtomCentres[i2] = pg.math.Vector2(qx, qy) - substance1.com + substance1.reactingPos #stores the position of this atom's centre
        else:
            substance1.reactingAtomCentres = [substance1.reactingPos] #if one atom, just give the atom's centre
        substance1.currentColl = [] #substances colliding with substance1 during the current frame
        substance1.rect.center = pg.math.Vector2(round(substance1.pos[0][0]), round(substance1.pos[0][1])) #rounds pos to nearest integer for rect.centre
        substance1.recentCollPrevious.append(copy.deepcopy(substance1.recentColl))
        if len(substance1.recentCollPrevious) > 10:  #if there are more than 10 frames' worth of info in substance1.recentCollPrevious
            del substance1.recentCollPrevious[0] #delete the oldest
        substanceIDs = {s.substanceID for s in substances}
        for recentID in substance1.recentColl[:]:  #iterate over a copy to avoid modifying while looping
            if recentID not in substanceIDs:
                substance1.recentColl.remove(recentID) #removes substances from substance1.recentColl that no longer exist
    return substances

def frameProcessing(substances, toggleReaction, frameN, substanceID, atomInfo,reactionInfo, avgMr, rollingData, framerate, substanceGenInfo, going, velocityDisplayScale): #processes each frame
    reactionCount = 0
    currentVels = [np.linalg.norm(s.velocity) for s in substances]  #norm = magnitude of vector
    rmsVel = np.sqrt(np.mean(np.square(currentVels)))  #root mean square velocity
    T = avgMr * (rmsVel**2) / (2 * R * 1000)  # avgMr in g/mol, so dividing by 1000. #alculates temperature from 0.5*avgMr*(vrms)^2 = 1.5 * R * T (i.e. 0.5*m*v^2 = 3/2*kB*T)
    for substance1 in substances:
        if substances.has(substance1): #checks whether substance1 hasn't reacted during the for loop 
            reactionSuccessful, threeSubstanceReaction = 0, 0 #note: code checks for reactions occuring in this frame before checking for reactions occuring in the next frame
            breakLoop = 0
            for substance2 in substances:
                if substances.has(substance2):
                    substances, substanceID = reactionProcessing(substances, frameN, substanceID, atomInfo, substance1, substance2,reactionSuccessful, threeSubstanceReaction, velocityDisplayScale) #processes reactions
                    substances, reactionCount = reactionChecking(substances, frameN, substance1, substance2, toggleReaction,reactionInfo, T, reactionCount) #checks for reactions                    
                substances, breakLoop = collisionProcessing(substances, substance1, substance2, breakLoop) #checks for and processes non-reaction collisions
                if breakLoop == 1: #loop will be broken if a collision is detected - no need to try and detect any other collisions
                    break

            vxDisp = substance1.velocity[0] * velocityDisplayScale
            vyDisp = substance1.velocity[1] * velocityDisplayScale
            if substance1.rect.center[0] + vxDisp - substance1.radius < substance1.frameWidth and substance1.velocity[0] < 0: #collision with left wall
                substance1.velocity[0], substance1.recentColl = -substance1.velocity[0], [] #inverts x velocity, only if the collision with the left wall will start (during this frame), + clears recentColl
            elif substance1.rect.center[0] + vxDisp + substance1.radius > substance1.screenArea[0] and substance1.velocity[0] > 0: #collision with right wall
                substance1.velocity[0], substance1.recentColl = -substance1.velocity[0], []
            elif substance1.rect.center[1] + vyDisp - substance1.radius < substance1.frameWidth and substance1.velocity[1] < 0: #collision with top wall
                substance1.velocity[1], substance1.recentColl = -substance1.velocity[1], []
            elif substance1.rect.center[1] + vyDisp + substance1.radius > substance1.screenArea[1] and substance1.velocity[1] > 0: #collision with bottom wall
                substance1.velocity[1], substance1.recentColl = -substance1.velocity[1], []

            for substance2 in substances:
                if substance2.substanceID in substance1.recentColl: #if substance2 is the substance that substance1 has recently collided with
                    collDetected, collDetectedTotal = 0, 0
                    for i in range(0, substance1.iterations+2): #needs to start from 0 (contrasting to above) to check for current collisions
                        collDetected = collCheck(substance1, substance2, collDetected)
                        collDetectedTotal +=collDetected
                    if collDetectedTotal == 0 and substance1.recentColl != [substance1.productPair]:
                        substance1.recentColl.remove(substance2.substanceID) #removes substance2 from substance1.recentColl if no current collision detected and the two aren't a recently created pair

    substances = substanceUpdates(substances, frameN, velocityDisplayScale)
    nN2 = sum(s.substanceType == "N2" for s in substances)
    nO2 = sum(s.substanceType == "O2" for s in substances)
    rate = reactionCount
    conversion = sum(s.substanceType == "NO" for s in substances) * 100 / substanceGenInfo[2][1]
    rollingData.append({"N2": nN2, "O2": nO2, "k": rate/(nN2*nO2), "conv": conversion, "T": T})  #dict to store basic information about the current frame in rollingData
    if conversion == 20: #once 20% conversion is reached
        avgk = sum(d["k"] for d in rollingData) / (len(rollingData)/framerate) #calculates average k from each frame so far
        avgT = sum(d["T"] for d in rollingData) / len(rollingData) #calculates average temperature from each frame
        filename = "k_vs_temp.xlsx"
        going = False #ends simulation
        if os.path.exists(filename): #if file exists
            df = pd.read_excel(filename) #opens file, stores in df
        else:
            df = pd.DataFrame(columns=["Temperature (K)", "k (s^−1)", "conversion (%)"]) #makes new df 
        df.loc[len(df)] = [avgT, avgk, conversion] #appends to end of df
        df.to_excel(filename, index=False) #saves df as excel file
    return substanceID, rollingData, going

def avgMrFunc(substances): #gets the average Mr of the substances
    totalMass = 0
    totalMolecules = 0
    for substance in substances:
        atoms = substance.atomInfo  #all atoms in the substance
        molarMass = sum(atom[3] for atom in atoms)  # Ar is at position 3. Sums relative atomic mass for each atom in the substance.
        totalMass += molarMass
        totalMolecules += 1
    return totalMass / totalMolecules

def avgMrFromGenInfo(substanceGenInfo, atomInfo): #gets the average Mr from just substanceGenInfo rather than the list of generated substances (need to use this before substances have been generated)
    totalMass = 0
    totalMolecules = 0
    for substance in substanceGenInfo:
        formula = substance[0]
        count = substance[1]
        if count == 0:
            continue #skip this substance
        substanceParams = [[formula, 0]]
        substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
        # Sum atomic masses
        molarMass = sum(atom[3] for atom in substanceParams[1:])  # skip metadata [formula, atom count]. As above, sums relative atomic mass for each atom in the substance.
        totalMass += molarMass * count
        totalMolecules += count
    return totalMass / totalMolecules if totalMolecules > 0 else 0

def main(TStart):
    #display setup
    dimensions = 800 #dmns, 800 usual, 200 for small-scale testing
    frameWidth = 40 #fwidth, sets width of outer frame surrounding the reaction screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (150, 50) #sets position of window on monitor screen
    pg.init()
    screen = pg.display.set_mode((dimensions, dimensions)) #window dimensions
    outerBackground = pg.Surface(screen.get_size()) #10-pixel width outer background
    outerBackground.convert()
    outerBackground.fill((0, 0, 0)) #screen fill colour
    screen.blit(outerBackground, (0, 0)) #puts outerBackground on screen
    innerBackgroundSize = (screen.get_size()[0]-(2*frameWidth), screen.get_size()[1]-(2*frameWidth))
    innerBackground = pg.Surface(innerBackgroundSize)
    innerBackground.convert()
    innerBackground.fill((15, 15, 15)) #slightly lighter colour for inner backgroud fill
    screen.blit(innerBackground, (frameWidth, frameWidth))
    pg.display.flip()
    
    #main running code
    atomInfo = [["H", "white", 31, 1], ["N", "purple", 71, 14], ["O", "blue", 66, 16], ["F", "yellow", 64, 19]] #defines properties of atoms of an element (name, colour, covalent radius in pm, mass)
    for i3 in range(0, len(atomInfo)):
        atomInfo[i3][2] = atomInfo[i3][2] * 0.09 #using multiplier as a scale factor to make atom size (radius) on screen appropriate
    #substanceGenInfo (sgi): [[[substance name, number of molecules / atoms of the substance to generate], [info about atom components (name + number)]... for each atom in the substance], for each type of substance]
    substanceGenInfo = [["N", 0], ["O", 0], ["N2", 40], ["O2", 40], ["NO", 0], ["H2", 0], ["H2O", 0], ["NO2", 0]]
    avgMr = avgMrFromGenInfo(substanceGenInfo, atomInfo)
    velocityDisplayScale = 0.01 #scales velociities for display on the screen
    substances, substanceID = substanceSetup(dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale) #setup for first frame
    avgMr = avgMrFunc(substances)
    reactionInfo = [["N2+O2","NO+NO",100]] #reactants (should be in alphabetical order), products, K value
    frameN = 0 #counts elapsed frame numbers
    toggleReaction = 1 #enable chemical reaction
    framerate = 10 #timescale / frames per second (fps)
    rollingData = [] #stores output after each frame
    outputList = []
    reactionCount = [0]
    closePlot = 0
    pause = 0
    going = True
    clock = pg.time.Clock()
    while going:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                going = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
##                    df = pd.DataFrame(outputList)
                   # df.to_excel("substanceCounts.xlsx", index=False)
                    going = False   #will quit when escape key is pressed
                    closePlot = 1 #will close the plot automatically as well
                if event.key == pg.K_p:
                    pause = 1 #will pause when the p key is pressed
        while pause == 1:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        pause = 0 #will unpause when the p key is pressed again
        screen.blit(outerBackground, (0, 0))
        screen.blit(innerBackground, (frameWidth, frameWidth))
        clock.tick(framerate)
        substanceID, rollingData, going = frameProcessing(substances, toggleReaction, frameN, substanceID, atomInfo, reactionInfo, avgMr, rollingData, framerate, substanceGenInfo, going, velocityDisplayScale)
        substances.draw(screen) #draws substances on the screen
        pg.display.flip()
        substancesList = list(substances)
##        avgN2 = sum(d["N2"] for d in rollingData) / len(rollingData)
##        avgO2 = sum(d["O2"] for d in rollingData) / len(rollingData)
##        frameCounts = {
##            "frame": frameN,
##            "avg_N2": avgN2,
##            "avg_O2": avgO2,
##            "avg_k": avgRate}
##        outputList.append(frameCounts)
        frameN+=1
            

if __name__ == "__main__":
    temperatureOptions = [220, 240, 260, 280, 300, 320] #list of temperatures for simulation
    for trial in range(20):  #runs 20 times
        TStart = random.choice(temperatureOptions) #chooses random temperature from temperatureOptions
        print(f"Starting simulation at T = {TStart} K")
        main(TStart)
        time.sleep(1)
pg.quit()

#references:
#pygame examples: https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/documentation/pygame/pygame_collision_and_intesection.md
#example website: https://phet.colorado.edu/sims/html/gas-properties/latest/gas-properties_all.html
#collisions: https://stackoverflow.com/questions/29640685/how-do-i-detect-collision-in-pygame
#collisions 2: https://github.com/rafael-fuente/Ideal-Gas-Simulation-To-Verify-Maxwell-Boltzmann-distribution/blob/master/Ideal%20Gas%20simulation%20code.py
#angle: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
