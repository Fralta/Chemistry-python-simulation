import pygame as pg
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import re
import copy
import pandas as pd
from scipy.constants import R

def fileRead(): #reads the input file
    inputData = pd.read_excel("Inputs.xlsx", header=None)
    optionNames = inputData.iloc[2].dropna().values  #row 3 (0-indexed)
    optionValues = inputData.iloc[3].dropna().values  #row 4
    options = dict(zip(optionNames, optionValues))
    atomInfoRows = inputData.iloc[7:11, 1:]  #rows 8–11, ignoring column A
    atomInfo = atomInfoRows.transpose().dropna(how="all").values.tolist()
    reactionInfo = inputData.iloc[13:19, 1:].transpose().dropna(how="all").values.tolist()
    for r in reactionInfo:
        if r[2] == True:
            r[2] = 1 #fixing weird bug where pandas converts 1 in the K row to True
    substanceNames = inputData.iloc[21, 1:].dropna().values  #row 22
    substanceCounts = inputData.iloc[22, 1:].dropna().values  #row 23
    substancePlotDisplay = inputData.iloc[23, 1:].dropna().values  #row 24
    substanceLineColours = inputData.iloc[24, 1:].dropna().values
    substanceCentralAtom = inputData.iloc[25, 1:].dropna().values
    substanceGenInfo = [[name, count, display, colour, central] for name, count, display,\
                        colour, central in zip(substanceNames, substanceCounts, substancePlotDisplay,\
                                               substanceLineColours, substanceCentralAtom)]
    return options, atomInfo, reactionInfo, substanceGenInfo

def fileWrite(outputList): #unused
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
        # substanceParams = [[substanceType, atomNumber, x, y, xdir, ydir, substanceID, angleStartPos, rotationDirection, frameN, recentColl, frameWidth, productPair, velocityDisplayScale, centralAtom, iterations],
            #[name, colour, radius, mass], ...] #properties of (proposed) new atom / molecule
        pg.sprite.Sprite.__init__(self)  #functioning code
        screen = pg.display.get_surface() #get screen
        self.substanceType = substanceParams[0][0]  #F2, H2, N2, O2, Ne, NO..
        self.atomNumber = substanceParams[0][1] #number of atoms in substance
        self.velocity = pg.math.Vector2(substanceParams[0][4], substanceParams[0][5])
        self.substanceID = substanceParams[0][6]
        self.mass = 0
        self.atomMasses = []
        self.colour = []
        self.atomRadii = []
        self.atomInfo = []
        velocityDisplayScale = substanceParams[0][13]
        self.pos = [pg.math.Vector2(substanceParams[0][2], substanceParams[0][3])]
        self.radius = 0
        self.iterations = substanceParams[0][15] #iterations of substance position between current and next, to check for collisions in the intermediate positions between the current and next positions 
        for i in range(1, self.iterations + 2):  #pos[0] is current pos; pos[1], pos[2] etc. are the iterations
            self.pos.append(pg.math.Vector2(substanceParams[0][2] + i * substanceParams[0][4] * velocityDisplayScale / (self.iterations + 1), substanceParams[0][3] + i * substanceParams[0][5] * velocityDisplayScale / (self.iterations + 1)))
        self.reactingPos = self.pos[self.iterations+1] #specifies the exact next position of the substance (defaults on current pos + velocity, but reactingPos gives a greater position accuracy for collisions then just using this value)
        if self.atomNumber > 2:
            for i in range(len(substanceParams)-1): #for each atom
                if substanceParams[i+1][0] == substanceParams[0][14]:
                    substanceParams.insert(1, substanceParams.pop(i+1)) #moves central atom to index 1 in substanceParams
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
            self.surfaceDimensions = pg.math.Vector2(self.radius * 10, self.radius * 10) #larger surfaceDImensions to account for rotation
            self.image = pg.Surface((self.surfaceDimensions), pg.SRCALPHA)
            self.com = pg.math.Vector2(self.surfaceDimensions / 2) #rotation around 'centre of mass'
            if self.atomNumber == 2:
                l1 = (self.atomMasses[1] * (self.atomRadii[0] + self.atomRadii[1])) / (self.atomMasses[0] + self.atomMasses[1]) #l1 = r1 when m1r1 = m2r2, along the C∞ axis r1+r2 
                self.imageAtomCentres = [self.com - (pg.math.Vector2(0, l1)) * 0.6, self.com + (pg.math.Vector2(0, self.atomRadii[0]
                                                                                + self.atomRadii[1] - l1)* 0.6)] #at 0º angle (no rotation), imageAtomCentres[0] is vertically above imageAtomCentres[1], 0.6 used to adjust atom separation to be more visually realistic
            else: #3+ atoms
                self.imageAtomCentres = [self.com - (pg.math.Vector2(0, 30/9))] #defines position of central atom
                for i in range(len(substanceParams)-2): #for the remaining atoms
                    bondLength = (0.8 * (substanceParams[1][2] + substanceParams[i+2][2])) #0.8 scale factor
                    if self.atomNumber == 3:
                        bondAngleRad = np.radians(60)
                        dx = bondLength * np.cos(bondAngleRad)
                        dy = bondLength * np.sin(bondAngleRad)
                        self.imageAtomCentres.append(self.imageAtomCentres[0] + pg.math.Vector2(dx*((-1)**i), -dy))
                    else: #>3 atoms
                        if self.atomNumber == 4:
                            angleDeg = i * 120
                        if self.atomNumber == 5:
                            angleDeg = i * 90
                        bondAngleRad = np.radians(angleDeg)
                        dx = bondLength * np.cos(bondAngleRad)
                        dy = bondLength * np.sin(bondAngleRad)
                        self.imageAtomCentres.append(self.imageAtomCentres[0] + pg.math.Vector2(dx, dy))
            for i2 in range(0, self.atomNumber):
                qx, qy = rotate(self.com, self.imageAtomCentres[i2], startingAngle) #origin, point, angle
                self.reactingAtomCentres.append(pg.math.Vector2(qx, qy) - self.com + self.reactingPos)
            self.currentAngle = startingAngle #clockwise angle of rotation of the molecule, where 0º is directly upwards
            #trueCom = sum((pos * mass for pos, mass in zip(self.imageAtomCentres, self.atomMasses)),start=pg.math.Vector2(0, 0)) / self.mass #can use this
            #offset = trueCom - self.com
            #self.com = trueCom 
            #self.imageAtomCentres = [pos - offset for pos in self.imageAtomCentres]
        self.rect = self.image.get_rect(center = (substanceParams[0][2], substanceParams[0][3]))
        self.willReact = 0 #once a reaction is detected, this is assigned to the substanceID of the substance(s) that this substance will react with, or just "uni" if it will undergo a unimolecular reaction
        self.currentColl = [] #only used to check current colls between non-reacting atoms (not used for reactions)
        self.recentColl = substanceParams[0][10]
        self.recentCollPrevious = [] #stores recentColl for the most recent 10 frames
        self.frameWidth = substanceParams[0][11]
        self.screenArea = (screen.get_rect()[2] - self.frameWidth, screen.get_rect()[3] - self.frameWidth)
        self.startingFrame = substanceParams[0][9] #frame when rotation begins
        self.product = ""
        self.reactingFrame = 0
        self.productPair = substanceParams[0][12] #substanceID of other molecule created during rxn

def collCheck(substance1, substance2, collDetected): #sub-function that does atom-atom collision checking
##    for i2 in range(0, substance1.atomNumber):  #this checks every atom with each other. More accurate but much slower.
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
    return substanceParams

def substanceSetup(dimensions, atomInfo, frameWidth, substanceGenInfo,TStart,avgMr,velocityDisplayScale, iterations):
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
                substanceParams = [[substanceGenInfo[i][0], 0, x, y, xdir, ydir, substanceID, "na", "na", 0, [], frameWidth, 0, velocityDisplayScale, substanceGenInfo[i][4], iterations]] #properties of (proposed) new atom / molecule
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

def reactionProcessing(substances, frameN, substanceID, atomInfo, substance1, substance2, reactionSuccessful, threeSubstanceReaction, velocityDisplayScale, substanceGenInfo, iterations):
    #print("substance2 "+str(substance2.substanceID)+" willreact "+str(substance2.willReact)+" frame "+str(frameN))
    for substance3 in substances:
        if substances.has(substance3):
            if substance3.willReact == [substance1.substanceID,substance2.substanceID] and frameN == substance3.reactingFrame: #if three substance reaction was successful
                newx, newy = (substance1.pos[0][0]+substance2.pos[0][0]+substance3.pos[0][0])/3, (substance1.pos[0][1]+substance2.pos[0][1]+substance3.pos[0][1])/3
                totalmv = (substance1.mass * substance1.velocity) + (substance2.mass * substance2.velocity) + (substance3.mass * substance3.velocity)
                totalMass = substance1.mass + substance2.mass + substance3.mass
                reactionCentre = (substance1.mass * np.array(substance1.pos[0]) + substance2.mass * np.array(substance2.pos[0]) + substance3.mass * np.array(substance3.pos[0])) / totalMass
                reactionSuccessful = 1
                threeSubstanceReaction = 1
                thirdSubstance = substance3.substanceID #assigns substanceID of third substance for later

            
        if substance2.willReact == substance1.substanceID and substance2.reactingFrame == frameN: #if two substance reaction was successful
            newx, newy = (substance1.pos[0][0]+substance2.pos[0][0])/2, (substance1.pos[0][1]+substance2.pos[0][1])/2
            totalmv = (substance1.mass * substance1.velocity) + (substance2.mass * substance2.velocity)
            totalMass = substance1.mass + substance2.mass
            reactionCentre = (substance1.mass * np.array(substance1.pos[0]) + substance2.mass * np.array(substance2.pos[0])) / totalMass
            reactionSuccessful = 1

        if reactionSuccessful == 1:
            productNumber = 0
            products = [s.strip() for s in substance1.product.split('+') if s.strip()]
            centralAtom = []
            for product in products:
                centralAtom.append(next((row[4] for row in substanceGenInfo if row[0] == product), "-")) #finds central atom if present
            try:
                productNumber = int(substance1.product[0]) #number of product molecules
            except:
                productNumber = len(products)
            newVelocity = totalmv / (productNumber * totalMass)
            newSubstanceRecentColl = []
            angleStartArgs = [[substance1.pos[0], substance1.velocity], [substance2.pos[0], substance2.velocity]]
            angleStartPos, rotationDirection = angleStartFunc(angleStartArgs)
            if productNumber == 1:
                substanceParams = [[substance1.product, 0, newx, newy, newVelocity[0], newVelocity[1], substanceID, angleStartPos, rotationDirection, frameN,\
                                    [], substance1.frameWidth, 0, velocityDisplayScale, centralAtom[0], iterations]] #new substance properties
                substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
                substanceNew = substanceGen(substanceParams)
                substances.add(substanceNew)
                substanceID+=1
            if productNumber > 1:
                vels = []
                vCom = totalmv / totalMass  #center-of-mass velocity
                if np.linalg.norm(vCom) == 0:#handle rare case where total momentum is zero — pick random direction
                    perp = np.array([1.0, 0.0])
                else:
                    perp = np.array([-vCom[1], vCom[0]])  #perpendicular direction (rotate vCom by 90°)
                    perp = perp / np.linalg.norm(perp)  #make it unit vector
                if threeSubstanceReaction != 1:
               # v1 = np.array([substance1.velocity[0], substance1.velocity[1]])
               # v2 = np.array([substance2.velocity[0], substance2.velocity[1]])
                    vRel = substance1.velocity - substance2.velocity #relative velocity of substance1 and substance2
                else:
                    vRel = substance1.velocity + substance2.velocity + substance3.velocity - (3 * vCom) #vector sum of velocities relative to COM                     
                speedRel = np.linalg.norm(vRel)
                vels.append(vCom + 0.5 * speedRel * perp)   #assign velocities
                vels.append(vCom - 0.5 * speedRel * perp)
                if productNumber == 2:
                    substanceVectorSeparation = substance1.pos[0] - substance2.pos[0]
                    newSubstancesSeparation, newSubstanceAtomNumber = [substanceVectorSeparation[1]/2, substanceVectorSeparation[0]/2], 2 #separates the two new substances horizontally by this amount (see substanceParams[0][2] below)
                    productPositions = [[newx-newSubstancesSeparation[0],newy-newSubstancesSeparation[0]], [newx+newSubstancesSeparation[0],newy+newSubstancesSeparation[0]]]
                if productNumber == 3:
                    triangleRadius = 20  #distance from center to each product
                    triangleOffsets = [pg.math.Vector2(triangleRadius, 0), pg.math.Vector2(triangleRadius * math.cos(2 * math.pi / 3), triangleRadius * math.sin(2 * math.pi / 3)),\
                                       pg.math.Vector2(triangleRadius * math.cos(4 * math.pi / 3), triangleRadius * math.sin(4 * math.pi / 3)),]
                    productPositions = [reactionCentre + offset for offset in triangleOffsets]
                    sepSpeed = 0.33 * np.linalg.norm(vRel)  #tweak factor as needed
                    for offset in triangleOffsets:
                        offsetArray = np.array([offset.x, offset.y], dtype=float)
                        if np.linalg.norm(offsetArray) == 0:
                            directionVector = np.array([1.0, 0.0])  #fallback
                        else:
                            directionVector = offsetArray / np.linalg.norm(offsetArray)
                        velocityRel = sepSpeed * directionVector
                        velocityActual = velocityRel + vCom  #add COM velocity to get lab-frame
                        vels.append(velocityActual)
                substanceParams = [[products[0], 0, productPositions[0][0], productPositions[0][1], 0, 0, substanceID, angleStartPos, rotationDirection,\
                                    frameN, [substanceID+1], substance1.frameWidth, substanceID+1, velocityDisplayScale, centralAtom[0], iterations]]
                substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
                for i in range(0, substance1.product.count("+")+1): #for each new substance
                    substanceParams[0][2] = productPositions[i][0]
                    substanceParams[0][3] = productPositions[i][1]
                    substanceParams[0][4]=vels[i][0]
                    substanceParams[0][5]=vels[i][1]
                    if i > 0:
                        substanceParams[0][0] = products[i]
                        substanceParams[0][6] = substanceID
                        substanceParams[0][8] = rotationDirection * -1
                        substanceParams[0][10] = [substanceID-1]
                        substanceParams[0][12] = [substanceID-1]
                        substanceParams[0][14] = centralAtom[i]
                        substanceParams = getSubstanceAtoms(substanceParams[:1], atomInfo)
                    if substance1.product.count("+") == 2: #three products
                        allSubstanceIDs = [substanceID, substanceID+1, substanceID+2]
                        del allSubstanceIDs[i] #remove self from allSubstanceIDs
                        substanceParams[0][10] = allSubstanceIDs
                        substanceParams[0][12] = allSubstanceIDs
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

def probabilityChecking(substances, reactionInfo, T, EaMethod):
    reactantList = sorted([str(s.substanceType) for s in substances])
    reactants = "+".join(reactantList)
    for r in reactionInfo:
        if str(r[0]) == reactants or str(r[1]) == reactants:
                deltaG = -R * T * np.log(r[2])  #in J/mol
                EaF = r[3] #artificially very low (could also use a very high temp), needed for good collision rate
                EaR = EaF - deltaG
                Ea = EaF if str(r[0]) == reactants else EaR
                if EaMethod == "Prob":
                    p = np.exp(-Ea / (R * T)) #probability of the reaction
                    return random.random() < p  #only return if reaction match is found, and if the probability outcome is successful
                else: #relative Ek approach
                    v1 = np.array([substances[0].velocity[0], substances[0].velocity[1]])
                    v2 = np.array([substances[1].velocity[0], substances[1].velocity[1]])
                    m1 = (substances[0].mass / 1000) / 6.022e23  # kg mass
                    m2 = (substances[1].mass / 1000) / 6.022e23  # kg
                    if len(substances) == 2: #two substance reaction
                        vRel = v1 - v2 #relative velocity
                        mu = (m1 * m2) / (m1 + m2) #reduced mass
                        vRelMagSquared = np.dot(vRel, vRel)
                        ERelMolecule = 0.5 * mu * vRelMagSquared  #relative kinetic energy (J)
                    if len(substances) == 3: #three substance reaction. Apparent approach by ChatGPT, but makes intuitive sense.
                        v3 = np.array([substances[2].velocity[0], substances[2].velocity[1]])
                        m3 = (substances[2].mass / 1000) / 6.022e23
                        vCOM = ((m1*v1) + (m2*v2) + (m3*v3)) / (m1 + m2 + m3) #centre of mass velocity
                        vRel1, vRel2, vRel3 = np.linalg.norm(v1 - vCOM), np.linalg.norm(v2 - vCOM), np.linalg.norm(v3 - vCOM), 
                        ERelMolecule = (0.5 * m1 * (vRel1 ** 2)) + (0.5 * m2 * (vRel2 ** 2)) + (0.5 * m3 * (vRel3 ** 2))
                    ERel = ERelMolecule * (6.022e23) #converting to J mol-1
                    return Ea < ERel #reaction allowed according to probability
    return False  #no valid reaction found

def reactionChecking(substances, frameN, substance1, substance2, toggleReaction,reactionInfo, T, reactionCount, timeInterval, EaMethod): #checks for reactions with >1 molecule
    if substance1 != substance2:  
            collDetected = 0
            for i in range(1, substance1.iterations+2):
                collDetected = collCheck(substance1, substance2, collDetected)
                if collDetected == 1 and toggleReaction == True and substance1.willReact == 0 and substance2.willReact == 0 and substance2.substanceID not in substance1.recentColl\
                   and substance1.substanceID not in substance2.recentColl:
                        nextReactionAllowed = 0 #valid reaction found, but not yet successful (needs to pass probability check)
                        reactantList = sorted([substance1.substanceType, substance2.substanceType])
                        reactants = "+".join(reactantList)
                        for r in reactionInfo:
                            if r[0] == reactants or r[1] == reactants:
                                if r[0] == reactants: #determine direction
                                    productString = r[1]
                                else:
                                    productString = r[0]
                                substance1.product = substance2.product = productString #set products
                                nextReactionAllowed = 1
                                break
                        if nextReactionAllowed == 1:
                            reactionSuccessful2 = probabilityChecking([substance1, substance2], reactionInfo, T, EaMethod)
                            if reactionSuccessful2 == True:
                                substance1.willReact = substance2.substanceID
                                substance2.willReact = substance1.substanceID
                                substance2.reactingFrame = frameN + 1
                                reactionCount+=1
                            break #breaks the iterations loop (so that the probability check only occurs once)
                                
                        for substance3 in substances: #checking for three substance reactions
                            if substance3 != substance1 and substance3 != substance2:
                                collDetected, collDetected2 = 0, 0
                                collDetected = collCheck(substance1, substance3, collDetected)     #
                                collDetected2 = collCheck(substance2, substance3, collDetected)   #checking substance 3 colliding with either substance1 or substance2
                                if (collDetected == 1 or collDetected2 == 1) and substance3.willReact == 0:
                                    reactantList = sorted([substance1.substanceType, substance2.substanceType, substance3.substanceType])
                                    reactants = "+".join(reactantList)
                                    for r in reactionInfo:
                                        if r[0] == reactants or r[1] == reactants:
                                            if r[0] == reactants: #determine direction
                                                productString = r[1]
                                            else:
                                                productString = r[0]
                                            substance1.product = substance2.product = substance3.product = productString #set products
                                            nextReactionAllowed = 1
                                            break
                                    if nextReactionAllowed == 1:
                                        reactionSuccessful2 = probabilityChecking([substance1, substance2, substance3], reactionInfo, T, EaMethod)
                                        if reactionSuccessful2 == True:
                                            substance1.willReact = [substance2.substanceID,substance3.substanceID]
                                            substance2.willReact = [substance1.substanceID,substance3.substanceID]
                                            substance3.willReact = [substance2.substanceID,substance1.substanceID]
                                            substance3.reactingFrame = frameN + 1 #specifies that reaction will take place in next frame
                                            reactionCount+=1
                                        break
    return substances, reactionCount

def collisionProcessing(substances, substance1, substance2, breakLoop):
        if substance1 != substance2 and substances.has(substance2):
            collDetected = 0
            for i in range(1, substance1.iterations+2):
                collDetected = collCheck(substance1, substance2, collDetected) 
                if collDetected == 1 and substance2.substanceID not in substance1.currentColl and\
                   substance2.substanceID not in substance1.recentColl and substance1.substanceID not in\
                   substance2.recentColl and substance1.willReact == 0: #if this collision hasn't been checked, and substance1 hasn't reacted (need to check both recentColls here)
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

def unimolecularReactionProcessing(substances, frameN, substanceID, atomInfo, substance1,reactionSuccessful, velocityDisplayScale, substanceGenInfo, iterations):
    if substance1.willReact == "uni":
        productNumber = 0
        products = [s.strip() for s in substance1.product.split('+') if s.strip()] #gets products from substance1.product
        centralAtom = []
        for product in products:
            centralAtom.append(next((row[4] for row in substanceGenInfo if row[0] == product), "-")) #finds central atom if present
        try:
            productNumber = int(substance1.product[0]) #number of product molecules
        except:
            productNumber = len(products)
        newSubstanceRecentColl = []
        vels = [] #product velocities
        if productNumber > 1:
            theta = np.random.uniform(0, 2*np.pi) #random angle
            randomDirection = np.array([np.cos(theta),np.sin(theta)])
            if productNumber == 2: #split momentum between products
                vels.append(substance1.velocity + (randomDirection * np.linalg.norm(substance1.velocity) * 0.5))
                vels.append(substance1.velocity - (randomDirection * np.linalg.norm(substance1.velocity) * 0.5))
                productPositions = [substance1.pos[0] - randomDirection * 10, substance1.pos[0] + randomDirection * 10] #offset positions a bit
            elif productNumber == 3:
                triangleRadius = 20
                triangleOffsets = [pg.math.Vector2(triangleRadius, 0), pg.math.Vector2(triangleRadius * math.cos(2 * math.pi / 3),
                                    triangleRadius * math.sin(2 * math.pi / 3)), pg.math.Vector2(triangleRadius * math.cos(4 * math.pi / 3),
                                    triangleRadius * math.sin(4 * math.pi / 3)),] #offsets position of each molecule from each other, each forming the corners of a triangle
                productPositions = [substance1.pos[0] + offset for offset in triangleOffsets] #gets absolute positions from these offsets (using the reactant's position as the centre of the triangle)
                separationSpeed = np.linalg.norm(substance1.velocity) * 0.33 #adjustment factor
                for offset in triangleOffsets:
                    offsetArray = np.array([offset.x, offset.y], dtype=float)
                    if np.linalg.norm(offsetArray) == 0:
                        directionVector = np.array([1.0, 0.0]) #fallback
                    else:
                        directionVector = offsetArray / np.linalg.norm(offsetArray)
                    velocityRel = separationSpeed * directionVector #direction of velocity relative to the centre of mass - same direction as position, therefore the products move out from the center of the triangle
                    velocityActual = velocityRel + substance1.velocity #absolute velocity (on screen), when centre of mass is considered
                    vels.append(velocityActual)
            for i in range(productNumber): #for each new substance
                substanceParams = [[products[i], 0, productPositions[i][0], productPositions[i][1], vels[i][0],
                                    vels[i][1], substanceID, substance1.angleDisp[0] * 6 / math.pi, substance1.angleDisp[1], frameN, [substanceID+1], substance1.frameWidth, substanceID+1,
                                    velocityDisplayScale, centralAtom[i], iterations]]
                if i != 0:
                    substanceParams[0][12]-=2 #gets substanceID of other product(s)
                    substanceParams[0][10][0]-=2 #gets substanceID of other product(s)
                substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
                substanceNew = substanceGen(substanceParams)
                substances.add(substanceNew)
                substanceID += 1
        substances.remove(substance1)
    return substances, substanceID

def unimolecularReactionChecking(reactionInfo, substance1, timeInterval, reactionCount, toggleReaction, frameN):
    for r in reactionInfo:
        if (r[0] == substance1.substanceType or r[1] == substance1.substanceType) and r[4] != "-":   #if substance1 is a reactant for a unimolecular reaction (r[4] is a k value)
            if r[0] == substance1.substanceType: #determine direction
                productString = r[1]
                k = float(r[4])
            else:
                productString = r[0]
                k = float(r[4]) / float(r[2]) #kBackwards = kForwards / K
            p = 1 - np.exp(-k * timeInterval) #probability of successful reaction
            if random.random() < p and toggleReaction == True and substance1.willReact == 0: #random.random() < p evaluates the probability
                substance1.willReact = "uni" #will undergo a unimolecular decomposition reaction
                substance1.reactingFrame = frameN + 1 #will react in the next frame
                reactionCount+=1
                substance1.product = productString
                break
    return reactionCount

def substanceUpdates(substances, frameN, velocityDisplayScale): #updates the properties (position, angle, recentColl) of each substance in a given frame
    for substance1 in substances: #need to update positions after all collisions from previous positions have been checked
        substance1.pos[0] = substance1.reactingPos #sets new position on screen to previous reactingPos
        delta = pg.math.Vector2(substance1.velocity[0] * velocityDisplayScale, substance1.velocity[1] * velocityDisplayScale) #vector change in substance1's position on the screen, scaled appropriately by the magnitude of velocityDisplayScale
        for i in range(1, substance1.iterations+2): #for each iteration
            step = i / (substance1.iterations + 1) #fraction of total distance to next position
            substance1.pos[i] = substance1.pos[0] + delta * step #sets position of this iteration
        substance1.reactingPos = substance1.pos[substance1.iterations+1] #specifies the exact next position of the substance (defaults on [current pos + velocity], but reactingPos gives a greater position accuracy for collisions then just using this value)
        if substance1.atomNumber > 1:
            substance1.image = pg.Surface((substance1.surfaceDimensions), pg.SRCALPHA) #clear image
            angle = substance1.angleDisp[0] + (frameN - substance1.startingFrame + 1) * substance1.angleDisp[1] * math.pi / 6 #current angle = original angle on creation + (frame count since substance created * rotation direction * unit of rotation (pi/6 radians, from 12 total rotation positions)
            substance1.currentAngle = angle
            for i2 in range(0, substance1.atomNumber):
                qx, qy = rotate(substance1.com, substance1.imageAtomCentres[i2], angle) #origin, point, angle
                pg.draw.circle(substance1.image, substance1.colour[i2], (qx, qy), substance1.atomRadii[i2]) #makes a circle of the given atom at qx, qy
                substance1.reactingAtomCentres[i2] = pg.math.Vector2(qx, qy) - substance1.com + substance1.reactingPos #stores the position of this atom's centre
        else:
            substance1.reactingAtomCentres = [substance1.reactingPos] #if one atom, just give the atom's centre
        substance1.currentColl = [] #substances colliding with substance1 during the current frame
        substance1.rect.center = pg.math.Vector2(round(substance1.pos[0][0]), round(substance1.pos[0][1])) #rounds pos to nearest integer for rect.center
        substance1.recentCollPrevious.append(copy.deepcopy(substance1.recentColl))
        if len(substance1.recentCollPrevious) > 10: #if there are more than 10 frames' worth of info in substance1.recentCollPrevious
            del substance1.recentCollPrevious[0] #delete the oldest
        substanceIDs = {s.substanceID for s in substances}
        for recentID in substance1.recentColl[:]:  #iterate over a copy to avoid modifying while looping
            if recentID not in substanceIDs:
                substance1.recentColl.remove(recentID) #removes substances from substance1.recentColl that no longer exist
    return substances

def frameProcessing(substances, toggleReaction, frameN, substanceID, atomInfo,reactionInfo, avgMr, rollingData, framerate, substanceGenInfo, going, velocityDisplayScale,\
                    fig, ax, substanceLines, substanceEnabled, showPlot, timeInterval, EaMethod, iterations): #processes each frame
    currentVels = [0]
    reactionCount = 0
    for substance1 in substances:
            currentVels.append(round(np.linalg.norm(substance1.velocity), 2))  #norm = magnitude of vector   
    rmsVel = np.sqrt(np.mean(np.square(currentVels))) #root mean square velocity
    T = avgMr * (rmsVel**2) / (2 * R * 1000) # avgMr in g/mol, so dividing by 1000. calculates temperature from 0.5*avgMr*(vrms)^2 = 1.5 * R * T (i.e. 0.5*m*v^2 = 3/2*kB*T)
    for substance1 in substances:
        if substances.has(substance1): #checks whether substance1 hasn't reacted during the for loop 
            reactionSuccessful, threeSubstanceReaction = 0, 0 #note: code processes reactions occuring in this frame before checking for reactions that will occur in the next frame
            breakLoop = 0
            for substance2 in substances:
                if substances.has(substance2):
                    substances, substanceID = reactionProcessing(substances, frameN, substanceID, atomInfo, substance1, substance2,reactionSuccessful, threeSubstanceReaction, velocityDisplayScale, substanceGenInfo, iterations) #processes reactions from previous frame
                    substances, reactionCount = reactionChecking(substances, frameN, substance1, substance2, toggleReaction,reactionInfo, T, reactionCount, timeInterval, EaMethod) #checks for reactions with >1 molecule                    
                substances, breakLoop = collisionProcessing(substances, substance1, substance2, breakLoop) #checks for and processes non-reaction collisions
                if breakLoop == 1: #loop will be broken if a collision is detected - no need to try and detect any other collisions
                    break
            substances, substanceID = unimolecularReactionProcessing(substances, frameN, substanceID, atomInfo, substance1,reactionSuccessful, velocityDisplayScale, substanceGenInfo, iterations) #as above, processes unimolecular reactions before checking for them
            reactionCount = unimolecularReactionChecking(reactionInfo, substance1, timeInterval, reactionCount, toggleReaction, frameN)

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
                    if collDetectedTotal == 0 and substance1.recentColl != substance1.productPair:
                        substance1.recentColl.remove(substance2.substanceID) #removes substance2 from substance1.recentColl if no current collision detected and the two aren't a recently created pair

    substances = substanceUpdates(substances, frameN, velocityDisplayScale)
    frameEntry = {"Frame": int(frameN), "T (K)": T, "Time (s)": frameN/framerate} #dict to store basic information about the current frame in rollingData
    for i in substanceGenInfo: #for all substances
        substanceName = i[0]
        count = sum(s.substanceType == substanceName for s in substances)
        frameEntry[substanceName] = count
    rollingData.append(frameEntry)

    if showPlot:
        times = [entry["Time (s)"] for entry in rollingData]
        for name, line in substanceLines.items():
            if substanceEnabled.get(name, False): #if the substance is enabled
                line.set_data(times, [entry[name] for entry in rollingData])
                line.set_visible(True)
            else:
                line.set_data([], [])
                line.set_visible(False)
        allCounts = [entry[name] for entry in rollingData for name in entry if name not in ["Frame", "T (K)", "Time (s)"]] #counts of each substance
        ymax = max(allCounts) + 12 #pad ymax by 12
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) #forces y axis to be in integer values
        ax.relim() #recomputes limits
        ax.autoscale_view()
        if times:
            ax.set_xlim(min(times), max(times))
        plt.pause(0.001) #gives time for the window to update without freezing the simulation. The value (0.001–0.1) controls the refresh rate.
##    if conversion > 20:
##        avgk = sum(d["k"] for d in rollingData) / (len(rollingData)/framerate)
##        avgT = sum(d["T"] for d in rollingData) / len(rollingData)
##        filename = "k_vs_temp.xlsx"
##        going = False
##        if os.path.exists(filename):
##            df = pd.read_excel(filename)
##        else:
##            df = pd.DataFrame(columns=["Temperature (K)", "k (s^−1)", "conversion (%)"])
##        df.loc[len(df)] = [avgT, avgk, conversion]
##        df.to_excel(filename, index=False)
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
        molarMass = sum(atom[3] for atom in substanceParams[1:])  #skip metadata [formula, atom count]. As above, sums relative atomic mass for each atom in the substance.
        totalMass += molarMass * count
        totalMolecules += count
    return totalMass / totalMolecules if totalMolecules > 0 else 0

def main():
    #display setup
    options, atomInfo, reactionInfo, substanceGenInfo = fileRead()
    dimensions = options.get("Dimensions (pixels)")
    frameWidth =options.get("Width of outer frame (pixels)")
    windowPos = options.get("Window position")
    x, y = map(int, windowPos.replace(" ", "").split(",")) #replace() used to remove space in windowPos
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    pg.init() #initiates pygame
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
    for i3 in range(0, len(atomInfo)):
        atomInfo[i3][2] = atomInfo[i3][2] * options.get("Atom size scale factor") #scales up atom sizes as per scale factor
    avgMr = avgMrFromGenInfo(substanceGenInfo, atomInfo)
    TStart = options.get("Starting temperature (K)")
    velocityDisplayScale = options.get("Velocity display scale factor")
    iterations = options.get("Iterations")
    substances, substanceID = substanceSetup(dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale, iterations) #setup for first frame
    avgMr = avgMrFunc(substances)
    EaMethod = options.get("Ea method")
    frameN = 0 #counts elapsed frame numbers
    toggleReaction = options.get("Toggle reactions")
    framerate = options.get("Framerate")
    timeInterval = 1 / framerate #time interval between frames
    showPlot = options.get("Show plot")
    rollingData = [] #stores output after each frame
    outputList = []
    reactionCount = [0]
    closePlot = 0
    pause = 0
    going = True
    if showPlot:
        plt.ion()#turn on interactive plots
        handles = []
        labels = []
        fig, ax = plt.subplots()
        colors = {entry[0]: entry[3] for entry in substanceGenInfo}
        substanceEnabled = {entry[0]: entry[2] for entry in substanceGenInfo}
        substanceNames = [entry[0] for entry in substanceGenInfo]
        substanceLines = {name: ax.plot([], [], label=name, color=colors[name])[0] for name in substanceNames} #graph line for each substance
        N2_eq, O2_eq, NO_eq= 34.15,39.15,51.71 #equilibrium values, to display as below
        #ax.axhline(N2_eq, linestyle='--', color='purple', alpha=0.6, label="N₂ eq")
        #ax.axhline(O2_eq, linestyle='--', color='blue', alpha=0.6, label="O₂ eq")
        #ax.axhline(NO_eq, linestyle='--', color='green', alpha=0.6, label="NO eq")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Molecule count")
        ax.set_title("Live molecule counts")
        for name, count, display, colour, enabled in substanceGenInfo:
            if str(display).strip().lower() in ["yes", "true", "1"]:
                line, = ax.plot(x, y, label=name, color=colour)
                handles.append(line)
                labels.append(name.replace("2", "₂").replace("3", "₃").replace("4", "₄")) #makes subscripts as appropriate
        ax.legend(handles, labels, loc = "upper left") #legend placed in upper left
        ax.grid(True)
        #time.sleep(10)
    else:
        fig, ax, substanceLines, substanceEnabled = 0, 0, 0, 0

    clock = pg.time.Clock()
    while going:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                going = False
            if event.type == pg.KEYDOWN: #if a button is pressed
                if event.key == pg.K_ESCAPE: #if escape key pressed
                    df = pd.DataFrame(rollingData) #stores rollingData in a dataframe, df
                    df.to_excel("substanceCounts.xlsx", index=False)
                    going = False   #quits simulation
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
        substanceID, rollingData, going = frameProcessing(substances, toggleReaction, frameN, substanceID, atomInfo, reactionInfo, avgMr, rollingData, framerate, substanceGenInfo, going, velocityDisplayScale, fig, ax,\
                                                          substanceLines, substanceEnabled, showPlot, timeInterval, EaMethod, iterations) #processes current frame
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
    main()
pg.quit()
t = 1


#references:
#pygame examples: https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/documentation/pygame/pygame_collision_and_intesection.md
#example website: https://phet.colorado.edu/sims/html/gas-properties/latest/gas-properties_all.html
#collisions: https://stackoverflow.com/questions/29640685/how-do-i-detect-collision-in-pygame
#collisions 2: https://github.com/rafael-fuente/Ideal-Gas-Simulation-To-Verify-Maxwell-Boltzmann-distribution/blob/master/Ideal%20Gas%20simulation%20code.py
#angle: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
