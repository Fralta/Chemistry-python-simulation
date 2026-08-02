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
from collections import defaultdict

def fileRead(): #reads the input file
    inputData = pd.read_excel("Inputs.xlsx", header=None)
    optionNames = inputData.iloc[2].dropna().values  #row 3 (0-indexed)
    optionValues = inputData.iloc[3].dropna().values  #row 4
    options = dict(zip(optionNames, optionValues))
    atomInfoRows = inputData.iloc[7:13, 1:]  #rows 8–13, ignoring column A
    atomInfoDF = atomInfoRows.transpose().dropna(how="all")
    atomInfoDF.columns = ["atom", "colour", "radius", "Ar", "outerE", "valence"]
    #atomInfoDF["oxidationStates"] = atomInfoDF["oxidationStates"].apply(lambda x: [int(i) for i in x.split(", ")])
    atomInfo = atomInfoDF.set_index("atom").to_dict(orient="index") #converts each row of atomInfoDF into a keyed dict
    reactionInfoDF = inputData.iloc[15:21, 1:].transpose().dropna(how="all")#.values.tolist()
    reactionInfoDF.columns = ["reactants", "products", "K", "Ea", "k", "study for second"]
    reactionInfo = reactionInfoDF.to_dict(orient="index")
    substanceFormulae = inputData.iloc[23, 1:].dropna().values  #row 24
    substanceCounts = inputData.iloc[24, 1:].dropna().values  #row 25
    substancePlotDisplay = inputData.iloc[25, 1:].dropna().values  #row 26
    substanceLineColours = inputData.iloc[26, 1:].dropna().values
    substanceCentralAtom = inputData.iloc[27, 1:].dropna().values
    substanceGenInfo = [{"formula": formula, "count": count, "display": display, "colour": colour, "central": central}
        for formula, count, display, colour, central in zip(substanceFormulae, substanceCounts, substancePlotDisplay,
                                                         substanceLineColours,substanceCentralAtom)]
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

def bondingToCentralAtom(LAI, bonds, atom, maxBondOrder):
    found = False
    for bond in bonds:
        if bond["firstAtom"] == 0 and bond["secondAtom"] == atom["ID"]: #if the respective bond is found
            bond["bondOrder"] = maxBondOrder #increases bond order
            found = True
            break
    if not found:
        bonds.append({"firstAtom": LAI[0]["ID"], "secondAtom": atom["ID"],
        "bondOrder": maxBondOrder}) #forms the bond if not already formed
        LAI[0]["ePairInfo"][0]+=1
    LAI[0]["remainingValence"]-=1
    atom["remainingValence"]-=1
    return LAI, bonds

def bondMatchesAtom(bond, atom): #matches a central (vs outer) atom to a given bond
    if atom["role"] == "central":
        return bond["firstAtom"] == atom["ID"]
    else:
        return max(int(bond["firstAtom"] == atom["ID"]), int(bond["secondAtom"] == atom["ID"]))

def valencyBondDifferenceCalc(LAI, bonds):
    for atom in LAI:
        atom["vBD"] =  sum(bond["bondOrder"] *
                                        bondMatchesAtom(bond, atom) for bond in bonds) - atom["valence"] - atom["charge"]
    return LAI

class substanceGen(pg.sprite.Sprite):  #class for chemical substance
    def __init__(self, substanceParams):   #sets up the atom / molecule
        # substanceParams = [[substanceType, atomNumber, x, y, xdir, ydir, substanceID, angleStartPos, rotationDirection, frameN, recentColl, frameWidth, velocityDisplayScale, centralAtom, iterations, substanceTypeInSubstances?,
            #boxes], [name, colour, radius, mass], ...] = properties of (proposed) new atom / molecule
        pg.sprite.Sprite.__init__(self)  #functioning code
        screen = pg.display.get_surface() #get screen
        self.substanceType = substanceParams["molecule"]["substanceType"]  #F2, H2, N2, O2, Ne, NO..
        self.atomNumber = substanceParams["molecule"]["atomNumber"] #number of atoms in substance
        self.velocity = pg.math.Vector2(substanceParams["molecule"]["xdir"], substanceParams["molecule"]["ydir"])
        self.substanceID = substanceParams["molecule"]["substanceID"] #note that substanceID is 1, 2... corresponding to the count of the substance as opposed to python's ID of the object (which is e.g. 2525047141904)
        ePairPosTerms = ["type", "order", "IDs", "angle"]
        self.mass = 0
        self.atomMasses = []
        self.colour = []
        self.atomRadii = []
        self.atomInfo = []
        self.reactingAtomCentres = []
        velocityDisplayScale = substanceParams["molecule"]["velocityDisplayScale"]
        self.pos = [pg.math.Vector2(substanceParams["molecule"]["x"], substanceParams["molecule"]["y"])]
        self.radius = 0
        self.centralAtom = substanceParams["molecule"]["centralAtom"]
        self.iterations = substanceParams["molecule"]["iterations"] #iterations of substance position between current and next, to check for collisions in the intermediate positions between the current and next positions
        self.substanceTypeInSubstances = substanceParams["molecule"]["substanceTypeInSubstances?"]
        for i in range(1, self.iterations + 2):  #pos[0] is current pos; pos[1], pos[2] etc. are the iterations
            self.pos.append(pg.math.Vector2(substanceParams["molecule"]["x"] + i * substanceParams["molecule"]["xdir"] * velocityDisplayScale / (self.iterations + 1), substanceParams["molecule"]["y"] + i * substanceParams["molecule"]["ydir"] * velocityDisplayScale / (self.iterations + 1)))
        self.reactingPos = self.pos[self.iterations+1] #specifies the exact next position of the substance (defaults on current pos + velocity, but reactingPos gives a greater position accuracy for collisions then just using this value)
        if self.atomNumber > 2:
            for key, atom in substanceParams.items(): #for each atom
                atom["charge"] = 0
                if key == self.centralAtom:
                    centralAtom = atom #stores info on central atom
        if self.centralAtom != "-":
            self.addAtom(self.centralAtom, substanceParams[self.centralAtom]) #central first
        for key, atom in substanceParams.items():
            if key not in ["molecule", self.centralAtom]:
                self.addAtom(key, atom)
        self.LewisChecking()
        if self.atomNumber == 1:
            self.radius = self.atomRadii[0]
            self.surfaceDimensions = pg.math.Vector2(self.radius * 3, self.radius *  3) #self.surfaceDimensions = 3x self.radius
            self.image = pg.Surface((self.surfaceDimensions), pg.SRCALPHA)
            self.relAtomCentres = [pg.math.Vector2(0, 0)]
            #print("ePairInfo "+str(self.LAI[0]["ePairInfo"])+" angleIt deg "+str(360 / (self.LAI[0]["ePairInfo"][1] + self.LAI[0]["ePairInfo"][2] + 1)))
            angleIteration = np.radians(360 / (self.LAI[0]["ePairInfo"][1] + self.LAI[0]["ePairInfo"][2]))
            for i in range(self.LAI[0]["ePairInfo"][1]): #for each lone pair
                self.LAI[0]["ePairPos"].append(dict(zip(ePairPosTerms, ["LP", 1, [self.LAI[0]["ID"]], i*angleIteration])))
            for i in range(self.LAI[0]["ePairInfo"][1], self.LAI[0]["ePairInfo"][1]+self.LAI[0]["ePairInfo"][2]): #for each unpaired electron
                self.LAI[0]["ePairPos"].append(dict(zip(ePairPosTerms, ["up", 1, [self.LAI[0]["ID"]], i*angleIteration])))
            pg.draw.circle(self.image, self.colour[0], self.surfaceDimensions * 1/2, self.radius)  #puts atom in position of the surface
            self.reactingAtomCentres = [self.reactingPos]
        else: #2+ atoms
            if substanceParams["molecule"]["angleStartPos"] == "na": #na here and below indicates that these aren't being defined upon creation of the substance (7 is angle, 8 is rotation direction)
                startingAngle = random.randint(0, 12) * math.pi / 6 #fraction of 2pi radians to rotate by, gives full 2pi rotation after 12 frames
            else:
                startingAngle = substanceParams["molecule"]["angleStartPos"] * math.pi / 6
            if substanceParams["molecule"]["rotationDirection"] == "na": 
                self.angleDisp = [startingAngle, random.choice([-1, 1])] #second number is direction of rotation
            else:
                self.angleDisp = [startingAngle, substanceParams["molecule"]["rotationDirection"]]
            self.surfaceDimensions = pg.math.Vector2(self.radius * 10, self.radius * 10) #larger surfaceDimensions to account for rotation
            self.image = pg.Surface((self.surfaceDimensions), pg.SRCALPHA)
            self.com = pg.math.Vector2(self.surfaceDimensions / 2) #rotation around 'centre of mass'
            if self.atomNumber == 2:
                l1 = (self.atomMasses[1] * (self.atomRadii[0] + self.atomRadii[1])) / (self.atomMasses[0] + self.atomMasses[1]) #l1 = distance between atom 1 and molecule's centre of mass when m1l1 = m2l2, along the C∞ axis. 0.6(r1+r2)=bond length 
                self.imageAtomCentres = [self.com - (pg.math.Vector2(0, l1)) * 0.6, self.com + (pg.math.Vector2(0, self.atomRadii[0]
                                                                                + self.atomRadii[1] - l1)* 0.6)] #at 0º angle (no rotation), imageAtomCentres[0] is vertically above imageAtomCentres[1], 0.6 used to adjust atom separation to be more visually realistic
                self.relAtomCentres = [pg.math.Vector2(0, -l1), pg.math.Vector2(0, self.atomRadii[0] + self.atomRadii[1] - l1)]
                self.atomAngles = [0, math.pi] #lone pairs on left atom should be on left side of atom; lone pairs on right atom should be on right (pi radians rotation) of atom
                for atom in self.LAI:
                   # print("atom "+str(atom["atom"])+" type "+str(self.substanceType))
                    if atom["ID"] == 0: #only adding the bond pair to one of the atoms for plotting
                        atom["ePairPos"].append(dict(zip(ePairPosTerms, ["BP", self.bonds[0]["bondOrder"], [0, 1], 0])))
                    angleIteration = np.radians(360 / (atom["ePairInfo"][1] + atom["ePairInfo"][2] + 1)) #angle between electron pairs
                    #print("ePairInfo "+str(atom["ePairInfo"])+" angleIt deg "+str(360 / (atom["ePairInfo"][1] + atom["ePairInfo"][2] + 1)))
                    for i in range(atom["ePairInfo"][1]): #for each lone pair on the atom
                        atom["ePairPos"].append(dict(zip(ePairPosTerms, ["LP", 1, [atom["ID"]], (i+1)*angleIteration])))
                    for i in range(atom["ePairInfo"][1], atom["ePairInfo"][1]+atom["ePairInfo"][2]): #for each unpaired e- of the atom
                        atom["ePairPos"].append(dict(zip(ePairPosTerms, ["up", 1, [atom["ID"]], (i+1)*angleIteration])))
            else: #3+ atoms
                bondPairs, lonePairs, eRemainder = self.LAI[0]["ePairInfo"][0], self.LAI[0]["ePairInfo"][1], self.LAI[0]["ePairInfo"][2]
                bondAngles = [[180,120,90,72],[117,107],[104.5]] #bondAngles[lonePairs][bondCount-2]
                self.imageAtomCentres = [self.com - (pg.math.Vector2(0, 30/9))] #defines position of central atom
                self.relAtomCentres, self.atomAngles = [pg.math.Vector2(0, 0)], [0, 0]
                bondLength = (0.8 * (self.LAI[0]["radius"]+self.LAI[1]["radius"]))
                self.relAtomCentres.append(pg.math.Vector2(bondLength, 0)) #places second atom to the right of central
                self.imageAtomCentres.append(self.imageAtomCentres[0] + pg.math.Vector2(bondLength, 0))
                angleRad = np.radians(bondAngles[lonePairs+eRemainder][bondPairs-2]) #bond angle in radians
                for atom in self.LAI:
                    if atom["role"] == "central":
                        for index, bond in enumerate(self.bonds): #for each bond the atom is associated with
                            atom["ePairPos"].append(dict(zip(ePairPosTerms, ["BP", bond["bondOrder"], [0, bond["secondAtom"]], angleRad*index])))
                        bondAngleSum = angleRad *(len(self.bonds)-1) #e.g. 2x100°=200°, so the non-bonding electrons are found between 200° and 360°
                    else:
                        bondAngleSum = 0 #since the only bond is towards the central atom, the LPs and up's have 360° of space
                    angleIteration = ((2*math.pi) - bondAngleSum) / (atom["ePairInfo"][1] + atom["ePairInfo"][2] + 1) #2pi-bAS gives remaining angle for non-bonding electrons
                    #if self.substanceType == "NO2" and atom["atom"] == "O":
                        #print("N len "+str(len(self.bonds))+"angleRad "+str(math.degrees(angleRad))+
                        #     " bAS "+str(math.degrees(bondAngleSum))+" aI "+str(math.degrees(angleIteration)))
                    for i in range(atom["ePairInfo"][1]): #for each lone pair of the atom
                       # print(" i "+str(i))
                        atom["ePairPos"].append(dict(zip(ePairPosTerms, ["LP", 1, [atom["ID"]], bondAngleSum+((i+1)*angleIteration)])))
                    for i in range(atom["ePairInfo"][1], atom["ePairInfo"][1] + atom["ePairInfo"][2]): #for each unpaired electron of the atom
                        atom["ePairPos"].append(dict(zip(ePairPosTerms, ["up", 1, [atom["ID"]], bondAngleSum+((i+1)*angleIteration)])))
                    if atom["ID"] > 1: #done IDs 0 and 1 above
                        qx, qy = rotate((0, 0), self.relAtomCentres[1], angleRad*(int(atom["ID"])-1))
                        self.relAtomCentres.append(pg.math.Vector2(qx, qy))
                        self.atomAngles.append(angleRad*(int(atom["ID"])-1))
                        self.imageAtomCentres.append(self.imageAtomCentres[0] + pg.math.Vector2(qx, qy))
            for i2 in range(0, self.atomNumber):
                qx, qy = rotate(self.com, self.imageAtomCentres[i2], startingAngle) #origin, point, angle
                self.reactingAtomCentres.append(pg.math.Vector2(qx, qy) - self.com + self.reactingPos)
            self.currentAngle = startingAngle #clockwise angle of rotation of the molecule, where 0º is directly upwards
            #trueCom = sum((pos * mass for pos, mass in zip(self.imageAtomCentres, self.atomMasses)),start=pg.math.Vector2(0, 0)) / self.mass #can use this
            #offset = trueCom - self.com
            #self.com = trueCom
            #self.imageAtomCentres = [pos - offset for pos in self.imageAtomCentres]
        #print("type "+str(self.substanceType))
        #for atom in self.LAI:
        #    print("atom "+str(atom["atom"])+" ID "+str(atom["ID"])+" ePairPos "+str(atom["ePairPos"]))
        self.rect = self.image.get_rect(center = (substanceParams["molecule"]["x"], substanceParams["molecule"]["y"]))
        self.combinedRect = self.rect.union(self.image.get_rect(center = self.reactingPos)) #makes a big rect object covering both the current and next frame positions
        self.currentBoxes = [i for i, box in enumerate(substanceParams["molecule"]["boxes"]) if self.combinedRect.colliderect(box)]
        self.willReact = set() #once a reaction is detected, this is assigned to the substanceID of the substance(s) that this substance will react with, or just "uni" if it will undergo a unimolecular reaction
        self.currentColl = [] #only used to check current colls between non-reacting atoms (not used for reactions)
        self.recentColl = substanceParams["molecule"]["recentColl"]
        self.recentCollPrevious = [] #stores recentColl for the most recent 10 frames
        self.frameWidth = substanceParams["molecule"]["frameWidth"]
        self.screenArea = (screen.get_rect()[2] - self.frameWidth, screen.get_rect()[3] - self.frameWidth)
        self.startingFrame = substanceParams["molecule"]["frameN"] #frame when rotation begins
        self.product = ""
        self.reactingFrame = 0

    def addAtom(self, key, atom): #adds atom info to self
        self.atomInfo.append({"atom": key, **atom})
        self.atomInfo[-1]["charge"], self.atomInfo[-1]["role"], self.atomInfo[-1]["remainingValence"], self.atomInfo[-1]["vBD"] =\
                                     0, "outer", self.atomInfo[-1]["valence"], 0 #role = central or outer atom
        for i in range(atom["count"]):
            self.colour.append(atom["colour"])
            self.atomRadii.append(atom["radius"])
            self.atomMasses.append(atom["Ar"])
            self.mass += atom["Ar"]
            self.radius = max(self.radius, atom["radius"]) #doing this isn't really that accurate, but it provides consistency with the appearance of a collision and the detection of a collision

    def LewisChecking(self):
        bonds = []
        bondsFields = ["firstAtom", "secondAtom", "bondOrder"]
        LewisAtomInfo = []
        LAI = LewisAtomInfo #shortening
        atomID = 0
        for atom in self.atomInfo:
            count = atom["count"]
            for i in range(count):
                newAtom = {"ID": atomID, "ePairInfo": [0, 0, 0], "ePairPos": [], **atom} #**atom adds all info from atom to newAtom. ePairInfo = bond count, lone pair count, unpaired e- count 
                LAI.append(newAtom) #ePairPos = lone pair / bond pair / unpaired e-, bond order, [ID1, ID2] or ID1 for one atom (lp/up), angle
                atomID += 1
        if self.atomNumber == 2:
            if self.atomInfo[0]["count"] == 2: #homonuclear diatomic
                maxOrder = self.atomInfo[0]["valence"]
            else: #heteronuclear diatomic
                maxOrder = min(LAI[0]["valence"], LAI[1]["valence"])
            bonds.append({"firstAtom": 0, "secondAtom": 1,
                "bondOrder": maxOrder})
            for atom in LAI:
                atom["remainingValence"] -=bonds[-1]["bondOrder"]
                atom["ePairInfo"][0] = 1
        if self.atomNumber > 2:
             for i, atom in enumerate(LAI): #loop to put central atom at index 0 of LAI for convenience
                if atom["role"] == self.centralAtom:
                    LAI.insert(0, LAI.pop(i))
                    break  
             LAI[0]["role"]= "central"
             for atom in LAI:
                maxBondOrder = 1
                if atom["role"] != "central": #non-central atoms
                    if LAI[0]["remainingValence"] > 0: #if central atom has available bonds
                        LAI, bonds = bondingToCentralAtom(LAI, bonds, atom, maxBondOrder)
                        while atom["remainingValence"] > 0 and LAI[0]["remainingValence"] > 0: #if there is still valence to be filled
                            maxBondOrder+=1
                            LAI, bonds = bondingToCentralAtom(LAI, bonds, atom, maxBondOrder)
        LAI = valencyBondDifferenceCalc(LAI, bonds)
        for atom in LewisAtomInfo: #loop to allow charge transfer when needed, e.g. donating an electron from N to O to give an N+ radical and completed-octet O
            if atom["role"] != "central" and self.atomNumber > 2:
                if atom["vBD"] < 0: #atom is electron deficient
                    for bond in bonds:
                        if (bond["firstAtom"] == LAI[0]["ID"] and bond["secondAtom"] == atom["ID"] and bond["bondOrder"] < 2): #identifies a relevant single bond
                            LAI[0]["charge"], LAI[0]["remainingValence"] = 1, LAI[0]["remainingValence"]-1
                            atom["charge"], atom["remainingValence"] = -1, atom["remainingValence"]+1
                            break
        LAI = valencyBondDifferenceCalc(LAI, bonds)
        self.LAI, self.bonds = LAI, bonds
        for atom in LAI:
            unpairedE = atom["outerE"] - (atom["valence"] - atom["remainingValence"])
            atom["ePairInfo"][1], atom["ePairInfo"][2] = divmod(unpairedE, 2)
        self.LewisErrors = [f"{atom['atom']} ({atom['role']}) has {abs(atom['vBD'])} " f"{'excess' if atom['vBD'] > 0 else 'deficient'} "
        f"[{atom['vBD']:+}] valence" for atom in LewisAtomInfo if atom["vBD"] != 0]
        if self.LewisErrors and self.substanceTypeInSubstances == False:
            print(f"Error in molecule {self.substanceType}: " + ", ".join(self.LewisErrors))


def collCheck(substance1, substance2): #sub-function that does atom-atom collision checking
##    for i2 in range(0, substance1.atomNumber):  #this checks every atom with each other. More accurate but much slower.
##        for i3 in range(0, substance2.atomNumber):
##            atomCollisionDistance = substance1.atomRadii[i2] + substance2.atomRadii[i3]
##            if substance1.reactingAtomCentres[i2].distance_to(substance2.reactingAtomCentres[i3]) < atomCollisionDistance:
##                collDetected = 1
    collisionDistanceSquared = (substance1.radius + substance2.radius)**2
    for iteration in range(1, substance1.iterations): #starting from 1 as current position was checked during previous frame as the last iteration (equal to the substance's substance.reactingPos)
       delta = substance1.pos[iteration] - substance2.pos[iteration]
       if delta.length_squared() < collisionDistanceSquared:
            return 1 #collision detected
    return 0 #no collision detected

def getSubstanceAtoms(substanceParams, atomInfo): #gets the constituent atoms and their counts from a substance formula
    substanceParams["molecule"]["atomNumber"]=0 #resetting atom number to 0 in case same substanceParams is passed into this func again
    capitalLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    substanceFormula = substanceParams["molecule"]["substanceType"].lstrip("0123456789") #removes count of molecule in substanceFormula if present
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
        for symbol, atom in atomInfo.items(): #for all possible atoms
            if symbol == atomList[i7][0]: #if atom type found within atomList
                substanceParams[symbol] = atom #adds info about the atom to substanceParams
                substanceParams["molecule"]["atomNumber"]+=int(atomList[i7][1])
                substanceParams[symbol]["count"] = int(atomList[i7][1])    
    return substanceParams

def gridGen(physicalParams, physicalParamsFields):
    dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,\
        iterations, substanceParamsMoleculeFields, boxes, grid = (physicalParams[i] for i in physicalParamsFields)
    #generate 1 of every type of substance (not adding these to list of simulation substances), find diameter of each (2*radius), therefore find maximum diameter
    substanceID = 1
    largestDiameter = 0 #largest diameter of any possible substance
    for i in range(0, len(substanceGenInfo)): #for each possible substance
        x, y, xdir, ydir = 50, 50, 50, 50
        substanceParams = {"molecule": dict(zip(substanceParamsMoleculeFields, [substanceGenInfo[i]["formula"], 0, x, y, xdir, ydir,
                                                                            substanceID, "na", "na", 0, set(), frameWidth, velocityDisplayScale, substanceGenInfo[i]["central"], iterations, "False", boxes]))}
        substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
        substanceNew = substanceGen(substanceParams)
        if substanceNew.radius > largestDiameter/2:
            largestDiameter = substanceNew.radius*2
    assumedBoxWidth = 1.2*largestDiameter #arbitrary 1.2 scale-up
    boxLengthwiseCount = math.ceil(dimensions/assumedBoxWidth) #number of boxes spanning one side of the simulation (e.g. if this = 5, the simulation has a grid of 5x5 = 25 boxes)
    boxes = []
    for row in range(boxLengthwiseCount):
        for col in range(boxLengthwiseCount):
            x = col * assumedBoxWidth #x position of the box = a multiple of the box width (as with y)
            y = row * assumedBoxWidth
            actualBoxWidth = min(assumedBoxWidth, dimensions - x) #if the box would overlap with the right or bottom sides of the simulation, the box width is brought down, so that it fits in the simulation
            actualBoxHeight = min(assumedBoxWidth, dimensions - y)
            boxes.append(pg.Rect(x, y, actualBoxWidth, actualBoxHeight))
    grid = {i: [] for i in range(len(boxes))}
    return boxes, grid
        
def substanceSetup(physicalParams, physicalParamsFields):
    dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,\
        iterations, substanceParamsMoleculeFields, boxes, grid = (physicalParams[i] for i in physicalParamsFields)
    substances = pg.sprite.Group(())
    substanceID = 1 #unique ID for each substance on screen. This variable stores the substanceID that will be used for the next substance that is generated. different to python's ID of the object (which is e.g. 2525047141904)
    largestRadius = 0 #largest radius of the atoms in a substance
    avgSpeed = np.sqrt((2 * R * TStart) / (avgMr / 1000))  # m/s
    largestRadius = max(int(atom["radius"]) for atom in atomInfo.values())
    for i in range(0, len(substanceGenInfo)):  #for each substanceType
        for i2 in range(substanceGenInfo[i]["count"]):  #for each atom / molecule of the substance
            spawnTries = 0
            while spawnTries < 100: #give 100 tries to spawn, otherwise skip (could do this differently, can lead to an uneven number of each substance type on the screen)
                x = random.randint(largestRadius+frameWidth, dimensions-largestRadius-frameWidth) #random x coordinate
                y = random.randint(largestRadius+frameWidth, dimensions-largestRadius-frameWidth) #random y coordinate
                xdir, ydir = 0, 0
                while xdir == 0 and ydir == 0: #prevents substance from spawning with zero velocity (stationary)
                    angle = random.uniform(0, 2 * math.pi) #random angle of direction
                    xdir, ydir = avgSpeed * math.cos(angle), avgSpeed * math.sin(angle)  #random magnitude + direction of x and y travel
                substanceParams = {"molecule": dict(zip(substanceParamsMoleculeFields, [substanceGenInfo[i]["formula"], 0, x, y, xdir, ydir,
                                                                                        substanceID, "na", "na", 0, set(), frameWidth, velocityDisplayScale, substanceGenInfo[i]["central"], iterations, "False", boxes]))}
                substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
                substanceParams["molecule"]["substanceTypeInSubstances?"] =  any(s.substanceType == substanceParams["molecule"]["substanceType"] for s in substances)
                substanceNew = substanceGen(substanceParams)
                collDetectedTotal = 0
                for substance2 in substances:
                    collDetectedTotal+=collCheck(substanceNew, substance2)
                if collDetectedTotal == 0:  #if new atom / molecule doesn't collide with any other substance currently present on screen
                    substances.add(substanceNew) #create the new atom / molecule
                    substanceID+=1
                    for box in substanceNew.currentBoxes:
                        physicalParams["grid"][box].append(substanceNew)
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
            if angleStartArgs[1][1][0] - angleStartArgs[0][1][0] == 0:
                print("angleStartArgs "+str(angleStartArgs))
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

def reactionProcessing(substance1, substance2,reactionSuccessful, threeSubstanceReaction, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields):
    dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,\
                 iterations, substanceParamsMoleculeFields, boxes, grid = (physicalParams[i] for i in physicalParamsFields)
    substances, toggleReaction, frameN, substanceID, reactionInfo, rollingData, framerate, going, showPlot, timeInterval, EaMethod\
                = (simulationParams[i] for i in simulationParamsFields)
    if len(substance1.willReact) == 2 and substance2 in substance1.willReact and substance1.reactingFrame == frameN: #three-substance reaction
        substance3 = next(iter(substance1.willReact - {substance2})) #gets substance3 from substance1.willReact
        newx, newy = (substance1.pos[0][0]+substance2.pos[0][0]+substance3.pos[0][0])/3, (substance1.pos[0][1]+substance2.pos[0][1]+substance3.pos[0][1])/3
        totalmv = (substance1.mass * substance1.velocity) + (substance2.mass * substance2.velocity) + (substance3.mass * substance3.velocity)
        totalMass = substance1.mass + substance2.mass + substance3.mass
        reactionCentre = (substance1.mass * np.array(substance1.pos[0]) + substance2.mass * np.array(substance2.pos[0]) + substance3.mass * np.array(substance3.pos[0])) / totalMass
        reactionSuccessful = 1
        threeSubstanceReaction = 1
    elif substance2.willReact == {substance1} and substance2.reactingFrame == frameN: #two-substance reaction
        newx, newy = (substance1.pos[0][0]+substance2.pos[0][0])/2, (substance1.pos[0][1]+substance2.pos[0][1])/2
        totalmv = (substance1.mass * substance1.velocity) + (substance2.mass * substance2.velocity)
        totalMass = substance1.mass + substance2.mass
        reactionCentre = (substance1.mass * np.array(substance1.pos[0]) + substance2.mass * np.array(substance2.pos[0])) / totalMass
        reactionSuccessful = 1
        
    if reactionSuccessful == 1:
        productNumber = 0
        products = [s.strip() for s in substance1.product.split('+') if s.strip()]
        formedSubstances = []
        centralAtom = []
        for product in products:
            centralAtom.append(next((row["central"] for row in substanceGenInfo if row["formula"] == product), "-")) #finds central atom if present
        try:
            productNumber = int(substance1.product[0]) #number of product molecules
        except:
            productNumber = len(products)
        newVelocity = totalmv / (productNumber * totalMass)
        newSubstanceRecentColl = set()
        angleStartArgs = [[substance1.pos[0], substance1.velocity], [substance2.pos[0], substance2.velocity]]
        angleStartPos, rotationDirection = angleStartFunc(angleStartArgs)
        if productNumber == 1:
            substanceParams = {"molecule": dict(zip(substanceParamsMoleculeFields, [substance1.product, 0, newx, newy,
                                                                                    newVelocity[0], newVelocity[1], substanceID, angleStartPos, rotationDirection, frameN,\
                                                                                    {}, substance1.frameWidth, velocityDisplayScale, centralAtom[0], iterations, "False", boxes]))}
            substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
            substanceParams["molecule"]["substanceTypeInSubstances?"] =  any(s.substanceType == substanceParams["molecule"]["substanceType"] for s in substances)               
            substanceNew = substanceGen(substanceParams)
            substances.add(substanceNew)
            substanceID+=1
            formedSubstances = [substanceNew]
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
                
            substanceParams = {"molecule": dict(zip(substanceParamsMoleculeFields, [products[0], 0, productPositions[0][0], productPositions[0][1], 0, 0, substanceID, angleStartPos, rotationDirection,\
                                frameN, set(), substance1.frameWidth, velocityDisplayScale, centralAtom[0], iterations, "False", boxes]))}
            substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
            for i in range(0, substance1.product.count("+")+1): #for each new substance
                substanceParams["molecule"]["x"] = productPositions[i][0]
                substanceParams["molecule"]["y"] = productPositions[i][1]
                substanceParams["molecule"]["xdir"]=vels[i][0]
                substanceParams["molecule"]["ydir"]=vels[i][1]
                if i > 0:
                    substanceParams = {"molecule": substanceParams["molecule"]}
                    substanceParams["molecule"]["substanceType"] = products[i]
                    substanceParams["molecule"]["substanceID"] = substanceID
                    substanceParams["molecule"]["rotationDirection"] = rotationDirection * -1
                    substanceParams["molecule"]["recentColl"] = set()
                    substanceParams["molecule"]["centralAtom"] = centralAtom[i]
                    substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
                substanceParams["molecule"]["substanceTypeInSubstances?"] =  any(s.substanceType == substanceParams["molecule"]["substanceType"] for s in substances)
                substanceNew = substanceGen(substanceParams)
                substances.add(substanceNew)
                formedSubstances.append(substanceNew)
                substanceID+=1
        substances.remove(substance1)
        substances.remove(substance2) #other substances will still be iterated in the substance1 loop (as this loop has already started), but these can be ignored using substances.remove() and checking for substances.has
        for i, substance in enumerate(formedSubstances):
            substance.recentColl = set(formedSubstances) - {substance} #removes self when forming recentColl
        if threeSubstanceReaction == 1:
            substances.remove(substance3)
            threeSubstanceReaction = 0
        reactionSuccessful = 0
    simulationParams["substances"], simulationParams["substanceID"] = substances, substanceID

def probabilityChecking(substances, reactionInfo, T, EaMethod):
    reactantList = sorted([str(s.substanceType) for s in substances])
    reactants = "+".join(reactantList)
    for r in reactionInfo.values():
        if str(r["reactants"]) == reactants or str(r["products"]) == reactants:
                deltaG = -R * T * np.log(r["K"])  #in J/mol
                EaF = r["Ea"] #artificially very low (could also use a very high temp), needed for good collision rate
                EaR = EaF - deltaG
                Ea = EaF if str(r["reactants"]) == reactants else EaR
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

def reactionChecking(substance1, substance2, T, frameN, reactionCount, box, i, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields): #checks for reactions with >1 molecule
    dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,\
                 iterations, substanceParamsMoleculeFields, boxes, grid = (physicalParams[i] for i in physicalParamsFields)
    substances, toggleReaction, frameN, substanceID, reactionInfo, rollingData, framerate, going, showPlot, timeInterval, EaMethod\
                = (simulationParams[i] for i in simulationParamsFields)
    reactionSuccessful2 = False

    if len(substance2.willReact) == 0: #checking for two-substance reactions. already checked (in collisionProcessing) if len(substance1.willReact) == 0
        nextReactionAllowed = 0 #indicates whether a given reaction is allowed (is in the list of possible reactions), but not yet successful (needs to pass probability check)
        reactantList = sorted([substance1.substanceType, substance2.substanceType])
        reactants = "+".join(reactantList)
        for r in reactionInfo.values():
            if r["reactants"] == reactants or r["products"] == reactants:
                if r["reactants"] == reactants: #determine direction of reaction
                    productString = r["products"]
                else:
                    productString = r["reactants"]
                substance1.product = substance2.product = productString #set products
                nextReactionAllowed = 1
                break
        if nextReactionAllowed == 1: #two-substance reaction allowed
            reactionSuccessful2 = probabilityChecking([substance1, substance2], reactionInfo, T, EaMethod)
            if reactionSuccessful2 == True: #two-substance reaction successful
                substance1.willReact.add(substance2)
                substance2.willReact.add(substance1)
                substance2.reactingFrame = frameN + 1
                reactionCount+=1
                
        else: #checking for three-substance reactions if no two-substance reactions were found
            for k in range(i+2, len(box)): #for each of the remaining substances in the box apart from substance1 and substance2
                substance3 = box[k]       
                if substance3 != substance1 and substance3 != substance2:
                    collDetected, collDetected2 = 0, 0
                    collDetected = collCheck(substance1, substance3)     #
                    collDetected2 = collCheck(substance2, substance3)   #checking substance 3 colliding with either substance1 or substance2
                    if (collDetected == 1 or collDetected2 == 1) and len(substance3.willReact) == 0:
                        reactantList = sorted([substance1.substanceType, substance2.substanceType, substance3.substanceType])
                        reactants = "+".join(reactantList)
                        for r in reactionInfo.values():
                            if r["reactants"] == reactants or r["products"] == reactants:
                                if r["reactants"] == reactants: #determine direction
                                    productString = r["products"]
                                else:
                                    productString = r["reactants"]
                                substance1.product = substance2.product = substance3.product = productString #set products
                                nextReactionAllowed = 1
                                break
                        if nextReactionAllowed == 1:
                            reactionSuccessful2 = probabilityChecking([substance1, substance2, substance3], reactionInfo, T, EaMethod)
                            if reactionSuccessful2 == True:
                                substance1.willReact.update([substance2,substance3])
                                substance2.willReact.update([substance1,substance3])
                                substance3.willReact.update([substance2,substance1])
                                substance1.reactingFrame = substance2.reactingFrame = substance3.reactingFrame = frameN + 1 #specifies that reaction will take place in next frame
                                reactionCount+=1
                            break
    simulationParams["substances"] = substances
    return reactionCount, reactionSuccessful2

def collisionProcessing(substances, substance1, substance2, breakLoop, T, frameN, reactionCount, box, i, physicalParams, physicalParamsFields,
                                                                                                       simulationParams, simulationParamsFields):
    collDetected = collCheck(substance1, substance2)
    if collDetected == 1 and substance2.substanceID not in substance1.currentColl and\
       substance2 not in substance1.recentColl and substance1 not in\
       substance2.recentColl and len(substance1.willReact) == 0: #if this collision hasn't been checked, and substance1 hasn't reacted (need to check both recentColls here)
        if simulationParams["toggleReaction"] == True:
            reactionCount, reactionSuccessful2 = reactionChecking(substance1, substance2, T, frameN, reactionCount, box, i, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields)
        else:
            reactionSuccessful2 = False
        if reactionSuccessful2 != True: #if reaction wasn't successful (hence non-reaction collision should be processed)
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
            substance1.recentColl.add(substance2)
            substance2.recentColl.add(substance1)
        breakLoop = 1
    return substances, breakLoop, reactionCount

def unimolecularReactionProcessing(substance1, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields):
    dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,\
                 iterations, substanceParamsMoleculeFields, boxes, grid = (physicalParams[i] for i in physicalParamsFields)
    substances, toggleReaction, frameN, substanceID, reactionInfo, rollingData, framerate, going, showPlot, timeInterval, EaMethod\
                = (simulationParams[i] for i in simulationParamsFields) 
    if substance1.willReact == "uni":
        productNumber = 0
        products = [s.strip() for s in substance1.product.split('+') if s.strip()] #gets products from substance1.product
        centralAtom = []
        for product in products:
            centralAtom.append(next((row["central"] for row in substanceGenInfo if row["formula"] == product), "-")) #finds central atom if present
        try:
            productNumber = int(substance1.product[0]) #number of product molecules
        except:
            productNumber = len(products)
        newSubstanceRecentColl = set()
        vels = [] #product velocities
        formedSubstances = []
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
                substanceParams = {"molecule": dict(zip(substanceParamsMoleculeFields, [products[i], 0, productPositions[i][0], productPositions[i][1], vels[i][0],
                                    vels[i][1], substanceID, substance1.angleDisp[0] * 6 / math.pi, substance1.angleDisp[1], frameN, set(), substance1.frameWidth,
                                    velocityDisplayScale, centralAtom[i], iterations, "False", boxes]))}
                substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
                substanceParams["molecule"]["substanceTypeInSubstances?"] =  any(s.substanceType == substanceParams["molecule"]["substanceType"] for s in substances)
                substanceNew = substanceGen(substanceParams)
                formedSubstances.append(substanceNew)
                substances.add(substanceNew)
                substanceID += 1
        substances.remove(substance1)
        for i, substance in enumerate(formedSubstances):
            substanceParams["molecule"]["recentColl"] = set(formedSubstances) - {substance}
    simulationParams["substances"], simulationParams["substanceID"] = substances, substanceID

def unimolecularReactionChecking(substance1, reactionCount, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields):
    dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,\
                 iterations, substanceParamsMoleculeFields, boxes, grid = (physicalParams[i] for i in physicalParamsFields)
    substances, toggleReaction, frameN, substanceID, reactionInfo, rollingData, framerate, going, showPlot, timeInterval, EaMethod\
                = (simulationParams[i] for i in simulationParamsFields) 
    for r in reactionInfo.values():
        if (r["reactants"] == substance1.substanceType or r["products"] == substance1.substanceType) and r["k"] != "-":   #if substance1 is a reactant for a unimolecular reaction
            if r["reactants"] == substance1.substanceType: #determine direction
                productString = r["products"]
                k = float(r["k"])
            else:
                productString = r["reactants"]
                k = float(r["k"]) / float(r["K"]) #kBackwards = kForwards / K
            p = 1 - np.exp(-k * timeInterval) #probability of successful reaction
            if random.random() < p and toggleReaction == True and len(substance1.willReact) == 0: #random.random() < p evaluates the probability
                substance1.willReact = "uni" #will undergo a unimolecular decomposition reaction
                substance1.reactingFrame = frameN + 1 #will react in the next frame
                reactionCount+=1
                substance1.product = productString
                break
    return reactionCount

def substanceUpdates(substances, frameN, velocityDisplayScale, boxes, grid): #updates the properties (position, angle, recentColl) of each substance in a given frame
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
        substance1.combinedRect = substance1.rect.union(substance1.image.get_rect(center = substance1.reactingPos)) #makes a big rect object covering both the current and next frame positions
        substance1.currentBoxes = [i for i, box in enumerate(boxes) if substance1.combinedRect.colliderect(box)]
        for box in substance1.currentBoxes:
            grid[box].append(substance1)
        substance1.recentCollPrevious.append(substance1.recentColl.copy())
        if len(substance1.recentCollPrevious) > 10: #if there are more than 10 frames' worth of info in substance1.recentCollPrevious
            del substance1.recentCollPrevious[0] #delete the oldest
        for recentSubstance in substance1.recentColl.copy():  #iterate over a copy to avoid modifying while looping
            if recentSubstance not in substances:
                substance1.recentColl.remove(recentSubstance) #removes substances from substance1.recentColl that no longer exist
    return substances, grid

def frameProcessing(physicalParams, physicalParamsFields, simulationParams, simulationParamsFields, plotParams): #processes each frame
    dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,\
                 iterations, substanceParamsMoleculeFields, boxes, grid = (physicalParams[i] for i in physicalParamsFields)
    substances, toggleReaction, frameN, substanceID, reactionInfo, rollingData, framerate, going, showPlot, timeInterval, EaMethod\
                = (simulationParams[i] for i in simulationParamsFields)
    fig, ax1, ax2, scatter, scatter2, lineFit, substanceOfInterest, textAnnotation = plotParams["fig"], plotParams["ax1"], plotParams["ax2"],\
                plotParams["scatter"], plotParams["scatter2"], plotParams["lineFit"], plotParams["substanceOfInterest"], plotParams["textAnnotation"]
    currentVels = [0]
    reactionCount = 0
    for substance1 in substances:
            currentVels.append(round(np.linalg.norm(substance1.velocity), 2))  #norm = magnitude of vector   
    rmsVel = np.sqrt(np.mean(np.square(currentVels))) #root mean square velocity
    T = avgMr * (rmsVel**2) / (2 * R * 1000) # avgMr in g/mol, so dividing by 1000. calculates temperature from 0.5*avgMr*(vrms)^2 = 1.5 * R * T (i.e. 0.5*m*v^2 = 3/2*kB*T)
    checked = set()
    for box in grid.values(): #for each box (this loop checks for substance collisions and reactions)
        for i in range(len(box)): #for each substance in the box (substance1)
            substance1 = box[i]
            if substances.has(substance1):
                reactionSuccessful, threeSubstanceReaction = 0, 0 #note: code processes reactions occuring in this frame before checking for reactions that will occur in the next frame
                breakLoop = 0
                for j in range(i+1, len(box)): #for each of the remaining substances in the box that we'll call substance2 (starting from i+1 to ignore i)
                    substance2 = box[j]
                    pair = tuple(sorted((id(substance1), id(substance2))))
                    if substances.has(substance2) and pair not in checked and bool(set(substance1.currentBoxes) & set(substance2.currentBoxes)): #if substance combo hasn't been checked, and the substances are in overlapping boxes
                        checked.add(pair)
                        reactionProcessing(substance1, substance2,reactionSuccessful, threeSubstanceReaction, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields) #processes reactions from previous frame
                        #reactionCount = reactionChecking(substance1, substance2, T, frameN, reactionCount, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields) #checks for reactions with >1 molecule                    
                        simulationParams["substances"], breakLoop, reactionCount = collisionProcessing(simulationParams["substances"], substance1, substance2, breakLoop, T, frameN, reactionCount, box, i, physicalParams, physicalParamsFields,
                                                                                                       simulationParams, simulationParamsFields) #checks for and processes collisions, including reactions with >1 molecule
                        if breakLoop == 1:
                            break #collision detected - can skip to next i
    for substance1 in substances: #processes unimolecular reactions and wall collisions
        unimolecularReactionProcessing(substance1, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields)
        reactionCount = unimolecularReactionChecking(substance1, reactionCount, physicalParams, physicalParamsFields, simulationParams, simulationParamsFields)
        vxDisp = substance1.velocity[0] * velocityDisplayScale
        vyDisp = substance1.velocity[1] * velocityDisplayScale
        if substance1.rect.center[0] + vxDisp - substance1.radius < substance1.frameWidth and substance1.velocity[0] < 0: #collision with left wall
            substance1.velocity[0], substance1.recentColl = -substance1.velocity[0], set() #inverts x velocity, only if the collision with the left wall will start (during this frame), + clears recentColl
        elif substance1.rect.center[0] + vxDisp + substance1.radius > substance1.screenArea[0] and substance1.velocity[0] > 0: #collision with right wall
            substance1.velocity[0], substance1.recentColl = -substance1.velocity[0], set()
        elif substance1.rect.center[1] + vyDisp - substance1.radius < substance1.frameWidth and substance1.velocity[1] < 0: #collision with top wall
            substance1.velocity[1], substance1.recentColl = -substance1.velocity[1], set()
        elif substance1.rect.center[1] + vyDisp + substance1.radius > substance1.screenArea[1] and substance1.velocity[1] > 0: #collision with bottom wall
            substance1.velocity[1], substance1.recentColl = -substance1.velocity[1], set()
        for substance2 in substance1.recentColl.copy():
            if bool(set(substance1.currentBoxes) & set(substance2.currentBoxes)): #if substances are in the same box(es)
                collDetected = collCheck(substance1, substance2)
                if collDetected == 0: #if no collisions are detected at all during iterations
                    substance1.recentColl.remove(substance2)
            else:
                substance1.recentColl.remove(substance2)

    grid = {i: [] for i in range(len(boxes))} #clears grid, before being re-populated within substanceUpdates
    simulationParams["substances"], physicalParams["grid"] = substanceUpdates(simulationParams["substances"], frameN, velocityDisplayScale, boxes, grid)

    frameEntry = {"Frame": int(frameN), "T (K)": T, "Time (s)": frameN/framerate} #dict to store basic information about the crurent frame in rollingData
    for i in substanceGenInfo: #for all substances
        substanceName = i["formula"]
        count = sum(s.substanceType == substanceName for s in substances)
        frameEntry[substanceName] = count
    if substanceOfInterest != 0: #if this is a substance of interest
        count = sum(s.substanceType == substanceOfInterest for s in substances)
        frameEntry[substanceOfInterest+" reciprocal"] = 1/count if count > 0 else float("nan")
    rollingData.append(frameEntry)
    if frameN % 10 == 0 and showPlot:
        frames = [entry["Frame"] for entry in rollingData]
        time = np.array([entry["Frame"] for entry in rollingData]) / framerate
        substanceCount = [entry[substanceOfInterest] for entry in rollingData] #can use count from above instead
        reciprocalOfCount = [entry[substanceOfInterest+" reciprocal"] for entry in rollingData]
        validPoints = [(f, l) for f, l in zip(time, reciprocalOfCount) if not np.isnan(l)]
        if len(validPoints) >= 2:
            framesFit, reciprocalFit = zip(*validPoints)
            coef = np.polyfit(framesFit, reciprocalFit, 1)  #linear fit
            xline = np.linspace(min(framesFit), max(framesFit), 100)
            yline = coef[0] * xline + coef[1]
            slope, intercept = coef
            predicted = slope * np.array(framesFit) + intercept
            residuals = reciprocalFit - predicted
            ssRes = np.sum(residuals**2)
            ssTot = np.sum((reciprocalFit - np.mean(reciprocalFit))**2)
            kPerSecond = slope * framerate / 2
            rSquared = 1 - (ssRes / ssTot)
            textAnnotation.set_text(f"calculated k = {kPerSecond:.4f} s⁻¹\nR² = {rSquared:.3f}")
            scatter.set_offsets(np.column_stack((time, reciprocalOfCount)))
            scatter2.set_offsets(np.column_stack((time, substanceCount)))
            ax1.relim()
            ax1.autoscale_view()
            ax2.set_ylim(max(0, min(substanceCount) - 2), max(substanceCount) + 2)
            ax2.relim()
            ax2.autoscale_view()
            lineFit.set_data(xline, yline)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    return substanceID, rollingData, going

def avgMrFunc(substances): #gets the average Mr of the substances
    totalMass = 0
    totalMolecules = 0
    for substance in substances:
        atoms = substance.atomInfo  # The whole structure
        molarMass = sum(atom["Ar"] for atom in atoms)  # Ar is at position 3. Sums relative atomic mass for each atom in the substance.
        totalMass += molarMass
        totalMolecules += 1
    return totalMass / totalMolecules

def avgMrFromGenInfo(substanceGenInfo, atomInfo): #gets the average Mr from just substanceGenInfo rather than the list of generated substances (need to use this before substances have been generated)
    totalMass = 0
    totalMolecules = 0
    for substance in substanceGenInfo:
        if substance["count"] == 0: 
            continue #skip this substance
        substanceParams = [[substance["formula"], 0]]
        substanceParams = {"molecule": dict(zip(["substanceType", "atomNumber"], [substance["formula"], 0]))}
        substanceParams = getSubstanceAtoms(substanceParams, atomInfo)
        molarMass = sum(atom["count"] * atom["Ar"] for key, atom in substanceParams.items() if key != "molecule")  #skip metadata [formula, atom count]. As above, sums relative atomic mass for each atom in the substance.
        totalMass += molarMass * substance["count"]
        totalMolecules += substance["count"]
    return totalMass / totalMolecules if totalMolecules > 0 else 0

def main():
    #display setup
    options, atomInfo, reactionInfo, substanceGenInfo = fileRead()
    mode = "simulation"
    dimensions = options.get("Dimensions (pixels)")
    frameWidth =options.get("Width of outer frame (pixels)")
    windowPos = options.get("Window position")
    x, y = map(int, windowPos.replace(" ", "").split(",")) #replace() used to remove space in windowPos
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    boxes, grid = [], []
    pg.init() #initiates pygame
    screen = pg.display.set_mode((dimensions, dimensions)) #window dimensions
    outerBackground = pg.Surface(screen.get_size()) #10-pixel width outer background
    outerBackground.convert()
    innerBackgroundSize = (screen.get_size()[0]-(2*frameWidth), screen.get_size()[1]-(2*frameWidth))
    innerBackground = pg.Surface(innerBackgroundSize)
    innerBackground.convert()
    innerBackground.fill((15, 15, 15)) #slightly lighter colour for inner backgroud fill
    #screen.blit(innerBackground, (frameWidth, frameWidth))
    pg.display.flip()
    
    #main running code
    for atom in atomInfo.values():
        atom["radius"] *= options.get("Atom size scale factor") #scales up/down atom sizes as per scale factor
    avgMr = avgMrFromGenInfo(substanceGenInfo, atomInfo)
    TStart = options.get("Starting temperature (K)")
    velocityDisplayScale = options.get("Velocity display scale factor")
    iterations = options.get("Iterations")
    saveInvalidLewisStructures = options.get("Save invalid Lewis structures")
    dSF = options.get("Lewis structure display scale factor")
    substanceParamsMoleculeFields = ["substanceType", "atomNumber", "x", "y", "xdir", "ydir", "substanceID", "angleStartPos",
                                 "rotationDirection", "frameN", "recentColl", "frameWidth", "velocityDisplayScale",
                                 "centralAtom", "iterations", "substanceTypeInSubstances?", "boxes"]
    physicalParamsFields = ["dimensions", "atomInfo", "frameWidth", "substanceGenInfo", "TStart", "avgMr", "velocityDisplayScale",
                        "iterations", "substanceParamsMoleculeFields", "boxes", "grid"]
    simulationParamsFields = ["substances", "toggleReaction", "frameN", "substanceID", "reactionInfo", "rollingData",
                              "framerate", "going", "showPlot", "timeInterval", "EaMethod"]
    physicalParams = dict(zip(physicalParamsFields, [dimensions, atomInfo, frameWidth, substanceGenInfo, TStart, avgMr, velocityDisplayScale,
                                             iterations, substanceParamsMoleculeFields, boxes, grid]))
    physicalParams["boxes"], physicalParams["grid"] = gridGen(physicalParams, physicalParamsFields) #generates boxes + grid
    substances, substanceID = substanceSetup(physicalParams, physicalParamsFields) #setup for first frame
    outerBackground.fill((0, 0, 0)) #screen fill colour
    screen.blit(outerBackground, (0, 0)) #puts outerBackground on screen
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
    browseIndex = 0
    simulationParams = dict(zip(simulationParamsFields, [substances, toggleReaction, frameN, substanceID, reactionInfo,
                                            rollingData, framerate, going, showPlot, timeInterval, EaMethod]))
    if showPlot:
        for reaction in reactionInfo.values():
            reactants = reaction["reactants"].split("+")
            products = reaction["products"].split("+")
            Yflag = reaction["study for second"]
            if Yflag == "Yf": #studying forwards reaction
                substanceOfInterest = reactants[0].strip()
            elif Yflag == "Yb": #studying backwards reaction
                substanceOfInterest = products[0].strip()
        plt.ion()  #turn on interactive plots
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        translatedSubstanceOfInterest = substanceOfInterest.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")) #turns numbers into subscript versions
        scatter = ax1.scatter([], [], label="1/["+translatedSubstanceOfInterest+"]", color="purple", s=10, zorder = 1)
        scatter2 = ax2.scatter([], [], color="green", label="["+translatedSubstanceOfInterest+"] (count)", s = 10, zorder = 0)
        lineFit, = ax1.plot([], [], label="1/["+translatedSubstanceOfInterest+"] best fit", color="purple", linestyle="--", zorder = 2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("1/["+translatedSubstanceOfInterest+"]", color = "purple")
        ax1.tick_params(axis='y', labelcolor="purple")
        ax1.spines["left"].set_color("purple")
        plt.title("1/["+translatedSubstanceOfInterest+"] and ["+translatedSubstanceOfInterest+"] vs time")
        ax2.set_ylabel("["+translatedSubstanceOfInterest+"] (count)", color="green")
        ax2.tick_params(axis='y', labelcolor="green")
        ax2.spines["right"].set_color("green")
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        textAnnotation = ax1.text(0.3, 0.95, "", transform=ax1.transAxes, fontsize=10, verticalalignment='top')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right") #Combine and create a single legend on ax1
    else:
        fig, ax1, ax2, scatter, scatter2, lineFit = 0, 0, 0, 0, 0, 0
    plotParams = dict(zip(["fig", "ax1", "ax2", "scatter", "scatter2", "lineFit", "substanceOfInterest", "textAnnotation"], [fig, ax1, ax2, scatter, scatter2, lineFit, substanceOfInterest, textAnnotation]))

    clock = pg.time.Clock()
    while simulationParams["going"]:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                simulationParams["going"] = False
            if event.type == pg.KEYDOWN: #if a button is pressed
                if event.key == pg.K_ESCAPE: #if escape key pressed
##                    df = pd.DataFrame(outputList)
                   # df.to_excel("substanceCounts.xlsx", index=False)
                    simulationParams["going"] = False   #quits simulation
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
        frameProcessing(physicalParams, physicalParamsFields, simulationParams, simulationParamsFields, plotParams) #processes current frame
        substances.draw(screen) #draws substances on the screen
        pg.display.flip()
        substancesList = list(substances)
        simulationParams["frameN"]+=1
            

if __name__ == "__main__":
    main()
pg.quit()
t = 1

#try recording and then slow down, check if there is a lag spike when plotting - may want to see if you can smooth this out.

#references:
#pygame examples: https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/documentation/pygame/pygame_collision_and_intesection.md
#example website: https://phet.colorado.edu/sims/html/gas-properties/latest/gas-properties_all.html
#collisions: https://stackoverflow.com/questions/29640685/how-do-i-detect-collision-in-pygame
#collisions 2: https://github.com/rafael-fuente/Ideal-Gas-Simulation-To-Verify-Maxwell-Boltzmann-distribution/blob/master/Ideal%20Gas%20simulation%20code.py
#angle: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
