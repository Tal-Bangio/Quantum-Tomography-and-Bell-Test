import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

class angle:
    def MakeCycle (self, ang_def):
        if (ang_def < 0):
            ang_def_new = -(abs (ang_def) % 180)
        else:
            ang_def_new = abs (ang_def) % 180
        return (ang_def_new)
    def __init__(self, ang_def):        
        self.ang = self.MakeCycle (ang_def)
    def __repr__(self):
        return str (self.ang)
    def __add__(self, value):
        ang_def = self.ang + value
        return angle(self.MakeCycle (ang_def))
    def __iadd__(self, value):
        ang_def = self.ang + value
        return angle(self.MakeCycle (ang_def))
    def __radd__(self, value):
        ang_def = self.ang + value
        return angle(self.MakeCycle (ang_def))
    def __sub__(self, value):
        ang_def = self.ang - value
        return angle(self.MakeCycle (ang_def))
    def __isub__(self, value):
        ang_def = self.ang - value
        return angle(self.MakeCycle (ang_def))
    def __rsub__(self, value):
        ang_def = value - self.ang
        return angle(self.MakeCycle (ang_def))

def AngleParse (anglestr):
    alice_angle_pre = anglestr [1:anglestr.find ("B")]
    Bob_angle_pre = anglestr [anglestr.find ("B") + 1:len (anglestr)]
    alice_angle_pre = alice_angle_pre.replace('_', '.')
    alice_angle_pre = alice_angle_pre.replace('m', '-')
    alice_angle_pre = alice_angle_pre.replace('p', '')
    Bob_angle_pre = Bob_angle_pre.replace('_', '.')
    Bob_angle_pre = Bob_angle_pre.replace('m', '-')
    Bob_angle_pre = Bob_angle_pre.replace('p', '')
    # print (Bob_angle_pre)
    return str (angle (float (alice_angle_pre))) + ","  + str(angle (float (Bob_angle_pre)))

def AnglesToStr (angle1, angle2):
    return str (angle1) + "," + str (angle2)
def StrToAngles (angles):
    return angle (float (angles [0:angles.find (",")])), angle (float (angles [angles.find (",") + 1:len (angles)]))
    

def OpenData (name):
    xl = pd.ExcelFile(name)
    angles = xl.sheet_names
    Data = {}
    Data ["angles"] = angles
    for angle_ in angles:
        Data [angle_] = (xl.parse (angle_))
        #print (Data [angle])
    return Data
def OpenDataCW (name):
    xl = pd.read_csv(name)
    return xl

def NormalizeParameters (Data):
    Amax = max (Data ["Alice-V"])
    Amin = min (Data ["Alice-V"])
    Bmax = max (Data ["Bob-V"])
    Bmin = min (Data ["Bob-V"])
    return (Bmax - Bmin)/(Amax - Amin)

def Normalize (Data, ratio):
    DataNorm = {}
    DataNorm ["angles"] = []
    for angle_ in Data ["angles"]:
        DataNorm ["angles"].append (AngleParse (angle_))
    for angle_ in Data ["angles"]:
        DataNorm [AngleParse (angle_)] = Data [angle_]
        DataNorm [AngleParse (angle_)] = {}
        DataNorm [AngleParse (angle_)] ["Frame"] = Data [angle_]  ["Frame"]
        DataNorm [AngleParse (angle_)] ["Alice-V"] = (Data [angle_] ["Alice-V"] - min (Data [angle_] ["Alice-V"]))/(max (Data [angle_] ["Alice-V"])- min (Data [angle_] ["Alice-V"]))
        DataNorm [AngleParse (angle_)] ["Bob-V"] = (Data [angle_] ["Bob-V"] - min (Data [angle_] ["Bob-V"]))/((max (Data [angle_] ["Alice-V"])- min (Data [angle_] ["Alice-V"]))*ratio)
    return DataNorm

def CalculateE_ (Data):
    E = {}
    p = {}
    for angle_ in Data ["angles"]:
        count0 = 0
        count1 = 0
        p [angle_] = {}
        p [angle_] ["0"] = 0
        p [angle_] ["1"] = 0
        for i, cell in enumerate (Data [angle_] ["bit"]):
            if (Data [angle_] ["bit"] [i] == 0):
                p [angle_] ["0"] += Data [angle_] ["Alice-V"] [i]
                p [angle_] ["0"] += Data [angle_] ["Bob-V"] [i]
                count0 += 1
            if (Data [angle_] ["bit"]  [i] == 1):
                p [angle_] ["1"] += Data [angle_] ["Alice-V"] [i]
                p [angle_] ["1"] += Data [angle_] ["Bob-V"] [i]
                count1 += 1
        # print (p [angle_] ["0"])
        p [angle_] ["0"] = p [angle_] ["0"]/count0
        p [angle_] ["1"] = p [angle_] ["1"]/count1
        # print (p [angle_] ["0"])
    # print (Data ["angles"])
    for angle_ in Data ["angles"]:
        AngleShift1, AngleShift2 = StrToAngles (angle_)
        AngleShift = AnglesToStr (AngleShift1, AngleShift2 + 90)
        try:
            E [angle_] = (p [angle_] ["0"] + p [angle_] ["1"] - p [AngleShift] ["0"] - p [AngleShift] ["1"])/(p [angle_] ["0"] + p [angle_] ["1"] + p [AngleShift] ["0"] + p [AngleShift] ["1"])
        except KeyError:
            E [angle_] = 1
    return E

def CalculateE (Data):
    E = {}
    p = {}
    thresh = 0.4
    aliceangle = []
    bobangle = []
    n = []
    for angle_ in Data ["angles"]:
        angleA, angleB = StrToAngles (angle_)
        count0 = 0
        count1 = 0
        
        p [angle_] = {}
        p [angle_] ["0"] = 0
        p [angle_] ["1"] = 0
        for i, cell in enumerate (Data [angle_] ["Alice-V"]):
            if ((Data [angle_] ["Alice-V"] [i] >= thresh) and (Data [angle_] ["Bob-V"] [i] >= thresh)):
                p [angle_] ["0"] += 1
                count0 += 1
        print (p [angle_] ["0"])
    
        aliceangle.append (float (str (angleA)))
        bobangle.append (float (str (angleB)))
        n.append (p [angle_] ["0"])
    print (n)
    print (aliceangle)
    print (bobangle)
    # .errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=5, label='Data with error bars')
    plt.figure(figsize=(10, 6))
    plt.errorbar(aliceangle, n, xerr=1, yerr=1, fmt='o', capsize=5, label='alice')
    plt.errorbar(bobangle, n, xerr=0.1, yerr=0.1, fmt='o', capsize=5,  label='bob')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    for angle_ in Data ["angles"]:
        AngleShift1, AngleShift2 = StrToAngles (angle_)
        AngleShift = AnglesToStr (AngleShift1, AngleShift2 + 90)
        print (AngleShift1, AngleShift2, AngleShift1 + 90, AngleShift2 + 90)
        try:
            E [angle_] = (p [angle_] ["0"] + p [AnglesToStr (AngleShift1 + 90, AngleShift2 + 90)] ["0"] - p [AnglesToStr (AngleShift1, AngleShift2 + 90)] ["0"] - p [AnglesToStr (AngleShift1 + 90, AngleShift2)] ["0"])/(p [angle_] ["0"] + p [AnglesToStr (AngleShift1 + 90, AngleShift2 + 90)] ["0"] + p [AnglesToStr (AngleShift1, AngleShift2 + 90)] ["0"] + p [AnglesToStr (AngleShift1 + 90, AngleShift2)] ["0"])
        except KeyError:
            E [angle_] = None
    return E

def CalculateS (E):
    a = angle (-45.0)
    b = angle (-22.5)
    phase = 45
    print (E [AnglesToStr (a, b)])
    print (E [AnglesToStr (a, b + phase)])
    print (E [AnglesToStr (a + phase, b)])
    print (E [AnglesToStr (a + phase, b + phase)])
    return E [AnglesToStr (a, b)] - E [AnglesToStr (a, b + phase)] + E [AnglesToStr (a + phase, b)] + E [AnglesToStr (a + phase, b + phase)]

        

# theta = angle (-22.5)
# print (theta + 220)

# print (AngleParse ("Ap90Bm22_5"))

Data = OpenData (sys.argv [1])
# print (Data ["angles"])
DataCW = OpenDataCW (sys.argv [2])
DataNorm = Normalize (Data, NormalizeParameters(DataCW))
# plt.figure(figsize=(10, 6))
# plt.plot(DataNorm [DataNorm ["angles"] [0]] ["Frame"], DataNorm [DataNorm ["angles"] [0]] ["Alice-V"], label='alice')
# plt.plot(DataNorm [DataNorm ["angles"] [0]] ["Frame"], DataNorm [DataNorm ["angles"] [0]] ["Bob-V"], label='bob')
# plt.legend()
# plt.grid(True)
# plt.show()
E = CalculateE (DataNorm)
S = CalculateS (E)
print (S)