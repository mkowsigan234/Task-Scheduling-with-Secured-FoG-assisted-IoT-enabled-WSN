import csv
import random as rn
import secrets

import numpy as np
from tinyec import registry
from tinyec.ec import SubGroup, Curve

from BA import BA
from BWO import BWO
from ECC_Security import encrypt_ECC, decrypt_ECC
from Global_Vars import Global_Vars
from Global_Vars import Task
from Jaya import Jaya
from MFO import MFO
from Objfun import objfun
from Plot_Results import Plot_Results
from Proposed import Proposed


def Initialization(vary=12):
    noOfTasks = vary
    taskLengths = np.arange(vary) * 100
    noOfVMs = vary * 5
    taskWorkLoad = taskLengths * np.random.randint(low=1000, high=10000, size=taskLengths.shape[0])
    half = np.ones(int(noOfVMs // 2))
    VmMIPS = np.append(half * 2000, half * 4000)
    Cost = np.append(half * 200, half * 400)
    task = Task()
    task.noOfTasks = noOfTasks
    task.taskLengths = taskLengths
    task.noOfVMs = noOfVMs
    task.taskWorkLoad = taskWorkLoad
    task.VmMIPS = VmMIPS
    task.Cost = Cost
    return task


def ConvertTextToBinary(plain):
    binary = ''
    for j in range(len(plain)):
        val = plain[j]
        bin_ = '{0:016b}'.format(ord(val))
        binary = binary + bin_
    return binary


def ConvertNumberToBinary(plain):
    binary = ''
    for j in range(len(plain)):
        val = plain[j]
        bin_ = '{0:016b}'.format(ord(chr(val)))
        binary = binary + bin_
    return binary


# Initialization
an = 0
if an == 1:
    variation = [10, 20, 30, 40, 50]
    Tasks = []
    for i in variation:
        Tasks.append(Initialization(i))
    np.save('Tasks.npy', Tasks)

# Read Dataset
an = 0
if an == 1:
    Files = ['./Dataset/train.csv', './Dataset/test.csv']
    Data = []
    for n in range(len(Files)):
        with open(Files[n], mode='r', encoding="utf8") as file:
            csvFile = csv.reader(file)
            CSV = []
            count = 0
            # displaying the contents of the CSV file
            for lines in csvFile:
                if count == 0:
                    pass
                else:
                    if len(lines) == 10:
                        CSV.append(lines[:9])
                    else:
                        CSV.append(lines)
                count += 1
        Data.append(CSV)
    CSV1 = np.asarray(Data[0])
    CSV2 = np.asarray(Data[1])
    Plain_Text = np.append(CSV1, CSV2, axis=0)
    np.save('Plain_Text.npy', Plain_Text)

# Generate Elliptic Curve for Cryptography
an = 0
if an == 1:
    curve1 = registry.get_curve('brainpoolP256r1')
    curve2 = registry.get_curve('secp192r1')
    field = SubGroup(p=17, g=(15, 13), n=18, h=1)
    curve3 = Curve(a=0, b=7, field=field, name='p1707')
    Curves = [curve1, curve2, curve3]
    np.save('Curves.npy', Curves)

# Encryption and Decryption
an = 0
if an == 1:
    curve = registry.get_curve('brainpoolP256r1')
    privKey = secrets.randbelow(curve.field.n)
    pubKey = privKey * curve.g
    Plain_Text = np.load('Plain_Text.npy', allow_pickle=True)
    for n in range(Plain_Text.shape[0]):
        for i in range(Plain_Text.shape[1]):
            print(n, i)
            plain_text = Plain_Text[n, i]
            plain_text = bytes(plain_text, 'utf-8')
            encryptedMsg = encrypt_ECC(plain_text, pubKey)
            decryptedMsg = decrypt_ECC(encryptedMsg, privKey)
            print(plain_text, decryptedMsg)

# Optimization of Private Key
an = 0
if an == 1:
    Bytes_Vary = [3, 6, 9]
    Curves = np.load('Curves.npy', allow_pickle=True)
    Plain_Text = np.load('Plain_Text.npy', allow_pickle=True)
    Bestsol_Curve = []
    Fitness_Curve = []
    for curve in range(len(Curves)):
        Bestsol_byte = []
        Fitness_byte = []
        for byt in range(len(Bytes_Vary)):
            Curve = Curves[curve]
            Global_Vars.Curve = Curves[curve]
            Global_Vars.Plain_Text = Plain_Text
            Global_Vars.Bytes = Bytes_Vary[byt]
            privKey = secrets.randbelow(Curve.field.n)

            Npop = 10
            Chlen = len(str(privKey))
            xmin = np.zeros((Npop, Chlen))
            xmax = 9 * np.ones((Npop, Chlen))
            initsol = np.zeros((Npop, Chlen))
            for i in range(xmin.shape[0]):
                for j in range(xmin.shape[1]):
                    initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
            fname = objfun
            Max_iter = 20

            print("Jaya...")
            [bestfit1, fitness1, bestsol1, time] = Jaya(initsol, fname, xmin, xmax, Max_iter)

            print("MFO...")
            [bestfit2, fitness2, bestsol2, time] = MFO(initsol, fname, xmin, xmax, Max_iter)

            print("BWO...")
            [bestfit3, fitness3, bestsol3, time] = BWO(initsol, fname, xmin, xmax, Max_iter)

            print("BA...")
            [bestfit4, fitness4, bestsol4, time] = BA(initsol, fname, xmin, xmax, Max_iter)

            print("Proposed...")
            [bestfit5, fitness5, bestsol5, time] = Proposed(initsol, fname, xmin, xmax, Max_iter)

            fitness1 = np.reshape(fitness1, (fitness1.shape[0], 1))
            fitness2 = np.reshape(fitness2, (fitness2.shape[0], 1))
            fitness3 = np.reshape(fitness3, (fitness3.shape[0], 1))
            fitness4 = np.reshape(fitness4, (fitness4.shape[0], 1))
            fitness5 = np.reshape(fitness5, (fitness5.shape[0], 1))
            Fitness_byte.append([fitness1, fitness2, fitness3, fitness4, fitness5])
            Bestsol_byte.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        Bestsol_Curve.append(Bestsol_byte)
        Fitness_Curve.append(Fitness_byte)
    np.save('BestSol.npy', Bestsol_Curve)
    np.save('Fitness.npy', Fitness_Curve)

# Key Optimized Encryption and Decryption
an = 0
if an == 1:
    Bytes_Vary = [3, 6, 9]
    Curves = np.load('Curves.npy', allow_pickle=True)
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    curve = registry.get_curve('brainpoolP256r1')
    Plain_Text = np.load('Plain_Text.npy', allow_pickle=True)
    Encrypted_Text = []
    Decrypted_Text = []
    for soli in range(BestSol.shape[0]):
        for solj in range(BestSol.shape[1]):
            print(soli, solj)
            for solk in range(BestSol.shape[2]):
                sol = np.round(BestSol[soli, solj, solk]).astype(np.int16)
                privkey = sol.astype(np.str)
                privkey = int(''.join(privkey))
                pubKey = privkey * curve.g
                Encrypted = []
                Decrypted = []
                for n in range(Plain_Text.shape[0]):
                    Encrypt = []
                    Decrypt = []
                    for i in range(Plain_Text.shape[1]):
                        # print(soli, solj, solk, n, i)
                        plain_text = Plain_Text[n, i]
                        plain_text = bytes(plain_text, 'utf-8')
                        encryptedMsg = encrypt_ECC(plain_text, pubKey)
                        decryptedMsg = decrypt_ECC(encryptedMsg, privkey)
                        Encrypt.append(encryptedMsg)
                        Decrypt.append(decryptedMsg)
                    Encrypted.append(Encrypt)
                    Decrypted.append(Decrypt)
                Encrypted_Text.append(Encrypted)
                Decrypted_Text.append(Decrypted)
    np.save('Encrypted_Text.npy', Encrypted_Text)
    np.save('Decrypted_Text.npy', Decrypted_Text)

Plot_Results()
