import sys
import time

import numpy as np

from ECC_Security import encrypt_ECC, decrypt_ECC
from Global_Vars import Global_Vars


def objfun(Soln):
    curve = Global_Vars.Curve
    byt = Global_Vars.Bytes
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            privkey = sol.astype(np.str)
            privkey = int(''.join(privkey))
            pubKey = privkey * curve.g
            Size = np.zeros(Global_Vars.Plain_Text.shape[0])
            Time = np.zeros(Global_Vars.Plain_Text.shape[0])
            for j in range(Global_Vars.Plain_Text.shape[0]):
                for k in range(byt):
                    ct = time.time()
                    plain = Global_Vars.Plain_Text[j, k]
                    plain_text = bytes(plain, 'utf-8')
                    encryptedMsg = encrypt_ECC(plain_text, pubKey)
                    decryptedMsg = decrypt_ECC(encryptedMsg, privkey)
                    Size[j] = Size[j] + sys.getsizeof(plain_text) + sys.getsizeof(decryptedMsg) + sys.getsizeof(encryptedMsg)
                    Time[j] = Time[j] + (time.time() - ct)
            Fitn[i] = np.mean(Size) + np.mean(Time)
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        privkey = sol.astype(np.str)
        privkey = int(''.join(privkey))
        pubKey = privkey * curve.g
        Size = np.zeros(Global_Vars.Plain_Text.shape[0])
        Time = np.zeros(Global_Vars.Plain_Text.shape[0])
        for j in range(Global_Vars.Plain_Text.shape[0]):
            for k in range(byt):
                ct = time.time()
                plain = Global_Vars.Plain_Text[j, k]
                plain_text = bytes(plain, 'utf-8')
                encryptedMsg = encrypt_ECC(plain_text, pubKey)
                decryptedMsg = decrypt_ECC(encryptedMsg, privkey)
                Size[j] = Size[j] + sys.getsizeof(plain_text) + sys.getsizeof(decryptedMsg) + sys.getsizeof(encryptedMsg)
                Time[j] = Time[j] + (time.time() - ct)
        Fit = np.mean(Size) + np.mean(Time)
        return Fit
