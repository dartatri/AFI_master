# -*- coding: utf-8 -*-

import numpy as np

class Lgm:
    def __init__(self, curva, tiempos_zeta, valores_zeta, a):
        self.curva = curva  # instancia de Curva
        self.tiempos_zeta = np.array(tiempos_zeta)
        self.valores_zeta = np.array(valores_zeta)
        self.a = a

    def zeta(self, t):
        t = np.atleast_1d(t)
        return np.interp(t, self.tiempos_zeta, self.valores_zeta)

    def H(self, t):
        return (1 - np.exp(-self.a * t)) / self.a 

    def precio_bono(self, t, T, x_t):
        H_val = self.H(T)
        P_t = self.curva.interpola(T)
        zeta_t = self.zeta(t)
        return P_t * np.exp(-H_val * x_t - 0.5 * (H_val ** 2) * zeta_t)