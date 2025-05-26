# -*- coding: utf-8 -*-
import numpy as np

class Curva:
    def __init__(self, tiempos, descuentos):
        self.tiempos = np.array(tiempos)
        self.descuentos = np.array(descuentos)

    def interpola(self, t_query):
        t_query = np.atleast_1d(t_query)
        return np.interp(t_query, self.tiempos, self.descuentos)

