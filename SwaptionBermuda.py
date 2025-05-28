import numpy as np

class SwaptionBermuda:
    """
    Clase para la definición y valoración de una Swaption Bermudan.

    Parámetros:
    - notional: float
        Nocional del swap.
    - strike: float
        Tasa fija del swap.
    - call_dates: list of float
        Fechas de ejercicio de la opción Bermudan (en años).
    - fixed_leg_start_dates: list of float
        Fechas de inicio de cada período fijo.
    - fixed_leg_end_dates: list of float
        Fechas de pago de cada período fijo.
    - float_leg_start_dates: list of float
        Fechas de inicio de cada período variable.
    - float_leg_end_dates: list of float
        Fechas de pago de cada período variable.
    - modelo: object
        Modelo de tasas con método .precio_bono(t, T, x).

    Notas:
    - Las fechas pueden ser listas o arrays de numpy; internamente se convierten a np.ndarray.
    - Los factores de devengo (DCF) se calculan como la diferencia entre fechas fin e inicio.
    """
    def __init__(self, notional, strike, call_dates, fixed_leg_start_dates,
        fixed_leg_end_dates, float_leg_start_dates, float_leg_end_dates,
        modelo):
        self.notional = notional
        self.strike = strike
        self.call_dates = np.array(call_dates)
        self.fixed_leg_start_dates = np.array(fixed_leg_start_dates)
        self.fixed_leg_end_dates = np.array(fixed_leg_end_dates)
        self.float_leg_start_dates = np.array(float_leg_start_dates)
        self.float_leg_end_dates = np.array(float_leg_end_dates)
        self.dcf_fixed = self.fixed_leg_end_dates - self.fixed_leg_start_dates
        self.dcf_float = self.float_leg_end_dates - self.float_leg_start_dates
        self.modelo = modelo

    def _precio_bono(self, t, T, x):
        return self.modelo.precio_bono(t, T, x)

    def valor_swap_fijo(self, t=0.0, x=0.0):
        valor = 0.0
        for i, T in enumerate(self.fixed_leg_end_dates):
            if T > t:
                bono = self._precio_bono(t, T, x)
                valor += self.dcf_fixed[i] * bono
        return self.notional * self.strike * valor

    def valor_swap_flotante(self, t=0.0, x=0.0):
        # Fechas de inicio y fin de cada tramo flotante
        fechas_ini = self.float_leg_start_dates
        fechas_fin = self.float_leg_end_dates

        # Encuentra la próxima fecha de fijación (inicio del primer periodo ≥ t)
        futuras = [T for T in fechas_ini if T >= t]
        
        T_fijacion = futuras[0]
        T_pago = fechas_fin[-1]  # Última fecha de pago

        P_fijacion = self._precio_bono(t, T_fijacion, x)
        P_pago = self._precio_bono(t, T_pago, x)

        return self.notional * (P_fijacion - P_pago)

    def valor_swap(self, t=0.0, x=0.0):
        return self.valor_swap_fijo(t, x) - self.valor_swap_flotante(t, x)
