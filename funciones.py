# -*- coding: utf-8 -*-
import numpy as np

def resolver_pde(vec_t, vec_x, lgm, swaption):
    """
    Resuelve la PDE con método implícito bajo LGM.

    Parámetros:
    - vec_t: np.array, tiempos (incluye t=0)
    - vec_x: np.array, estados
    - lgm: objeto LGM
    - swaption: objeto SwaptionBermuda

    Retorna:
    - sol: matriz Nx x Nt con la evolución temporal
    - valor_0_0: valor interpolado en t=0, x=0
    """
    Nx = len(vec_x)
    Nt = len(vec_t)
    dx = vec_x[1] - vec_x[0]
    call_dates = swaption.call_dates
    # Condición terminal en t = T
    T = vec_t[-1]
    valores = swaption.valor_swap(t=T, x=vec_x)
    payoff_terminal = np.maximum(valores, 0.0)
    sol = np.zeros((Nx, Nt))
    sol[:, -1] = valores

    # Resolver hacia atrás
    for i in reversed(range(Nt - 1)):
        t0 = vec_t[i]
        t1 = vec_t[i + 1]
        dt = t1 - t0

        z1 = lgm.zeta(t1)
        z0 = lgm.zeta(t0)
        a = 0.5 * (z1 - z0) / dt

        alpha = dt * a / (dx ** 2)

        # Construcción de la matriz tridiagonal
        lo = -alpha * np.ones(Nx - 1)
        di = (1 + 2 * alpha) * np.ones(Nx)
        up = -alpha * np.ones(Nx - 1)

        # Fronteras (derivada segunda cero)
        di[0] = di[-1] = 1
        lo[0] = up[-1] = 0

        # RHS = u_{i+1}
        rhs = sol[:, i + 1].copy()
        rhs[0] -= lo[0] * 0.0  # U_L
        rhs[-1] -= up[-1] * 0.0  # U_R

        # Resolver sistema lineal
        sol[:, i] = resolver_tridiagonal(lo, di, up, rhs)

        # Condición Bermudan: early exercise
        if np.any(np.isclose(t0, call_dates, atol=1e-8)):
            payoff = swaption.valor_swap(t=t0, x=vec_x)
            sol[:, i] = np.maximum(sol[:, i], payoff)

    precio = np.interp(0.0, vec_x, sol[:, 0])
    return precio


def resolver_tridiagonal(a, b, c, d):
    """
    Thomas algorithm para resolver Ax = d con A tridiagonal (a=inf, b=diag, c=sup)
    """
    n = len(d)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    dp[-1] = (d[-1] - a[-2] * dp[-2]) / (b[-1] - a[-2] * cp[-2])

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in reversed(range(n - 1)):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def crear_malla_LGM(T, N_t, fechas_call, varianza, N_x, conf_interval=4):
    """
    Inicializa malla temporal y espacial para PDE bajo LGM incluyendo fechas de ejercicio si es necesario.

    Parámetros:
    - T: float, horizonte temporal en años.
    - N_t: int, número de pasos temporales.
    - varianza: float, variza asumida
    - N_x: int, número de nodos espaciales.
    - fechas_call: lista de floats, fechas que deben incluirse sí o sí en el mallado temporal.
    - conf_interval: número de desviaciones estándar para cubrir en vec_x.

    Retorna:
    - vec_t: vector de tiempos (incluyendo fechas críticas).
    - vec_x: vector de estados.
    - dt: paso temporal.
    - dx: paso espacial.
    """
    vec_t_base = np.linspace(0, T, N_t + 1)
    vec_t = np.union1d(vec_t_base, fechas_call)

    sigma = np.sqrt(varianza)
    max_std = conf_interval * sigma
    vec_x = np.linspace(-max_std, max_std, N_x)

    return vec_t, vec_x


def integrando(x, T, zeta_T, swaption):
    payoff = max(swaption.valor_swap(t=T, x=x), 0.0)
    densidad = np.exp(-0.5 * x**2 / zeta_T) / np.sqrt(2 * np.pi * zeta_T)
    return payoff * densidad

def integrar_swaption_lgm(zeta, swaption, T, num_puntos=201, tipo="receiver"):
    """
    Evalúa la fórmula integral para una swaption europea receiver o payer bajo LGM sin usar scipy.

    Parámetros:
    - zeta: varianza acumulada zeta(T)
    - swaption: objeto con método .valor_swap(t, x)
    - T: tiempo de ejercicio
    - num_puntos: número de nodos de integración
    - tipo: 'receiver' o 'payer'

    Retorna:
    - precio: valor aproximado de la swaption (escalar)
    """
    std = np.sqrt(zeta)
    x_grid = np.linspace(-4 * std, 4 * std, num_puntos)
    dx = x_grid[1] - x_grid[0]

    # Evaluar el payoff en cada punto (forzando a escalar)
    if tipo == "receiver":
        payoffs = np.array([max(float(swaption.valor_swap(T, x)), 0.0) for x in x_grid])
    elif tipo == "payer":
        payoffs = np.array([max(-float(swaption.valor_swap(T, x)), 0.0) for x in x_grid])
    else:
        raise ValueError("tipo debe ser 'receiver' o 'payer'")

    # Evaluar densidad gaussiana
    densidad = (1 / (np.sqrt(2 * np.pi * zeta))) * np.exp(-0.5 * x_grid**2 / zeta)

    integrando = payoffs * densidad
    precio = np.trapz(integrando, x_grid)  # Esto será ahora un escalar

    return precio