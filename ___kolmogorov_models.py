import math
import random
import numpy as np
import scipy
import scipy.integrate

def state_x2_solution(t, alpha, beta, P0):
    """
    Вычисляет аналитическое решение для вероятности открытого состояния P^2(t)
    в двухсостоянийной модели с переходами S1 ⇄ S2.
    Уравнение:
        dP^2/dt = α (1 - P^2) - β P^2
        P^2(0) = P0
    Решение:
        P^2(t) = α / (α + β) + (P0 - α / (α + β)) * exp(- (α + β) * t)
    Параметры:
    ----------
    t : float или np.ndarray
        Время (или массив времени).
    alpha : float
        Скорость перехода из закрытого состояния в открытое (α > 0).
    beta : float
        Скорость перехода из открытого состояния в закрытое (β > 0).
    P0 : float
        Начальное значение вероятности открытого состояния (0 ≤ P0 ≤ 1).
    Возвращает:
    -----------
    P2_t : float или np.ndarray
        Вероятность открытого состояния в момент времени t.
    """
    P_inf = alpha / (alpha + beta)
    return P_inf + (P0 - P_inf) * np.exp(-(alpha + beta) * t)

def state_x2_rph(t, P2, V_func, alpha_func, beta_func):
    """
    Правая часть ОДУ для модели с двумя состояниями ворот (S1 ⇄ S2),
    где параметры α и β зависят от внешнего напряжения V(t).

    Уравнение:
        dP2/dt = α(V) * (1 - P2) - β(V) * P2

    Параметры:
    ----------
    t : float
        Время.
    P2 : float
        Текущее значение P^2(t) — вероятность открытого состояния.
    V_func : callable
        Функция напряжения V(t), должна возвращать скаляр.
    alpha_func : callable
        Функция α(V), зависящая от напряжения.
    beta_func : callable
        Функция β(V), зависящая от напряжения.
    Возвращает:
    -----------
    dP2dt : float
        Значение производной dP2/dt.
    """
    V = V_func(t)
    alpha = alpha_func(V)
    beta = beta_func(V)
    return alpha * (1 - P2) - beta * P2
    

def state_x3_independ_ab_rph(t, y, V_func, alpha1_func, alpha2_func, beta1_func, beta2_func):
    """
    Правая часть системы ОДУ для трех состояний ворот:
        S1 ⇄ S2 ⇄ S3, и независимых интенсевностей перехода
    где параметры переходов зависят от напряжения V(t).
    Переменные:
    ----------
    t : float
        Время
    y : array-like
        Вектор состояния [P2, P3]
    V_func : callable
        Функция напряжения V(t)
    alpha1_func, alpha2_func : callable
        Функции α1(V), α2(V)
    beta1_func, beta2_func : callable
        Функции β1(V), β2(V)
    Возвращает:
    -----------
    dydt : list[float]
        Правая часть системы [dP2/dt, dP3/dt]
    """
    P2, P3 = y
    V = V_func(t)

    # Вычисление параметров
    alpha1 = alpha1_func(V)
    alpha2 = alpha2_func(V)
    beta1  = beta1_func(V)
    beta2  = beta2_func(V)

    # Уравнения
    dP2dt = alpha1 * (1 - P2 - P3) + beta2 * P3 + (-alpha2 - beta1) * P2
    dP3dt = alpha2 * P2 - beta2 * P3

    return [dP2dt, dP3dt]

def state_x3_depend_ab_rph(t, y, V_func, alpha_func, beta_func):
    """
    Правая часть ОДУ для трех состояний ворот с кратными интенсивностями:
        S1 ⇄ S2 ⇄ S3, с коэффициентами 3α, 2α, α и β, 2β, 3β
    Параметры:
    ----------
    t : float
        Время
    y : array-like
        Вектор состояния [P2, P3]
    V_func : callable
        Функция напряжения V(t)
    alpha_func : callable
        Функция α(V)
    beta_func : callable
        Функция β(V)
    Возвращает:
    -----------
    dydt : list[float]
        Производные [dP2/dt, dP3/dt]
    """
    P2, P3 = y
    V = V_func(t)

    # Переходные коэффициенты
    alpha = alpha_func(V)
    beta = beta_func(V)

    # Уравнения
    dP2dt = 2 * alpha * (1 - P2 - P3) + 2 * beta * P3 + (-alpha - beta) * P2
    dP3dt = alpha * P2 - 2 * beta * P3

    return [dP2dt, dP3dt]



