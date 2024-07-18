import math
import random
import numpy as np
import scipy
import scipy.integrate


# Постоянные моделей
##вероятности переходов состояний канальных рецепторов

k_mod={'C02': 4.59, '2C0': 4.26, '23': 2.84, '32': 3.26, '35': 4.24, '53': 0.9, '26': 2.89, '62': 0.0392,
  '64': 1.27, '46': 0.0457, '34': 0.172, '43': 0.00727, '47': 0.0168, '74': 0.1904, '57': 0.0177, '75': 0.004,
  'C0N8': 10, '8C0N': 0.0047, '89': 5, '98': 0.0094, '911': 0.0465, '119': 0.0916, '910': 0.0084, '109': 0.0018}

N=4000  # число визикул глютомата возле одного синаптического тела
#постоянные модели перемещения визикул
stP_min_1=0.41
stP_max_1=0.88
stP_min_2=6*10**(-7)
#stP_max_2=0.04 # этот параметр stP_max_2=stP_2*lam_2*N, lam_2*stP_max_2*N<=420 -- поток везикулб 420 -- опытное число измеренное Лензи
lam_2max=14
stP_max_2=420/lam_2max/N
#'S': 7.71
koef_vezicular={'V12': -33.41, 'S': 15,
                'stP_min_1': stP_min_1, 'stP_max_1': stP_max_1, 
                'stP_min_2': stP_min_2, 'stP_max_2': stP_max_2, 
                'stP_min_3': 1-stP_max_1-stP_min_2, 'stP_max_3': 1-stP_min_1-stP_max_2,
                'lam_2max': lam_2max, 'lam_2min': 1,
               }

#Вероятности нахождения рецептора в каждом из состояний
def stat_P_ij(V, ij, koef_vezicular):
    a='stP_min_'+str(ij)
    b='stP_max_'+str(ij)
    stP_min=koef_vezicular[a]
    stP_max=koef_vezicular[b]
    c=koef_vezicular[a]+(koef_vezicular[b]-koef_vezicular[a])/(1+math.exp(-(-V-koef_vezicular['V12'])/koef_vezicular['S']))
    return c

#Интенсивности пререходов между стостояниями
def lam_2(V, koef_vezicular):
    l2=koef_vezicular['lam_2min']+(koef_vezicular['lam_2max']-koef_vezicular['lam_2min'])/(1+math.exp(-(V-koef_vezicular['V12'])/koef_vezicular['S']))
    return l2/1000
def lam_1(V, koef_vezicular):
    return lam_2(V, koef_vezicular)*stat_P_ij(V, 2, koef_vezicular)/stat_P_ij(V, 1, koef_vezicular)
def lam_3(V, koef_vezicular):
    P1=stat_P_ij(V, 1, koef_vezicular)
    P2=stat_P_ij(V, 2, koef_vezicular)
    P3=1-P1-P2
    return lam_2(V, koef_vezicular)*stat_P_ij(V,2, koef_vezicular)/P3

def model_Kolmogorov_vesicular_stats_on_Vconst(t0, tend, dt, V,  x10=0.9, x20=0.01,):
    l_2=lam_2(V, koef_vezicular)
    l_1=lam_1(V, koef_vezicular)
    l_3=lam_3(V, koef_vezicular)
    X=np.array([[x10], [x20]])
    args = (V, dt,l_1, l_2, l_3, )
    sol = scipy.integrate.solve_ivp(model_Kolmogorov_vesicular_stats_rp,
                                    [t0, tend], [x10, x20], args=args)
    X = []
    for x in sol.y:
        y_new=np.interp(np.arange(t0, tend, dt), sol.t, x)
        X.append(np.transpose(y_new))
    return X

def model_Kolmogorov_vesicular_stats_rp(t, y, V, dt, lam_1, lam_2, lam_3):
    x1, x2 =y
    dx1=-x1*lam_1 +(1-x1-x2)*lam_3
    dx2=-x2*lam_2+  x1*lam_1
    return dx1, dx2

# реализация случайных процессов в векторном виде
def veziculas_n_set(V, k_mod, dt, n1, n2, n3, nset, n, l1, l2, l3, x1, x2, ksi1, ksi2, ksi3):
    x3=1-x1-x2
    #krel -- наличие релиза глютомата 
    krel=0
    if  ksi1 < l1*dt*n1 and n2 < nset and n1 > 1:
        n1 = n1 - 1
        n2 = n2 + 1
        krel=1
    if ksi2 < l2*dt*n2 and n2>1:
        n2 = n2 - 1
        n3 = n3 + 1
    if ksi3 <l3*dt*n3 and n3>1:
        n3 = n3 - 1
        n1 = n1 + 1
    return n1, n2, n3, krel

## эта функция возвращает приращение концентрации и вероятность открытых/закрытых рецепторных каналов при отсутсвии расхода глютомата, только убывание за tau
#N_NDMA -- количество включенных NDMA рецепторов 

def dydt_ty_no_tau(t, y, k_mod, M, N_NDMA, tau):
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17=y
    E=M-N_NDMA-y2-y3-y4-y5-y6-y7
    C0N=N_NDMA-y8-y9-y10-y11
    ## MEMBRANA NEURON
    #CURISYN=(y5*10.0+y11*45)*(-70)*0.001 #  nA
    # GLUTAMAT:
    dy1=-y1/tau
    # AMPA :           
    dy2=-y2*(k_mod['2C0']+k_mod['23']*y1+k_mod['26'])+E*k_mod['C02']*y1+y3*k_mod['32'] +y6*k_mod['62']
    dy3=-y3*(k_mod['32'] +k_mod['34']   +k_mod['35'])+y2*k_mod['23']*y1+y4*k_mod['43'] +y5*k_mod['53']
    dy4=-y4*(k_mod['43'] +k_mod['46']   +k_mod['47'])+y3*k_mod['34']+y6*k_mod['64']*y1+y7*k_mod['74']
    dy5=-y5*(k_mod['53']+k_mod['57'])+y3*k_mod['35']+y7*k_mod['75']
    dy6=-y6*(k_mod['62']+k_mod['64']*y1)+k_mod['46']*y4+k_mod['26']*y2
    dy7=-y7*(k_mod['74']+k_mod['75'])+k_mod['57']*y5+k_mod['47']*y4                             
    # NMDA:
    dy8=-y8*(k_mod['8C0N']+k_mod['89']*y1)+k_mod['C0N8']*y1*C0N + k_mod['98']*y9
    dy9=-y9*(k_mod['98']+k_mod['910']+k_mod['911'])+k_mod['89']*y1*y8+k_mod['119']*y11+k_mod['109']*y10
    dy10=-y10*k_mod['109']+k_mod['910']*y9
    dy11=-y11*k_mod['119']+k_mod['911']*y9
    
    # модификация НВК формул Джонсона
    tauAMPA=0.75
    tauNMDA=20
    
    dy12=y13
    dy13=-y13*(2./tauAMPA)-(1./(tauAMPA*tauAMPA))*y12
    dy14=y15
    dy15=-y15*(2./tauNMDA)-(1./(tauNMDA*tauNMDA))*y14
    
    dy16=y11
    dy17=y5
    return [dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9, dy10, dy11, dy12, dy13]

## эта функция возвращает приращение концентрации и вероятность открытых/закрытых рецепторных каналов
#N_NDMA -- количество включенных NDMA рецепторов
def dydt_ty(t, y, k_mod, M, N_NDMA, tau):
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13=y
    E=M-y2-y3-y4-y5-y6-y7
    C0N=N_NDMA-y8-y9-y10-y11
    # GLUTAMAT:
    dy1=-y1/tau -y1*(k_mod['C02']*E+k_mod['23']*y2+k_mod['64']*y6+k_mod['C0N8']*C0N+k_mod['89']*y8);
    # AMPA :           
    dy2=-y2*(k_mod['2C0']+k_mod['23']*y1+k_mod['26'])+E*k_mod['C02']*y1+y3*k_mod['32'] +y6*k_mod['62']
    dy3=-y3*(k_mod['32'] +k_mod['34']   +k_mod['35'])+y2*k_mod['23']*y1+y4*k_mod['43'] +y5*k_mod['53']
    dy4=-y4*(k_mod['43'] +k_mod['46']   +k_mod['47'])+y3*k_mod['34']+y6*k_mod['64']*y1+y7*k_mod['74']
    dy5=-y5*(k_mod['53']+k_mod['57'])+y3*k_mod['35']+y7*k_mod['75']
    dy6=-y6*(k_mod['62']+k_mod['64']*y1)+k_mod['46']*y4+k_mod['26']*y2
    dy7=-y7*(k_mod['74']+k_mod['75'])+k_mod['57']*y5+k_mod['47']*y4                             
    # NMDA:
    dy8=-y8*(k_mod['8C0N']+k_mod['89']*y1)+k_mod['C0N8']*y1*C0N + k_mod['98']*y9
    dy9=-y9*(k_mod['98']+k_mod['910']+k_mod['911'])+k_mod['89']*y1*y8+k_mod['119']*y11+k_mod['109']*y10
    dy10=-y10*k_mod['109']+k_mod['910']*y9
    dy11=-y11*k_mod['119']+k_mod['911']*y9
    
# модификация НВК формул Джонсона// для сравнения моделей
 #   tauAMPA=0.75
 #   tauNMDA=20
    
 #   dy12=y13;
 #   dy13=-y13*(2./tauAMPA)-(1./(tauAMPA*tauAMPA))*y12;
 #   dy14=y15;
 #   dy15=-y15*(2./tauNMDA)-(1./(tauNMDA*tauNMDA))*y14;
    
    dy12=y11
    dy13=y5
    return [dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9, dy10, dy11, dy12, dy13]

# Токи вестибулярной клетки
#Описание тока I_K_Low через 3 состояния, 2 ворот с независимыми интенсивностями

def alpha_1_K_Low(V):
    return 0.4/(1+np.exp(-(V+36.8)/8.3)) # alfa3 у НВК
    
def alpha_2_K_Low(V):
    return 0.320513 # alfa4  у НВК

def beta_1_K_Low(V):
    return 0.072*np.exp(-V/45.48) # beta3 у НВК
    
def beta_2_K_Low(V):
    return 0.0083*np.exp(-V/41.55) # beta4 у НВК

def p_K_Low(t, y, V, alpha1, alpha2, beta1,beta2):
    x1,x2, x3 = y
    dx1= -alpha1*x1 + beta1*x2
    dx2= -(alpha2+beta1)*x2 + alpha1*x1 +beta2*x3 #(alpha1-beta2)*x1 + (-alpha2-beta1-beta2)*x2 + beta2
    dx3= alpha2*x2 - beta2 * x3
    return dx1, dx2, dx3

def K_low_HC_typeI_P_C_ankr(t, y, V, G_K=10, RTF=25, C_HC_cl=140): # для интенсивностей alpha1, alpha2, beta1,beta2 
    # учтено изменение концентрации в щели
    # x1 x2 x3 вероятности состояний x3 -- Open
    x1,x2, x3, C_K_cl = y
    alpha1, alpha2, beta1,beta2 = alpha_1_K_Low(V), alpha_2_K_Low(V), beta_1_K_Low(V), beta_2_K_Low(V)
    dx1= -alpha1*x1 + beta1*x2
    dx2= -(alpha2+beta1)*x2 + alpha1*x1 +beta2*x3
    dx3= alpha2*x2 - beta2 * x3
    E_K_cl=RTF * np.log(C_K_cl/C_HC_cl)
    dC_K_cl=G_K*x3*(V-E_K_cl) #I_K_Low
    return dx1, dx2, dx3, dC_K_cl


