
######################################################################################################################
# Nataly Martínez Riascos 
# Departamento de Matemáticas de la Universidad del Valle
# 2021
######################################################################################################################

import numpy as np
import math 
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


print ('Convergencia y cosistencia del sistema de Wanwright-Hsu para modelos de Bianchi clase A tipo I-VIII')
input()

######################################################################################################################
#                                                   IMPLEMENTACIÓN DEL PROGRAMA
######################################################################################################################



#---------------------------------------------------METODO RUNGE-KUTTA 4 (RK4)----------------------------------------

def RungeKutta4(y0, a,b, h, f, Constraint, Lambda):     # definimos R-K de orden 4
     
    y = y0                                         # valores inciales
    t = a                                          # tiempo inicial
    T = np.arange(a,b+h,h)                         # array de los tiempos
    n = len(T)                                     # longitud del array
    Y = np.array([y])                              # array de la soluciones
    C , Q , OK = Constraint(y,Lambda)  
    constraint, q, Omega_k =np.array([C]), np.array([Q]),np.array([OK])   # Funcion que calcula si el constraint se cumple

     

    for i in range(n-1):                           # empezamos con el algoritmo para hallar las soluciones
        
        k1 = h*f(t,y,Lambda)                       # hallamos los k's que se necesitan para el metodo
        k2 = h*f(t+0.5*h,y + 0.5*k1,Lambda)
        k3 = h*f(t+0.5*h,y + 0.5*k2,Lambda)
        k4 = h*f(t+h, y + k3,Lambda)
    
        y  = y + (1.0/6.0)*((1.0*k1)+(2.0*k2)+(2.0*k3)+(1.0*k4))  # las soluciones 
        t  = t + h                                                # corremos el tiempo
                                              
        Y          = np.append( Y, [y], axis=0 )                  # adicionamos las soluciones 
        C , Q , OK = Constraint(y,Lambda)  
        constraint , q , Omega_k = np.append(constraint,C),np.append(q,Q),np.append(Omega_k,OK)  # adicionamos las soluciones para el constrint                         
        
    return T, Y , constraint , q, Omega_k

#----------------------------------------------------SISTEMA W-H-------------------------------------------------------------------        
          
def f(t,y,Lambda):                                                                        # definimos el sistema de W-H
    H          = y[0]                                                                     
    sigmaMas   = y[1]   
    sigmaMenos = y[2]
    n1         = y[3]  
    n2         = y[4] 
    n3         = y[5] 
    Rmas       = (1.0 / 6.0) * ((n2-n3)**2 - n1 * (2.0 * n1 - n2 - n3))
    Rmenos     = (1.0 / (2.0 * math.sqrt(3.0))) * (n3 - n2) * (n1 - n2 - n3)  


    f0 =   - H**2 - 2.0 * ((sigmaMas**2)+(sigmaMenos**2)) + (Lambda/3.0)                 # parametro de desaceleracion   
    f1 =   -3.0* H * sigmaMas  - Rmas                                                    # sistema dinamico, wangright-Hsu
    f2 =   -3.0* H * sigmaMenos- Rmenos
    f3 =   (-H - 4.0  * sigmaMas) * n1
    f4 =   (-H + (2.0 * sigmaMas) + (2.0 * math.sqrt(3.0) * sigmaMenos)) * n2
    f5 =   (-H + (2.0 * sigmaMas) - (2.0 * math.sqrt(3.0) * sigmaMenos)) * n3
    
    F  = np.array([f0,f1,f2,f3,f4,f5])

    return F

#---------------------------------------------------RESTRICCIÓN HAMILTONIANA---------------------------------------------------------


def Constraint(y,Lambda):   # definmos como una funcion el contraint, veremos si efectivamente en la evolución se cumple que es igual a 1.0
    H          = y[0]
    sigmaMas   = y[1]   
    sigmaMenos = y[2]
    n1         = y[3]  
    n2         = y[4] 
    n3         = y[5] 


    Omega_k         = ((n1**2)+(n2**2)+(n3**2)-(1.0/2.0)*((n1+n2+n3)**2))/(6.0*(H**2))
    SigmaCuadrado   = ((sigmaMas**2)+(sigmaMenos**2))/(H**2)
    Omega_lambda    =  Lambda/(3.0*(H**2)) 
    
    
    f0 =  - H**2 - 2.0 * ((sigmaMas**2)+(sigmaMenos**2)) + (Lambda/3.0)
    q  =  - 1.0 - f0/(H**2)
    
    
    constraint      = Omega_lambda + Omega_k + SigmaCuadrado

    return constraint , q , Omega_k


###############################################################################################################
#                                                MODELOS DE BIANCHI 
###############################################################################################################

#------------------------------------------------- Bianchi-tipo I-----------------------------------------------

'''

y0 = np.array([1.0, 1.0/math.sqrt(3.0),-1.0/math.sqrt(3.0),0.0,0.0,0.0])

'''

# --------------------------------------------------Bianchi-tipo II----------------------------------------------

'''
y0 = np.array([1.0,1.0/math.sqrt(6.0),-1.0/math.sqrt(6.0),2.0,0.0,0.0])

'''

# --------------------------------------------------Bianchi-tipo VI_0--------------------------------------------

'''

y0 = np.array([1.0,1.0/math.sqrt(6.0),-1.0/math.sqrt(6.0),1.0/2.0,-3.0/2.0,0.0])

'''

# -------------------------------------------------Bianchi-tipo VII_0--------------------------------------------

'''

y0 = np.array([1.0,1.0/math.sqrt(6.0),-1.0/math.sqrt(6.0),1.0/2.0,5.0/2.0,0.0])

'''

# ---------------------------------------------------Bianchi-tipo VIII--------------------------------------------


y0 = np.array([1.0,1.0/math.sqrt(6.0),-1.0/math.sqrt(6.0),1.0/2.0,1.0/2.0,-1.23606797749979 ])


#-----------------------------------------------------------------------------------------------------------------


##################################################################################################################
#                                         PARAMETROS INCIALESS PARA EL TEST DE CONVERGENCIA 
#                                                      Y CONSISTENCIA
###################################################################################################################


#------------------------------------------GRAFICA CON RK4 PARA TEST CONVERGENCIA----------------------------------

a=0
b=20
h = 0.1
Lambda=1.0
precision=1e-14
T1 ,Y1, constraint1 , q1, Omega_k1= RungeKutta4(y0, a,b, h, f, Constraint,Lambda)
T2 ,Y2, constraint2 , q2, Omega_k2= RungeKutta4(y0, a,b, h/2.0, f,  Constraint,Lambda)
T3 ,Y3, constraint3 , q3, Omega_k3= RungeKutta4(y0, a,b, h/4.0, f, Constraint,Lambda)



#---------- GRAFICA PARA VERIFICAR EL ORDEN DE EL ERROR DE TRUNCAMIENTO CON RESPECTO A LA RESTRICCION----------------
#                                                 CONSISTENCIA DEL SISTEMA

a=0
b=20
T4 ,Y4, constraint4 , q4, Omega_k4= RungeKutta4(y0, a,b, h, f, Constraint,Lambda)
T5 ,Y5, constraint5 , q5, Omega_k5= RungeKutta4(y0, a,b, h/10.0, f,  Constraint,Lambda)
T6 ,Y6, constraint6 , q6, Omega_k6= RungeKutta4(y0, a,b, h/100.0, f, Constraint,Lambda)


###########################################################################################################################
#                                     IMPLEMENTACIÓN DEL PROGRAMA DE CONVERGENCIA
###########################################################################################################################


D11=np.array([0])                                        
D22=np.array([0])
D4=np.array([[0.0]])
SUMA=0
for i in range(1,len(T1)):          # calculamos las diferencias de el metodo RKA con h y h/2, e igualmente con h/2 y h/4 .
                                    # Teniendo en cuenta que los comparamos en los mismos ti

    DY1=np.abs(Y1[i]-Y2[2*i])
    DY2=np.abs(Y2[2*i]-Y3[4*i])


    NormaDY1=math.sqrt(DY1[0]**2+DY1[1]**2+DY1[2]**2+DY1[3]**2+DY1[4]**2+DY1[5]**2)
    NormaDY2=math.sqrt(DY2[0]**2+DY2[1]**2+DY2[2]**2+DY2[3]**2+DY2[4]**2+DY2[5]**2)
    
    D11=np.append(D11, NormaDY1)
    D22=np.append(D22, NormaDY2)

    D4=np.append(D4,np.abs(np.log2(NormaDY1)-np.log2(NormaDY2))) 

                                                            
    SUMA+=D4[i]


PROMEDIO=SUMA/(len(T1)-1)                             # sacamos el promedio de cada una de las distancias anteriores

print('PROMEDIO con norma', PROMEDIO)

D1= np.delete(D11, 0)                                 # como vamos a graficar para el tiempo t=0 las solcuciones coinciden 
D2= np.delete(D22,0)                                  # y por lo tanto la diferencia es cero, si graficamos con logaritmo
T = np.delete(T1,0)                                   # entonces nos mostrara un error en ese primer tiempo ya que no esta definido

#----------------------------------------------------------------------------------------------------------------------------------
parameters = {'axes.labelsize': 18, 'axes.titlesize': 20}
plt.rcParams.update(parameters)


#####################################################################################################################################
#                                                      GRAFICO DE TEST DE CONVERGENCIA
#####################################################################################################################################

plt.plot(T, np.log2(D1), color='m',linewidth = 2, label=r'$D_1=\log_2 ||y_2-y_1||$')
plt.plot(T, np.log2(D2), color='c',linewidth = 2, label=r'$D_2=\log_2 ||y_3-y_2||$')
plt.legend(facecolor='thistle', loc='lower left', title=r'Distancia de las soluciones para \\ $\quad h_1=0.1,\, h_2=0.05,\, h_3=0.025$')
#plt.semilogy(base=2)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\log_2(D)$')
plt.title(r'\textbf{Test de Convergencia}', pad=20)
plt.show()

####################################################################################################################################
#                                           GRAFICO DE LA RESTRICCION CON DISTINTOS TAMAÑO DE PASO
#                                                         TEST DE CONSISTENCIA
#                                 VERIFICACIÓN DEL ORDEN DE CONVERGENCIA DEL ERROR LOCAL DE TRUNCAMIENTO
#####################################################################################################################################

plt.plot(T4, abs(1.0-constraint4), color='m',linewidth = 2, label=r'$h_1=0.1$')
plt.plot(T5, abs(1.0-constraint5), color='c',linewidth = 2, label=r'$h_2=0.01$')
plt.plot(T6, abs(1.0-constraint6), color='y',linewidth = 2, label=r'$h_3=0.001$')
plt.legend(facecolor='thistle', loc='center right')
plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$|\,1-\Omega_{\Lambda}-\Sigma^2-\Omega_{k}\,|$')
plt.title(r'\textbf{Error de la restricción Hamiltoniana}', pad=20)
plt.show()
