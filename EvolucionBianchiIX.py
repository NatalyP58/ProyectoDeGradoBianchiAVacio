######################################################################################################################
# Nataly Martínez Riascos 
# Departamento de Matemáticas de la Universidad del Valle
# 2021
######################################################################################################################

import numpy as np
import math 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt



print ('Evolción del sistema de Wanwright-Hsu para modelos de Bianchi clase A tipo IX')
input()

######################################################################################################################
#                                                   IMPLEMENTACIÓN DEL PROGRAMA
######################################################################################################################


#---------------------------------------------------METODO RUNGE-KUTTA 4 (RK4)--------------------------------------------

def RungeKutta4(y0, a,b, h, f, KretschmannScalar, Constraint, Lambda):     # definimos R-K de orden 4
     
    y = y0                                         # valores inciales
    t = a                                          # tiempo inicial
    T = np.arange(a,b+h,h)                         # array de los tiempos
    n = len(T)                                     # longitud del array
    Y = np.array([y])                              # array de la soluciones
    C , Q , OK = Constraint(y,Lambda)  
    constraint, q, Omega_k =np.array([C]), np.array([Q]),np.array([OK])   # Funcion que calcula si el constraint se cumple
    Krest      = np.array([KretschmannScalar(y,Lambda)])  # Krestschmann a partir de las componentes de la parte electrica y magnetica
     

    for i in range(n-1):                           # empezamos con el algoritmo para hallar las soluciones
        
        k1 = h*f(t,y,Lambda)                              # hallamos los k's que se necesitan para el metodo
        k2 = h*f(t+0.5*h,y + 0.5*k1,Lambda)
        k3 = h*f(t+0.5*h,y + 0.5*k2,Lambda)
        k4 = h*f(t+h, y + k3,Lambda)
    
        y  = y + (1.0/6.0)*((1.0*k1)+(2.0*k2)+(2.0*k3)+(1.0*k4))  # las soluciones 
        t  = t + h                                                # corremos el tiempo
                                              
        Y          = np.append( Y, [y], axis=0 )                   # adicionamos las soluciones 
        C , Q , OK = Constraint(y,Lambda)  
        constraint , q , Omega_k = np.append(constraint,C),np.append(q,Q),np.append(Omega_k,OK)            # adicionamos las soluciones para el constrint                         
        Krest      = np.append(Krest, KretschmannScalar(y,Lambda))        # adicionamos las soluciones para el kretschmann 1
        
        
    return T, Y , Krest, constraint , q, Omega_k



#-----------------------------------------------METODO RUNGE-KUTTA-FEHLBERGS (RK45)------------------------------------


def RungeKutta45(a,b, y0, TOL,hmax,hmin,f, KretschmannScalar, Constraint, Lambda):               # definimos R-K de orden 4
    t = a 
    y = y0
    x = y0
    h = hmax
    BAND = 1


    T=np.array([t])                                                                             # listas que almacenran las soluciones
    Y=np.array([y])
    X=np.array([x])
    R=np.array([[0.0,0.0,0.0,0.0,0.0,0.0]])
    C , Q , OK = Constraint(y,Lambda)  
    constraint, q, Omega_k =np.array([C]), np.array([Q]),np.array([OK])                         # Funcion que calcula si el constraint se cumple
    Krest      = np.array([KretschmannScalar(y,Lambda)])                                        # Krestschmann a partir de las componentes de la parte electrica y magnetica
    
    while BAND==1:
        
        k1 = h * f(t,y,Lambda)                                                                  # hallamos los k's que se necesitan para el metodo
        k2 = h * f(t + 1.0/4.0 * h, y + 1.0/4.0* k1,Lambda)
        k3 = h * f(t + 3.0/8.0 * h, y + 3.0/32.0 * k1 + 9.0/32.0 * k2,Lambda)
        k4 = h * f(t + 12.0/13.0 * h, y + 1932.0/2197.0 * k1 -7200.0/2197.0 * k2 + 7296.0/2197.0 * k3,Lambda )
        k5 = h * f(t + 1.0* h ,y + 439.0/216.0  * k1 - 8.0 * k2 + 3680.0/513.0 * k3 -845.0/4104.0 * k4,Lambda)
        k6 = h * f(t + 1.0/2.0 * h, y - 8.0/27.0* k1 + 2.0 * k2 - 3544.0/2656.0 * k3 + 1859.0/4104.0 * k4 - 11.0/40.0 * k5,Lambda)
        
        r = (1/h) * abs( 1.0/360.0 * k1 - 128.0/4275.0 * k3 - 2197.0/75240.0 * k4 + 1.0/50.0* k5 + 2.0/55.0 * k6 ) 
        R = np.append(R,[r],axis=0)

        if max(r[0],r[1],r[2],r[3],r[4],r[5]) <= TOL:
            t = t + h
            y, x = y + 25.0/216.0 * k1 + 1408.0/2565.0 * k3 + 2197.0/4104.0 * k4 - 1.0/5.0 * k5, y + 16.0/135.0 * k1 + 6656.0/12825.0 *k3 + 28561.0/56430.0 *k4 -9.0/50.0 * k5 + 2.0/55.0* k6
            
            T = np.append( T, t)                                                                      # adicionamos las soluciones a las listas
            Y = np.append( Y, [y], axis=0 )
            X = np.append( X, [x] , axis=0)

            C , Q , OK = Constraint(y,Lambda)  
            constraint, q, Omega_k =np.append(constraint, C), np.append(q,Q),np.append(Omega_k,OK)   # Funcion que calcula si el constraint se cumple
            Krest      = np.append(Krest,KretschmannScalar(y,Lambda))                                # Krestschmann a partir de las componentes de la parte electrica y magnetica

            
        Delta = 0.84 * ( TOL / max(r[0],r[1],r[2],r[3],r[4],r[5]) )**(1.0/4.0)
        
        if Delta <= 0.1:
            h = 0.1*h

        elif Delta >= 4:
            h = 4*h

        else: 
            h = Delta*h

        if h > hmax: 
            h= hmax
        
        if t >= b:
            BAND=0
        elif t+h > b:
            h = b-t

        elif h < hmin:
            BAND=0
            print('rebasado h minimo')
            break
        
    return T,Y,X,R, Krest, constraint,q,Omega_k

#--------------------------------------------------SISTEMA W-H------------------------------------------------------------        
          
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

#--------------------------------------------------RESTRICCIÓN HAMILTONIANA----------------------------------------------

def Constraint(y,Lambda):                  # definmos como una funcion la restricción Hamiltoniana, veremos si efectivamente en la evolución se cumple que es igual a 1.0
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

#------------------------------------------------------ESCALAR DE KRETSCHMANN--------------------------------------------

def KretschmannScalar(y,Lambda):          # definimos el Krtschmann apartir  las comoponentes de los tensores de la parte electrica y magnetica del tensor e Weyl
    
    H          = y[0]
    sigmaMas   = y[1]   
    sigmaMenos = y[2]
    n1         = y[3]  
    n2         = y[4] 
    n3         = y[5] 
    
        
    E11 =  (1.0/6.0)* (6.0 * n2* n3 - 4 * Lambda - 12.0 * H* sigmaMas + 12.0 * (H**2) + 3.0 * (n1**2) - 3.0 *(n2**2) - 3.0 *(n3**2) - 24.0 *(sigmaMas**2))
    E22 =  n1 * n3 - (2.0 *Lambda)/3.0 + H* sigmaMas + math.sqrt(3.0)* H *sigmaMenos + 2.0 *(H**2) - (n1**2)/2.0 + (n2**2)/2.0 - (n3**2)/2.0 - (sigmaMas**2) - 2.0*math.sqrt(3.0)*sigmaMas*sigmaMenos - 3.0* (sigmaMenos**2)
    E33 =  n1 * n2 - (2.0* Lambda)/3.0 + H* sigmaMas - math.sqrt(3.0)* H *sigmaMenos + 2.0 *(H**2) - (n1**2)/2.0 - (n2**2)/2.0 + (n3**2)/2.0 - (sigmaMas**2) + 2.0*math.sqrt(3.0)*sigmaMas*sigmaMenos - 3.0* (sigmaMenos**2)


    B11 = -(3.0 * (n1 - n2 - n3) * sigmaMas) 
    B22 = -(3.0/2.0) * (n1 - n2 + n3) * (sigmaMas + math.sqrt(3) * sigmaMenos)
    B33 = -(3.0/2.0) * (n1 + n2 - n3) * (sigmaMas - math.sqrt(3) * sigmaMenos)

    Eij = np.array([ [E11,0,0], [0,E22,0], [0,0,E33] ])
    Bij = np.array([ [B11,0,0], [0,B22,0], [0,0,B33] ])
    
    R   = 4.0  * (Lambda**2)

    R2  = 16.0 * (Lambda**2)

    E2  = 0
    B2  = 0
    

    for i in range(3) :

            E2 += Eij[i,i] * Eij[i,i]

            B2 += Bij[i,i] * Bij[i,i] 

    return 8.0*(E2-B2) + 2.0*R - (1.0/3.0)*R2



###############################################################################################################
#                                                MODELOS DE BIANCHI TIPO IX
#                                                     DISTINTOS CASOS
###############################################################################################################



################################################################################################################
#                                           REGION DE COLAPSO
################################################################################################################-

# 1) Primer sistema donde K<-Sqrt[(3*Lambda]) Recolapso, escogencia de H=-1  y Lambda=1
'''
Lambda=1  #recolapso
a6=0                  #valores inciales
b6=50
y06 = np.array([-1.0, math.sqrt(0.75), -math.sqrt(0.75), 1.0, 2.58578643762691, 3.0])
'''

#--------------------------------------------------------------------------------------------------------------

# 2) Recolapso con K=-Sqrt[(3*Lambda])]   ,  H=-1.0  y Lambda=3

'''
Lambda=3
a6=0                  #valores inciales
b6=20
y06 = np.array([-1.0, math.sqrt(1.0/4.0), -math.sqrt(1.0/4.0), 2.0, 1.58578643762690, 1.0])
'''


################################################################################################################
#                                           REGION DE EXPANSION
################################################################################################################

#----------------------------------------------------------------------------------------------------------------


# 3) Expansion con K=Sqrt[(3*Lambda])]   ,  H=1.0  y Lambda=3

'''
Lambda=3                         # Constante comologica
a6=0                             # valores inciales
b6=20
y06 = np.array([1.0, math.sqrt(1.0/4.0), -math.sqrt(1.0/4.0), 2.0, 1.58578643762690, 1.0])
''' 

#----------------------------------------------------------------------------------------------------------------


# 4) Sistema expansion hacia De Sitter  (Sistema con H postivo cumple la codcion (a) y (b) de Kitada y Maeda,
#    esta en la region Sqrt(3*Lambda)<K). 
#    Elección H=1 y Lambda=1.  Sqrt(3*Lambda)<K


Lambda=1
a6=0                  #valores inciales
b6=20
y06 = np.array([1.0, 0.60380736442456, -0.60380736442456, 1.0/2.0, 1.0/2.0, 1.0/2.0])


#----------------------------------------------------------------------------------------------------------------


#  5) Sistema de caso de expansion en   0<K<Sqrt[(3*Lambda])]

'''
Lambda=1 
a6=0                  #valores inciales
b6=50
y06 = np.array([1.0/2.0, math.sqrt(0.01), -math.sqrt(0.01), 1.0,  0.338675227416385, 1.0])
'''

################################################################################################################
#                                    REGION FUERA DE LA CONDICIONES SUFICIENTES
#                                          DE INFLACIÓN Y COLAPSO
################################################################################################################


# 6)  - Sqrt[(3*Lambda])] < K < Sqrt[(3*Lambda])]: No cumple con la codicion de que (a) de Kitada y Maeda

'''
Lambda=1 # expansion -sqrt(3)< K<sqrt(3) , como es una parabola hay dos puntos en donde omega_K=0.5 que son 
#                                          n2-> 0.90098  para el primero expansión al de De Sitter 
#                                          n2-> 11.099   Recolapsa
a6=0                                      #valores inciales      
b6=50


y06= np.array([1.0/2.0, math.sqrt(5.0/24.0), -math.sqrt(5.0/24.0), 2.0,0.900980486407215, 4.0]) # Expansion
'''

#--------------------------------------------------------------------------------------------------------------


# 7) Tiene las mismas condiciones del anterior - Sqrt[(3*Lambda])] <K<Sqrt[(3*Lambda])]
#    No cumple con la codicion de que (a) de Kitada y Maeda 

'''
Lambda=1 # expansion -sqrt(3)< K<sqrt(3) 
a6=0                  #valores inciales
b6=50

y06 = np.array([1.0/2.0, math.sqrt(5.0/24.0), -math.sqrt(5.0/24.0), 2.0,11.0990195135928, 4.0]) # Colapso

'''

#--------------------------------------------------------------------------------------------------------------


# 8) El siguiente caso no cumple con la condicion (a) de Kitada y Maeda.

# En este caso H=2 y  Lambda=1. Por tanto,    Sqrt[(3*Lambda])]<K 

'''
Lambda=1.0
a6=0                  #valores inciales
b6=80
y06 = np.array([2.0,math.sqrt(3.0), math.sqrt(3.0), 8.0, 2.0, 4.0])}

'''



##################################################################################################################
#                                                       GRAFICAS 
###################################################################################################################

#--------------------------------------------------GRAFICA CON RK4----------------------------------------------------


h = 0.01                                         # tamaño de paso
precision=1e-14                                  # preisición a la hora de graficar

T6 ,Y6, Krest6, constraint6 , q6, Omega_k6= RungeKutta4(y06, a6,b6, h, f, KretschmannScalar, Constraint,Lambda)


#---------------------------------------GRAFICA CON RK45(RUNGE-KUTTA-FEHLBERGS)--------------------------------------
'''

hmax6= 1                                          # tamaño de paso maximo
hmin6=0.00001                                     # tamaño de paso minimo
TOL6= 10**(-3)                                    # tolerancia
precision=1e-14                                   # presicion a la hora de graficar
        
T6,Y6,X6,R6,Krest6, constraint6,q6,Omega_k6 = RungeKutta45(a6,b6, y06, TOL6,hmax6,hmin6,f,KretschmannScalar,Constraint,Lambda)

'''
#---------------------------------------------------------------------------------------------------------------------
print ('Tiempo:  ',T6)
print ('y(H,sigmaMas,sigmaMenos,N1,N2,N3) : ',Y6)    # los datos vienen dados de la forma (H,sigmaMas,sigmaMenos,N1,N2,N3)
print('Kretschamann: ', Krest6)
print('Hubble',Y6[:,0])
print('Desaceleracion',q6)
print('Restriccion',constraint6)
print('Omega_k', Omega_k6)
#graficamos la evolucion de nuestro sistema


parameters = {'axes.labelsize': 18, 'axes.titlesize': 20,'legend.title_fontsize':18}
plt.rcParams.update(parameters)




plt.plot(T6, abs(Y6[:,0]-math.sqrt(Lambda/3.0)),'m-', linewidth = 2 )
plt.title(r'\textbf{Error del escalar de Hubble}', pad=20)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\large{\tau}$')
plt.ylabel(r'$|\,H-\sqrt{\Lambda/3}\,|$ ')
plt.show()



plt.plot(T6,Y6[:,1],'m-', linewidth = 2)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sigma_{+}$')
plt.title(r'\textbf{Evolución de la anisotropía} $\sigma_{+}$', pad=20)
plt.show()


plt.plot(T6,Y6[:,2],'m-', linewidth = 2)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.title(r'\textbf{\textbf{Evolución de la anisotropía} $\sigma_{-}$', pad=20)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sigma_{-}$')
plt.show()


plt.plot(T6,Y6[:,3],'m-', linewidth = 2)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$n_{1}$')
plt.title(r'\textbf{\textbf{Evolución del parámetro} $n_{1}$', pad=20)
plt.show()

plt.plot(T6,Y6[:,4],'m-', linewidth = 2)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$n_{2}$')
plt.title(r'\textbf{Evolución del parámetro} $n_{2}$', pad=20)
plt.show()

plt.plot(T6,Y6[:,5],'m-', linewidth = 2)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$n_{3}$')
plt.title(r'\textbf{Evolución del parámetro} $n_{3}$', pad=20)
plt.show()


plt.plot(T6,abs(1.0-constraint6),'m-', linewidth = 2)
plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$|\,1-\Omega_{\Lambda}-\Sigma^2-\Omega_{k}\,|$')
plt.title(r'\textbf{Error de la restricción Hamiltoniana}', pad=20)
plt.show()


plt.plot(T6,abs((8.0/3.0)*(Lambda**2)-Krest6),'m-', linewidth = 2)
plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$|\,$\textbf{K}$-\frac{8}{3}\,\Lambda^2 \,|$')
plt.title(r'\textbf{Error del escalar de Kretschmann}', pad=20)
plt.show()

plt.plot(T6,abs(-1.0-q6),'m-', linewidth = 2)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$|\,1+q\,|$')
plt.title(r'\textbf{Error del parámetro desaceleración}', pad=20)
plt.show()

plt.plot(T6, Omega_k6,'m-', linewidth = 2)
#plt.yscale("symlog",linthresh=precision)
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega_k$')
plt.title(r'\textbf{Evolución del parámetro $\Omega_k$}', pad=20)
plt.show()