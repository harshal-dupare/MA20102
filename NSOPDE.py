import numpy as np

# Euler Forward Method
def EFM(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        Y.append(Y[i]+h*F(x_i,Y[i]))
    
    return Y


# Newton Raphson Method
def NR(F,dF,x_init,eplison,max_itt=50):
    
    x_new=x_init
    
    while max_itt > 0 :
        x_old=x_new
        x_new = x_old - F(x_old)/(dF(x_old))
        max_itt-=1
        if abs(x_old-x_new) < eplison : break
            
    return x_new
    

 # Euler Backward Method
def EBM(n,F,dFy,y_init,eplison,max_itter=50,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        max_itt=max_itter
        while max_itt > 0 :
            y_old=y_new
            y_new = y_old - (y_old-h*F(x_i,y_old)-Y[i])/(1-h*dFy(x_i,y_old))
            max_itt-=1
            if abs(y_old-y_new) < eplison : break
        Y.append(y_new)
    
    return Y


# Modified Euler Method
def MEM(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+(h/2),y_new+(k1/2))
        Y.append(y_new+k2)
    
    return Y



# Euler Cauchy Method
# Hune Method
def ECM(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+h,y_new+k1)
        Y.append(y_new+((k2+k1)/2))
    
    return Y


# RK method of thita
def RK(n,F,y_init,thita=0.5,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+thita*h,y_new+thita*k1)
        Y.append(y_new+k2)
    
    return Y





# general 2nd order method
def General_order2(n,F,y_init,aplha=0.5,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    w2=1/(2*alpha)
    w1=1-w2
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+alpha*h,y_new+alpha*k1)
        Y.append(y_new+w1*k1+w2*k2)
    
    return Y




# Nystrom 3rd order
def Nystrom(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+(2/3)*h,y_new+(2/3)*k1)
        k3=h*F(x_i+(2/3)*h,y_new+(2/3)*k2)
        Y.append(y_new+((k1+3*k2+3*k3)/8))
    
    return Y



# Hune 3rd order method
def Hune(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+(1/3)*h,y_new+(1/3)*k1)
        k3=h*F(x_i+(2/3)*h,y_new+(2/3)*k2)
        Y.append(y_new+((k1+3*k3)/4))
    
    return Y




# Classical 3rd Order method
def Classical_order3(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+(1/2)*h,y_new+(1/2)*k1)
        k3=h*F(x_i+h,y_new-k1+2*k2)
        Y.append(y_new+((k1+4*k2+k3)/6))
    
    return Y




# Nearly optimal 3rd Order method
def Nearly_optimal(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+(1/2)*h,y_new+(1/2)*k1)
        k3=h*F(x_i+(3/4)*h,y_new+(3/4)*k2)
        Y.append(y_new+((2*k1+3*k2+4*k3)/9))
    
    return Y



# Kutta Method 4th order
def Kutta_order4(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+(1/3)*h,y_new+(1/3)*k1)
        k3=h*F(x_i+(2/3)*h,y_new+(-1/3)*k1+(1)*k2)
        k4=h*F(x_i+(1)*h,y_new+(1)*k1+(-1)*k2+(1)*k3)
        Y.append(y_new+((1*k1+3*k2+3*k3+k4)/8))
    
    return Y


# Classical 4rd Order method
def Classical_order4(n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        k1=h*F(x_i,y_new)
        k2=h*F(x_i+(1/2)*h,y_new+(1/2)*k1)
        k3=h*F(x_i+(1/2)*h,y_new+(1/2)*k2)
        k4=h*F(x_i+(1)*h,y_new+(1)*k3)
        Y.append(y_new+((1*k1+2*k2+2*k3+k4)/6))
    
    return Y


# General Explicit method

def gen(order,W,A,n,F,y_init,limit=[0,1]):
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        
        k=np.zeros(order)
        for j in range(order):
            k[j]=h*F(x_i+h*A[j].sum(),y_new+(A[j]*k).sum())
            
        Y.append(y_new+(W*k).sum())
    
    return Y

def NR_solver(F,dyF,x_init,y_init,eplison=1e-4,max_itt=50):

    y_new = y_init
    
    while max_itt > 0:
        y_old=y_new
        y_new = y_old - (F(x=x_init,y=y_old)/dyF(x=x_init,y=y_old))
        
        if abs(y_old-y_new) < eplison: break
            
        max_itt-=1
    
    return y_new


def implicit_rk2(n,F,dyF,y_init,limit=[0,1],eplison=1e-4,max_itt=50):
    a=limit[0]
    b=limit[1]
    step=(b-a)/n
    Y=[]
    Y.append(y_init)
    
    
    for i in range(n-1):
        x_i=a+step*i
        y_i=Y[i]
                
        k_new = step*F(x=x_i,y=y_i)
        max_ittr=max_itt
        while max_ittr > 0:
            k_old=k_new
            k_new = k_old - (k_old-step*F(x=x_i+(1/2)*step,y=y_i+(1/2)*k_old)/(1-step*dyF(x=x_i+(1/2)*step,y=y_i+(1/2)*k_old)))

            if abs(k_old-k_new) < eplison: break
            
            max_ittr-=1
            
        Y.append(y_i+k_new)
    
    return Y
       


# Implicit 4th order RK method
# not using vectorized Newton raphson method, insted updating k1,k2 togther by single Newton raphson on each
def implicit_rk4(n,F,dyF,y_init,limit=[0,1],eplison=1e-4,max_itt=50):
    a=limit[0]
    b=limit[1]
    step=(b-a)/n
    Y=[]
    Y.append(y_init)
    alpha=(3-np.sqrt(3))/6
    
    
    for i in range(n-1):
        x_i=a+step*i
        y_i=Y[i]
        
        temp=step*F(x=x_i,y=y_i)
        k1_new,k2_new = step*F(x=x_i+alpha*step,y=y_i+alpha*temp),step*F(x=x_i+(1-alpha)*step,y=y_i+(1-alpha)*temp)
        max_ittr=max_itt
        while max_ittr > 0:
            k1_old,k2_old=k1_new,k2_new
            k1_new = k1_old - (k1_old-step*F(x=x_i+(alpha)*step,y=y_i+(0.25)*k1_old+(alpha-0.25)*k2_old)/(1-step*dyF(x=x_i+(alpha)*step,y=y_i+(0.25)*k1_old+(alpha-0.25)*k2_old)))
            k2_new = k2_old - (k2_old-step*F(x=x_i+(1-alpha)*step,y=y_i+(0.25)*k2_old+(0.75-alpha)*k1_old)/(1-step*dyF(x=x_i+(1-alpha)*step,y=y_i+(0.25)*k2_old+(0.75-alpha)*k1_old)))
            
            if abs(k1_old-k1_new)  + abs(k2_old-k2_new) < eplison: break
            
            max_ittr-=1
            
        Y.append(y_i+(k1_new+k2_new)/2)
    
    return Y
        




























