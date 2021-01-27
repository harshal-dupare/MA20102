import numpy as np

def EFM(n,F,y_init,limit=[0,1]):
    """
    Euler Forward method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        Y.append(Y[i]+h*F(x_i,Y[i]))
    
    return Y

def NR(F,dF,x_init,epsilon,max_itter=50):
    """
    Newton Raphson Method for solving for x in 
    
    F(x) = 0

    F : F(x) takes 1 input as x and then outputs the value
    dF : dF(x)/dx, derivative wrt x of F
    x_init : initial guess of x
    eplsion : accuracy of answer
    max_itter : maximum times to itterate if answer never reaches the accuracy
    """

    
    x_new=x_init
    
    while max_itter > 0 :
        x_old=x_new
        x_new = x_old - F(x_old)/(dF(x_old))
        max_itter-=1
        if abs(x_old-x_new) < epsilon : break
            
    return x_new
    
def EBM(n,F,dFy,y_init,epsilon,max_itter=50,limit=[0,1]):
    """
    Euler Backward method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    dFy : dF(x,y)/dy, partial derivative wrt y of F
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    eplsion : accuracy of answer
    max_itter : maximum times to itterate if answer never reaches the accuracy

    uses newton raphson method to solve for next step value
    """


    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    
    Y=[]
    Y.append(y_init)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]

        while max_itter > 0 :
            y_old=y_new
            y_new = y_old - (y_old-h*F(x_i,y_old)-Y[i])/(1-h*dFy(x_i,y_old))
            max_itter-=1
            if abs(y_old-y_new) < epsilon : break
        Y.append(y_new)
    
    return Y

def MEM(n,F,y_init,limit=[0,1]):
    """
    Modified Euler method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
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

def ECM(n,F,y_init,limit=[0,1]):
    """
    Euler Cauchy Method or Hune Method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
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

def RK(n,F,y_init,thita=0.5,limit=[0,1]):
    """
    Ranga Kutta method of thita for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    thita : thita parameter
    limit : [a,b], region on x to solve the equation for
    """
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

def General_order2(n,F,y_init,alpha=0.5,limit=[0,1]):
    """
    General 2nd Order method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    alpha : alpha parameter from range [0,1]
    limit : [a,b], region on x to solve the equation for
    """
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

def Nystrom(n,F,y_init,limit=[0,1]):
    """
    Nystrom 3rd order method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
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

def Hune(n,F,y_init,limit=[0,1]):
    """
    Hune 3rd order method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
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

def Classical_order3(n,F,y_init,limit=[0,1]):
    """
    Classical 3rd Order method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
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

def Nearly_optimal(n,F,y_init,limit=[0,1]):
    """
    Nearly optimal 3rd Order method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
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

def Kutta_order4(n,F,y_init,limit=[0,1]):
    """
    Kutta Method 4th order method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    """
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

def Classical_order4(n,F,y_init,limit=[0,1]):
    """
    Classical 4rd Order method for solution of 
    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    alpha : alpha parameter from range [0,1]
    limit : [a,b], region on x to solve the equation for
    """
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

def General_Explicit(order,W,A,n,F,y_init,limit=[0,1],C=None):
    """
    General Explicit Ranga Kutta method for solution of 
    y' = F(x,y)

    order : order of method say m
    W = weight numpy Array for, y_n+1 = y_n + ( w1*k1 + w2*k2.....+ wm*km ), here W = [w1,w2..,wm]
    A = weight numpy matrix Matrix for, ki = h*F(x_n + ci*h,y_n + ( a_{i,i}*k1 + a_{i,2}*k2 + ...+ a_{i,i-1}*k_i-1 )) for i = 0...m, h is step size = (b-a/n)
        note values beyound i-1 are all supposed to be zero for Explicit method
    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    C : weight numpy Array [c1,c2...c], usually ci = a_{i,1}+ a_{i,2}+..+  a_{i,i-1}
        this can be changed with help of this input
    """
    a=limit[0]
    b=limit[1]
    h=(b-a)/n
    Y=[]
    Y.append(y_init)

    A=np.asarray(A)
    W=np.asarray(W)
    if C!=None:
        C=np.asarray(C)
    else:
        C=np.sum(A,axis=1)
        print(C)
    
    for i in range(n-1):
        x_i=a+i*h
        y_new=Y[i]
        
        k=np.zeros(order)
        for j in range(order):
            k[j]=h*F(x_i+h*C[j],y_new+(A[j]*k).sum())
            
        Y.append(y_new+(W*k).sum())
    
    return Y

def NR_solver(F,dyF,x_init,y_init,epsilon=1e-4,max_itter=50):
    """
    Newton Raphson Method for solving for y for a given x in
    
    F(x,y) = 0

    F : F(x,y) takes 2 positional inputs as x and y, then outputs the value
    dF : dF(x,y)/dx, partial derivative wrt x of F
    x_init : Value of x
    y_init : initial guess of y
    eplsion : accuracy of answer
    max_itter : maximum times to itterate if answer never reaches the accuracy
    """

    y_new = y_init
    
    while max_itter > 0:
        y_old=y_new
        y_new = y_old - (F(x=x_init,y=y_old)/dyF(x=x_init,y=y_old))
        
        if abs(y_old-y_new) < epsilon: break
            
        max_itter-=1
    
    return y_new

def implicit_rk2(n,F,dyF,y_init,limit=[0,1],epsilon=1e-4,max_itter=50):
    """
    Implicit RK Method of order 2 for solving for x in 
    
    F(x,y) = y'

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    dFy : dF(x,y)/dy, partial derivative wrt y of F
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    x_init : initial guess of x
    eplsion : accuracy of answer
    max_itter : maximum times to itterate if answer never reaches the accuracy
    """
    a=limit[0]
    b=limit[1]
    step=(b-a)/n
    Y=[]
    Y.append(y_init)
    
    
    for i in range(n-1):
        x_i=a+step*i
        y_i=Y[i]
                
        k_new = step*F(x=x_i,y=y_i)

        while max_itter > 0:
            k_old=k_new
            k_new = k_old - (k_old-step*F(x=x_i+(1/2)*step,y=y_i+(1/2)*k_old)/(1-step*dyF(x=x_i+(1/2)*step,y=y_i+(1/2)*k_old)))

            if abs(k_old-k_new) < epsilon: break
            
            max_itter-=1
            
        Y.append(y_i+k_new)
    
    return Y
       
def implicit_rk4(n,F,dyF,y_init,limit=[0,1],epsilon=1e-4,max_itter=50):
    """
    Implicit RK Method of order 4 for solving for x in 
    
    F(x,y) = y'

    n : number of steps
    F : F(x,y) takes 2 input as x and y then outputs the value for this equation
    dFy : dF(x,y)/dy, partial derivative wrt y of F
    y_init = initial value at point x=a
    limit : [a,b], region on x to solve the equation for
    x_init : initial guess of x
    eplsion : accuracy of answer
    max_itter : maximum times to itterate if answer never reaches the accuracy

    note : Not using vectorized Newton raphson method, insted updating k1,k2 togther by single Newton raphson on each
    """
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

        while max_itter > 0:
            k1_old,k2_old=k1_new,k2_new
            k1_new = k1_old - (k1_old-step*F(x=x_i+(alpha)*step,y=y_i+(0.25)*k1_old+(alpha-0.25)*k2_old)/(1-step*dyF(x=x_i+(alpha)*step,y=y_i+(0.25)*k1_old+(alpha-0.25)*k2_old)))
            k2_new = k2_old - (k2_old-step*F(x=x_i+(1-alpha)*step,y=y_i+(0.25)*k2_old+(0.75-alpha)*k1_old)/(1-step*dyF(x=x_i+(1-alpha)*step,y=y_i+(0.25)*k2_old+(0.75-alpha)*k1_old)))
            
            if abs(k1_old-k1_new)  + abs(k2_old-k2_new) < epsilon: break
            
            max_itter-=1
            
        Y.append(y_i+(k1_new+k2_new)/2)
    
    return Y
        
def factorial(n):
    """
    Utility function
    """

    fact = 1
    while n > 0:
        fact*=n
        n-=1

    return fact

def binomial_coeff(n,k):
    """
    Utility function to get nCk
    """
    if k == 0:
        return 1
    b = factorial(n)/(factorial(k)*factorial(n-k))

    return b

def forward_diff(F,x_init,y_list,y_init_idx,step=0.1,order=1):
    """
    Returns (Ex-1)^order*F(x,y), i.e. order'th forward difference of F wrt x
    F : F(x,y) 2 positional arguments
    x_init : value of x corrosponding to y_init
    y_list : list of y values corresponding to step size differences from x_init
            Here we need sufficient number of values in list to calculate the difference
    y_init_idx : index where we have y_init corrosponding to x_init
    step : step size
    order : the order of forward difference

    Note: Assuming , y_list[i+y_init_idx] = y(x_init+i*step)
    """
    d=0

    if (y_init_idx + order+1) > len(y_list):
        print("Error: y_list must have sufficient point values to compute "+str(order)+" differences\n")
    for i in range(order+1):
        d+=(((i+1)%2)-1)*binomial_coeff(order,i)*F(x=x_init+step*i,y=y_list[y_init_idx+i])
    
    return d

def backward_diff(F,x_init,y_list,y_init_idx,step=0.1,order=1):
    """
    Returns (Ex-1)^order*F(x,y), i.e. order'th backward difference of F wrt x
    F : F(x,y) 2 positional arguments
    x_init : value of x corrosponding to y_init
    y_list : list of y values corresponding to step size differences from x_init
            Here we need sufficient number of values in list to calculate the difference
    y_init_idx : index where we have y_init corrosponding to x_init
    step : step size
    order : the order of forward difference

    Note: Assuming , y_list[i+y_init_idx] = y(x_init+i*step)
    """
    d=0

    if y_init_idx < order :
        print("Error: y_list must have sufficient point values to compute "+str(order)+" differences\n")
    for i in range(order+1):
        d+=(((i+1)%2)-1)*binomial_coeff(order,i)*F(x=x_init-step*i,y=y_list[y_init_idx-i])
    
    return d

def Adam_Bashford(n,F,y_list,limit=[0,1],order=1):
    """
    Adam Bashford method for solving
    
    F(x,y) = y'

    n : number of steps
    F : F(x,y) takes 2 positional input as x and y then outputs the value for this equation
    y_list : list of y values corresponding to step size differences from x_init
            Here we need sufficient number of values in list to calculate the difference
    limit : [a,b], region on x to solve the equation for
    order : the order of method to use

    Note: Assuming , y_list[-1] contains the value at x=a, i.e. the last element has value at x=a for y
    """
    x_init=limit[0]
    a=limit[0]
    b=limit[1]
    step=(b-a)/n

    ## somehow we get the coefficients with each backward diff power
    coeff = [1,1/2,5/12,3/8,251/720]

    for i in range(n-1):
        net=0
        for j in range(order):
            net+=coeff[j]*backward_diff(F,x_init+i*step,y_list,y_init_idx=len(y_list)-1,step=step,order=j)
        y_list.append(y_list[-1]+step*net)
    
    return y_list

## need to check
def Adam_Moulton(n,F,y_list,limit=[0,1],order=1):
    """
    Adam Moulton method for solving
    
    F(x,y) = y'

    n : number of steps
    F : F(x,y) takes 2 positional input as x and y then outputs the value for this equation
    y_list : list of y values corresponding to step size differences from x_init
            Here we need sufficient number of values in list to calculate the difference
    limit : [a,b], region on x to solve the equation for
    order : the order of method to use

    Note: Assuming , y_list[-1] contains the value at x=a, i.e. the last element has value at x=a for y
    """
    x_init=limit[0]
    a=limit[0]
    b=limit[1]
    step=(b-a)/n

    ## somehow we get the coefficients with each backward diff power
    coeff = [1,-1/2,-1/12,-1/24,-19/720]

    for i in range(n-1):
        net=0
        for j in range(order):
            net+=coeff[j]*backward_diff(F,x_init+i*step,y_list,y_init_idx=len(y_list)-1,step=step,order=j)
        y_list.append(y_list[-1]+step*net)
    
    return y_list


def NRy(F,dFy,x_init,y_init,intercept,coeff=1,epsilon=1e-4,max_itter=50):
    """
    Newton Raphson Method for solving for y for a given x in
    
    y-coeff*F(x,y) = intercept

    F : F(x,y) takes 2 positional inputs as x and y, then outputs the value
    dFy : dF(x,y)/dy, partial derivative wrt y of F
    x_init : Value of x
    y_init : initial guess of y
    eplsion : accuracy of answer
    max_itter : maximum times to itterate if answer never reaches the accuracy
    """
    
    y_new=y_init
    
    while max_itter > 0 :
        y_old=y_new
        y_new = y_old - (y_old-coeff*F(x=x_init,y=y_old)-intercept)/(1-coeff*dFy(x=x_init,y=y_old))
        max_itter-=1
        if abs(y_old-y_new) < epsilon : break
            
    return y_new
 
## need to check !! major check
## IMPLICIT METHOD need to solve by NR method after forming the equation
def Milne_Simpson(n,F,dFy,y_list,epsilon,max_itter=50,limit=[0,1],order=1):
    """
    Milne Simpson method for solving
    
    F(x,y) = y'

    n : number of steps
    F : F(x,y) takes 2 positional input as x and y then outputs the value for this equation
    dFy : dF(x,y)/dy, partial derivative wrt y of F
    y_list : list of y values corresponding to step size differences from x_init
            Here we need sufficient number of values in list to calculate the difference
    limit : [a,b], region on x to solve the equation for
    order : the order of method to use
    eplsion : accuracy of answer
    max_itter : maximum times to itterate if answer never reaches the accuracy

    Note: Assuming , y_list[-1] contains the value at x=a, i.e. the last element has value at x=a for y
    """
    x_init=limit[0]
    a=limit[0]
    b=limit[1]
    step=(b-a)/n

    ## somehow we get the coefficients with each f_i term for each order
    coeff = [1/3,4/3,1/3]

    for i in range(n-1):
        net=0
        for j in range(order):
            net+=coeff[j+1]*F(x=x_init-step*j,y=y_list[-j-1])
        net = y_list[-2]+step*net
        y_list.append(NRy(F=F,dFy=dFy,x_init=x_init+i*step+step,y_init=y_list[-1],intercept=net,coeff=step*coeff[0],epsilon=epsilon,max_itter=max_itter))
    
    return y_list

def  Adams_Bashforth_Moulton_PC(n,F,y_list,limit=[0,1],correct_count=1):
    """
    Adams Bashforth Moulton PC method for solving 

    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 positional inputs as x and y then outputs the value for this equation
    limit : [a,b], region on x to solve the equation for
    y_list : list of y values corresponding to step size differences from x_init upto 3 steps, i.e. 4 total value
            as this method requires 4 values.
    correct_count : Number of correction steps to do between 2 prediction steps

    Note: Assuming , y_list[i] = y(a+i*step), for i=0,1,2,3

    """

    if len(y_list) < 4:
        print("Give sufficient number of values in y_list\n")
        return
    x_init=limit[0]
    a=limit[0]
    b=limit[1]
    step=(b-a)/n

    coeff_P = [55/24,-59/24,37/24,-9/24]
    coeff_C = [9/24,19/24,-5/24,1/24]

    for i in range(n):

        ## predictor
        net = 0
        for k in range(4):
            net+=coeff_P[k]*F(x=x_init-step*k,y=y_list[-k-1])
        y_p = y_list[-1]+step*net
        print("pred  "+str(i)+" "+str(y_p)+"\n")
        ## corrector

        for j in range(correct_count):
            net = coeff_C[0]*F(x=x_init+step,y=y_p)
            for k in range(3):
                net+=coeff_C[k+1]*F(x=x_init-step*k,y=y_list[-k-1])
            y_p=y_list[-1]+step*net
            print(str(j+1)+" "+str(y_p))
        
        y_list.append(y_p)
        x_init+=step
        
    return y_list

def Milne_PC(n,F,y_list,limit=[0,1],correct_count=1):
    """
    Milne PC method for solving 

    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 positional inputs as x and y then outputs the value for this equation
    limit : [a,b], region on x to solve the equation for
    y_list : list of y values corresponding to step size differences from x_init upto 3 steps, i.e. 4 total value
            as this method requires 4 values.
    correct_count : Number of correction steps to do between 2 prediction steps

    Note: Assuming , y_list[i] = y(a+i*step), for i=0,1,2,3
    """
    # y_list must contain 4 values
    if len(y_list) < 4:
        print("Give sufficient number of values in y_list\n")
        return
    x_init=limit[0]
    a=limit[0]
    b=limit[1]
    step=(b-a)/n

    coeff_P = [8/3,-4/3,8/3]
    coeff_C = [1/3,4/3,1/3]

    for i in range(n):

        ## predictor
        net = 0
        for k in range(3):
            net+=coeff_P[k]*F(x=x_init-step*k,y=y_list[-k-1])
        y_p = y_list[-4]+step*net
        print("pred  "+str(i)+" "+str(y_p)+"\n")
        ## corrector

        for j in range(correct_count):
            net = coeff_C[0]*F(x=x_init+step,y=y_p)
            for k in range(2):
                net+=coeff_C[k+1]*F(x=x_init-step*k,y=y_list[-k-1])
            y_p=y_list[-2]+step*net
            print(str(j+1)+" "+str(y_p))
        
        y_list.append(y_p)
        x_init+=step
        
    return y_list

def Euler_PC(n,F,y_list,limit=[0,1],correct_count=1):
    """
    Euler PC method for solving 

    y' = F(x,y)

    n : number of steps
    F : F(x,y) takes 2 positional inputs as x and y then outputs the value for this equation
    limit : [a,b], region on x to solve the equation for
    y_list : list of y values corresponding to step size differences from x_init upto 3 steps, i.e. 4 total value
            as this method requires 4 values.
    correct_count : Number of correction steps to do between 2 prediction steps

    Note: Assuming , y_list[i] = y(a+i*step), for i=0,1,2,3

    """

    if len(y_list) < 4:
        print("Give sufficient number of values in y_list\n")
        return
    x_init=limit[0]
    a=limit[0]
    b=limit[1]
    step=(b-a)/n

    for i in range(n):

        ##predictor
        y_p = y_list[-1]+step*F(x=x_init,y=y_list[-1])
        print("pred  "+str(i)+" "+str(y_p)+"\n")

        ##corrector
        for j in range(correct_count):
            y_p=y_list[-1]+step*F(x=x_init+step,y=y_p)
            print(str(j+1)+" "+str(y_p))
        
        y_list.append(y_p)
        x_init+=step
        
    return y_list

"""Numerical solutions of BVP - Linear BVP, 

finite difference methods, shooting methods, Newtonâs method for system of equations,
stability, error and convergence analysis, non linear BVP, higher order BVP.


Partial Differential Equations: Classification of PDEs, Finite difference
approximations to partial derivatives, convergence and stability analysis.
Explicit and Implicit schemes - Crank-Nicolson scheme, tri-diagonal system,
Laplace equation using standard five point formula and diagonal five point
formula. ADI scheme, hyperbolic equation, explicit scheme, method of
characteristics. Solution of one dimensional heat conduction equation by
Schmidt and Crank Nicolson methods. Solution of wave equation."""

def Thomas_algorithm(a,b,c,d):
    """
    Thomas algorithm to solve Tridiagonal system of equation

    [a,b,c]*X = d

    [a,b,c] : Tridiagonal matrix with main diagonal as b and a the lower diagonal and c upper diagonal.
    a,b,c,d are lists or arrays

    returns X
    """
    n = len(d)
    if b[0] == 0 :
        print("Division by zero encountered at c[0]/=b[0]\n")
        return
    c[0]/=b[0]
    for i in range(n-2):
        temp = b[i+1]-(a[i]*c[i])
        if temp == 0 :
            print("Division by zero encountered at c[i+1]/=b[i+1]-(a[i+1]*c[i]) at i={i} \n")
            return
        c[i+1]/=temp
    d[0]/=b[0]
    for i in range(n-1):
        temp = b[i+1]-(a[i]*c[i])
        if temp == 0 :
            print("Division by zero encountered at d[i+1]=(d[i+1]-a[i+1]*d[i])/b[i+1]-(a[i+1]*c[i]) at i={i} \n")
            return
        d[i+1]=(d[i+1]-a[i]*d[i])/temp
        
    x = [float(d[n-1])]
    for i in range(n-1):
        x.append(d[n-i-2]-c[n-i-2]*x[-1])
    return x[::-1]

# utility function to print the tridiagonal matrix
def printm(A,B,C,D):
    n=len(D)
    X=np.zeros((n,n+1),dtype=np.float32)
    for i in range(n):
        X[i][i]=B[i]
        X[i][n]=D[i]
        if i>0 : 
            X[i][i-1]=A[i-1]
        if i<n-1:
            X[i][i+1]=C[i]
        
    print(X)
    return

# to solve general BVP problem
def BVP(p,q,r,n,a,b,a0,b0,c0,an,bn,cn):
    """
    a1,a0,b are the functions with equation
        y" = p*y'+q*y+r   on [a,b] 
    with "n" points required between them at uniform interval gap
    and
    a0*y(a)+b0*y'(a)=c0
    an*y(b)+bn*y'(b)=cn
    
    returns the value of y on the interval points
    """
    deri2_c1={ -1:1    , 0:-2  , 1:1 }
    deri1_c1={ -1:-0.5 , 0:0   , 1:0.5 }
    deri1_b2={ -1:1.5  , 0:-2  , 1:0.5}
    
    h = (b-a)/n
    A=[]
    B=[]
    C=[]
    D=[]
    
    B0 = {-1:(deri1_c1[-1]*b0),0:(deri1_c1[0]*b0+a0*h),1:(deri1_c1[1]*b0)}
    Bn = {-1:(deri1_c1[-1]*bn),0:(deri1_c1[0]*bn+an*h),1:(deri1_c1[1]*bn)}
    
    #print("B0",B0)
    #print("Bn",Bn)
    
    D0 = {-1:(deri2_c1[-1]-deri1_c1[-1]*h*p(a)),
           0:(deri2_c1[0]-h*p(a)*deri1_c1[0]-h*h*q(a)),
           1:(deri2_c1[1]-deri1_c1[1]*h*p(a))}
    Dn = {-1:(deri2_c1[-1]-deri1_c1[-1]*h*p(b)),
           0:(deri2_c1[0]-h*p(b)*deri1_c1[0]-h*h*q(b)),
           1:(deri2_c1[1]-deri1_c1[1]*h*p(b))}
    
    #print("D0",D0)
    #print("Dn",Dn)
    
    C0 = {0:D0[-1]*B0[0]-B0[-1]*D0[0],1:D0[-1]*B0[1]-B0[-1]*D0[1],'c':(c0*h*D0[-1]-B0[-1]*h*h*r(a))}
    Cn = {-1:Dn[1]*Bn[-1]-Bn[1]*Dn[-1],0:Dn[1]*Bn[0]-Bn[1]*Dn[0],'c':(cn*h*Dn[1]-Bn[1]*h*h*r(b))}
    
    #print("C0",C0)
    #print("Cn",Cn)
    
    # for x=a
    B.append(C0[0])
    C.append(C0[1])
    D.append(C0['c'])

    for i in range(1,n):
        x = a + i*h
        A.append(deri2_c1[-1]-deri1_c1[-1]*h*p(x))
        B.append(deri2_c1[0]-h*p(x)*deri1_c1[0]-h*h*q(x))
        C.append(deri2_c1[1]-deri1_c1[1]*h*p(x))
        D.append(h*h*r(x))
    
    # for x=b
    A.append(Cn[-1])
    B.append(Cn[0])
    D.append(Cn['c'])
    
    # printm(A,B,C,D)
    
    X = Thomas_algorithm(A,B,C,D)
    dX = dict()
    for i in range(0,n+1):
        dX[a+i*h]=X[i]

    return dX

