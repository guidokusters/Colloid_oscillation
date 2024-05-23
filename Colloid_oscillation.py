#Import
import numpy as np
from numba import jit

@jit(nopython=True)
def func(w0, beta, omega, no_osc, factor, eps):

    # Create time array information
    tf = max(2.*np.pi*no_osc/omega,1500)
    tsteps = int(np.ceil(tf/dt))

    # Create arrays to store dynamical order parameters
    z  = np.zeros((tsteps+1,3))
    
    # Create arrays to store polymer boundaries
    leftpolymer  = np.zeros(tsteps+1)
    rightpolymer  = np.zeros(tsteps+1)

    # Create parameters to check if in contact
    contact = 1.0
    # Direction of colloid motion
    direction = 1
    # Placeholders for boundaries of cavity if no contact (pure viscoelastic relaxation)
    xleft0 = 0.0
    xright0 = 0.0
    # Elastic rest positions cavity walls
    xrestleft = 0.0
    xrestright = 0.0
    # Contact while colloid moves away
    backcontact = 1.0

    # Solve dynamics iteratively
    for i in range(1, tsteps + 1):
        
        # Update time (use previous time step in RK4 scheme)
        t = (i-1)*dt

        # If moving left
        if direction == -1:
            # Driven harmonic oscillator
            k1 = np.array(osc_l([z[i-1,0], z[i-1,1], z[i-1,2]], t, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            k2 = np.array(osc_l(np.array([z[i-1,0], z[i-1,1], z[i-1,2]])+0.5*dt*k1, t+0.5*dt, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            k3 = np.array(osc_l(np.array([z[i-1,0], z[i-1,1], z[i-1,2]])+0.5*dt*k2, t+0.5*dt, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            k4 = np.array(osc_l(np.array([z[i-1,0], z[i-1,1], z[i-1,2]])+dt*k3, t+dt, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            z[i,:] = z[i-1,:] + dt/6.*(k1+2.*k2+2.*k3+k4)
                
        # If moving right
        else:
            # Driven harmonic oscillator
            k1 = np.array(osc_r([z[i-1,0], z[i-1,1], z[i-1,2]], t, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            k2 = np.array(osc_r(np.array([z[i-1,0], z[i-1,1], z[i-1,2]])+0.5*dt*k1, t+0.5*dt, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            k3 = np.array(osc_r(np.array([z[i-1,0], z[i-1,1], z[i-1,2]])+0.5*dt*k2, t+0.5*dt, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            k4 = np.array(osc_r(np.array([z[i-1,0], z[i-1,1], z[i-1,2]])+dt*k3, t+dt, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega))
            z[i,:] = z[i-1,:] + dt/6.*(k1+2.*k2+2.*k3+k4)
            
        # If there was contact
        if contact == 1.0:
        
            # When sphere changes direction contact will generally be lost
            if np.sign(z[i,1]) != np.sign(z[i-1,1]):

                # Contact is lost
                contact = 0.0
                
                # No back contact unless stated otherwise
                backcontact = 0.0

                # If switching to the right
                if np.sign(z[i,1]) == 1.0:
                    direction = 1
                    
                    leftpolymer[i] = (leftpolymer[i-1]+rightpolymer[i-1]-z[i,2])/2.
                    rightpolymer[i] = (leftpolymer[i-1]+rightpolymer[i-1]+z[i,2])/2.
                    
                    if z[i,0] >= rightpolymer[i] - eps:
                        contact = 1.0
                        if rightpolymer[i]<xrestright:
                            xrestright = rightpolymer[i]
                        rightpolymer[i] = max(z[i,0], rightpolymer[i])
                        leftpolymer[i] = rightpolymer[i] - z[i,2]
                        
                    if z[i,0] <= leftpolymer[i] + eps:
                        backcontact = 1.0
                        leftpolymer[i] = min(z[i,0], leftpolymer[i])
                        rightpolymer[i] = leftpolymer[i] + z[i,2]
                        
                    xleft0 = leftpolymer[i]
                    xright0 = rightpolymer[i]

                # If switching to the left
                else:
                    direction = -1
                    
                    leftpolymer[i] = (leftpolymer[i-1]+rightpolymer[i-1]-z[i,2])/2.
                    rightpolymer[i] = (leftpolymer[i-1]+rightpolymer[i-1]+z[i,2])/2.
                    
                    if z[i,0] <= leftpolymer[i] + eps:
                        contact = 1.0
                        if leftpolymer[i]>xrestleft:
                            xrestleft = leftpolymer[i]
                        leftpolymer[i] = min(z[i,0], leftpolymer[i])
                        rightpolymer[i] = leftpolymer[i] + z[i,2]
                    
                    if z[i,0] >= rightpolymer[i] - eps:
                        backcontact = 1.0
                        rightpolymer[i] = max(z[i,0], rightpolymer[i])
                        leftpolymer[i] = rightpolymer[i] - z[i,2] 
                    
                    xleft0 = leftpolymer[i]
                    xright0 = rightpolymer[i]

            # No change in direction: check if back contact 
            else:
                backcontact = 0.0
                if z[i,2] <= eps:
                    backcontact = 1.0
                if direction == 1:
                    leftpolymer[i] = z[i,0] - z[i,2]
                    rightpolymer[i] = z[i,0]
                else:
                    leftpolymer[i] = z[i,0]
                    rightpolymer[i] = z[i,0] + z[i,2]

        # If there was no contact
        else:

            # Update boundaries of free volume for contact condition
            xleft = (xleft0 + xright0 - z[i,2])/2.
            xright = (xleft0 + xright0 + z[i,2])/2.
            leftpolymer[i] = xleft
            rightpolymer[i] = xright
            
            # Update direction sphere is moving in
            direction = int(np.sign(z[i,1]))

            # If moving to the right
            if direction == 1:

                # If intercepting polymer
                if z[i,0] >= xright - eps:

                    # Again contact
                    contact = 1.0

                    # Update rest length for elastic force
                    if xright < xrestright:
                        xrestright = xright
                        
                    rightpolymer[i] = max(z[i,0], rightpolymer[i])
                    leftpolymer[i] = rightpolymer[i] - z[i,2]

                # Check for rear contact
                if z[i,0] <= xleft + eps:
                    backcontact = 1.0
                    leftpolymer[i] = min(z[i,0], leftpolymer[i])
                    rightpolymer[i] = leftpolymer[i] + z[i,2]
                    xleft0 = leftpolymer[i]
                    xright0 = rightpolymer[i]
                else:
                    backcontact = 0.0
                    xleft0 = leftpolymer[i]
                    xright0 = rightpolymer[i]

            # Moving to the left
            else:

                # If intercepting polymer while moing to the left
                if z[i,0] <= xleft + eps:

                    # Again contact
                    contact = 1.0

                    # Update rest length for elastic force
                    if xleft > xrestleft:
                        xrestleft = xleft
                        
                    leftpolymer[i] = min(z[i,0], leftpolymer[i])
                    rightpolymer[i] = leftpolymer[i] + z[i,2]
                    
                # Check for rear contact
                if z[i,0] >= xright - eps:
                    backcontact = 1.0
                    rightpolymer[i] = max(z[i,0], rightpolymer[i])
                    leftpolymer[i] = rightpolymer[i] - z[i,2]
                    xleft0 = leftpolymer[i]
                    xright0 = rightpolymer[i]
                else:
                    backcontact = 0.0
                    xleft0 = leftpolymer[i]
                    xright0 = rightpolymer[i]
                    
    return z

# Function for rightward motion
@jit(nopython=True)
def osc_r(z, t, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega):
    x, xdot, v = z
    dzdt = [xdot, -2*beta*xdot*max((1.-contact)*(1.-backcontact),-contact*direction*np.sign(xdot)*(1.-backcontact),(1.-contact)*backcontact*direction*np.sign(xdot))-factor*2*beta*xdot*max(contact*np.sign(xdot)*direction,-backcontact*np.sign(xdot)*direction)-w0**2.*(x-xrestright)*contact-w0**2.*min((x-xrestleft),0)*backcontact+np.sin(omega*t), max(xdot*direction,0.)*contact - v]
    return dzdt

# Function for leftward motion
@jit(nopython=True)
def osc_l(z, t, beta, w0, contact, backcontact, xrestleft, xrestright,direction,omega):
    x, xdot, v = z
    dzdt = [xdot, -2*beta*xdot*max((1.-contact)*(1.-backcontact),-contact*direction*np.sign(xdot)*(1.-backcontact),(1.-contact)*backcontact*direction*np.sign(xdot))-factor*2*beta*xdot*max(contact*np.sign(xdot)*direction,-backcontact*np.sign(xdot)*direction)-w0**2.*(x-xrestleft)*contact-w0**2.*max((x-xrestright),0)*backcontact+np.sin(omega*t), max(xdot*direction,0.)*contact - v]
    return dzdt

# Set (scaled) parameter values
w0 =0.7
beta = 0.05
# Option to include selective friction/screening
# Default value is 1.0
factor = 1.0
# Possibility to include a space cushion for the contact function
eps = 0.0
# Driving frequency
omega = 1.0

# Create time array
# Number of oscillations to average over
# The script will average over the last no_osc oscillations
# Verify that the system is indeed in steady state in this regime
no_osc = 15
# Time step
dt = 1e-4

# Run code
z = func(w0, beta, omega, no_osc, factor, eps)
# Compute limits of averaging procedure
no_osc = max(int(omega*max(2.*np.pi*no_osc/omega,1500)/(2.*np.pi)),no_osc)
index1 = round(2.*np.pi/(omega*dt)*int(no_osc/2))
index2 = round(2.*np.pi/(omega*dt)*no_osc)
# Compute average volume and amplitude
vavg = np.mean(z[index1:index2,2])
amp = (np.max(z[index1:index2,0])-np.min(z[index1:index2,0]))/2.
