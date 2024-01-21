#!/usr/bin/env python
# coding: utf-8
#
# Written by Lili Zhang, 2022
# Modified and annoted by Cécile Fradin, 2023
# 
# This program simulates the continuous photobleaching occuring inside a cell nucleus
# caused by a focused laser beam (e.g. during a single point FCS experiment).
#
# Two separate populations of fluorophore are considered:
#    - A fast monomeric population (Df, Bm)
#    - A slow oligomeric population (Ds, Bs = Ss*Bf, fraction Fs)
#
# The program returns the expected value of the relative amplitude of the slow dynamic
# term if one were to conduct an FCS experiment in that nucleus 
#
#
# In[1]:
    
### Importing Packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


# In[2]:

    
### Physical properties

radius = 3; # um (radius of the nucleus)
radius_squared = radius * radius 
Nparticles = 10000 # (number of particles)
Bm = 1e4 # Hz (brightness of a monomer)
Df = 15 # um^2/s (diffusion coefficient of fast particles)
Ds = 0.4 # um^2/s (diffusion coefficient of slow particles)
Avonano = 6.02e14 # particles per nM
Concentration = Nparticles / Avonano / (4/3*np.pi*radius**3*1e-15)
print('The total particle concentration is: ' + f'{Concentration:.3}' + ' nM')

### Calculate or input the concentration and brightness of the oligomers

CalculateOligomerBrightness = False
if (CalculateOligomerBrightness): 
    p0 = 0.65 # Initial amplitude of slow term in the autocorrelation function
    pinf = 0.27 # Limit value of the amplitude of slow term in the autocorrelation function
    Ss = (1/pinf-1)/(1/p0-1)   # (Stoichiometry of slow particles - needs to be larger than 1)
    Fs =  (1/p0-1)/((1/p0-1)+(1/pinf-1)**2)   # (Fraction of slow particles )
    NsoverNf = (1/p0-1)/((1/pinf-1)**2)  # (Number of slow particles over number of fast particles)
else:
    Ss = 3.5 
    Fs = 0.08 
    NsoverNf = Fs/(1-Fs)
print('The total fluorophore concentration is: ' + f'{Concentration*(1+Fs*(Ss-1)):.3}' + ' nM')
print('The fraction of slow oligomers is: ' + f'{Fs:.3}')
print('Nf/Ns = ' + f'{NsoverNf:.3}')
print('The average stoichiometry of the slow oligomers is: ' + str(Ss))
    

### Optical properties

wl = 0.488 # um (excitation wavelength)
w0 = 0.3 # um (1/e^2 radius of the point spread function)
z0 = 1.5 # um (1/e^2 half-height of the point spread function)
cnst_a = 3 # (photobleaching strength)


### Image properties

image_size = 80 # px (images will be square)
pixel_size = 0.1 # um
dwell_time = 0.001 # s


### Simulation properties

w_rad = image_size * pixel_size / 2 # um (size of the simulation window)
r_c = [w_rad, w_rad, w_rad] # (coordinates of the simulation window center)
step_time = 0.01 # s (time between two consecutive steps)
i_time = image_size * image_size * dwell_time / step_time  # Time necessary to record an image in number of steps
steps = int(30 * i_time)   # Needs to be larger than i_time if we want to have time to record an image


# In[3]:
    
### Initialize system

# Start the clock

time1 = time.time()

# Generate initial positions of particles

start_pos = np.zeros((Nparticles,3))
for n in range(Nparticles):         # Make sure all particles are inside the nucleus 
    x = start_pos[n,:]
    while ((x[0:3] - r_c[0:3])**2).sum() > radius_squared :
        x = np.random.rand(3) * w_rad * 2
    start_pos[n,:] = x            

# Assign a specific brightness to each particle (in monomer unit)

Nps = int(Nparticles*Fs)     # Number of slow oligomers
Npf = Nparticles - Nps       # Number of fast monomers
start_B = np.ones((Nparticles,1))
# Method 1: Round up to next integer
#    start_B[n] = start_B[n] * int(Ss+1)
# Method 2: Round up or down to obtain correct average stoichiometry
#    r = Ss % 1                     
#    if (np.random.rand(1) < (1-r)):
#        start_B[n] = start_B[n] * int(Ss)
#    else:
#        start_B[n] = start_B[n] * int(Ss+1) 
# Method 3: Poisson distribution of stoichiometries
for n in range(Nps):
    start_B[n] = start_B[n] * np.random.poisson(Ss)
   
# Plot histogram of particle stoichiometry 
plt.hist(start_B, bins=np.arange(-0.5, int(max(start_B)) + 1.5, 1), color='skyblue', edgecolor='black')
plt.xlabel('Stoichiometry')
plt.ylabel('Frequency')
plt.title('Distribution of stoichiometries')
plt.show()


# In[4]:

### Define useful functions


# Pixel intensity contributed by a particle (not yet multiplied by the specific brightness)

def GaussianBeam(particle_pos, beam_pos, w0, z0):
    if particle_pos.shape[0] == 2:
        GB = dwell_time * np.exp(- 2* ((particle_pos - beam_pos)**2).sum()/w0**2) 
    else:
        GB = dwell_time * np.exp(- 2* ((particle_pos[0:2] - beam_pos[0:2])**2).sum()/w0**2) * np.exp(-2*((particle_pos[2]-beam_pos[2])**2/z0**2))        
    return GB
    
# Photobleaching probability (for laser beam focused at the center of the nucleus)

def BeamProfile(x_pos, y_pos, z_pos)   :
    wz_over_w0_squared = (1 + (wl*z_pos/(np.pi*w0**2))**2)
    Pb = step_time * cnst_a * np.exp(- 2*(x_pos**2 + y_pos**2)/(w0**2*wz_over_w0_squared))/wz_over_w0_squared
    return Pb

# Generate the "relative amplitude of slow fraction", p, from the Brightness matrix

def RatioofAmplitude(matrix, Ns, Nf, Nparticles):    
    ns = np.count_nonzero(matrix[:,0:Ns],axis=1)      # Number of slow particles that are still fluorescent
    nf = np.count_nonzero(matrix[:,Ns:Nparticles],axis=1) # Number of fast particles that are still fluorescent
    ints = np.sum(matrix[:,0:Ns],axis=1)   # intensity contributed by slow particles
    intf = np.sum(matrix[:,Ns:Nparticles],axis=1)   # intensity contributed by fast particles
    inttot = ints + intf
    matrix2 = np.square(matrix)
    B = np.sum(matrix2[:,0:Ns],axis=1)
    A = np.sum(matrix2[:,Ns:Nparticles],axis=1)
    p = np.divide(B,(A+B))
    return p, ns, nf, ints, intf, inttot

# Exponential fitting function

def func(x, a, b, c):
    return a * np.exp(- x/b) + c
            

# In[5]:

### Initialize tables that will hold simulation results    

pre_pos = np.zeros((steps+1,3,Nparticles))   # Particle positions at all time steps
pre_pos[0,:,:] = np.transpose(start_pos)

pre_B = np.zeros((steps+1,Nparticles))    # Particle brightness at all time steps
pre_B[0,:] = np.transpose(start_B)

N_in_out = np.zeros((steps + 1, 2))  # Number of particles in detection volume (first row: slow particles, second row: fast particles)
B_in = np.zeros((steps + 1, 2))   #  Number of fluorescent particles in detection volume 
P_det = np.zeros((steps + 1, 2)) # Sum squared of signal contributed by particles in detection volum 

for n in range(Nparticles):
# Count bright particles in detection volume (defined as the 1/eˆ2-radius ellipsoid)
    if ((pre_pos[0,0,n] - r_c[0])**2 < w0**2) and ((pre_pos[0,1,n] - r_c[1])**2 < w0**2) and ((pre_pos[0,2,n] - r_c[2])**2 < z0**2) :
        CB_gard = int(pre_B[0,n])   
        if n <= Nps:  # slow particles
            N_in_out[0,0] += 1
            if CB_gard > 0:
                B_in[0,0] += 1
                P_det[0,0]+= CB_gard**2
        else: # fast particles
            N_in_out[0,1] += 1
            if CB_gard > 0:
                B_in[0,1] += 1
                P_det[0,1]+= CB_gard**2
                    

# In[6]:

### Calculate particle trajectories

time2 = time.time()

for i in range(steps):           # At each step in the simulation...
    for n in range(Nparticles):       # And for each particle...

        if n <= Nps:              # Assign a diffusion constant
            diff_const = Ds
        else: 
            diff_const = Df
        
    # Calculate particle displacement
        track = np.random.normal(loc=0,scale=np.sqrt(2*diff_const*step_time),size=(1,3))
        
    # Refuse moves that see the particle cross the nuclear membrane
        forwd_squared = ((pre_pos[i,:,n] + track - r_c)**2).sum()   # Distance from center of nucleus
        if forwd_squared <= radius_squared:
            pre_pos[i+1,:,n] = pre_pos[i,:,n] + track
        else: 
            pre_pos[i+1,:,n] = pre_pos[i,:,n]
 
    # Update the brightness of the particle due to eventual photobleaching
        CB_gard = int(pre_B[i,n])
        if CB_gard > 0 : # for all fluorescent particles
            for m in range(CB_gard):    # loop over all the fluorophores in the particle...
                proba = np.random.rand()    
                if proba < BeamProfile(pre_pos[i+1,0,n]-r_c[0], pre_pos[i+1,1,n]-r_c[1], pre_pos[i+1,2,n]-r_c[2]):
                    CB_gard -= 1     # photobleach the fluorophore with a probability depending on its position
                    
    # Count bright particles in detection volume (defined as the 1/eˆ2-radius ellipsoid)
        if ((pre_pos[i+1,0,n] - r_c[0])**2 < w0**2) and ((pre_pos[i+1,1,n] - r_c[1])**2 < w0**2) and ((pre_pos[i+1,2,n] - r_c[2])**2 < z0**2) :
            if n <= Nps:                 # slow particles/oligomers
                N_in_out[i+1,0] += 1
                if CB_gard > 0:
                    B_in[i+1,0] += 1
                    P_det[i+1,0]+= CB_gard**2
            else:                        # fast particles
                N_in_out[i+1,1] += 1
                if CB_gard > 0:
                    B_in[i+1,1] += 1
                    P_det[i+1,1]+= CB_gard**2
                    
        pre_B[i+1,n] = CB_gard     
        

time3 = time.time()
print('Time to calculate ' + str(Nparticles) + ' particle trajectories for ' + str(steps) + ' steps: ' + f'{(time3-time2)/60:.3}' + 'min')

  
 # In[7]               

### Calculate the amplitude of the slow term in the ACF (p)

# Call the procedure that calculates p for the whole nucleus

[p_gard,ns_gard,nf_gard,A_gard,B_gard,S_gard] = RatioofAmplitude(pre_B, Nps, Npf, Nparticles)   

# Calculate p for the detection volume (as the sum of the slow particles' squared-intensity)
p_det_gard=np.zeros(steps+1)
for s in range(steps+1):
    if (P_det[s,0] + P_det[s,1])>0:
        p_det_gard[s]=P_det[s,0] / ( P_det[s,0] + P_det[s,1] )
    else:
        p_det_gard[s] = 0     # If there is no particle in the detection volume we consider that  p = 0


# In[8]:
    
### Generate two images of the system, one at the beginning, one at the end

time4 = time.time()

z_slice = 39      # Only a sagital image of the nucleus will be recorded 
kk = [0, steps + 1 - i_time]    # Imaging will be initiated at two time points

image_array = np.zeros((image_size,image_size,2))

for k in range(0,2): # two images, one at the begninning, one at the end
   for j in range(image_array.shape[1]): # x
        for i in range(image_array.shape[0]): # y
            beam_pos = np.array([i,j,z_slice]) * pixel_size # Position of the beam
            s_time = int(kk[k] + (image_size * j + i) * dwell_time / step_time)  # Time in number of steps
            for n in range(Nparticles):  # particles
                particle_pos = pre_pos[s_time,:,n]    # Particle position at that time
                image_array[i,j,k] += GaussianBeam(particle_pos,beam_pos,w0,z0)* pre_B[s_time,n] * Bm     # Intensity in photon per pixel (no photon noise)
            image_array[i,j,k] =  np.random.poisson(image_array[i,j,k])        # Intensity in photon per pixel (with photon noise)

time5 = time.time()

print('Time taken to generate two confocal images: ' + f'{(time5-time4):.3}' + 's')



# In[9]:

### Plot and save the first image

fig = plt.figure()
ax = plt.axes()

t = 0
maxintL=image_array[:, : ,t].max()

im = plt.imshow(np.transpose(image_array[:, : ,t]),cmap ='viridis', vmin=0, vmax=maxintL)
plt.xlabel('x (pixels)', fontsize=14)
plt.ylabel('y (pixels)', fontsize=14)
plt.title('Time = ' + str(int(kk[t] * step_time)) + ' s')
cbar = fig.colorbar(im, ax= ax)
cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)
plt.gca().invert_yaxis()
    
c1 = plt.Circle((40, 40), 30, color='white', fill = '',linestyle = '--')
  #  fig.add_subplot(111).add_artist(c1)

filename = 'B_8um_Nparticles_' + str(Nparticles) + '_time_' + str(kk[t] * step_time) + '_mobile_wo_noise3.eps'
plt.savefig(filename , dpi=300, bbox_inches='tight')    


# In[10]:

### Plot and save the last image

fig = plt.figure()
ax = plt.axes()

t = 1
maxintS=image_array[:, : ,t].max()

im = plt.imshow(np.transpose(image_array[:, : ,t]),cmap ='viridis',vmin=0, vmax=maxintS)
plt.xlabel('x (pixels)', fontsize=14)
plt.ylabel('y (pixels)', fontsize=14)
plt.title('Time = ' + str(int(kk[t] * step_time)) + ' s')
cbar = fig.colorbar(im, ax= ax)
cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)
plt.gca().invert_yaxis()
    
c1 = plt.Circle((40, 40), 30, color='white', fill = '',linestyle = '--')
#  fig.add_subplot(111).add_artist(c1)

filename = 'B_8um_Nparticles_' + str(Nparticles) + '_time_' + str(kk[t] * step_time) + '_mobile_wo_noise3.eps'
plt.savefig(filename , dpi=300, bbox_inches='tight')  
  

# In[11]:
    
### Visualize the temporal evolution of the fluorescence intensity
    
skip = int(100)    # Only show data every 100th time steps

TimeLine =  np.arange(0, steps+1, skip) * step_time
npoints=len(TimeLine)

Aplot = A_gard[::skip]
Bplot = B_gard[::skip]
Splot = S_gard[::skip]     # This one will be fitted

poptA, pcovA = curve_fit(func, TimeLine, Aplot)
poptB, pcovB = curve_fit(func, TimeLine, Bplot)
poptS, pcovS = curve_fit(func, TimeLine, Splot)

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height] 
ax.set_position(pos2) # set a new position

plt.xlim(-2,max(TimeLine)+5)
plt.ylim(0,max(Splot)+100)

plt.plot( TimeLine, Bplot, marker = 'o', color = 'darkorange', mfc = 'none')
plt.plot( TimeLine, Aplot, marker = 'o', color = 'green', mfc = 'none')
plt.plot( TimeLine, Splot, marker = 'o', color = 'black', mfc = 'none')
plt.plot( TimeLine, func(TimeLine, *poptA), 'r-', label="Fitted Curve")
plt.plot( TimeLine, func(TimeLine, *poptB), 'r-', label="Fitted Curve")
plt.plot( TimeLine, func(TimeLine, *poptS), 'r-', label="Fitted Curve")

plt.title( 'Photobleaching rate, a = '+ str(cnst_a) + ' /s' )
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Total Nuclear Fluorescence' , fontsize=12)
plt.legend(['Fast monomers',  'Slow oligomers', 'Sum'], loc='upper right', fontsize=10)

fig.tight_layout()
filename = 'Cnst_a_' + str(cnst_a)+'Nparticles_' + str(Nparticles)  + '_steps_' + str(steps)+'_Fluorescence.eps'
plt.savefig(filename, dpi=300, format='eps')
plt.show() 

print("The characteristic photobleaching time is: " + f'{poptS[1]:.3}' + 's' )
    
    
# In[12]:
    
### Visualize the temporal evolution of the total number of fluorescent particles

nf = nf_gard[::skip]
ns = ns_gard[::skip]
nsum = nf + ns

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height] 
ax.set_position(pos2) # set a new position

plt.xlim(-2,max(TimeLine)+5)
plt.ylim(0,Nparticles*1.05)

plt.plot( TimeLine, nf, marker = 'o', color = 'darkorange', mfc = 'none')
plt.plot( TimeLine, ns, marker = 'o', color = 'green', mfc = 'none')
plt.plot( TimeLine, nsum, marker = 'o', color = 'black', mfc = 'none')

plt.title( 'Photobleaching rate, a = '+ str(cnst_a) + ' /s' )
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Total # of fluorescent particles' , fontsize=12)
plt.legend(['Fast monomers',  'Slow oligomers', 'All particles'], loc='upper right', fontsize=10)

fig.tight_layout()
filename = 'Cnst_a_' + str(cnst_a)+'Nparticles_' + str(Nparticles)  + '_steps_' + str(steps)+'_NfNs.eps'
plt.savefig(filename, dpi=300, format='eps')
plt.show() 


# In[13]:
    
### Visualize the temporal evolution of p

for i in range(steps+1):
    if np.isnan(p_gard[i]): p_gard[i] = -0.1     # removes NaN values and store them as negative values
p = p_gard[::skip]
p_det = p_det_gard[::skip]

# Calculate a more experimentally realistic value of p

tint = 5 # s (duration of FCS experiment)
skipA = int(5/step_time)    # Only show data every 100th time steps

TimeLineA =  np.arange(0, steps-skipA, skipA) * step_time # Alternative timeline
npointsA=len(TimeLineA)

p_measA = np.zeros(npointsA)
for k in range(npointsA):
    cc=skipA
    for i in range(skipA):
        if (np.isnan(p_det_gard[k*skipA+i])):
            cc -= 1
        else:
            p_measA[k] += p_det_gard[k*skipA+i]
    if (cc>0):
        p_measA[k] = p_measA[k] / cc
    else:
#        p_measA[k] = float("NaN")
        p_measA[k] = -0.1

poptp, pcovp = curve_fit(func, TimeLine, p)
poptA, pcovA = curve_fit(func, TimeLineA, p_measA)

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position
plt.xlim(-2,max(TimeLine)+5)

plt.plot( TimeLine, p_det, '-', color = 'green')
plt.plot( TimeLineA, p_measA, '-', color = 'black')
plt.plot( TimeLine, p, '-', color = 'lightgreen')
plt.plot( TimeLine, func(TimeLine, *poptp), 'r-', label="Fitted Curve")
plt.plot( TimeLineA, func(TimeLineA, *poptA), 'r-', label="Fitted Curve")
 
plt.title( 'a = '+ str(cnst_a) )
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('p ' , fontsize=12)
plt.legend(['Detection volume','Detection volume integrated','Whole nucleus'], loc='upper right', fontsize=10)

plt.ylim(min(p_det),max(p_det)+0.1)


fig.tight_layout()
filename = 'Cnst_a_' + str(cnst_a)+'Nparticles_' + str(Nparticles)  + '_steps_' + str(steps)+'_p_detV.eps'
plt.savefig(filename, dpi=300, format='eps')
plt.show() 


# In[14]:
        
### Visualize the temporal evolution of the number of particles in the detection volume
    
nin1 = N_in_out[::100,0]
nout1 = N_in_out[::100,1]
nBin = B_in[::100,0]
nBout = B_in[::100,1]

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position
plt.xlim(-2,max(TimeLine)+5)
plt.ylim(-1,max(nout1)+5)

plt.plot( TimeLine, nout1, marker = '>', color = 'yellow', mfc = 'none')
plt.plot( TimeLine, nBout, marker = '>', color = 'darkorange', mfc = 'none')
plt.plot( TimeLine, nin1, marker = 'o', color = 'lightgreen', mfc = 'none')
plt.plot( TimeLine, nBin, marker = 'o', color = 'green', mfc = 'none')

plt.title( 'a = '+ str(cnst_a) )
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Particles in detection volume' , fontsize=12)
plt.legend(['Fast particles',  'Fast particles, $B>0$', 'Slow particles',  'Slow particles, $B>0$ '], loc='right', fontsize=10)

fig.tight_layout()
filename = 'Cnst_a_' + str(cnst_a)+'Nparticles_' + str(Nparticles)  + '_steps_' + str(steps)+'_Nin.eps'
plt.savefig(filename, dpi=300, format='eps')
plt.show() 


# In[15]:
    
### Save data

data1 = np.column_stack((TimeLine, p, p_det, ns, nf, nin1, nBin, nout1, nBout, Aplot, Bplot, Splot))
data1 = data1.astype(np.float32)

header1 = "a = " + str(cnst_a) +  ", R = " + str(radius)+ " um, Df = " + str(Df) + ' um^2/s, Ds = ' + str(Ds) +' um^2/s, Ss = ' + str(Ss) + ' , Fs = ' + str(Fs) + ' , N = ' + str(Nparticles) + ' , steps = ' + str(steps) + " \n"
header1 += "time, p, p_detV, Ns, Nf, Ns_detV, Ns_detV_B>0, Nf_detV, Nf_detV_B>0, Is, If, IT"
dataname1 = 'Results_a_' + str(cnst_a) + '_R_' + str(radius)+ '_Df_' + str(Df) + '_Ds_' + str(Ds) +'_S_' + str(Ss) + '_Fs_' + str(Fs) + '_N_' + str(Nparticles) + '_ns_' + str(steps) + '.txt'
np.savetxt(dataname1, data1, header=header1, fmt='%4.4f')

data2 = np.column_stack((TimeLineA, p_measA))
data2 = data2.astype(np.float32)

header2 = "a = " + str(cnst_a) +  ", R = " + str(radius)+ " um, Df = " + str(Df) + ' um^2/s, Ds = ' + str(Ds) +' um^2/s, Ss = ' + str(Ss) + ' , Fs = ' + str(Fs) + ' , N = ' + str(Nparticles) + ' , steps = ' + str(steps) + " \n"
header2 += "time, p_measA"
dataname2 = 'Results_Appendix_a_' + str(cnst_a) + '_R_' + str(radius)+ '_Df_' + str(Df) + '_Ds_' + str(Ds) +'_S_' + str(Ss) + '_Fs_' + str(Fs) + '_N_' + str(Nparticles) + '_ns_' + str(steps) + '.txt'
np.savetxt(dataname2, data2, header=header2, fmt='%4.4f')

# In[16]

### The end

time6 = time.time()
print('Total run time: ' + f'{(time6-time1)/60:.3}' + 'min')

print('\a')

    