#!/usr/bin/env python
# coding: utf-8

# 

# 
# - This program intends to simulate the continuous Photobleaching for Two Component Model,

#  where there are two populations:
#       fast population (Df, Bf) and 
#       slow population (Ds =  1/10 Df, Bs = 10 Bf).
# 

# ### Importing Packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time

# ### Image properties

# In[2]:


image_size = 80 # px (The image will be square 100 x 100 x 100)
pixel_size = 0.1 # um
boundary = image_size * pixel_size
#radius = image_size * pixel_size / 4 ;
radius =  3; # um
dwell_time = 0.001 # s
psf_width = 0.3 # um (Width of the point spread function in focus)
psf_height = 1.5 # 
Nparticles = 5000
step_time = 0.01 # s 

#B = 1e4 # Brightness, Hz
#diff_const = 1 # um^2/s (diffusion coefficient of mobile particles)


Ea  = 0.3 # um
Eb  = 0.3
Ec  = 2.1 

#z_slice = [0, 29, 59, 89, 119] # this is a list, to be used as a range in for loop
z_slice = [39, 69] # this is a list, to be used as a range in for loop

steps = 50000

wl = 0.488 # incident wavelength in µm
w0 = 0.3  # µm
cnst_a = 1


# In[18]:

# genreate initial positions of particles, which are inside of nucleus.
start_pos = np.zeros((Nparticles,3))

# Assigning brightness to the molecules.
start_B = np.ones((Nparticles,1))*2
start_B[0:Nparticles//7] = np.ones((Nparticles//7,1))*5;


for n in range(Nparticles):
    temp = start_pos[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x = np.random.rand(3) * 12
        if ((x[0] - 4)**2 + (x[1] - 4)**2 + (x[2] - 4)**2) < radius**2 :
            start_pos[n,:] = x

#center_pos = np.tile(np.array([radius*2, radius*2, radius*2]),(Nparticles,1))
center_pos = [4, 4, 4]
   

# In[19]:


# The measured pixel intensity is dependent on the distance between the particle and the scanning beam.

def GaussianBeam( start_pos, beam_pos, psf_width, psf_height):
    if start_pos.shape[0] == 2:
        GB = step_time*np.exp(- 2* ((start_pos - beam_pos)**2).sum()/ psf_width**2) #  B will be added later
    else:
        GB = step_time*np.exp(- 2* ((start_pos[0:2] - beam_pos[0:2])**2).sum()/ psf_width**2) * np.exp(-2*((start_pos[2]-beam_pos[2])**2/psf_height**2))
        
    return GB

    
# The photobleaching is dependent on the beam profile.

def BeamProfile(x_pos, y_pos, z_pos)   :
    Pb = cnst_a * np.exp(- (x_pos**2 + y_pos**2)/(w0**2*(1+(wl*z_pos/(np.pi*w0**2))**2)))/(1+(wl*z_pos/(np.pi*w0**2))**2) * step_time
    return Pb



# Generate "fast fraction" p from the Brightness matrix
def RatioofAmplitude(matrix, Ns, Nf, Nsteps,Nparticles):
    # By design, the first Ns rows in the matrix represent brightness for Ns number of slow particle.
    # and the rest of rows for Nf number of fast particle.
    matrix2 = np.square(matrix)
    
    ns = np.count_nonzero(matrix[:,0:Ns],axis=1)              
    nf = np.count_nonzero(matrix[:,Ns:Nparticles],axis=1)
    
    B = np.sum(matrix2[:,0:Ns],axis=1)
    A = np.sum(matrix2[:,Ns:Nparticles],axis=1)
    p = np.divide(A, (A+B))
    
    return p, ns, nf
            
    


# In[ ]:


pre_pos = np.zeros((steps+1,3,Nparticles))
pre_pos[0,:,:] = np.transpose(start_pos)

pre_B = np.zeros((steps+1,Nparticles))
pre_B[0,:] = np.transpose(start_B)

depth = np.zeros((steps,Nparticles))

loca = np.zeros((steps,3,Nparticles))
P = np.zeros((500,1))



CB = np.zeros((steps,Nparticles))

N_in_out = np.zeros((steps, 2)) # first row N_s, second row N_f
B_in = np.zeros((steps, 2))
P_det = np.zeros((steps, 2)) # first row: \sum \eta_s^2, second row: \sum \eta_f^2. 

# In[ ]:

t02 = time.time()

for i in range(steps):        
    # movement of particles
    for n in range(Nparticles):
        # Check if it is Bf or Bs.
        if n <= Nparticles//7:
            diff_const = 1
        else: 
            diff_const = 10
        
        # Do the random walk.
        track = np.random.normal(loc=0,scale=np.sqrt(2*diff_const*step_time),size=(1,3))
        
        depth[i,n] = np.sqrt(((pre_pos[i,:,n] - center_pos)**2).sum())
        forwd = np.sqrt(((pre_pos[i,:,n] + track - center_pos)**2).sum())

        if forwd <= radius:
                loca[i,:,n] = pre_pos[i,:,n] + track
        else:
               loca[i,:,n] = pre_pos[i,:,n]
                    
        pre_pos[i+1,:,n] = loca[i,:,n]
        
        CB[i,n] = pre_B[i,n]
        if CB[i,n] == 1 : # for fast pouplation            
            proba = np.random.rand()    
            if proba < BeamProfile(loca[i,0,n]-4, loca[i,1,n]-4, loca[i,2,n]-4) : #
                
                CB[i,n] = 0 # set B =0  from t =i to t =end .
        
        elif CB[i,n] > 1 : # for slow population
            Ns = int(CB[i,n])
            for m in range(Ns):
                proba = np.random.rand()    
                if proba < BeamProfile(loca[i,0,n]-4, loca[i,1,n]-4, loca[i,2,n]-4):
                    CB[i,n] -= 1
        
            
        
        # Check if the particle is inside detection volume
        # and if yes, if it is a bright or dark one.
        if (loca[i,0,n] - 4)**2/Ea**2 + (loca[i,1,n] - 4)**2/Eb**2 + (loca[i,2,n] - 4)**2/Ec**2 < 1 :
            if n <= Nparticles//7:  # slow particles
                N_in_out[i,0] += 1
                if CB[i,n] > 0:
                    B_in[i,0] += 1
                    P_det[i,0]+= CB[i,n]**2
            else: # fast particles
                N_in_out[i,1] += 1
                if CB[i,n] > 0:
                    B_in[i,1] += 1
                    P_det[i,1]+= CB[i,n]**2
                    
        pre_B[i+1,n] = CB[i,n]     
        

    
tn2 = time.time()
print('The time for calculating trajactories of molecules is ' + str((tn2- t02)/60) + 'min')
   
 


  
 # In[ ]               

[ p,ns,nf ] = RatioofAmplitude(CB, Nparticles//7, Nparticles*6//7 , steps, Nparticles)

p_det = P_det[:,1] / ( P_det[:,0] + P_det[:,1] )

# In[ ]:

z_slice = [39, 39] 
kk = [0, steps - image_size*image_size]   

image_array = np.zeros((image_size,image_size,len(z_slice)))
image_array_mobile = np.zeros((image_size,image_size,len(z_slice)))

t03 = time.time()
for n in range(Nparticles):

    for k in range(0,2): # two images, one at the begninning, one at the end
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,39]) * pixel_size # only scan the Z/2
                
                
                particle_pos = loca[ i + image_size * j + kk[k] ,:,n]
                image_array[i,j,k] += GaussianBeam(particle_pos,beam_pos,psf_width,psf_height)*CB[i + image_size * j + kk[k],n]*1e3
image_array_mobile = np.array(image_array)

tn3 = time.time()

print('The time for generating image from mobile molecules is ' + str((tn3 - t03)/60) + ' min')

# In[ ]:


#plt.imshow(image_array[:,:,1])

fig = plt.figure()
ax = plt.axes()

for z_m in range(0,2):
    
    im = plt.imshow(np.transpose(image_array_mobile[:, : ,z_m]),cmap ='viridis') #, vmin=0, vmax=85

    plt.xlabel('x (pixels)', fontsize=14)
    plt.ylabel('y (pixels)', fontsize=14)
    plt.title('z = ' + str(z_m))
    cbar = fig.colorbar(im, ax= ax)
    cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)
    plt.gca().invert_yaxis()
    
    c1 = plt.Circle((40, 40), 30, color='white', fill = '',linestyle = '--')
#    fig.add_subplot(111).add_artist(c1)


    filename = 'B_8um_Nparticles_' + str(Nparticles) + '_z_' + str(z_m) + '_mobile_wo_noise3.eps'
    plt.savefig(filename , dpi=300, bbox_inches='tight')    



# In[ ]:

## fit the recovery curve to obtain tau_P
## plot Nout Nin and ration in the first 1e4 steps


TimeLine =  np.arange(0, steps, 100)/100
p = p[::100]
nf = nf[::100]
ns = ns[::100]

p_det = p_det[::100]



# Plot N_in, N_in_B>0 as well as fit for N_in_B>0. 
fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position
plt.xlim(-2,max(TimeLine)+5)

plt.plot( TimeLine, nf, marker = 'o', color = 'darkorange', mfc = 'none')
plt.plot( TimeLine, ns, marker = 'o', color = 'green', mfc = 'none')



plt.title( 'a = '+ str(cnst_a) )
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('N in the nucleus' , fontsize=12)
plt.legend(['$n_{f}$ ',  '$n_{s}$ '], loc='upper right', fontsize=10)

plt.ylim(-20,max(nf)+20)


fig.tight_layout()
filename = 'Cnst_a_' + str(cnst_a)+'Nparticles_' + str(Nparticles)  + '_steps_' + str(steps)+'_NfNs.eps'
plt.savefig(filename, dpi=300, format='eps')
plt.show() 

# In[ ]:
# Plot N_in, N_in_B>0 as well as fit for N_in_B>0. 

#p_det = p_det[::100]
#TimeLine = TimeLine[::100]
fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position
plt.xlim(-2,max(TimeLine)+5)

plt.plot( TimeLine, p_det, '-', color = 'deeppink')
plt.plot( TimeLine, p, '-', color = 'aqua')
 
plt.title( 'a = '+ str(cnst_a) )
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('p ' , fontsize=12)
plt.legend(['detection V',' nucleus V'], loc='lower right', fontsize=10)

plt.ylim(min(p_det),max(p_det)+0.1)


fig.tight_layout()
filename = 'Cnst_a_' + str(cnst_a)+'Nparticles_' + str(Nparticles)  + '_steps_' + str(steps)+'_p_detV.eps'
plt.savefig(filename, dpi=300, format='eps')
plt.show() 

# In[ ]:
nin1 = N_in_out[::100,0]

nout1 = N_in_out[::100,1]

nBin = B_in[::100,0]

nBout = B_in[::100,1]

# Plot N_in, N_in_B>0 as well as fit for N_in_B>0. 
fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position
plt.xlim(-2,max(TimeLine)+5)

plt.plot( TimeLine, nin1, marker = 'o', color = 'darkorange', mfc = 'none')
plt.plot( TimeLine, nBin, marker = 'o', color = 'green', mfc = 'none')
plt.plot( TimeLine, nout1, marker = '>', color = 'red', mfc = 'none')
plt.plot( TimeLine, nBout, marker = '>', color = 'teal', mfc = 'none')



plt.title( 'a = '+ str(cnst_a) )
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Nf Ns in detection volume' , fontsize=12)
plt.legend(['$N_{s}$ ',  '$N_{s,B>0}$ ','$N_{f}$ ',  '$N_{f,B>0}$ ',], loc='upper right', fontsize=10)

plt.ylim(-10,max(nin1)+40)


fig.tight_layout()
filename = 'Cnst_a_' + str(cnst_a)+'Nparticles_' + str(Nparticles)  + '_steps_' + str(steps)+'_Nin.eps'
plt.savefig(filename, dpi=300, format='eps')
plt.show() 

# In[ ]:


data1 = np.column_stack((TimeLine, p, p_det, nf,ns, nin1, nBin, nout1, nBout ))
data1 = data1.astype(np.float32)

header = "R = " + str(radius)+ " ,D = " + str(diff_const) +" ,N = " + str(Nparticles) +  " ,pixels = " + str(image_size) + " \n"
header += "time, p, p_detV, nf, ns, ns_detV, ns_V_B>0, nf_detV, nf_V_B>0"
dataname = 'Cnst_a_' + str(cnst_a)+ 'N' + str(Nparticles) + '_steps_' + str(steps)+ 'R_ ' + str(radius)+ ' D_' + str(diff_const)  +'nBin_detV.txt'
np.savetxt(dataname, data1, header=header, fmt='%4.4f')







  
    