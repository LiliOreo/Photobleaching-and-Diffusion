#!/usr/bin/env python
# coding: utf-8

# # The principle behind the confocal image simulation - FRAP simualtion

# 
# - This program intends to simulate the short fast FRAP where only half the nucleus is bleached and then recovered.
# 
# 

# ### Importing Packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

# ### Image properties

# In[2]:


image_size = 120 # px (The image will be square 100 x 100 x 100)
pixel_size = 0.1 # um
boundary = image_size * pixel_size
#radius = image_size * pixel_size / 4 ;
radius =  3;
dwell_time = 0.001 # s
psf_width = 0.3 # um (Width of the point spread function in focus)
psf_height = 1.5 # 
diff_const = 10 # um^2/s (diffusion coefficient of mobile particles)
step_time = 0.001 # s 
B = 1e4 # Brightness, Hz

Nparticles = 5000

#z_slice = [0, 29, 59, 89, 119] # this is a list, to be used as a range in for loop
z_slice = [59, 89] # this is a list, to be used as a range in for loop

steps = image_size * image_size * len(z_slice)



# In[18]:


# genreate initial positions of particles, which are outside of nucleus.

start_pos = np.zeros((Nparticles,3))
for n in range(Nparticles):
    temp = start_pos[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x = np.random.rand(3) * 12
        if ((x[0] - 6)**2 + (x[1] - 6)**2 + (x[2] - 6)**2) < radius**2 and  x[0] < 5:
            start_pos[n,:] = x

#center_pos = np.tile(np.array([radius*2, radius*2, radius*2]),(Nparticles,1))
center_pos = [6, 6, 6]
# start_pos in um


## plot the start position for particles
ax = plt.axes(projection='3d')
ax.set_xlabel('X (\u03bcm)')
ax.set_ylabel('Y (\u03bcm)')
ax.set_zlabel('Z (\u03bcm)')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_zlim(0, 12)
ax.set_title('t = start')

for n in range(Nparticles):
    ax.scatter3D(start_pos[n,0],start_pos[n,1],start_pos[n,2],marker = 'o',color = 'orange', s = 6)
    
filename = 'B_12um_Nparticles_' + str(Nparticles) + '_t_start_FRAP.png'
plt.savefig(filename , dpi=300 )


# ### Calculating the pixel intensity

# The pixel intensity is dependent on the distance from the optical axis.

# In[19]:


def GaussianBeam( start_pos, beam_pos, psf_width, psf_height):
    if start_pos.shape[0] == 2:
        GB = B*step_time*np.exp(- 2* ((start_pos - beam_pos)**2).sum()/ psf_width**2)
    else:
        GB = B*step_time*np.exp(- 2* ((start_pos[0:2] - beam_pos[0:2])**2).sum()/ psf_width**2) * np.exp(-2*((start_pos[2]-beam_pos[2])**2/psf_height**2))
        
    return GB





# ### Display the particles at initial positions on an image
# - display a z slice of 3D data

# In[ ]:


t_0 = time.time()

image_array = np.zeros((image_size,image_size,len(z_slice)))
image_array_stationary = np.zeros((image_size,image_size,len(z_slice)))

for n in range(Nparticles):
    particle_pos = start_pos[n]   
    for k in z_slice : # z
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                
                    kk = round(k/30-2) ;
                    beam_pos = np.array([i,j,k]) * pixel_size 
                    
                    image_array[i,j,kk] += GaussianBeam(particle_pos,beam_pos,psf_width,psf_height)

image_array_stationary = np.array(image_array)

t_n = time.time()
print('The time for generating image from stationary molecules is ' + str((t_n- t_0)/60) + 'min')


# ### Display the particles at initial positions on an image
# - display a z slice of 3D data


    # In[ ]:


fig = plt.figure()
ax = plt.axes()

for z_s in range(0,2):
    
    im = plt.imshow(np.transpose(image_array_stationary[:, : ,z_s]),cmap ='viridis')
    plt.xlabel('x (pixels)', fontsize=14)
    plt.ylabel('y (pixels)', fontsize=14)
    plt.title('z = ' + str((z_s+2)*30) )
    cbar = fig.colorbar(im, ax= ax) #,boundaries=np.linspace(0,20,5)
    cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)
    #cbar.ax.set_clim(0, 20)
    
    c1 = plt.Circle((60, 60), 30, color='white', fill = '',linestyle = '--')
    fig.add_subplot(111).add_artist(c1)
    
    filename = 'B_12um_Nparticles_' + str(Nparticles) + '_z_' + str((z_s+2)*30) + '_stationary_wo_noise.png'
    plt.savefig(filename , dpi=300, bbox_inches='tight')


# In[ ]:




steps = image_size*image_size*len(z_slice)
#steps = 10000

pre_pos = np.zeros((steps+1,3,Nparticles))
pre_pos[0,:,:] = np.transpose(start_pos)
depth = np.zeros((steps,Nparticles))


track = np.random.normal(loc=0,scale=np.sqrt(2*diff_const*step_time),size=(steps,3,Nparticles))

loca = np.zeros((steps,3,Nparticles))


# In[ ]:
t02 = time.time()
for n in range(Nparticles):
    for i in range(steps):
    
        depth[i,n] = np.sqrt(((pre_pos[i,:,n] - center_pos)**2).sum())
        forwd = np.sqrt(((pre_pos[i,:,n] + track[i,:,n] - center_pos)**2).sum())

        if forwd <= radius:
                loca[i,:,n] = pre_pos[i,:,n] + track[i,:,n]

        else:
               loca[i,:,n] = pre_pos[i,:,n] 
            
    
        
        pre_pos[i+1,:,n] = loca[i,:,n]




tn2 = time.time()
print('The time for calculating trajactories of molecules is ' + str((tn2- t02)/60) + 'min')

# In[ ]:


#np.save("loca_Box_12um_Nparticles_30_traj_probability.npy",loca) # 1.2 GB!!
    
ax = plt.axes(projection='3d')
ax.set_xlabel('X (\u03bcm)')
ax.set_ylabel('Y (\u03bcm)')
ax.set_zlabel('Z (\u03bcm)')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_zlim(0, 12)
ax.set_title('t = end')

for n in range(Nparticles):
    ax.scatter3D(loca[-1,0,n],loca[-1,1,n],loca[-1,2,n],marker = 'o', color='cornflowerblue', s= 6)

filename = 'B_12um_Nparticles_' + str(Nparticles) + '_t_end.png'
plt.savefig(filename , dpi=300 )



# In[ ]:

image_array = np.zeros((image_size,image_size,len(z_slice)))
image_array_mobile = np.zeros((image_size,image_size,len(z_slice)))

t03 = time.time()
for n in range(Nparticles):

    for k in z_slice: # z
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,k]) * pixel_size
                
                kk = round(k/30 -2)
                particle_pos = loca[ i + image_size * j + image_size*image_size * kk ,:,n]
                image_array[i,j,kk] += GaussianBeam(particle_pos,beam_pos,psf_width,psf_height)
image_array_mobile = np.array(image_array)

tn3 = time.time()

print('The time for generating image from mobile molecules is ' + str((tn3 - t03)/60) + ' min')

# In[ ]:


#plt.imshow(image_array[:,:,1])

fig = plt.figure()
ax = plt.axes()

for z_m in range(0,2):
    
    im = plt.imshow(np.transpose(image_array_mobile[:, : ,z_m]),cmap ='viridis')

    plt.xlabel('x (pixels)', fontsize=14)
    plt.ylabel('y (pixels)', fontsize=14)
    plt.title('z = ' + str((z_m+2)*30))
    cbar = fig.colorbar(im, ax= ax)
    cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)

    filename = 'B_12um_Nparticles_' + str(Nparticles) + '_z_' + str((z_m+2)*30) + '_mobile_wo_noise.png'
    plt.savefig(filename , dpi=300, bbox_inches='tight')


# ## Adding Poisson noise to the image

# ### Stationary



# In[ ]:
## plot image of stationary molecules with/without noise

for z_s in range(0,2):

    image_array_intensity = np.transpose(image_array_stationary[:,:,z_s]*6)
    image_array_intensity += 1.2
    noisy = np.random.poisson(image_array_intensity)


    ## plot stationary image with noise
    fig = plt.figure(figsize=(4.27,3.2))

    im = plt.imshow(noisy,cmap ='viridis')


    plt.xlabel('x (pixels)', fontsize=14)
    plt.ylabel('y (pixels)', fontsize=14)
    plt.title('z = ' + str((z_s+2)*30) + ' w/ noise')
    cbar = fig.colorbar(im, ax= ax)
    cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)

    filename = 'Nparticles_' + str(Nparticles) + '_z_' + str((z_s+2)*30) + '_stationary_w_noise.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')


# ### Mobile

# In[ ]:
## plot image of mobile molecules with/without noise

for z_m in range( 0,2):

    image_array_intensity = np.transpose(image_array_mobile[:,:,z_m]*6)
    image_array_intensity += 2.2
    noisy = np.random.poisson(image_array_intensity)


    ## plot image for mobile particles with noise
    fig = plt.figure(figsize=(4.27,3.2))


    im = plt.imshow(noisy,cmap ='viridis')


    plt.xlabel('x (pixels)', fontsize=14)
    plt.ylabel('y (pixels)', fontsize=14)
    plt.title('z = ' + str((z_m+2)*30) + ' w/ noise')
    cbar = fig.colorbar(im, ax= ax)
    cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)

    filename = 'Nparticles_' + str(Nparticles) + '_z_' + str((z_m+2)*30) + '_mobile_w_noise.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
##



# In[ ]:
TimeLine = np.arange(0, steps, 1) 
print('the shape is '+ str(TimeLine.shape))



# In[ ]:
    
## Determine the number of molecules inside and outside
t04 = time.time()
N_low_top = np.zeros((steps, 2)) # first row N_out, second row N_in
for t in range(steps):
    for n in range(Nparticles):
        if loca[t,0,n] < 5:
            N_low_top[t,0] += 1
        else:
            N_low_top[t,1] += 1
        #timecheck = 't=' + str(t) + ',n=' + str(n)
        #print(timecheck)
tn4 = time.time()
print('The time for calculating ratio of Nin/Nout is ' + str((tn4 - t04)/60) + ' min')

# In[ ]:
## plot Nout Nin and ration in the full range with an interval of 1e4

TimeLine =  np.arange(0, steps, 100)/1000 
nlow = N_low_top[::100,1]

ntop = N_low_top[::100,0]
Ratio = ntop/nlow

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position

plt.plot(TimeLine, nlow, '>', TimeLine, ntop,'o', TimeLine, Ratio,'+' )
plt.ylabel('molecule number', fontsize=14)
plt.xlabel('time (s)', fontsize=14)


plt.legend(['$N_{low}$', '$N_{top}$','$N_{top}/N_{low}$'],loc='upper right')

fig.tight_layout()
filename = 'Nparticles_' + str(Nparticles) + '_N_top_low_Ratio.png'
plt.savefig(filename, dpi=300)
plt.show()  






# In[ ]:

## fit the recovery curve to obtain tau_P
## plot Nout Nin and ration in the first 1e4 steps
t1 = 10000

TimeLine1 =  np.arange(0, t1,1) /1000
nlow1 = N_low_top[0:t1,1]

ntop1 = N_low_top[0:t1,0]



def func(t, N0, tauP):
    return N0 * (1 -  np.exp(-t / tauP) )

popt, pcov = curve_fit(func, TimeLine1, nlow1)

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position


plt.plot( TimeLine1, ntop1, marker = 'o', color = 'darkorange', mfc = 'none')
plt.plot( TimeLine1, nlow1, marker = '>', color = 'green', mfc = 'none')
plt.plot( TimeLine1, func(TimeLine1, *popt), '-', color = 'indigo')
plt.xlabel('time (s)', fontsize=12)
plt.ylabel('molecule number', fontsize=12)
plt.legend(['$N_{low}$ ', ' $N_{top}$', 'fit'], loc='upper right', fontsize=12)

plt.text(2, 1060, '$N = N_0 *(1- exp(-\u03C4 / \u03C4_P ) )$ ' , fontsize=12)
plt.text(2, 500, '$N_0$ = ' + str(round(popt[0],0)), fontsize=12)
plt.text(2, 40, '$\u03C4_P$ = ' + str(round(popt[1],2)) + ' s', fontsize=12)


fig.tight_layout()
filename = 'Nparticles_' + str(Nparticles) + '_first_' + str(t1) +'_Ntop_fit.png'
plt.savefig(filename, dpi=300)
plt.show() 







# In[ ]:


data1 = np.column_stack((nlow1, ntop1))
data1 = data1.astype(np.int32)

header = "R = " + str(radius)+ " ,D = " + str(diff_const) +" ,N = " + str(Nparticles) +  " ,pixels = " + str(image_size) + " \n"
header += "N_low, N_top "
dataname = 'N' + str(Nparticles) + 'R_ ' + str(radius)+ ' D_' + str(diff_const)  +'Nlow_Ntop.txt'
np.savetxt(dataname, data1, header=header, fmt='%-4d')







  
    