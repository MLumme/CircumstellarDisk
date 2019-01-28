import numpy as np
from numpy.random import uniform as urand
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy
import sys
from astropy.constants import au,M_sun,pc,h,c


#=====================================================================
#Construct array containing grid vertices
#=====================================================================
def vertbuild(dphi,dz,dr,phi_min,z_min,r_min,cutoff):
    
    grid_vertex = np.ndarray(shape=(n_r+1,n_z+1,n_phi+1,3),dtype=float)

    if(cutoff):
        range_0 = 1        
        #set first vertices at x=y=0
        for i in range(0,n_z+1):
            grid_vertex[:,i,:,0] = 0.0
            grid_vertex[:,i,:,1] = 0.0
            grid_vertex[:,i,:,2] = z_min+dz*i
    else:
        range_0 = 0
            
    #disk structure in cartesian coordinates
    for i in range(range_0,n_r+1):
        for j in range(0,n_z+1):
            for k in range(0,n_phi+1):
                grid_vertex[i,j,k,0] = (r_min+dr*(i-range_0)) \
                    *scipy.cos(phi_min+dphi*k)
                grid_vertex[i,j,k,1] = (r_min+dr*(i-range_0)) \
                    *scipy.sin(phi_min+dphi*k)
                grid_vertex[i,j,k,2] = z_min+dz*j

    return grid_vertex 
    
#=====================================================================
#Construct array containing cells and cell walls
#=====================================================================
def cellbuild(grid_vertex):
    global grid_cell

    grid_cell = np.ndarray(shape=(n_r,n_z,n_phi,6,4,3),dtype=float)

    for i in range(0,n_r):
        for j in range(0,n_z):
            for k in range(0,n_phi):

                #inner wall
                grid_cell[i,j,k,0,0,:] = grid_vertex[i,j,k,:]
                grid_cell[i,j,k,0,1,:] = grid_vertex[i,j,k+1,:]
                grid_cell[i,j,k,0,2,:] = grid_vertex[i,j+1,k,:]
                grid_cell[i,j,k,0,3,:] = grid_vertex[i,j+1,k+1,:]
        
                #outer wall
                grid_cell[i,j,k,1,0,:] = grid_vertex[i+1,j,k,:]
                grid_cell[i,j,k,1,1,:] = grid_vertex[i+1,j,k+1,:]
                grid_cell[i,j,k,1,2,:] = grid_vertex[i+1,j+1,k,:]
                grid_cell[i,j,k,1,3,:] = grid_vertex[i+1,j+1,k+1,:]
    
                #bottom wall
                grid_cell[i,j,k,2,0,:] = grid_cell[i,j,k,0,0,:] 
                grid_cell[i,j,k,2,1,:] = grid_cell[i,j,k,0,1,:]  
                grid_cell[i,j,k,2,2,:] = grid_cell[i,j,k,1,0,:]
                grid_cell[i,j,k,2,3,:] = grid_cell[i,j,k,1,1,:]

                #top wall
                grid_cell[i,j,k,3,0,:] = grid_cell[i,j,k,0,2,:] 
                grid_cell[i,j,k,3,1,:] = grid_cell[i,j,k,0,3,:]   
                grid_cell[i,j,k,3,2,:] = grid_cell[i,j,k,1,2,:]
                grid_cell[i,j,k,3,3,:] = grid_cell[i,j,k,1,3,:]
    
                #clockwise wall
                grid_cell[i,j,k,4,0,:] = grid_cell[i,j,k,0,0,:] 
                grid_cell[i,j,k,4,1,:] = grid_cell[i,j,k,1,0,:]
                grid_cell[i,j,k,4,2,:] = grid_cell[i,j,k,0,2,:]
                grid_cell[i,j,k,4,3,:] = grid_cell[i,j,k,1,2,:]
        
                #counterclockwise wall
                grid_cell[i,j,k,5,0,:] = grid_cell[i,j,k,0,1,:]
                grid_cell[i,j,k,5,1,:] = grid_cell[i,j,k,1,1,:]  
                grid_cell[i,j,k,5,2,:] = grid_cell[i,j,k,0,3,:] 
                grid_cell[i,j,k,5,3,:] = grid_cell[i,j,k,1,3,:]
    
    return grid_cell

#=====================================================================
#cell density array
#=====================================================================
def densitybuild(cutoff,R_0,h_0,m_disk,r_min,r_max,z_mid):
    
    global rho_arr
    
    #density array, because radially symmetric can use only z and r,
    #density at center of mass
    rho_arr  = np.ndarray(shape=(n_r,n_z,n_phi),dtype=float)   
    xy       = np.ndarray(shape=(4,2),dtype=float)

    epsilon_0 = m_disk/(12./5.*h_0*R_0*(2.*np.pi)**(2./3.)\
                        *(r_max**(5./8.)-r_min**(5./8.)))\
                        *(M_sun.cgs.value/(au.cgs.value)**2)

    if(cutoff):
        range_0  = 1
        #first radial batch of cells zero density
        rho_arr[0,:,:] = 0.0
    else:
        range_0  = 0

    for i in range(range_0,n_r):
    #CoM for cell radial from lower wall, do here because radial symmetry
        xy[:,:] = grid_cell[i,0,0,2,:,0:2]
        A       = .5*np.sum(xy[:-1,0]*xy[1:,1]-xy[:-1,1]*xy[1:,0])
        C_x     = np.sum((xy[:-1,0]+xy[1:,0])* \
                         (xy[:-1,0]*xy[1:,1]-xy[:-1,1]*xy[1:,0]))*(6.*A)**-1
        C_y     = np.sum((xy[:-1,1]+xy[1:,1])* \
                         (xy[:-1,0]*xy[1:,1]-xy[:-1,1]*xy[1:,0]))*(6.*A)**-1
        r       = np.sqrt(C_x**2+C_y**2)
        
        for j in range(0,n_z):
                z = (-z_mid)+(.5+j)*dz
                rho_arr[i,j,:] = 1.5*epsilon_0/R_0*(r/R_0)**(-2.5)*\
                                    np.exp(-0.5*(z/h_0)**2*(r/R_0)**-2.25)        
                
    return rho_arr

#=====================================================================
#raytrace to domain edge
#=====================================================================
def raytrace(O,direction,ind_cell,path_type):
    
    #no error 0
    #intersect error -1
    #too small optical depth at forced first scatter -2
    err=-1
            
    #init vectors for lightpath containing intersect coordinates,
    #cell path containig indexes of corresponding cells, loc_old
    #to contain location of last intersect, interaction location and
    #tau up to that point and to domain edge  
    lightpath      = np.ndarray(shape=(2,4),dtype=float)
    cell_path      = np.ndarray(shape=(1,3),dtype=int)
    loc_old        = np.ndarray(shape=(3),dtype=float)
    loc_old[:]     = O[:]
    interaction    = np.ndarray(shape=(4),dtype=float)
    interaction[:] = 0.0
    
    tau_path = 0.0
    ind_wall = 0
    
    #tells to intersection checker to check all walls of initial cell,
    #afterwards check others than the one beam is entering cell
    wall_bool      = False

    #puls double duty, first tells function that beam reched domain edge,
    #later tels caller if domain wall was reached without scatter
    domain_edge    = False

    #insert origin to first point in lightpath
    lightpath[0,0:3] = O[:]
    lightpath[0,3]   = 0.0
    
    n = 0

    #repeat until edge found, or maximum reached, 
    while(n < n_r*n_z*2):
        
        #checks walls for intersects
        intersect,loc,ind_wall = \
            Wall_intersect(O,direction,ind_cell,ind_wall,wall_bool)

        #when intersection point found insert into lightpath, else return to
        #caller with error
        if(not(intersect)):
            return ind_cell,err,domain_edge,interaction
            
        #tau within cell
        tau_cell = np.linalg.norm(loc[:]-loc_old[:])*\
            rho_arr[ind_cell[0],ind_cell[1],ind_cell[2]]
      
        #count total path
        tau_path = tau_path+tau_cell
        
        #update lightpaths
        lightpath[-1,0:3] = loc[:]
        lightpath[-1,3]   = tau_cell
        cell_path[-1,:]   = ind_cell[:]
        
        #update ind_cell for next cell in path, detect if it is on domain edge
        ind_cell,ind_wall,domain_edge=ind_update(ind_cell,ind_wall)
        
        #if current cell is on domain edge and moving out commence interaction
        #handling
        if(domain_edge):
            
            #if forced first scatter call correct random function, unless
            #tau_path so small that tau_sca rounds to 0
            if(path_type == -2):
                if(tau_path <= 10**-15):        
                    err=-2
                    return ind_cell,err,domain_edge,interaction
                
                tau_sca = Tau_ffs(tau_path)
                
            #standard scatter, escape if tau_sca > tau_path
            elif(path_type == -1):
                tau_sca = Tau_scatter()
                if(tau_sca >= tau_path):
                    err=0
                    return ind_cell,err,domain_edge,interaction

            #else beam is one that has been peeled of towards observer
            #and wont scatter
            else:
                
                interaction[0:3] = O[:]
                interaction[3] = tau_path
                err=0
                return ind_cell,err,domain_edge,interaction
            
            #start counting untill tau_sca reached (or not)
            n_cell  = cell_path.size/3
            tau_loc = 0.0  

            for i in range(0,n_cell):
     
                tau_cell = lightpath[i+1,3]
                tau_loc  = tau_loc+tau_cell

                #check when tau_loc passes tau_sca, scattering location
                if(tau_loc > tau_sca):
                  
                    loc[:]           = lightpath[i+1,0:3]
                    loc_old[:]       = lightpath[i,0:3]
                    
                    scalefac         = (tau_sca-(tau_loc-tau_cell))/tau_cell
                    interaction[0:3] = np.add(loc_old,(np.subtract(loc,loc_old)\
                                        *scalefac))
                    interaction[3]   = tau_path
                    ind_cell[:]      = cell_path[i,:]
                    err              = 0
                    domain_edge      = False
                    
                    return ind_cell,err,domain_edge,interaction
                      
                #if forced firs scatterer but can scatter return error
                elif(path_type == -2 and i >= n_cell):
                    err  -1
                
                    return ind_cell,err,domain_edge,interaction
                      
                #ifbeam failed to scatter mark it as escaped
                elif(path_type != -2):
                    interaction[0:3] = O[:]
                    interaction[3]   = tau_path
                    domain_edge      = True
                    err              = 0
       
                    return ind_cell,err,domain_edge,interaction
            
                      
        #increase size of lightpath and cell_path for next cell
        lightpath       = np.concatenate((lightpath,[[0.0,0.0,0.0,0.0]]),axis=0)
        cell_path       = np.concatenate((cell_path,[[0,0,0]]),axis=0)

        #repace loc_old with current loc
        loc_old[:] = loc[:]
        
        #tell itersect to check walls other than one trough which beam is
        #arriving from now on
        wall_bool = True
        
        n = n + 1
        
    return ind_cell,err,domain_edge,interaction

#=====================================================================
#Cell wall intersect test
#=====================================================================
def Wall_intersect(O,path,ind_cell,ind_wall,wall_bool):
    
    V0  = np.ndarray(shape=(3),dtype=float)
    V1  = np.ndarray(shape=(3),dtype=float)   
    V2  = np.ndarray(shape=(3),dtype=float)
    loc = np.ndarray(shape=(3),dtype=float)
    
    loc[:]    = 0.0
    intersect = False
    cell      = (ind_cell[0],ind_cell[1],ind_cell[2])
    
    #depending on wall_bool either check wall with wall_index, or the others   
    for i in range(0,6):
        #if at center where inward cell wall A=0 ignore it
        if(cell[0] == 0 and i == 0):
            continue
        
        #if light coming trough a wall ignore wall in question
        #depending on wall bool
        elif((i == ind_wall) and  wall_bool):
            continue

        #split wall in two triangles, pass to intersect check
        for j in range(2):
            for k in range(3):
                V0[k] = grid_cell[(cell+(i,0+j*3,k))] 
                V1[k] = grid_cell[(cell+(i,1,k))]
                V2[k] = grid_cell[(cell+(i,2,k))]
            
            intersect,H,U,V = MT_test(path,O,V0,V1,V2)
            #if intersect found return bool to main, location, index of wall
            if(intersect):
                loc = np.add((1-U-V)*V0,U*V1)
                loc = np.add(loc,V*V2)
                return intersect,loc,i
            
    #if at center and ni intersects, expand search to all center cells at same 
    #z-level
    #if no intersect return False to caller, wall_index unchanged
    return intersect,loc,ind_wall
        
#=====================================================================
#Moller-Trumbore triangle intersection test
#=====================================================================
def MT_test(path,O,V0,V1,V2):

    epsilon = 10e-15
    
    E1 = np.ndarray(shape=(3),dtype=float)   
    E2 = np.ndarray(shape=(3),dtype=float)
    D  = np.ndarray(shape=(3),dtype=float)   
    P  = np.ndarray(shape=(3),dtype=float)   
    T  = np.ndarray(shape=(3),dtype=float)   
    Q  = np.ndarray(shape=(3),dtype=float)   
        
    E1 = np.subtract(V1,V0)
    E2 = np.subtract(V2,V0)

    D   = path
    T   = O-V0
    Q   = np.cross(T,E1)
    P   = np.cross(D,E2)
    det = np.dot(E1,P)

    U = 0.0
    V = 0.0
    H = 0.0
    
    if(np.abs(det) < epsilon):
        intersect = False
        return intersect,H,U,V

    U = np.dot(P,T)/det  
    if(U < 0.0 or U > 1.0):
        intersect = False
        return intersect,H,U,V
    
    V = np.dot(Q,D)/det
    if(V < 0.0 or U+V > 1.0):
        intersect = False
        return intersect,H,U,V

    #additional condition, point must be in direction of lightbeam, and not
    # too close to current origin
    H = np.dot(Q,E2)/det
    if(H <= 0.0):
        intersect = False
        return intersect,H,U,V

    #If all is well return true and parameters if intersection 
    intersect = True
    return intersect,H,U,V

#=====================================================================
#Updates cell index when shifting to new cell 
#=====================================================================
def ind_update(ind_cell,ind_wall):
               
    #adjust ind_cell for moving to next cell, depending on exit
    #wall of last, at the same time check if moving outside of domain
    domain_edge = False

    #inwards, at center of domain wall A=0 so raytrace set to bypass it         
    if(ind_wall == 0):
        ind_cell[0] = ind_cell[0]-1
        ind_wall    = 1

    #outwards        
    elif(ind_wall == 1):
        if(ind_cell[0] == n_r-1):
            domain_edge = True
        else: 
            ind_cell[0] = ind_cell[0]+1
            ind_wall    = 0

    #downwards                   
    elif(ind_wall == 2):
        if(ind_cell[1] == 0):
            domain_edge = True
        else:
            ind_cell[1] = ind_cell[1]-1
            ind_wall    = 3
            
    #upwards
    elif(ind_wall == 3):
        if(ind_cell[1] == n_z-1):
           domain_edge = True
        else:
            ind_cell[1] = ind_cell[1]+1
            ind_wall     = 2
           
    #for clocwise, ccw shift take modulo of index for n_phi-1 when moving
    #over from index 0=>n_phi-1 or vice verse 
    elif(ind_wall == 4):
        ind_cell[2] = (ind_cell[2]-1)%(n_phi)
        ind_wall    = 5
                         
    elif(ind_wall == 5):
        ind_cell[2] = (ind_cell[2]+1)%(n_phi)
        ind_wall    = 4

    return ind_cell,ind_wall,domain_edge


#=====================================================================
#New direction vector after scattering
#=====================================================================
#Rotates old diection vector to point into scattered direction
def Vector_rot(direction,cos_Theta,Phi):
    new_dir=np.ndarray(shape=(3),dtype=float)
    
    u=direction[0]
    v=direction[1]
    w=direction[2]
    
    sin_Theta = np.sqrt(1.-cos_Theta**2.)
    
    if(w <=.99 and w >=-.99):
        new_dir[0]=sin_Theta*(u*w*scipy.cos(Phi)-v*scipy.sin(Phi))\
            /np.sqrt(1.-w**2)+u*cos_Theta
        new_dir[1]=sin_Theta*(v*w*scipy.cos(Phi)+u*scipy.sin(Phi))\
            /np.sqrt(1.-w**2)+v*cos_Theta
        new_dir[2]=-sin_Theta*scipy.cos(Phi)*np.sqrt(1.-w**2)\
            +w*cos_Theta
    else:
        new_dir[0] = sin_Theta*scipy.cos(Phi)
        new_dir[1] = sin_Theta*scipy.sin(Phi)
        if(w >= .99):
            new_dir[2] = cos_Theta
        else:
            new_dir[2] = -cos_Theta
    
    return new_dir

#=====================================================================
#HG probability function for peel-off
#=====================================================================
def p_HG(g,cos_Theta):
    return (1-g**2.)/(1+g**2.-2.*g*cos_Theta)**(3./2.)/(4*np.pi)

#=====================================================================
#cos(Theta)-angle from HG-function with uniform random number
#=====================================================================
def HG_scatter(g):
    r=(1.-urand(low=(0.),high=(1.)))
    return 1.0/(2.0*g)*((1.0+g**2.)-((1-g**2.)/(1-g+2*g*r))**2.)

#=====================================================================
#Weight of escaping photons
#=====================================================================
def W_esc(tau_path):
    
    return np.exp(-tau_path)

#=====================================================================
#Weight of scatterd photons
#=====================================================================
def W_sca(tau_path,a):
    return a*(1-np.exp(-tau_path))

#=====================================================================
#Weight of absorbed photons in cell j,
#tau_j is the optical thickness of path trough cell j
#=====================================================================
#def W_abs(tau_j,a):
#    
#    return (1-a)*(1-np.exp(-tau_j))

#=====================================================================
#Optical depth of forced first scattering event
#=====================================================================
def Tau_ffs(tau_path):
     
    r       = urand(0.,1.)
    tau_sca = -np.log(1.-(r)*(1.-np.exp(-tau_path)))

    return tau_sca

#====================================================================
#optical depth of standard scattering optical depth
#====================================================================
def Tau_scatter():
    r=urand(0.,1.)

    tau_sca = -np.log(r)

    return tau_sca

#====================================================================
#input
#====================================================================
def input_reader():
    #globals due to being used trough every level of program
    global n_phi
    global n_z
    global n_r
    
    input_values = np.genfromtxt('inputs.txt',usecols =(0))

    #domain parameters
    n_r   = int(input_values[0])
    n_z   = int(input_values[1])
    n_phi = int(input_values[2])

    r_min = float(input_values[3])
    r_max = float(input_values[4])

    z_max = float(input_values[5])

    #disk parameters
    R_0   = float(input_values[6])
    h_0   = float(input_values[7])

    #dust parameters
    m_disk= float(input_values[8])
    kappa = float(input_values[9])
    albedo= float(input_values[10])
    
    #stellar and photon parameters
    L_star= float(input_values[11])
    n_packet= int(input_values[12])
    Lambda= float(input_values[13])

    #observer parametes
    obs_incl=float(input_values[14])
    detector=int(input_values[15])
    
    obs_incl = obs_incl/180.*np.pi

    return r_min,r_max,z_max,R_0,h_0,m_disk,kappa,albedo,\
      L_star,n_packet,Lambda,obs_incl,detector

#====================================================================  
#====================================================================
#Main program
#====================================================================  
#====================================================================

#====================================================================
#Grid structure, sky coverage, dust properties  
#====================================================================

r_min,r_max,z_max,R_0,h_0,m_disk,kappa,albedo,\
    L_star,n_packet,Lambda,obs_incl,detector=input_reader()

#output filename from command line arg
file_out=sys.argv[1]

#min max of angular width
phi_min = 0.0
phi_max = 2*np.pi

#level of domain bottom
z_min = 0.0

#cell sizes
dphi  = (phi_max-phi_min)/float(n_phi)
dz    = (z_max-z_min)/float(n_z)
dr    = (r_max-r_min)/float(n_r)
z_mid = (z_max-z_min)/2 

#star coordinate
Star   = [0.0,0.0,z_mid]
g      = 0.6

#min max of nonzer density containing domain angular height extent,
#adjustment on n_R to compensate additional wall of cells if density region
#doesn't start from center  
if(r_min != 0.):
    cos_theta_max = (z_max-z_mid)/np.sqrt(r_min**2+(z_max-z_mid)**2)
    cos_theta_min = (z_min-z_mid)/np.sqrt(r_min**2+(z_max-z_mid)**2)
    cutoff    = True
    n_r       = n_r+1
    range_0  = 1
    
else:   
    theta_min = 0.0
    theta_max = np.pi
    cutoff    = False
    range_0  = 0

#fraction of sky solid angle covered by region photon emitted to
A_frac  =(cos_theta_max-cos_theta_min)/2.

#photons per packet
ny = c.value*(Lambda*1e-6)
n_phot = L_star/(n_packet*h.value*ny)*A_frac

#set minimum weight of packet until considered spent so that cutoff when 1e-10 
#of photons remain 
packet_tol = 1e-6

#====================================================================
#Grid and grid contents  
#====================================================================    

#vertices of grid
grid_vertex= vertbuild(dphi,dz,dr,phi_min,z_min,r_min,cutoff)

#vertices of each grid cell and walls
grid_cell  = cellbuild(grid_vertex)

#densities of cells, update to rho*kappa_ext
rho_arr    = densitybuild(cutoff,R_0,h_0,m_disk,r_min,r_max,z_mid)
rho_arr    = rho_arr*kappa

#run loop until wanted amount of photon packets simulated
packet_count   = 0
direction      = np.ndarray(shape=(3),dtype=float)
observer       = np.ndarray(shape=(3),dtype=float)     
O              = np.ndarray(shape=(3),dtype=float)
ind_cell       = np.ndarray(shape=(3),dtype=int)
ind_cell_O     = np.ndarray(shape=(3),dtype=int)
observed_light = np.ndarray(shape=(1,4),dtype=float)

#points to observer at inclination in positive X-axis direction 
observer[0]  = scipy.sin(obs_incl)
observer[1]  = 0.0  
observer[2]  = scipy.cos(obs_incl)
Errorcount = 0
No_interact= 0

#run loop until wanted amount of photon packets simulated
while(packet_count < n_packet):

    #set origin of path back at star
    O[:]=Star[:]

    #set packet weight
    W_packet = 1.0
    
    #generate packet direction
    phot_phi   = 2.*np.pi*urand(low=0.,high=1.)
    phot_theta = urand(low=cos_theta_min,high=cos_theta_max)
      

    direction[0] = np.sqrt(1.-phot_theta**2.)*scipy.cos(phot_phi)
    direction[1] = np.sqrt(1.-phot_theta**2.)*scipy.sin(phot_phi)
    direction[2] = phot_theta

    #if emitter is exactly on same z-coord as vertex, see if beam goes
    #to cells above or below, in case of purely radial beam flip coin,
    #raytrace detects edge intersects
    ind_cell[:]=0         
    for i in range(0,n_z+1):
        if(O[2] == grid_vertex[0,i,0,2]):
            
            if(phot_theta > 0.0):
                ind_cell[1] = i
                break
            
            elif(phot_theta < 0.0):
                ind_cell[1] = i-1
                break
            
            else:
                ind_cell[1] = i-np.random.randint(2)

            
        elif(O[2] < grid_vertex[0,i,0,2]):
            ind_cell[1] = i-1
            break

    #now for azimuthal cell index, see if direction same as one
    #of the vertices, if yes flip coin and take modulo to clear
    #indexing discontinuity at full circle, else see when phot phi        
    for j in range(0,n_phi+1):
        
        if(phot_phi == phi_min+dphi*j):
            ind_cell[2] = (j-np.random.randint(2))%(n_phi-1)
            break
        
        elif(phot_phi < phi_min+dphi*j):
            ind_cell[2] = j-1
            break
        
    #Forced first scatter
    ffs=-2

    #integrate path untill first escape 
    ind_cell,err,domain_edge,interaction =\
        raytrace(O,direction,ind_cell,ffs)

    #in case of raytrace error redraw direction
    if(err == -1):
        Errorcount = Errorcount + 1         
        continue
    
    #if optical depth along path too long consider packet 
    #forced firs scatter to have contributed nothing 
    elif(err == -2):
        packet_count = packet_count + 1
        No_interact = No_interact +1        
        continue

    #weight of scattered light, update W_packet to contain only scattered
    #portion of original weight
    tau_esc  = interaction[3]
    #W_e      = W_esc(tau_esc)*W_packet        #escaping
    W_s      = W_sca(tau_esc,albedo)*W_packet #scattering    
    W_packet = W_s                            #new weight scattered portion of last
    
    #print W_e+W_s+(1-albedo)*(1-np.exp(-tau_esc)),'forced first'
    
    #now begin looping following two raytracers, first out off interaction point
    #to observer, second scattered
    n_sca = 0
    
    while(n_sca < 200):
        #set origin, new cell containing origin from last results
        O[:]          = interaction[0:3]
        ind_cell_O[:] = ind_cell[:]
        
        #Peeled off branch
        ffs = -3
        ind_cell,err,domain_edge,interaction =\
            raytrace(O,observer,ind_cell,ffs)

        if(err == -1):
            
            Errorcount = Errorcount + 1            
            break
       
        #cos(theta) between packet direction and observer direction
        cos_Theta = np.dot(direction,observer)

        #weight of peeled photon packet, currently contains photons scattered 
        #at last interaction location, weight p_HG goes towards observer,
        #rest scatters to random
        tau_obs = interaction[3]
        HG_weight = p_HG(g,cos_Theta)
        W_obs     = HG_weight*W_esc(tau_obs)*W_packet
        W_s       = (1-HG_weight)*W_packet
        W_packet  = W_s
        
        #print HG_weight*(1-W_esc(tau_obs))+HG_weight*W_esc(tau_obs)+(1-HG_weight),'peel off'
        
        #store origin of emission,emission weight, grow observed_light arr
        observed_light[-1,0:3] = O[:]
        observed_light[-1,3]   = W_obs
        observed_light  = np.concatenate((observed_light,[[0.,0.,0.,0.]]),axis=0)
        
        #if scattered light runs into error redraw scattering direction
        while(True):
            
            #scattered branch 
            ffs = -1
            
            #draw new direction for scattered
            cos_Theta = HG_scatter(g)
            Phi   = 2.*np.pi*urand(low=0.,high=1.)
        
            ind_cell[:] = ind_cell_O[:]

            direction = Vector_rot(direction,cos_Theta,Phi)
       
        
            ind_cell,err,domain_edge,interact =\
                raytrace(O,direction,ind_cell,ffs)
        
            if(err == -1):
                Errorcount = Errorcount + 1
            else:
                break
        
        #Readjust packet weights, store if toward observer, cycle to next peel off,
        #unless at domain wall or Weight too low to continue
        tau_esc  = interact[3]
        #W_e      = W_esc(tau_esc)*W_packet
        W_s      = W_sca(tau_esc,albedo)*W_packet
        W_packet = W_s
        
        #print W_esc(tau_esc)+W_sca(tau_esc,albedo)+W_abs(tau_esc,albedo),'scatter'
        
        
        if(domain_edge):  
            
            #check if escaping packet in direction of obsever
            if(np.abs(np.dot(direction,observer)) <= 1e-12):
                observed_light[-1,0:3] = O[:]
                observed_light[-1,3]   = W_e
                observed_light  = np.concatenate((observed_light,[[0.,0.,0.,0.]]),axis=0)
                break
    
            else:        
                break
            
        elif(W_packet < packet_tol):  
            break

        n_sca = n_sca + 1
        
    #if there wasn't errors in execution increase packet_count, else restart
    #from packet generation
    if(err != -1):
        packet_count = packet_count + 1
    
#construct binned image centered on disk
observed_light = np.delete(observed_light,-1,0)
n_incoming = observed_light.size/4

image = np.ndarray(shape=((detector+1),(detector+1),3),dtype=float)

observed_pixels     = np.ndarray(shape=(n_incoming,2),dtype=int)

for i in range(n_incoming):
    
    x = observed_light[i,0]
    y = observed_light[i,1]
    z = observed_light[i,2]-z_mid

    z_proj = z*scipy.sin(obs_incl)-x*scipy.cos(obs_incl)
    y_proj = y

    observed_pixels[i,0] = np.floor(detector*(y_proj+r_max*1.1)/(2*r_max*1.1))+1
    observed_pixels[i,1] = np.floor(detector*(z_proj+r_max*1.1)/(2*r_max*1.1))+1

image[:,:,:] = 1.

#binning into image
for i in range(detector+1):
    for j in range(detector+1):
        image[i,j,0] = -r_max*1.1+(2.2*r_max)/float(detector)*i
        image[i,j,1] = -r_max*1.1+(2.2*r_max)/float(detector)*j

        for k in range(n_incoming):         
            if(observed_pixels[k,0] == i and observed_pixels[k,1] == j):
                
                image[i,j,2]=image[i,j,2]+observed_light[k,3]*n_phot*ny*h.value

print 'Forced first scatters passed due to too low optical thickness =', No_interact               
print 'Number of errors causing redraw of direction during execution =', Errorcount

norm = colors.LogNorm(image[:,:,2].min(),image[:,:,2].max())
plt.pcolor(image[:,:,0],image[:,:,1],image[:,:,2],norm=norm)
plt.title(r'$\frac{W}{Hz \ m^{2}\ px}$ i= %3.2f'%(obs_incl/np.pi*180.))
plt.xlabel('AU')
plt.ylabel('AU')
plt.colorbar()
plt.savefig(file_out,dpi=300)
plt.show()
