import numpy as np
import flopy
import time
import h5py
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



Folder = 'FW4'  ## THis is only for FW4

##########Load RES###################
filepath= './%s/relz_res.mat' %(Folder)
f2 = h5py.File(filepath)
mat2 = list(f2['relz'])
data = np.asarray(mat2)
resistivity = (np.exp(data[1,:,:]).T)*10+100 # for i = 0  
EC = 1/resistivity

##porosity
sigma_w = 0.02
m = 1.99
poro = (EC/sigma_w)**(1/m)

##permeability
Cc = 3.87*10**(-12)
kp = Cc*(poro**m)/(1-poro)

##Hydraulic conductivity
rho = 1000
grav = 9.8
mu = 8.9/(10**4)

K_ms = kp*rho*grav/mu
Kdata = K_ms*86400

Qv = (10**(-9.2))*(kp**(-0.82))


##grid
N = int(Kdata.shape[1])
n= int(N/2)
kk = np.asarray(list(range(0,n+1))+list(range(-n+1,0)))
kk = kk*(2*np.pi/2000)
xk, yk = np.meshgrid(kk,kk)

##########GW MODEL########################

# Kdata = np.ones((100,100))
# Kdata[:,0:10]= 100
# Kdata[:,90:100] = 100

def GWmodel(K):
    # Assign name and create modflow model object
    modelname = 'Syn2D'
    model_ws= './%s' %(Folder)
    mf = flopy.modflow.Modflow(modelname, exe_name='C:/UHM/Research/Upscaling/mf2005', model_ws = model_ws)

    # Model domain and grid definition
    Lx = 2000.
    Ly = 2000.
    ztop = 0.
    zbot = -20.
    nlay = 1
    nrow = 100
    ncol = 100
    delr = Lx/ncol
    delc = Ly/nrow
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)

    # Create the discretization object
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                top=ztop, botm=botm[1:])
    # Variables for the BAS package
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = 10.
    strt[:, :, -1] = 0.
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    # Add LPF package to the MODFLOW model
    lpf = flopy.modflow.ModflowLpf(mf, hk=K[:,:], vka=10., ipakcb=53)

    # Add OC package to the MODFLOW model
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf)

    # Write the MODFLOW model input files
    mf.write_input()

    # Run the MODFLOW model
    success, buff = mf.run_model()

    # Post process the results
    import matplotlib.pyplot as plt
    import flopy.utils.binaryfile as bf


    hds = bf.HeadFile(model_ws+'/'+modelname + '.hds')
    head = hds.get_data(totim=1.0)
    sp_q = flopy.utils.postprocessing.get_specific_discharge(mf,model_ws+'/'+modelname + '.cbc')


    return head, sp_q
#########Self-potential####################
def spec2D(xx):
    xx = xx.reshape(-1,1)
    Ax =-xk*np.fft.fftn(EC*np.fft.ifftn(np.reshape(xk.reshape(-1,1)*xx,(N,N))))-yk*np.fft.fftn(EC*np.fft.ifftn(np.reshape(yk.reshape(-1,1)*xx,(N,N))))
    # Ax =-xk*np.fft.fftn(EC*np.fft.ifftn(xk*np.fft.fftn(xx.reshape(N,N))))\
    #     -yk*np.fft.fftn(EC*np.fft.ifftn(yk*np.fft.fftn(xx.reshape(N,N))))
    return Ax.reshape(-1)


def SPforward(K,head,spq):
    # poro = 0.3
    rho = 1000
    grav = 9.8
    mu = 8.9/(10**4)
    
    K_ms = K/86400   ##input K
    # Keff= K_ms/poro
    Keff = K_ms
    keff = Keff*mu/(rho*grav)
    Qv = (10**(-9.2))*(keff**(-0.82))

    # b1 = 1j*xk*np.fft.fftn(-1*Qv*Keff*np.fft.ifftn(1j*xk*np.fft.fftn(head[0,:,:])))+1j*yk*np.fft.fftn(-1*Qv*Keff*np.fft.ifftn(1j*yk*np.fft.fftn(head[0,:,:])))
    b2 = 1j*xk*np.fft.fftn(Qv*spq[0]/86400)+1j*yk*np.fft.fftn(Qv*spq[1]/86400)
    A = LinearOperator((N**2,N**2), matvec = spec2D)

    # sp1, exitcode1 = cg(A,b1.reshape(-1))
    # if exitcode1 != 0:
    #     print("cg not converged: %d" % (exitcode1))
    #     sp1, exitcode1 = gmres(A,b1.reshape(-1),x0=sp1)
    
    sp2, exitcode1 = cg(A,b2.reshape(-1))
    if exitcode1 != 0:
        print("cg not converged: %d" % (exitcode1))
        sp2, exitcode1 = gmres(A,b2.reshape(-1),x0=sp2)        

    # SP1 = np.fft.ifftn(sp1.reshape(N,N))
    SP2 = np.fft.ifftn(sp2.reshape(N,N))
    # SP1 = sp1
    return SP2




[head, spq] = GWmodel(Kdata)
sp  = SPforward(Kdata,head,spq)

# plt.figure(1)
# plt.imshow(np.real(t1))
# plt.colorbar()

# plt.figure(2)
# plt.imshow(np.real(t2[0,:,:]))
# plt.colorbar()
# plt.show()

# qTest = -1*Kdata[:,0:99]*(head[0,:,1:]-head[0,:,0:99])/20
# plt.figure(1)
# plt.imshow(qTest)
# plt.colorbar()

# plt.figure(2)
# plt.imshow(spq[0][0,:,:])
# plt.colorbar()
# plt.show()
#############################[figure]###############################################
plt.figure(1)
plt.title('Hydraulic head [m]')
im0=plt.imshow(head[0,:,:], cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
# plt.gca().set_aspect('equal','box-forced')
# cbar = plt.colorbar(im0, ticks=[np.log(1),np.log(10), np.log(100),np.log(1000)])
# cbar.ax.set_yticklabels(['1','10','100','1000'])
plt.tight_layout()
plt.savefig('./%s/Head.png' %(Folder)) 

plt.figure(2)
plt.title(r'Hydraulic conductivity '+ r'$[\log_{10}$' + '(m/d)]')
im1 = plt.imshow(np.log10(Kdata), cmap=plt.get_cmap('jet'),extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/K.png' %(Folder))

plt.figure(3)
plt.title('Self-Potential [V]')
im3 = plt.imshow(np.real(sp), cmap=plt.get_cmap('jet'),extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Sp.png' %(Folder)) 

plt.figure(4)
plt.title(r'True resistivity ' + r'$[\Omega \cdot m]$')
im4 = plt.imshow(resistivity, cmap=plt.get_cmap('jet'),extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Resistivity.png' %(Folder)) 

plt.figure(5)
plt.title('Electrical Conductivity [S/m]')
im5 = plt.imshow(EC, cmap=plt.get_cmap('jet'), extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/EC.png' %(Folder)) 

plt.figure(6)
plt.title('Porosity')
im5 = plt.imshow(poro, cmap=plt.get_cmap('jet'), extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Porosity.png' %(Folder)) 

plt.figure(7)
plt.title('Permeability' +r'$ [m^2]$')
im5 = plt.imshow(kp, cmap=plt.get_cmap('jet'), extent = (0,2000,2000,0))
plt.xticks(np.linspace(0,2000,3))
plt.yticks(np.linspace(0,2000,3))
plt.xlabel('x [m]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.tight_layout()
plt.savefig('./%s/Permeability.png' %(Folder)) 

#######################[boundary graph]################################################
px = np.linspace(1,100,100)
plt.figure(8)
plt.title('SP on Boundary')
plt.plot(px,sp[:,0], label = 'Left')
plt.plot(px,sp[:,-1], label = 'Right')
plt.xlabel('Depth [m]')
plt.ylabel('SP [V]')
plt.legend()
plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4),useMathText=True)
plt.savefig('./%s/Boundary.png' %(Folder))


plt.close('all')
