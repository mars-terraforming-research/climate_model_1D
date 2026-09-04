current_path='/home/akling/Code/climate_model_1D'
import sys; sys.path.append(current_path)
import matplotlib.pyplot as plt
import numpy as np
import time
from climate_utils import SHforced_TPUTs, Sun, background_dust_tau,cos_declination,comimart_surface_fluxes
from climate_utils import progress, rolling_window,FD_diffusion_solver_1D
from amescap.FV3_utils import sol2ls
from amescap.Ncdf_wrapper  import  Ncdf
import pandas as pd


#=================
SOL_in_sec=86400
SOL_per_Mars_year=668

#================physics====================
sigma = 5.670373*10**-8 # Stefan Boltzmann constant J/(m2.K-4)
eps=1.
cp_regolith=735.9   # J/(kg.K)
rho_regolith=1481.#kg/m3
#==========================
k_regolith=0.03#0.035#0.035#0.065#W.m-1.K-1  ---> derived from TI. At Gale:0.067 W.m-1.K-1  Grott 2007: 0.02-0.1 .m-1.K-1
A_regolith=k_regolith/(rho_regolith*cp_regolith) #thermal diffusivity   m2/s
#=================physics===============

latitude=40*np.pi/180
longitude=200*np.pi/180

P=7*100 # [Pa] surface pressure
albd=0.234 #atmospheric opacity =0. since atmospheric refraction included in albedo
wind=5. #  [m/s] wind speed for sensible heat flux
b_atm=0.2 #coefficient for atmospheric temperature
Fgeo=0.03 #W/m-2 Grott 2007
TM=207. #K, initialization temperature
wo=0.914 #dust single scat albedo
g=0.724  #dust single asymetry factor
REFLECTOR_FACT= 1.44 # TODO in current for, use 1.44 for Essunfeld 2026 constalation and 0. for no sail

runtype='Implicit'

Lx=5. #m
Nx=600 #
tstep_per_sol=96#1=24h, 2=12h, 24=1h 96=15 min
nSOL=668*2
Tf=nSOL*SOL_in_sec
#Downward infrared
A=-2.63898123;B=0.145363974;C=8.36913082e-05
IRD_T_quad=lambda T: A+B*T+C*T**2

#Fit for downaward infrared
A=-2.63898123;B=0.145363974;C=8.36913082e-05
IRD_T_quad=lambda T: A+ B*T+C*T**2


d_sail=pd.read_csv('/home/akling/Data/sfc_tempk/marsyear_vacuum_g18_96per_sol.csv')
LT_sail=d_sail['lst_h'].values[0:96]
LT_sail_all=d_sail['lst_h']
areo_sail=d_sail['solar_longitude_deg'].values
I_ref=d_sail['I_reflected_vacuum_bin_mean_W_m2'].values
I_nat=d_sail['I_natural_toa_bin_mean_W_m2'].values
zenith_sail_all=d_sail['reflected_power_weighted_zenith_deg'].values*np.pi/180
zenith_sail_all[np.isnan(zenith_sail_all)]=np.pi/2 #Zenith angle set to 90deg if nan

#=============Numerics=============

dt=SOL_in_sec/tstep_per_sol
t=np.arange(0,Tf+1,dt);
Nt=len(t)
x=np.zeros((Nx));x[:]=np.linspace(0,Lx,Nx);dx=(Lx-0)/(Nx-1)
solver=FD_diffusion_solver_1D(Nx,dx,dt,A_regolith,k_regolith,runtype)
solver.print_cfl()
Ls_t=sol2ls(t/86400)*np.pi/180
LT_t=np.zeros((Nt))


#===Intitialization===
TM=220.
Tsfc_t=np.zeros((Nt));Tsfc_t[0]=TM
Tatm_t=np.zeros((Nt));Tatm_t[0]=TM
TG=np.zeros((Nt,Nx));TG[0,:]=TM;TG[-1,:]=TM #Use last timestep for initialisation
Qsun_toa_t=np.zeros((Nt));Qsun_toa_t[0]=Sun(latitude,Ls_t[0],0.,0.)
Q_IR_down_t=np.zeros((Nt))
#====
Q_IR_down_t[0]=IRD_T_quad(TM)
#====
Q_IR_up_t=np.zeros((Nt))
SH_forced_t=np.zeros((Nt))
Qcond_surf_t=np.zeros((Nt))
DIR_sun=np.zeros((Nt))
DIF_sun=np.zeros((Nt))
TOT_sun=np.zeros((Nt))
DIR_sail=np.zeros((Nt))

Qreflect_toa_t=np.zeros((Nt))

DUDT_gnd_t=np.zeros((Nt)) #energy in the subsurface = rho .cp .sum (dx *T)
tau_t=np.zeros((Nt))
Tsfc_mean_day_old=TM
Tsfc_min_day_old=TM
Tatm=TM
Ls_sol_s=np.zeros((nSOL))
Tsfc_sol_s=np.zeros((nSOL));Tsfc_sol_s[0]=TM

#Initialize

iaero_sail=np.argmin(np.abs(Ls_t[0]*180/np.pi-areo_sail))
iLT_sail=0
print('Started at Ls %.4f'%(Ls_t[0]*180/np.pi))
##===================
t0=time.time()
for isol in range(0,nSOL):
    #---
    if isol%10==0:
        progress(isol,nSOL)
        sys.stdout.write(" %iK"%(Tsfc_mean_day_old))
        sys.stdout.flush()
    #---
    if isol >1:
        Tsfc_mean_day_old=np.mean(Tsfc_t[(isol-1)*tstep_per_sol:(isol*tstep_per_sol)])
        Tsfc_min_day_old=np.min(Tsfc_t[(isol-1)*tstep_per_sol:(isol*tstep_per_sol)])
        Ls_sol_s[isol]=np.mean(Ls_t[(isol-1)*tstep_per_sol:(isol*tstep_per_sol)])
        Tsfc_sol_s[isol]=Tsfc_mean_day_old


    for wsol in range(0,tstep_per_sol):
        it=isol*tstep_per_sol+wsol
        LT_t[it]=24.*(float(wsol)/tstep_per_sol)

        T_sfc_old=TG[it-1,0]
        Tatm=Tsfc_min_day_old**b_atm*(T_sfc_old)**(1-b_atm)
        tau_t[it]=background_dust_tau(Ls_t[it]*180/np.pi,latitude*180/np.pi)

        Q_sun=Sun(latitude,Ls_t[it],LT_t[it],0.) #noontime
        cosz_sun=cos_declination(latitude,Ls_t[it],LT_t[it])

        TOT_sun[it],DIR_sun[it],DIF_sun[it] = comimart_surface_fluxes(
            toa_horizontal_flux=Q_sun,
            optical_depth=tau_t[it],
            single_scattering_albedo=wo,
            asymmetry_factor=g,
            surface_albedo=0.23,
            mu0=max(cosz_sun,10**-8))

        Q_IR_down=IRD_T_quad(T_sfc_old) #use the latest temperature


        #=======Logic for reading chunk of solar sail data=======
        #Get nearest LS
        iaero_sail=np.argmin(np.abs(Ls_t[it]*180/np.pi-areo_sail))
        #Find timestep for LT=0 nearest that chunk, with an exception for the last
        #24hr (96 timestep) of the file)
        if iaero_sail<len(areo_sail)-96:
            LT_sail_tmp=LT_sail_all[iaero_sail:iaero_sail+96]
            iLT_sail=np.argmin(np.abs(LT_t[it]-LT_sail_tmp))
            iIneed=iaero_sail+iLT_sail
        else:
            iaero_sail=len(areo_sail)-96
            LT_sail_tmp=LT_sail_all[iaero_sail:iaero_sail+96]
            iLT_sail=np.argmin(np.abs(LT_t[it]-LT_sail_tmp))
            iIneed=iaero_sail+iLT_sail


        zenith_sail=zenith_sail_all[iIneed]
        I_reflec=REFLECTOR_FACT*I_ref[iIneed]

        _,DIR_sail[it],_ = comimart_surface_fluxes(
            toa_horizontal_flux=I_reflec,
            optical_depth=tau_t[it],
            single_scattering_albedo=wo,
            asymmetry_factor=g,
            surface_albedo=0.23,
            mu0=max(np.cos(zenith_sail*np.pi/180),10**-8))

        #========================
        Q_IR_up=eps*sigma*(T_sfc_old)**4
        SH_forced=SHforced_TPUTs(Tatm,P,wind,T_sfc_old)
        F_top=(1-albd)*TOT_sun[it]+Q_IR_down-Q_IR_up+SH_forced+(1-albd)*DIR_sail[it]
        F_bot=Fgeo
        #============Update================
        TG[it,:]=solver.advance_dt(TG[it-1,:],['N','N'],[F_top,F_bot])
        #========Save data=================
        Tsfc_t[it]=TG[it,0] #TODO, this is temperature at DZ/2 depth, not skin temperature  #Calculate skin temperature
        Tatm_t[it]=Tatm
        Q_IR_down_t[it]=Q_IR_down
        SH_forced_t[it]=SH_forced
        Q_IR_up_t[it]=Q_IR_up
        Qcond_surf_t[it]=-k_regolith*(TG[it,0]-TG[it,1])/dx

        Qsun_toa_t[it]=Q_sun
        Qreflect_toa_t[it]=I_reflec


print("Elapsed= %.2f s"%(time.time()-t0))


##======Log in Netcdf files =========================================================

print('Logging data...')

Log=Ncdf(current_path)#create run +date

Log.add_constant('dt',dt,"time step","s")
Log.add_constant('nSOL',len(Ls_sol_s[nSOL//2+1:]),"number of sol in simulation including spin-up","")
Log.add_constant('tstep_per_sol',tstep_per_sol,"number of timestep per sol","")
#---
Log.add_constant('b_atm',b_atm,"atmospheric parameter")
Log.add_constant('P',P,"surface pressure","Pa")
Log.add_constant('wind',wind,"winds at 5m","m/s")
Log.add_constant('albd',albd,"planetary albedo","")
Log.add_constant('cp_regolith',cp_regolith,"soil heat capacity","J/(kg.K)")
Log.add_constant('rho_regolith',rho_regolith,"soil density","kg/m3")
Log.add_constant('k_regolith',k_regolith,"soil thermal conductivity","W.m-1.K-1 ")
Log.add_constant('Lx',Lx,"soil model depth","m")
Log.add_constant('Fgeo',Fgeo,"geothermal flux","W/m2")

#---
Log.add_dimension('Nx',Nx)
Log.add_dimension('time',len(Ls_t[Nt//2:]))
Log.add_dimension('sols',len(Ls_sol_s[nSOL//2+1:]))
#---

Log.log_variable('Ls_sol_s',180/np.pi*Ls_sol_s[nSOL//2+1:],'sols','daily mean solar longitude','degree')
Log.log_variable('Tsfc_sol_s',Tsfc_sol_s[nSOL//2+1:],'sols','daily mean sfc temperature','K')
#---
Log.log_variable('Ls_t',180/np.pi*Ls_t[Nt//2:],'time','solar longitude','degree')
Log.log_variable('LT_t',LT_t[Nt//2:],'time','local time','hours')
Log.log_variable('Tsfc_t',Tsfc_t[Nt//2:],'time','soil surface temperature','K')
Log.log_variable('Tatm_t',Tatm_t[Nt//2:],'time','atmospheric temperature at 5m','K')
Log.log_variable('Qsun_toa_t',Qsun_toa_t[Nt//2:],'time',' vis solar irradiance at the TOA','W/m2')
Log.log_variable('Qsun_sfc_total_t',TOT_sun[Nt//2:],'time',' total solar irradiance at the surface','W/m2')
Log.log_variable('Qsun_sfc_diff_t',DIF_sun[Nt//2:],'time',' diffuse component of the solar irradiance at the surface','W/m2')
Log.log_variable('Qreflect_toa_t',Qreflect_toa_t[Nt//2:],'time',' vis reflector irradiance at the TOA','W/m2')
Log.log_variable('Qreflect_sfc_t',DIR_sail[Nt//2:],'time',' direct reflector irradiance at the surface','W/m2')
Log.log_variable('Q_IR_down_t',Q_IR_down_t[Nt//2:],'time','downward IR','W/m2')
Log.log_variable('Q_IR_up_t',Q_IR_up_t[Nt//2:],'time','IR up (sigmaT**4)','W/m2')
Log.log_variable('SH_forced_t',SH_forced_t[Nt//2:],'time','forced sensible heat flux','W/m2')
Log.log_variable('Qcond_surf_t',Qcond_surf_t[Nt//2:],'time','conductive flux','W/m2')
Log.log_variable('TG',TG[Nt//2:,:],('time','Nx'),'soil temperature','K')
Log.close()

print('Log completed')

##===================Plots========================================


plt.close('all')
plt.figure(figsize=(40/2.54, 30/2.54),facecolor='white')
ax=plt.subplot(111)

Ls_1D_roll=rolling_window(Ls_t*180/np.pi,tstep_per_sol)
T_1D_roll=rolling_window(Tsfc_t,tstep_per_sol)
DT_1D_roll=rolling_window(Tsfc_t,tstep_per_sol,'DV')
Q_1D_roll=rolling_window(Qcond_surf_t,tstep_per_sol)
SH_1D_roll=rolling_window(SH_tot_t,tstep_per_sol)
nroll=len(T_1D_roll)

plt.plot(Ls_1D_roll[:nroll//2],T_1D_roll[:nroll//2],'.b',label= 'MY 1')
plt.plot(Ls_1D_roll[nroll//2+1:],T_1D_roll[nroll//2+1:],'.r',label= 'MY 2')
plt.legend(fontsize=10)
plt.xlabel('Ls',fontsize=14)
plt.ylabel('Daily surface temperature [K]',fontsize=14)
plt.grid()
plt.xlim([Ls_t.min()*180/np.pi,Ls_t.max()*180/np.pi])
plt.title('Diurnally-average surface temperature')
plt.show()

#=======================


