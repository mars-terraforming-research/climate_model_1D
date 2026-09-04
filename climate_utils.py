#A. Kling, 2026

import numpy as np
from math import ceil #round to the 2nd digit

import scipy.sparse as sparse
import scipy.sparse.linalg
import math
import sys
from typing import NamedTuple

#Mars---Parameter---
Rsun=1361/(1.523)**2. #
Lsp=248.*np.pi/180  #solar longitude at perihelion
e = 0.0934 #excentricity
obliquity=25.*np.pi/180 #obliquity  25
g=3.72 #m.s-2

z5 = 5 # [m] ref altitude
z0 = 0.01 #[m]roughness for ice, also in the GCM (initpbl.f) and Dundas_2010
von_k = 0.4 # von karman constant
Mco2=(12+16*2)*0.001# [kg/mol]
R_=8.3144621 #[J.K-1.mol-1]
cp_co2= 770 #[J/(K.kg)]
katm=0.02 #W.m-1.K-1 conductivity of the atmosphere From Holman 2002, Heat Transfert, Table 6-A
Mh2o=(2+16)*0.001# [kg/mol]

def Psat_T(T):
    T=np.array(T)
    if len(np.atleast_1d(T))==1: #if T and p dim 1
        if T>273.:
            result=Psat_liq_T(T)
        else:
            result=Psat_ice_T(T)
    else:
        result=np.zeros_like(T)
        for i in range(0,len(np.atleast_1d(T))):
                    if T[i]>273.:
                        result[i]=Psat_liq_T(T[i])
                    else:
                        result[i]=Psat_ice_T(T[i])
    return result

def nu_TP(T,P):
    '''
    Kinematic viscosity  ν = μ / ρ
    Args:
        T : temperature [K]
        P : pressure in [Pa]
    Returns:
        kinematic viscosity  ν in [m2/s]
    '''
    T=np.array(T);P=np.array(P)
    return 1.48*10**-5*R_*T/(Mco2*P)*(240+293.15)/(240+T)*(T/293.15)**(3./2)



def  Psat_liq_T(T): #From buck 1981 [-80+50]
   T=np.array(T)
   return  611*np.exp(17.3*(T-273)/(T+237.3-273))
def Psat_ice_T(T):
    T=np.array(T)
    return 611*np.exp(22.5*(1 -(273.16/T))) #[Pa] #GCM   anf close to Buck 1981 22.5

def progress(k,Nmax,skip=1.):
    '''
    Display a progress bar within a loop.
    Args:
        k: the current loop index
        Nmax: the total number of indexes
        skip: skip every n iteration when displaying the results
    Return:
        A progress bar in the console
    '''
    if  k//skip ==k/float(skip):
        progress=float(k)/Nmax
        barLength = 10 # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var2 must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength*progress))
        text = "\rRunning... [{0}] {1} {2}%".format( "#"*block + "-"*(barLength-block), ceil(progress*100*100)/100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

def rolling_window(a, window,operation='mean'):
    '''
    Given a serie of number returns the mean, max, min of max-min over a rolling window.
    Note that this is different than a moving average.
       max  max  max  max
    x:____|____|____|____|__
    Args:
        a : (float) a 1D array of values
        window: (interger) number of elements to perform the operation on
        operation: values to return, 'mean', 'max', 'min' or 'DV'
    Returns:
            a: an array of size n/window with the mean, max, min or max-min.
    ***Example***
    Example for x and y:
    y_m=rolling_window(y,10,'max')
    x_m=rolling_window(y,10,'mean')

    plot(x_m,y_m)
    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    b=np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if operation=='mean':
        b=np.mean(b,axis=1)
    elif operation=='max':
        b=np.max(b,axis=1)
    elif operation=='min':
        b=np.min(b,axis=1)
    elif operation=='DV':
        b=np.max(b,axis=1)-np.min(b,axis=1)
    elif operation=='sum':
        b=np.sum(b,axis=1)
    else:
        raise ValueError("In rolling window: invalid operation")
    return b[::window]

class FD_diffusion_solver_1D(object):
    # dx, dt : steps [m] and timestep [s]
    # A_d diffusivity [m2/s] = k/(rho *cp)
    # k_cond conductivity [W.m-1.K-1] used when BC are Von Neuman
    # runtype: 'Explicit' or  'Implicit'
    def __init__(self,Nx,dx,dt,Ad,k_cond,runtype='Implicit'):
        self.dx=dx
        self.dt=dt
        self.cfl=(dx**2)/(2*Ad)
        self.Ad=Ad
        self.k_cond=k_cond
        self.runtype=runtype
        if runtype=='Implicit': #initialize additional variables

            data = np.ones((3, Nx))
            I = sparse.identity(Nx)               # Identity Matrix
            data[1] = -2*data[1]
            diags = [-1,0,1] # data[lower, main, upper]
            D2 = sparse.spdiags(data,diags,Nx,Nx) #Check D2 with  D2.todense()

            self.s=Ad*dt/dx**2  #coefficient s in the matrix, example 1+2s
            self.A = (I -self.s*D2)   #matrix A.u=b
            self.Nx=Nx
    def print_cfl(self):
        print('dt=%g %% of CFL'%(100*self.dt/self.cfl))
        if self.runtype=='Explicit' and 100*self.dt/self.cfl>100:
            print("Warning stability: decrease dt or increase dx")
    def advance_explicit(self,R_old,BC,TorQ):
        R_new=np.zeros_like(R_old)
    # R_old: temperature array, one dimensional
    # BC boundary condition scheme [top, bottom], Dirichlet or Neuman.         Example =['D','N']
    # TorQ boundary conditions top and bottom temperature [K] or flux [W/m2].  Example =[273.,25.]

        DIFFX=(self.Ad*self.dt/self.dx**2)*(R_old[2:]-2*R_old[1:-1]+R_old[0:-2])
        R_new[1:-1]=R_old[1:-1]+DIFFX
        #--Dirichlet BC--
        if BC[0]=='D':
            R_new[0]=TorQ[0]
        elif BC[0]=='N':
            c1=-TorQ[0]/self.k_cond
            R_new[0]=(self.Ad*self.dt/self.dx**2)*(2*R_old[1]-2*c1*self.dx-2*R_old[0])+R_old[0]
        else:
            raise ValueError('Error BC top')
        if BC[1]=='D':
            R_new[-1]=TorQ[1]
        elif BC[1]=='N':
            c2=TorQ[1]/self.k_cond
            R_new[-1]=(self.Ad*self.dt/self.dx**2)*(2*c2*self.dx-2*R_old[-1]+2*R_old[-2])+R_old[-1]
        else:
            raise ValueError('Error BC bottom')
        return R_new

    def advance_implicit(self,R_old,BC,TorQ):
        b =R_old.copy()

        if BC[0]=='D':
        #Apply BC Dirichlet Top
            self.A[0,0]=1.
            self.A[0,1]=0.
            b[0]=TorQ[0]
        elif BC[0]=='N':
        #Apply BC Von Neuman Top
            c1=-TorQ[0]/self.k_cond
            self.A[0,0]=1+2*self.s
            self.A[0,1]=-2*self.s
            b[0]=b[0]-2*self.s*self.dx*c1
        else:
            raise ValueError('Error BC top')
        #--------------------------
        if BC[1]=='D':
        #Apply BC Dirichlet Bottom
            self.A[self.Nx-1,self.Nx-1]=1.
            self.A[self.Nx-1,self.Nx-2]=0.
            b[-1]=TorQ[1]
        elif BC[1]=='N':
        #Apply BC Von Neuman Bottom
            c2=TorQ[1]/self.k_cond
            self.A[self.Nx-1,self.Nx-1]=1+2*self.s
            self.A[self.Nx-1,self.Nx-2]=-2*self.s
            b[-1]=b[-1]+2*self.s*self.dx*c2
        else:
            raise ValueError('Error BC bottom')
        #-------
        return sparse.linalg.spsolve(self.A,  b )

    def advance_dt(self,R_old,BC,TorQ):
        if self.runtype=='Explicit':
            temp= self.advance_explicit(R_old,BC,TorQ)

        elif self.runtype=='Implicit':
            temp= self.advance_implicit(R_old,BC,TorQ)
        else:
            raise ValueError('Error in advance dt')
        return temp



def background_dust_tau(Ls_deg,lat_deg):
    """
    5-cosine fit to the Ames GCM Background dust scenario.

    Parameters
    ----------
    Ls_deg : float or array
        Areocentric solar longitude Ls [degrees].
    lat_deg : float or array
        Planetocentric latitude [degrees].

    Returns
    -------
    tau : float or array
        Dust optical depth.
    """

    lat = np.asarray(lat_deg)
    Ls = np.asarray(Ls_deg)

    log_tau = (
        -1.39503896
        + 0.86962746 * np.cos(np.deg2rad(Ls + 142.586279))
        + 0.10903534 * np.cos(np.deg2rad(3*lat - 130.927067))
        + 0.89178155 * np.cos(np.deg2rad(lat - Ls + 6.915782))
        + 0.55329458 * np.cos(np.deg2rad(2*lat + 14.397994))
        + 0.37107464 * np.cos(np.deg2rad(2*lat - Ls - 141.564067))
    )

    return np.exp(log_tau)

def cos_declination(lat,Ls,LTST_hr):
    '''
    Return the solar flux at the surface or top of the atmosphere.
    Following Levine (1977): Solar radiation on Mars and Outer Planet, Icarus
    Args:
        lat: latitude, in [radian]
        Ls: solar longitude in [radian]
        LTST_hr : local time in [hours]
        tau: dust
    '''
    lat=np.array(lat);Ls=np.array(Ls);LTST_hr=np.array(LTST_hr) #lat is radian, Ls in radian, LTST is in hour (noon=12)
    solar_decl= np.arcsin(np.sin(obliquity)*np.sin(Ls)) # radian
    cos_z=(np.sin(lat)*np.sin(solar_decl)+np.cos(lat)*np.cos(solar_decl)*np.cos((LTST_hr-12.)/24.*2*np.pi)).clip(min=10**-8)
    return cos_z



def Sun(lat,Ls,LTST_hr,tau): #
    '''
    Return the solar flux at the surface or top of the atmosphere.
    Following Levine (1977): Solar radiation on Mars and Outer Planet, Icarus
    Args:
        lat: latitude, in [radian]
        Ls: solar longitude in [radian]
        LTST_hr : local time in [hours]
        tau: dust
    '''
    lat=np.array(lat);Ls=np.array(Ls);LTST_hr=np.array(LTST_hr) #lat is radian, Ls in radian, LTST is in hour (noon=12)
    Ratio= 1+e*np.cos(Ls-Lsp)/(1-e**2)                          #tau is the optical depth
    solar_decl= np.arcsin(np.sin(obliquity)*np.sin(Ls)) # radian
    cos_z=(np.sin(lat)*np.sin(solar_decl)+np.cos(lat)*np.cos(solar_decl)*np.cos((LTST_hr-12.)/24.*2*np.pi)).clip(min=10**-8)
    #return Rsun*Ratio**2*np.exp(-tau/max(cos_z,10**-8))*cos_z # 2nd cos_z is because the solar panels are not facing the beam
    return Rsun*Ratio**2*np.exp(-tau/cos_z)*cos_z


def rho_TP(T,P):   #density
    T=np.array(T);P=np.array(P)
    return P*Mco2/(R_*T)


def SHforced_TPUTs(T,P,u5,Ts):
    A_drag=(von_k/np.log(z5/z0))**2
    return rho_TP(T,P)*cp_co2*A_drag*u5*(T-Ts)

#===COMIMART=============

"""Standalone COMIMART delta-Eddington surface-flux model.

This is a scalar, standard-library-only transcription of
Vicente-Retortillo et al. (2015), J. Space Weather Space Clim. 5, A33,
Eqs. (13)--(23).  Eqs. (10)--(12) are included because the later equations
use the delta-scaled optical properties ``tau_prime``, ``omega_prime``, and
``g_prime``.

``toa_horizontal_flux`` may be broadband or spectral; the returned fluxes
have the same units.  Call the function once per wavelength or atmospheric
band when the optical properties vary spectrally.
"""



class SurfaceFluxes(NamedTuple):
    """Downward total, unscattered-direct, and diffuse surface flux."""

    total: float
    direct: float
    diffuse: float


def _conservative_total_ratio(
    tau_prime: float,
    g_prime: float,
    A: float,
    mu0: float,
) -> float:
    """Continuous ``omega_prime -> 1`` limit of Eqs. (13)--(22)."""
    source = math.exp(-tau_prime / mu0)
    q = 2.0 / (3.0 * (1.0 - g_prime))
    alpha = 0.75 * mu0
    beta = 0.5
    c3 = A + (1.0 - A) * alpha - (1.0 + A) * beta
    denominator = (1.0 - A) * tau_prime + 2.0 * q
    return (
        (c3 * source * tau_prime + 2.0 * (alpha + beta) * q) / denominator
        - (alpha + beta - 1.0) * source
    )


def _resonant_total_ratio(
    tau_prime: float,
    omega_prime: float,
    g_prime: float,
    A: float,
    k: float,
    p_prime: float,
) -> float:
    """Continuous ``mu0 * k -> 1`` limit of Eqs. (13)--(22)."""
    mu0 = 1.0 / k
    source = math.exp(-tau_prime / mu0)
    tanh_k_tau = math.tanh(k * tau_prime)
    reduced_denominator = 2.0 * p_prime + (
        1.0 - A + p_prime**2 * (1.0 + A)
    ) * tanh_k_tau
    c3_coefficient = (
        (1.0 - p_prime**2) * tanh_k_tau / reduced_denominator
    )
    exp_negative = math.exp(-k * tau_prime)
    sech_k_tau = 2.0 * exp_negative / (1.0 + exp_negative**2)
    source_coefficient = 2.0 * p_prime * sech_k_tau / reduced_denominator

    alpha = 0.75 * mu0 * omega_prime * (
        1.0 + g_prime * (1.0 - omega_prime)
    )
    beta = 0.5 * omega_prime * (
        1.0 + 3.0 * mu0**2 * g_prime * (1.0 - omega_prime)
    )
    c3 = (1.0 - A) * alpha - (1.0 + A) * beta

    alpha_derivative = 0.75 * omega_prime * (
        1.0 + g_prime * (1.0 - omega_prime)
    )
    beta_derivative = 3.0 * mu0 * omega_prime * g_prime * (
        1.0 - omega_prime
    )
    c3_derivative = (
        (1.0 - A) * alpha_derivative
        - (1.0 + A) * beta_derivative
    )
    source_derivative = source * tau_prime / mu0**2
    numerator_derivative = (
        source_derivative * (c3_coefficient * c3 - alpha - beta)
        + source * c3_coefficient * c3_derivative
        + (alpha_derivative + beta_derivative)
        * (source_coefficient - source)
    )
    denominator_derivative = -2.0 * mu0 * k**2
    particular_solution = numerator_derivative / denominator_derivative
    return source * (1.0 + A * c3_coefficient) + particular_solution


def comimart_surface_fluxes(
    toa_horizontal_flux: float,
    optical_depth: float,
    single_scattering_albedo: float,
    asymmetry_factor: float,
    surface_albedo: float,
    mu0: float,
) -> SurfaceFluxes:
    """Calculate the downward surface fluxes in COMIMART Eqs. (13)--(23).

    Args:
        toa_horizontal_flux: Incident horizontal flux at the top of the
            atmosphere, denoted ``E`` in the paper.
        optical_depth: Unscaled extinction optical depth ``tau``.
        single_scattering_albedo: Unscaled ``omega`` in [0, 1].
        asymmetry_factor: Unscaled phase-function asymmetry ``g`` in (-1, 1).
        surface_albedo: Lambertian surface albedo ``A`` in [0, 1].
        mu0: Cosine of the solar zenith angle, in (0, 1].

    Returns:
        ``SurfaceFluxes(total, direct, diffuse)`` in the input flux units.
    """
    E, tau, omega, g, A, mu0 = map(
        float,
        (
            toa_horizontal_flux,
            optical_depth,
            single_scattering_albedo,
            asymmetry_factor,
            surface_albedo,
            mu0,
        ),
    )
    named_inputs = {
        "toa_horizontal_flux": E,
        "optical_depth": tau,
        "single_scattering_albedo": omega,
        "asymmetry_factor": g,
        "surface_albedo": A,
        "mu0": mu0,
    }
    for name, value in named_inputs.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if E < 0.0 or tau < 0.0:
        raise ValueError(
            "toa_horizontal_flux and optical_depth must be nonnegative"
        )
    if not 0.0 <= omega <= 1.0:
        raise ValueError("single_scattering_albedo must lie in [0, 1]")
    if not -1.0 < g < 1.0:
        raise ValueError("asymmetry_factor must lie strictly between -1 and 1")
    if not 0.0 <= A <= 1.0:
        raise ValueError("surface_albedo must lie in [0, 1]")
    if not 0.0 < mu0 <= 1.0:
        raise ValueError("mu0 must lie in (0, 1]")
    if E == 0.0 or tau == 0.0:
        return SurfaceFluxes(E, E, 0.0)

    # Delta scaling, Eqs. (10)--(12), required by Eqs. (13)--(22).
    scaling_denominator = 1.0 - omega * g**2
    g_prime = g / (1.0 + g)
    omega_prime = omega * (1.0 - g**2) / scaling_denominator
    tau_prime = tau * scaling_denominator

    if 1.0 - omega_prime <= 64.0 * sys.float_info.epsilon:
        total_ratio = _conservative_total_ratio(tau_prime, g_prime, A, mu0)
    else:
        # Eqs. (14)--(17).
        k = math.sqrt(
            3.0 * (1.0 - omega_prime) * (1.0 - g_prime * omega_prime)
        )
        p_prime = (2.0 / 3.0) * math.sqrt(
            3.0 * (1.0 - omega_prime) / (1.0 - g_prime * omega_prime)
        )
        particular_denominator = 1.0 - mu0**2 * k**2
        resonance_tolerance = math.sqrt(sys.float_info.epsilon)
        if abs(particular_denominator) <= resonance_tolerance:
            total_ratio = _resonant_total_ratio(
                tau_prime, omega_prime, g_prime, A, k, p_prime
            )
        else:
            alpha = (
                0.75
                * mu0
                * omega_prime
                * (1.0 + g_prime * (1.0 - omega_prime))
                / particular_denominator
            )
            beta = (
                0.5
                * mu0
                * omega_prime
                * (
                    1.0 / mu0
                    + 3.0 * mu0 * g_prime * (1.0 - omega_prime)
                )
                / particular_denominator
            )

            # Eqs. (20)--(22).
            c3 = A + (1.0 - A) * alpha - (1.0 + A) * beta
            c4 = 1.0 - A + p_prime * (1.0 + A)
            c5 = 1.0 - A - p_prime * (1.0 + A)

            exp_positive = math.exp(k * tau_prime)
            exp_negative = math.exp(-k * tau_prime)
            exp_source = math.exp(-tau_prime / mu0)
            common_denominator = (
                (1.0 + p_prime) * c4 * exp_positive
                - (1.0 - p_prime) * c5 * exp_negative
            )

            # Eqs. (18)--(19).
            c1 = -(
                (1.0 - p_prime) * c3 * exp_source
                - (alpha + beta) * c4 * exp_positive
            ) / common_denominator
            c2 = (
                (1.0 + p_prime) * c3 * exp_source
                - (alpha + beta) * c5 * exp_negative
            ) / common_denominator

            # Eq. (13): total downward surface-flux ratio T/E.
            total_ratio = (
                c1 * exp_negative * (1.0 + p_prime)
                + c2 * exp_positive * (1.0 - p_prime)
                - (alpha + beta - 1.0) * exp_source
            )

    # Eq. (23) uses the original, unscaled optical depth.
    direct = E * math.exp(-tau / mu0)
    total = E * total_ratio
    diffuse = total - direct  # Definition immediately following Eq. (23).
    roundoff = 256.0 * sys.float_info.epsilon * max(1.0, abs(total), abs(direct))
    if -roundoff <= diffuse < 0.0:
        diffuse = 0.0
    return SurfaceFluxes(direct + diffuse, direct, diffuse)


