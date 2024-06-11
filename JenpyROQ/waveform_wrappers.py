## -*- coding: utf8 -*-
#!/usr/bin/env python

import numpy as np, warnings

# ===================== #
# Waveform approximants #
# ===================== #

__non_lal_approx_names__ = ['teobresums-giotto',    'mlgw-bns', 'mlgw-bns-standalone',
                            'nrpmw', 'nrpmw-recal', 'nrpmw-merger', 'nrpmw-recal-merger',
                            'teobresums-spa-nrpmw', 'teobresums-spa-nrpmw-recal',
                            'mlgw-bns-nrpmw',       'mlgw-bns-nrpmw-recal']

WfWrapper = {} # collect all the wvf wrappers

# ------------------------------------------------------- #
# Example of waveform wrapper for JenpyROQ                #
# A wrapper must be a class with these attributes/methods #
# ------------------------------------------------------- #

class ZeroWf:
    
    def __init__(self,
                 approximant,
                 waveform_params = {}):
        
        self.approximant     = approximant
        self.waveform_params = waveform_params
    
    def generate_waveform(self, p, deltaF, f_min, f_max):
        
        freq = np.arange(f_min, f_max+deltaF, deltaF)
        hp   = np.zeros(len(freq))
        hc   = np.zeros(len(freq))
        
        return hp, hc

# --- #
# LAL #
# --- #

try:
    
    # LAL imports
    import lal, lalsimulation
    from lal.lal import PC_SI as LAL_PC_SI, MSUN_SI as LAL_MSUN_SI

    # Add the approximants that can be called
    approximants = []
    approximants.append(lalsimulation.IMRPhenomD)
    approximants.append(lalsimulation.IMRPhenomPv2)
    approximants.append(lalsimulation.IMRPhenomPv3)
    approximants.append(lalsimulation.IMRPhenomPv3HM)
    approximants.append(lalsimulation.IMRPhenomXHM)
    approximants.append(lalsimulation.IMRPhenomXPHM)
    approximants.append(lalsimulation.TaylorF2Ecc)
    approximants.append(lalsimulation.IMRPhenomPv2_NRTidal)
    approximants.append(lalsimulation.IMRPhenomNSBH)
    
    # LAL wrapper
    class LALWf:
        def __init__(self,
                     approximant,
                     additional_waveform_params = {}):

            self.approximant = approximant

            #FIXME: for LAL waveforms, additional_waveform_params is currently ignored

        def generate_waveform(self, p, deltaF, f_min, f_max, distance):
            
            waveform_params = lal.CreateDict()

            # Update baseline waveform_params with p
            # This is redundant and incomplete, see
            # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/waveform.py#L77
            # but it should be Ok
        
            if 'lambda1' in p.keys():
                if p['lambda1'] is not None: lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(waveform_params, p['lambda1'])
            if 'lambda2' in p.keys():
                if p['lambda2'] is not None: lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(waveform_params, p['lambda2'])

            if 'ecc' not in p.keys(): p['ecc'] = 0.

            spin_names = ['s1x', 's1y', 's1z', 's2x', 's2y', 's2z']
            for spin_name in spin_names:
                if not(spin_name in p.keys()): p[spin_name] = 0.0

            [plus, cross] = lalsimulation.SimInspiralChooseFDWaveform(p['m1']*LAL_MSUN_SI,
                                                                      p['m2']*LAL_MSUN_SI,
                                                                      p['s1x'], p['s1y'], p['s1z'],
                                                                      p['s2x'], p['s2y'], p['s2z'],
                                                                      distance*(LAL_PC_SI*1e6),
                                                                      p['iota'],
                                                                      p['phiref'],
                                                                      0, # 'long_asc_nodes'
                                                                      p['ecc'],
                                                                      0, # 'mean_per_ano'
                                                                      deltaF,
                                                                      f_min,
                                                                      f_max,
                                                                      0,
                                                                      waveform_params,
                                                                      self.approximant)
            hp = plus.data.data
            hc = cross.data.data
            hp = hp[np.int(f_min/deltaF):np.int(f_max/deltaF)+1]
            hc = hc[np.int(f_min/deltaF):np.int(f_max/deltaF)+1]
            
            return hp, hc

    # Add a wrapper for each approximant
    for a in approximants: WfWrapper[a] = LALWf

except ModuleNotFoundError:

    print('\nWarning: `LALSimulation` module not found.\n')

    # Set the constants, they might be needed
    # https://lscsoft.docs.ligo.org/lalsuite/lal/group___l_a_l_constants__h.html
    LAL_PC_SI   = 3.085677581491367278913937957796471611e16
    LAL_MSUN_SI = 1.988409902147041637325262574352366540e30


# ---------- #
# TEOBResumS #
# ---------- #
    
try:
    # TEOBResumS imports
    import EOBRun_module

    # Add the approximants that can be called
    approximants = []
    approximants.append('teobresums-giotto')

    # Helpers
    TEOBResumS_domain = {'TD':0,'FD':1}
    TEOBResumS_spins  = {'nospin':0,'aligned':1,'precessing':2}

    # TEOBResumS wrapper
    class WfTEOBResumS:
        def __init__(self,
                     approximant,
                     additional_waveform_params = {}):
            
            self.approximant = approximant

            # Start with the defaults, and update according to user request
            self.waveform_params = self.set_parameters()
            self.waveform_params.update(additional_waveform_params)

        def modes_to_k(self,modes):
            
            """
            Map (l,m) -> k 
            """
            
            return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

        def set_parameters(self):
            
            """
            Utility to set EOB parameters based on the selected mode
            Uses defaults for unset parameters
            """
            
            p = {}
            p['use_geometric_units'] = "no"
            p['domain']              = TEOBResumS_domain['FD']
            p['interp_uniform_grid'] = "yes" # ignored for FD, needed because of FFT for TD
        
            p['use_mode_lm'        ] = self.modes_to_k([[2,2]])
            p['use_spins'          ] = TEOBResumS_spins['aligned']

            p['output_hpc'         ] = "no"
            p['output_multipoles'  ] = "no"

            return p

        def JBJF(self,hp,hc,dt):
            
            """
            Fourier transform of TD wfv
            """
            
            hptilde = np.fft.rfft( hp) * dt 
            hctilde = np.fft.rfft(-hc) * dt
            
            return hptilde, hctilde
  
        def generate_waveform(self, p, deltaF, f_min, f_max, distance):

            m1,m2 = p['m1'],p['m2']
            q     = p['m1']/p['m2']

            s1x,s1y,s1z = 0.,0.,0.
            s2x,s2y,s2z = 0.,0.,0.
            if self.waveform_params['use_spins'] == TEOBResumS_spins['precessing']:
                if (('s1x' not in p.keys()) or ('s1y' not in p.keys()) or ('s1z' not in p.keys())): raise ValueError("Spin1 parameters missing, while the 'precessing' spin option for TEOB was selected.")
                if (('s2x' not in p.keys()) or ('s2y' not in p.keys()) or ('s2z' not in p.keys())): raise ValueError("Spin2 parameters missing, while the 'precessing' spin option for TEOB was selected.")
                s1x,s1y,s1z = p['s1x'], p['s1y'], p['s1z']
                s2x,s2y,s2z = p['s2x'], p['s2y'], p['s2z']
                
            elif self.waveform_params['use_spins'] == TEOBResumS_spins['aligned']:
                if 's1z' not in p.keys(): raise ValueError('spin1 parameters missing')
                if 's2z' not in p.keys(): raise ValueError('spin2 parameters missing')
                s1z = p['s1z']
                s2z = p['s2z']
            
            if((abs(p['lambda1']) < 0.) or (abs(p['lambda2']) < 0.)): raise ValueError("The lambda parameters have to be larger than 0, but ({},{}) were passed.".format(p['lambda1'], p['lambda2']))
            lambda1,lambda2 = 0.,0.
            if 'lambda1' in p.keys(): lambda1 = p['lambda1']
            if 'lambda2' in p.keys(): lambda2 = p['lambda2']

            if 'ecc' not in p.keys(): p['ecc'] = 0.
            if(abs(p['ecc']) > 1e-12): raise ValueError("Eccentricity is not supported, but eccentricity={} was passed.".format(p['ecc']))
            
            # convert naming from bajes to TEOB for dynamics pars
            if 'TEOBResumS_a6c'   in p.keys(): p['a6c']   = p['TEOBResumS_a6c']    
            if 'TEOBResumS_cN3LO' in p.keys(): p['cN3LO'] = p['TEOBResumS_cN3LO']

            # Impose the correct convention on masses
            if q < 1. :
               m1,m2           = m2,m1
               q               = 1./q
               t1,t2,t3        = s1x,s1y,s1z
               s1x,s1y,s1z     = s2x,s2y,s2z
               s2x,s2y,s2z     = t1,t2,t3
               lambda1,lambda2 = lambda2,lambda1

            # System parameters
            p['M'                  ] = m1+m2
            p['q'                  ] = q
            p['LambdaAl2'          ] = lambda1
            p['LambdaBl2'          ] = lambda2
            if self.waveform_params['use_spins'] == TEOBResumS_spins['precessing']:
                p['chi1x'          ] = s1x
                p['chi1y'          ] = s1y
                p['chi1z'          ] = s1z
                p['chi2x'          ] = s2x
                p['chi2y'          ] = s2y
                p['chi2z'          ] = s2z
            else:
                p['chi1'           ] = s1z
                p['chi2'           ] = s2z
            p['distance'           ] = distance # Mpc
            p['inclination'        ] = p['iota']
            p['coalescence_angle'  ] = p['phiref']

            # Generator parameters
            p['initial_frequency'  ] = f_min    # Hz
            p['srate_interp'       ] = f_max*2  # If TD, sets dt at which to interpolate waveform and dynamics. If FD, sets f_max = srate_interp/2.
            p['df'                 ] = deltaF   # Freq axis goes from initial_frequency to srate_interp/2. in units of df

            p['srate'              ] = f_max*2  # If TD, srate at which the ODE is solved. Possibly unused, CHECK.

            self.waveform_params.update(p)
            
            if self.waveform_params['domain'] == TEOBResumS_domain['TD']:
                t, hptd, hctd = EOBRun_module.EOBRunPy(self.waveform_params)
                hp, hc = JBJF(hptd,hctd,t[1]-t[0])
                warnings.warn("\n\nCheck waveform lenght\n\n")
            else:
                f, rhplus, ihplus, rhcross, ihcross = EOBRun_module.EOBRunPy(self.waveform_params)
                hp, hc = rhplus-1j*ihplus, rhcross-1j*ihcross

            return hp, hc
    
    # Add a wrapper for each approximant
    for a in approximants: WfWrapper[a] = WfTEOBResumS

except ModuleNotFoundError: print('\nWarning: `TEOBResumS` module not found.\n')

# FIXME: This has been left for testing reasons, and should be removed once the implementation is stable.

# -------- #
# MLGW-BNS #
# -------- #
try:
    # MLGW imports
    from mlgw_bns import ParametersWithExtrinsic, Model

    # Add the approximants that can be called
    approximants = []
    approximants.append('mlgw-bns-standalone')

    # MLGW-BNS wrapper
    class WfMLGW:
        def __init__(self,
                     approximant,
                     additional_waveform_params = {}):

            # Currently unused for this waveform
            self.approximant     = approximant
            self.waveform_params = additional_waveform_params

        def generate_waveform(self, p, deltaF, f_min, f_max, distance):

            m1,m2 = p['m1'],p['m2']
            q     = p['m1']/p['m2']

            s1x,s1y,s1z = 0,0,0
            s2x,s2y,s2z = 0,0,0
            if 's1z' not in p.keys(): raise ValueError('spin1 parameters missing')
            if 's2z' not in p.keys(): raise ValueError('spin2 parameters missing')
            s1z = p['s1z']
            s2z = p['s2z']

            # Precessing spins are not supported
            if 's1x' in p.keys(): s1x = p['s1x']
            if 's1y' in p.keys(): s1y = p['s1y']
            if 's2x' in p.keys(): s2x = p['s2x']
            if 's2y' in p.keys(): s2y = p['s2y']
            if((abs(s1x) > 1e-6) or (abs(s1y) > 1e-6)): raise ValueError("Precession is not supported, but (spin1x, spin1y)=({},{}) were passed.".format(s1x, s1y))
            if((abs(s2x) > 1e-6) or (abs(s2y) > 1e-6)): raise ValueError("Precession is not supported, but (spin2x, spin2y)=({},{}) were passed.".format(s2x, s2y))

            lambda1,lambda2 = 0.,0.
            if 'lambda1' in p.keys(): lambda1 = p['lambda1']
            if 'lambda2' in p.keys(): lambda2 = p['lambda2']
            if((abs(lambda1) < 5.   ) or (abs(lambda2) < 5.   )): raise ValueError("lambdas>5 but ({},{}) were passed.".format(lambda1, lambda2))
            if((abs(lambda1) > 5000.) or (abs(lambda2) > 5000.)): raise ValueError("lambdas<5000 but ({},{}) were passed.".format(lambda1, lambda2))

            if 'ecc' not in p.keys(): p['ecc'] = 0.
            if(abs(p['ecc']) > 1e-12): raise ValueError("Eccentricity is not supported, but eccentricity={} was passed.".format(p['ecc']))

            # Impose the correct convention on masses
            if q < 1. :
               m1,m2           = m2,m1
               q               = 1./q
               s1z,s2z         = s2z,s1z
               lambda1,lambda2 = lambda2,lambda1

            # Call it
            model       = Model.default() # FIXME: here you can use self.approximant to call any MLGW-BNS Model
            frequencies = np.arange(f_min, f_max+deltaF, step=deltaF)

            params      = ParametersWithExtrinsic(q,
                                                  lambda1,
                                                  lambda2,
                                                  s1z,
                                                  s2z,
                                                  distance,
                                                  p['iota'],
                                                  m1+m2,
                                                  reference_phase=p['phiref'])
            hp, hc      = model.predict(frequencies, params)

            return hp, hc

    # Add a wrapper for each approximant
    for a in approximants: WfWrapper[a] = WfMLGW

except ModuleNotFoundError: print('\nWarning: `mlgw-bns module` not found.\n')

# ---------------------------------------------------- #
# NRPMw & MLGW-BNS & MLGW-BNS+NRPMw & TEOBResumS+NRPMw #
# ---------------------------------------------------- #
try:
    
    # Waveform wrappers
    from bajes.obs.gw.approx.nrpmw      import nrpmw_wrapper, nrpmw_recal_wrapper, nrpmw_attach_wrapper, nrpmw_attach_recal_wrapper
    from bajes.obs.gw.approx.teobresums import teobresums_spa_nrpmw_wrapper, teobresums_spa_nrpmw_recal_wrapper
    from bajes.obs.gw.approx.mlgw       import mlgw_bns_wrapper, mlgw_bns_nrpmw_wrapper, mlgw_bns_nrpmw_recal_wrapper

    from bajes.obs.gw.approx.nrpmw      import __recalib_names__ as recalib_names_no_attach, __recalib_names_attach__ as recalib_names_attach

    # Add the approximants that can be called
    approximants = []
    approximants.append('nrpmw')
    approximants.append('nrpmw-recal')
    approximants.append('nrpmw-merger')
    approximants.append('nrpmw-recal-merger')
    approximants.append('teobresums-spa-nrpmw')
    approximants.append('teobresums-spa-nrpmw-recal')
    approximants.append('mlgw-bns')
    approximants.append('mlgw-bns-nrpmw')
    approximants.append('mlgw-bns-nrpmw-recal')

    class WfBajes:
        def __init__(self,
                     approximant,
                     additional_waveform_params = {}):
            
            self.approximant     = approximant
            # Currently unused for this waveform
            self.waveform_params = additional_waveform_params

            # Avoid multple initialization of Model.default()
            # Note: init values of bajes waveform wrappers are not used in waveform generation
            if   self.approximant=='mlgw-bns'            : self.waveform_func = mlgw_bns_wrapper(            [0,1,2], 1, 1)
            elif self.approximant=='mlgw-bns-nrpmw'      : self.waveform_func = mlgw_bns_nrpmw_wrapper(      [0,1,2], 1, 1)
            elif self.approximant=='mlgw-bns-nrpmw-recal': self.waveform_func = mlgw_bns_nrpmw_recal_wrapper([0,1,2], 1, 1)

            if('recal' in self.approximant):
                if('merger' in self.approximant): self.recalib_names = recalib_names_no_attach
                else                            : self.recalib_names = recalib_names_attach

        def generate_waveform(self, p, deltaF, f_min, f_max, distance):

            # Note: differentely from the TEOBResumS wrapper, in this one we work directly with the dictionary that will be passed to the waveform generator.

            p['q']    = p['m1']/p['m2']
            p['mtot'] = p['m1'] + p['m2']
            p['nu']   = p['m1']*p['m2']/(p['m1'] + p['m2'])**2

            if 's1z' not in p.keys(): raise ValueError('`spin1` parameters missing.')
            if 's2z' not in p.keys(): raise ValueError('`spin2` parameters missing.')

            if('s1x' in p.keys()): raise ValueError("Precession is not supported, but spin1_x = {} was passed.".format(p['s1x']))
            if('s1y' in p.keys()): raise ValueError("Precession is not supported, but spin1_y = {} was passed.".format(p['s1y']))
            if('s2x' in p.keys()): raise ValueError("Precession is not supported, but spin2_x = {} was passed.".format(p['s2x']))
            if('s2y' in p.keys()): raise ValueError("Precession is not supported, but spin2_y = {} was passed.".format(p['s2y']))

            # Lambda parameters are required for this model.
            if 'lambda1' not in p.keys(): raise ValueError('`lambda1` parameters missing.')
            if 'lambda2' not in p.keys(): raise ValueError('`lambda2` parameters missing.')
            if((abs(p['lambda1']) < 0.) or (abs(p['lambda2']) < 0.)): raise ValueError("The lambda parameters have to be larger than 0, but ({},{}) were passed.".format(p['lambda1'], p['lambda2']))
            if((abs(p['lambda1']) > 5000.) or (abs(p['lambda2']) > 5000.)): raise ValueError("The model was calibrated on lambdas < 5000 but ({},{}) were passed.".format(p['lambda1'], p['lambda2']))

            if 'ecc' not in p.keys(): p['ecc'] = 0.
            if(abs(p['ecc']) > 1e-12): raise ValueError("Eccentricity is not supported, but eccentricity={} was passed.".format(p['ecc']))

            # Impose the correct convention on masses
            if p['q'] < 1.:
               p['q']                     = 1./p['q']
               p['s1z'],p['s2z']          = p['s2z'],p['s1z']
               p['lambda1'], p['lambda2'] = p['lambda2'],p['lambda1']

            # Extrinsic parameters
            p['distance'] = distance
            p['cosi']     = np.cos(p['iota'])
            p['phi_ref']  = p['phiref']

            # Auxiliary parameters
            p['seglen'] = 1./deltaF
            p['f_min']  = f_min
            p['f_max']  = f_max
            p['srate']  = f_max*2

            # Post-merger parameters
            if('nrpmw' in self.approximant):
                
                if((self.approximant=='nrpmw') or (self.approximant=='nrpmw-recal')): p['NRPMw_phi_pm'] = 0.             # At the NRPMw level, NRPMw_phi_pm has the same effect of phi_ref
                else                                                                : p['NRPMw_phi_pm'] = p['nrpmw-phi']
                
                p['NRPMw_t_coll'] = p['nrpmw-tcoll']
                p['NRPMw_df_2']   = p['nrpmw-df2']
                
                # 22-mode only
                p['lmax']   = 0

            # Add recalibration parameters
            if('recal' in self.approximant):
                for ni in self.recalib_names: p['NRPMw_recal_'+ni] = p['nrpmw-{}'.format(ni)]

            frequencies = np.arange(f_min, f_max+deltaF, step=deltaF)

            # Call it
            if(  self.approximant=='nrpmw'                     ): hp, hc = nrpmw_attach_wrapper(              frequencies, p)
            elif(self.approximant=='nrpmw-recal'               ): hp, hc = nrpmw_attach_recal_wrapper(        frequencies, p)
            elif(self.approximant=='nrpmw-merger'              ): hp, hc = nrpmw_wrapper(                     frequencies, p)
            elif(self.approximant=='nrpmw-recal-merger'        ): hp, hc = nrpmw_recal_wrapper(               frequencies, p)
            elif(self.approximant=='teobresums-spa-nrpmw'      ): hp, hc = teobresums_spa_nrpmw_wrapper(      frequencies, p)
            elif(self.approximant=='teobresums-spa-nrpmw-recal'): hp, hc = teobresums_spa_nrpmw_recal_wrapper(frequencies, p)
            elif('mlgw-bns' in self.approximant                ): hp, hc = self.waveform_func(                frequencies, p)

            if any(np.isnan(hp)): raise  RuntimeError("Waveform generator returned NaN for this parameter: {}".format(p))
            else:                 return hp, hc

    # Add a wrapper for each approximant
    for a in approximants: WfWrapper[a] = WfBajes

except ModuleNotFoundError: print('\nWarning: `bajes` module or `teobresums` module not found.\n')