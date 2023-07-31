## -*- coding: utf8 -*-
#!/usr/bin/env python

import numpy as np, os, subprocess, sys, warnings
try:                import configparser
except ImportError: import ConfigParser as configparser

def store_git_info(output):

    git_info = open(os.path.join(output, 'git_info.txt'), 'w')
    pipe1 = str(subprocess.Popen("git branch | grep \* | cut -d ' ' -f2", shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    pipe2 = str(subprocess.Popen("git log --pretty=format:'%H' -n 1 ",    shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    git_info.write('PyROQ\nbranch: {}\t commit: {}\n'.format(pipe1, pipe2))
    pipe1 = str(subprocess.Popen('git config user.name',  shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    pipe2 = str(subprocess.Popen('git config user.email', shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    git_info.write('Author: {} {}'.format(pipe1, pipe2))
    pipe = str(subprocess.Popen('git diff',               shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    git_info.write('\n\nGit diff:\n{}'.format(pipe))
    git_info.close()

    return

#Description of the package. Printed on stdout if --help option is give.
usage="""\n\n python -m PyROQ --config-file config.ini\n
Package description FIXME.

Options syntax: type, default values and sections of the configuration
file where each parameter should be passed are declared below.
By convention, booleans are represented by the integers [0,1].
A dot is present at the end of each description line and is not
to be intended as part of the default value.

       **************************************************************************
       * Parameters to be passed to the [I/O] section.                          *
       **************************************************************************

               output                  Output directory. Default: './'.
               verbose                 Option to regulate logger verbose mode. Available options: [0,1, ???]. Default: 1.
               debug                   Flag to activate debugging additional checks. Default: 0.
               timing                  Flag to activate timing profiling. Default: 0.
               show-plots              Flag to show produced plots. Default: 0.
               post-processing-only    Flag to skip interpolants constructions, running post-processing tests and plots. Default: 0.
               random-seed             Initial seed for pseudo-random number generators. Default: 170817.
       
       **************************************************************************
       * Parameters to be passed to the [Parallel] section.                     *
       **************************************************************************

               parallel                Option to activate parallelisation. Allowed values: [0, 1, 2] corrsponding to [serial, multiprocessing, MPI]. Default: 0.
               n-processes             Number of processes on which the parallelisation is carried on. Default: 4.
       
       **************************************************************************
       * Parameters to be passed to the [Waveform_and_parametrisation] section. *
       **************************************************************************

               approximant             Waveform approximant. Allowed values: ['teobresums-giotto', 'mlgw-bns-standalone', 'mlgw-bns' (called through bajes), 'nrpmw', 'nrpmw-recal', 'nrpmw-merger', 'nrpmw-recal-merger', 'teobresums-spa-nrpmw', 'teobresums-spa-nrpmw-recal', 'mlgw-bns-nrpmw', 'mlgw-bns-nrpmw-recal','IMRPhenomPv2', 'IMRPhenomPv3', 'IMRPhenomXHM', 'IMRPhenomXPHM', 'TaylorF2Ecc', 'IMRPhenomPv2_NRTidal', 'IMRPhenomNSBH']. Default: 'teobresums-giotto'.
               spins                   Option to select spin degrees of freedom. Allowed values: ['no-spins', 'aligned', 'precessing']. Default: 'aligned'.
               tides                   Flag to activate tides training. Default: 0.
               eccentricity            Flag to activate eccentricity training. Default: 0.
               post-merger             Flag to activate post-merger parameters training. Default: 0.
               dynamics                Flag to activate EOB dynamics parameters training. Default: 0.
               mc-q-par                Flag to activate parametrisation in chirp mass and mass ratio. Default: 1.
               m-q-par                 Flag to activate parametrisation in total mass and mass ratio. Default: 0.
               spin-sph                Flag to activate parametrisation in spins spherical components. Default: 0.
               f-min                   Minimum of the frequency axis on which the interpolant will be constructed. Default: 20.
               f-max                   Maximum of the frequency axis on which the interpolant will be constructed. Default: 1024.
               seglen                  Inverse of the step of the frequency axis on which the interpolant will be constructed. Default: 4.0.

       **************************************************************************
       * Parameters to be passed to the [ROQ] section.                          *
       **************************************************************************
       
               gram-schmidt            Flag to activate gram-schmidt orthonormalisation on a new basis element. Default: 0.
              
               basis-lin               Flag to activate linear    basis construction. Default: 1.
               basis-qua               Flag to activate quadratic basis construction. Default: 1.
       
               n-tests-post            Number of random validation test waveforms checked to be below tolerance a-posteriori. Typically same as `n_tests_basis`. Default: 1000.
               minimum-speedup         Minimum ratio of X:=len(Original-frequency-axis)/len(ROQ-frequency-axis), implying a minimum speedup during parameter estimation. The ROQ construction is interrupted if X < `minimum-speedup`. Default: 1.0.
           
               pre-basis               Option determining the pre-basis computation. Available options: ['corners', partial-pre-selected-basis', 'pre-selected-basis', 'pre-enriched-basis']; 'corners' initialises the basis using the lower and upper values of all parameters simultaneously; 'partial-pre-selected-basis' uses a previously computed subset of a pre-selected basis (typically used when the code crashed in the middle of one of the pre-selection cycle); 'pre-selected-basis' uses a previously computed complete pre-selected basis; 'pre-enriched-basis' uses a previously enriched basis (typically used when the code crashed in the middle of one of the enrichment cycles). Default: 'corners'.
               tolerance-pre-basis-lin Basis projection error threshold for linear basis elements. Default: 1e-8.
               tolerance-pre-basis-qua Same as above, for quadratic basis. Default: 1e-10.
               n-pre-basis-lin         Total number (including corner elements) of basis elements to be constructed in the pre-selection loop for the linear case, before starting the cycles of basis enrichments over training sets. Cannot be smaller than 2 (number of `corner waveforms`). If larger than 2, overrides `tolerance-pre-basis`. Default 80.
               n-pre-basis-qua         Total number (including corner elements) of basis elements to be constructed in the pre-selection loop for the quadratic case, before starting the cycles of basis enrichments over training sets. Cannot be smaller than 2 (number of `corner waveforms`). If larger than 2, overrides `tolerance-pre-basis`. Default 80.

               n-pre-basis-search-iter Number of points for each search of a new basis element during basis construction. Typical values: 30-100 for testing; 300-2000 for production. Typically roughly comparable to the number of basis elements. Depends on complexity of waveform features, parameter space and signal length. Increasing it slows down offline construction time, but decreases number of basis elements. Default: 80.
           
               n-training-set-cycles   Number of basis enrichment cycles, each using `training-set-sizes` number of training elements, and stopping until `training-set-n-outliers` are below `training-set-rel-tol` Default: 4.
               training-set-sizes      List (in string-format) of sizes of the training set for each basis enrichment cycles. Default: '10000,100000,1000000,10000000'.
               training-set-n-outliers List (in string-format) of number of tolerated outliers for each basis enrichment cycles. Default: '10000,100000,1000000,10000000'.Default: '20,20,1,0'.
               training-set-rel-tol    List (in string-format) of relative tolerance (e.g. tolerance = `tolerance-lin` * `training-set-rel-tol`) of the training set for each basis enrichment cycles. Default: '10000,100000,1000000,10000000'.Default: '0.1,0.1,0.05,0.3,1.0'.
               
               tolerance-lin           Interpolation error threshold for linear basis elements. Default: 1e-8.
               tolerance-qua           Same as above, for quadratic basis. Default: 1e-10.
               
       **************************************************************************
       * Parameters range and test values syntax.                               *
       **************************************************************************
       
       Allowed parameter names and units are:
       
               m    (m-q-par=1)                : total mass [Msun]
               mc   (mc-q-par=1)               : chirp mass [Msun]
               q    (mc-q-par=1 or  m-q-par=1) : mass ratio
               m1   (mc-q-par=0 and m-q-par=0) : mass object 1 [Msun]
               m2   (mc-q-par=0 and m-q-par=0) : mass object 2 [Msun]
               s1s1 (spin-sph=1) : spin components object 1, spherical coords (FIXME: SPECIFY)
               s1s2 (spin-sph=1) : spin components object 1, spherical coords (FIXME: SPECIFY)
               s1s3 (spin-sph=1) : spin components object 1, spherical coords (FIXME: SPECIFY)
               s2s1 (spin-sph=1) : spin components object 2, spherical coords (FIXME: SPECIFY)
               s2s2 (spin-sph=1) : spin components object 2, spherical coords (FIXME: SPECIFY)
               s2s3 (spin-sph=1) : spin components object 2, spherical coords (FIXME: SPECIFY)
               s1x  (spin-sph=0) : spin components object 1, cartesian coords
               s1y  (spin-sph=0) : spin components object 1, cartesian coords
               s1z  (spin-sph=0) : spin components object 1, cartesian coords
               s2x  (spin-sph=0) : spin components object 2, cartesian coords
               s2y  (spin-sph=0) : spin components object 2, cartesian coords
               s2z  (spin-sph=0) : spin components object 2, cartesian coords
               lambda1           : tidal polarizability parameter object 1
               lambda2           : tidal polarizability parameter object 2
               TEOBResumS_a6c    : EOB dynamics parameter, pseudo 5PN in A(r)
               TEOBResumS_cN3LO  : EOB dynamics parameter, N3LO in Gs and Gss
               ecc               : eccentricity
               iota              : inclination
               phiref            : reference phase
               
               nrpmw-tcoll       : time of collapse     for the NRPMw model [Msun]
               nrpmw-df2         : frequency derivative for the NRPMw model [Msun^2]
               nrpmw-phi         : phase                for the NRPMw model
               
               distance          : distance [Mpc] (dummy value fixed at 10Mpc, unused in ROQ construction)
               
      Waveform wrappers must work with these keywords.
      Parameter ranges can be set using: par-X=value, where X can be ['min', 'max'] and par is any of the above names.

    """

def check_skip_parameter(key, input_par, nrpmw_recalib_names):

    if((key=='m1')                  and    (input_par['Waveform_and_parametrisation']['mc-q-par'] or input_par['Waveform_and_parametrisation']['m-q-par'])): return 1
    if((key=='m2')                  and    (input_par['Waveform_and_parametrisation']['mc-q-par'] or input_par['Waveform_and_parametrisation']['m-q-par'])): return 1
    if((key=='m')                   and not(input_par['Waveform_and_parametrisation']['m-q-par'])                                                         ): return 1
    if((key=='mc')                  and not(input_par['Waveform_and_parametrisation']['mc-q-par'])                                                        ): return 1
    if((key=='q')                   and not(input_par['Waveform_and_parametrisation']['mc-q-par'] or input_par['Waveform_and_parametrisation']['m-q-par'])): return 1
    if((key=='s1s1')                and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s1s2')                and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s1s3')                and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s2s1')                and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s2s2')                and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s2s3')                and not(input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s1x')                 and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s1y')                 and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s1z')                 and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s2x')                 and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s2y')                 and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='s2z')                 and    (input_par['Waveform_and_parametrisation']['spin-sph'])           ): return 1
    if((key=='ecc')                 and not(input_par['Waveform_and_parametrisation']['eccentricity'])       ): return 1
    if(('lambda' in key)            and not(input_par['Waveform_and_parametrisation']['tides'])              ): return 1
    if((key=='s1x')                 and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): return 1
    if((key=='s2x')                 and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): return 1
    if((key=='s1y')                 and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): return 1
    if((key=='s2y')                 and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): return 1
    if((key=='s1z')                 and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): return 1
    if((key=='s2z')                 and    (input_par['Waveform_and_parametrisation']['spins']=='no-spins'  )): return 1
    if((key=='TEOBResumS_a6c')      and not(input_par['Waveform_and_parametrisation']['dynamics'])           ): return 1
    if((key=='TEOBResumS_cN3LO')    and not(input_par['Waveform_and_parametrisation']['dynamics'])           ): return 1
    if((key=='nrpmw-tcoll')         and not(input_par['Waveform_and_parametrisation']['post-merger'])        ): return 1
    if((key=='nrpmw-df2')           and not(input_par['Waveform_and_parametrisation']['post-merger'])        ): return 1
    if((key=='nrpmw-phi')           and not(input_par['Waveform_and_parametrisation']['post-merger'])        ): return 1

    # Set NRPMW recalibration parameters.
    for ni in nrpmw_recalib_names:
        if((key=='nrpmw-{}'.format(ni)) and( not(input_par['Waveform_and_parametrisation']['post-merger']) or not('nrpmw-recal' in input_par['Waveform_and_parametrisation']['approximant']))): return 1

    return 0

# This is the training range of the 'mlgw-bns' approximant for the inspiral parameters, and of the 'NRPMw' approximant for the post-merger parameters.
default_params = {
    'mc'                : {'range' : [  0.9,     1.4], 'test-value': 1.3    },
    'm'                 : {'range' : [  2.0,     4.0], 'test-value': 2.8    },
    'q'                 : {'range' : [  1.0,     3.0], 'test-value': 2.0    },
    'm1'                : {'range' : [  1.0,     3.0], 'test-value': 1.5    },
    'm2'                : {'range' : [  0.5,     2.0], 'test-value': 1.5    },
    's1x'               : {'range' : [  0.0,     0.0], 'test-value': 0.0    },
    's1y'               : {'range' : [  0.0,     0.0], 'test-value': 0.0    },
    's1z'               : {'range' : [ -0.5,     0.5], 'test-value': 0.2    },
    's2x'               : {'range' : [  0.0,     0.0], 'test-value': 0.0    },
    's2y'               : {'range' : [  0.0,     0.0], 'test-value': 0.0    },
    's2z'               : {'range' : [ -0.5,     0.5], 'test-value': 0.1    },
    's1s1'              : {'range' : [  0.0,     0.5], 'test-value': 0.3    },
    's1s2'              : {'range' : [  0.0,   np.pi], 'test-value': 0.4    },
    's1s3'              : {'range' : [  0.0, 2*np.pi], 'test-value': 0.5    },
    's2s1'              : {'range' : [  0.0,     0.5], 'test-value': 0.3    },
    's2s2'              : {'range' : [  0.0,   np.pi], 'test-value': 0.4    },
    's2s3'              : {'range' : [  0.0, 2*np.pi], 'test-value': 0.5    },
    'lambda1'           : {'range' : [  5.0,  5000.0], 'test-value': 1000.0 },
    'lambda2'           : {'range' : [  5.0,  5000.0], 'test-value': 1000.0 },
    'TEOBResumS_a6c'    : {'range' : [-100.,   -20.0], 'test-value': -40.0  },
    'TEOBResumS_cN3LO'  : {'range' : [-100.,   -20.0], 'test-value': -40.0  },
    'ecc'               : {'range' : [  0.0,     0.0], 'test-value': 0.0    },
    'iota'              : {'range' : [  0.0,   np.pi], 'test-value': 1.9    },
    'phiref'            : {'range' : [  0.0, 2*np.pi], 'test-value': 0.6    },
    'nrpmw-tcoll'       : {'range' : [  0.0,    3000], 'test-value': 1000   },
    'nrpmw-df2'         : {'range' : [-1e-5,    1e-5], 'test-value': 0.0    },
    'nrpmw-phi'         : {'range' : [  0.0, 2*np.pi], 'test-value': 0.0    },
}

# Include recalibration parameters for NRPMw.
# This initialisation differs from the other parameters to avoid writing explicitly the recalibration parameters.
try:
    
    from bajes.obs.gw.approx.nrpmw import __recalib_names__        as nrpmw_recalib_names
    from bajes.obs.gw.approx.nrpmw import __BNDS__                 as nrpmw_recalib_bounds

    for ni in nrpmw_recalib_names:
        default_params['nrpmw-{}'.format(ni)]               = {}
        default_params['nrpmw-{}'.format(ni)]['range']      = nrpmw_recalib_bounds[ni]
        default_params['nrpmw-{}'.format(ni)]['test-value'] = 0.

except ImportError:
    nrpmw_recalib_names = []
    print('\nWarning: bajes module not found. Cannot initialise NRPMw recalibration parameters.\n')


def read_config(config_file, directory, logger):

    Config = configparser.ConfigParser()
    Config.read(config_file)

    # Store configuration file and git info to allow for run reproducibility.
    os.system('cp {} {}/.'.format(config_file, directory))
    store_git_info(directory)

    logger.info('')
    logger.info('Reading config file: {}'.format(config_file)+'.')
    logger.info('With sections: '+str(Config.sections())+'.')
    logger.info('')
    logger.info('####################')
    logger.info('# \u001b[\u001b[38;5;39mInput parameters\u001b[0m #')
    logger.info('####################')
    logger.info('')
    logger.info('I\'ll be running with the following values:')

    # ==========================================================#
    # Initialize and read from config the ROQ input parameters. #
    # ==========================================================#

    sections  = ['I/O', 'Parallel', 'Waveform_and_parametrisation', 'ROQ']
    input_par = {}

    input_par['I/O']                           = {
                                                 'output'                  : './',
                                                 'verbose'                 : 1,
                                                 'debug'                   : 0,
                                                 'timing'                  : 0,
                                                 'show-plots'              : 0,
                                                 'post-processing-only'    : 0,
                                                 'random-seed'             : 170817,
                                                }
    input_par['Parallel']                     = {
                                                 'parallel'                : 0,
                                                 'n-processes'             : 4,
                                                }

    input_par['Waveform_and_parametrisation'] = {
                                                 'approximant'             : 'teobresums-giotto',
                                                 'spins'                   :'aligned',
                                                 'tides'                   : 0,
                                                 'eccentricity'            : 0,
                                                 'post-merger'             : 0,
                                                 'dynamics'                : 0,
                                                 'mc-q-par'                : 1,
                                                 'm-q-par'                 : 0,
                                                 'spin-sph'                : 0,
                                                 'f-min'                   : 20.0,
                                                 'f-max'                   : 2048.0,
                                                 'seglen'                  : 128.0,
                                                }
    input_par['ROQ']                          = {
        
                                                 'gram-schmidt'            : 0,
                                                 
                                                 'basis-lin'               : 1,
                                                 'basis-qua'               : 1,

                                                 'pre-basis'               : 'corners',
                                                 'tolerance-pre-basis-lin' : 1e-8,
                                                 'tolerance-pre-basis-qua' : 1e-10,
                                                 'n-pre-basis-lin'         : 80,
                                                 'n-pre-basis-qua'         : 5,
                                                 'n-pre-basis-search-iter' : 80,
                                                 
                                                 'n-training-set-cycles'   : 4,
                                                 'training-set-sizes'      : '10000,100000,1000000,10000000',
                                                 'training-set-n-outliers' : '20,20,1,0',
                                                 'training-set-rel-tol'    : '0.1,0.1,0.05,0.3,1.0',

                                                 'tolerance-lin'           : 1e-8,
                                                 'tolerance-qua'           : 1e-10,
                              
                                                 'n-tests-post'            : 1000,
                                                 'minimum-speedup'         : 1.0,
                              
                                                }

    max_len_keyword = len('n-pre-basis-search-iter')
    for section in sections:
        logger.info('')
        logger.info('[\u001b[\u001b[38;5;39m{}\u001b[0m]'.format(section))
        logger.info('')
        for key in input_par[section]:
            try:
                keytype = type(input_par[section][key])
                input_par[section][key]=keytype(Config.get(section,key))
                leg = ''
            except (KeyError, configparser.NoOptionError, TypeError):
                leg = '(default)'
            # Format lists.
            if((key=='training-set-sizes') or (key=='training-set-n-outliers') or (key=='training-set-rel-tol')):
                input_par[section][key] = input_par[section][key].split(',')
                if(key=='training-set-rel-tol'):
                    for x in range(len(input_par[section][key])): input_par[section][key][x] = float(input_par[section][key][x])
                else:
                    for x in range(len(input_par[section][key])): input_par[section][key][x] = int(input_par[section][key][x])
            logger.info('{name} : {value} {leg}'.format(name=key.ljust(max_len_keyword), value=input_par[section][key], leg=leg))

    # Sanity checks
    list_keys = ['training-set-sizes', 'training-set-n-outliers', 'training-set-rel-tol']
    for list_key in list_keys:
        if not(len(input_par['ROQ'][list_key])==input_par['ROQ']['n-training-set-cycles']):
            raise ValueError('Length of {} list has to be equal to the number of training cycles (`n-training-set-cycles`).'.format(key))

    if(np.any(np.array(input_par['ROQ']['training-set-n-outliers']) < 0)): raise ValueError('The `training-set-n-outliers` variable cannot be negative.')
    if(input_par['ROQ']['minimum-speedup'] < 1.0): raise ValueError('The speedup factor has to be larger than unity, otherwise the ROQ construction will not accelerate parameter estimation.')
    if not((input_par['ROQ']['n-pre-basis-lin']>1) and (input_par['ROQ']['n-pre-basis-qua']>1)): raise ValueError('The minimum number of basis elements has to be larger than 1, since currently the initial basis is composed by the lower/upper corner of the parameter space (hence two waveforms).')
    if(input_par['Waveform_and_parametrisation']['spin-sph'] and not(input_par['Waveform_and_parametrisation']['spins']=='precessing')): raise ValueError('Spherical spin coordinates are currently supported only for precessing waveforms.')

    if not(input_par['Waveform_and_parametrisation']['spins'] in ['no-spins', 'aligned', 'precessing']): raise ValueError('Invalid spin option requested.')

    if(input_par['Waveform_and_parametrisation']['mc-q-par'] and input_par['Waveform_and_parametrisation']['m-q-par']): raise ValueError('Simultaneous parametrisations in total mass and chirp mass are incompatible.')

    if((input_par['Parallel']['parallel']) and (input_par['Parallel']['n-processes']<2)): raise ValueError('When parallelisation is active, at least two processes have to be requested.')

    # Set run types.
    input_par['I/O']['run-types'] = []
    if(input_par['ROQ']['basis-lin']): input_par['I/O']['run-types'].append('linear')
    if(input_par['ROQ']['basis-qua']): input_par['I/O']['run-types'].append('quadratic')

    if((input_par['I/O']['debug']) and not os.path.exists(os.path.join(directory, 'Debug'))):
        os.makedirs(os.path.join(directory, 'Debug'))

    # ====================================#
    # Read training range and test point. #
    # ====================================#

    params_ranges, test_values = {}, {}

    logger.info('')
    logger.info('[\u001b[\u001b[38;5;39mTraining_range\u001b[0m]')
    logger.info('')

    for key in default_params:
        
        if(check_skip_parameter(key, input_par, nrpmw_recalib_names)): continue

        keytype = type(default_params[key]['range'][0])
        try:
            params_ranges[key] = [keytype(Config.get('Training_range',key+'-min')), keytype(Config.get('Training_range',key+'-max'))]
            logger.info('{name} : [{min},{max}]'.format(          name=key.ljust(max_len_keyword), min=params_ranges[key][0], max=params_ranges[key][1]))
        except (KeyError, configparser.NoOptionError, TypeError):
            params_ranges[key] = default_params[key]['range']
            logger.info('{name} : [{min},{max}] (default)'.format(name=key.ljust(max_len_keyword), min=params_ranges[key][0], max=params_ranges[key][1]))

        try:                                                      test_values[key] = keytype(Config.get('Test_values',key))
        except (KeyError, configparser.NoOptionError, TypeError): test_values[key] = default_params[key]['test-value']

        if    (params_ranges[key][1] < params_ranges[key][0]):                      raise ValueError('{} upper bound is smaller than its lower bound.'.format(key))
        if not(params_ranges[key][0] <= test_values[key] <= params_ranges[key][1]): logger.info('WARNING: Chosen test value for {} outside training range.'.format(key))

    logger.info('')

    return input_par, params_ranges, test_values
