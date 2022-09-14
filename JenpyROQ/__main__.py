## -*- coding: utf8 -*-
#!/usr/bin/env python

# General python imports
import multiprocessing as mp, numpy as np, os, sys, random, time, warnings
from optparse import OptionParser

try               : import configparser
except ImportError: import ConfigParser as configparser

# Package internal imports
from . import initialise, post_processing

# Initialize logger
import logging

# Inizialize error handlers
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Logger setter
def set_logger(label=None, outdir=None, level='INFO', verbose=True):
    
    # Set formatters
    datefmt = '%m-%d-%Y %H:%M'
    fmt     = '[{}] [%(asctime)s] %(message)s'.format(label)

    # Initialize logger
    logger = logging.getLogger(label)
    logger.propagate = False
    logger.setLevel(('{}'.format(level)).upper())

    # Set stream-handler (i.e. console)
    if verbose:
        if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            stream_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(stream_handler)
    
    # Set file-handler (i.e. file)
    if outdir != None:
        if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            file_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':

    # Initialise and read config.
    parser      = OptionParser(initialise.usage)
    parser.add_option('--config-file',  type   = 'string',      metavar = 'config_file',    default = None)
    (opts,args) = parser.parse_args()
    config_file = opts.config_file
    
    if not config_file:
        parser.print_help()
        parser.error('Please specify a config file.')
    if not os.path.exists(config_file):
        parser.error('Config file {} not found.'.format(config_file))
    
    # FIXME: Manual and ugly. Assumes default values.
    Config = configparser.ConfigParser()
    Config.read(config_file)
    try:                                debug_tmp     = int(Config.get('I/O','debug'))
    except(configparser.NoOptionError): debug_tmp     = 0
    try:                                directory_tmp = str(Config.get('I/O','output'))
    except(configparser.NoOptionError): directory_tmp = './'
    try:                                verbose_tmp   = int(Config.get('I/O','verbose'))
    except(configparser.NoOptionError): verbose_tmp   = 1
    
    # Create dir structure.

    dirs_list = [directory_tmp,
                 os.path.join(directory_tmp, 'Plots'),
                 os.path.join(directory_tmp, 'Plots/Basis_parameters'),
                 os.path.join(directory_tmp, 'Plots/Waveform_comparisons'),
                 os.path.join(directory_tmp, 'ROQ_data'),
                 os.path.join(directory_tmp, 'ROQ_data/linear'),
                 os.path.join(directory_tmp, 'ROQ_data/quadratic')]
    
    for dir_to_create in dirs_list:
        if not os.path.exists(dir_to_create): os.makedirs(dir_to_create)

    # set logger(s)
    if debug_tmp:
        logger = set_logger(label='JenpyROQ',
                            level='DEBUG',
                            outdir=directory_tmp,
                            verbose=bool(verbose_tmp),)
    else:
        logger = set_logger(label='JenpyROQ',
                            outdir=directory_tmp,
                            verbose=bool(verbose_tmp),)
    
    config_pars, params_ranges, test_values = initialise.read_config(config_file, directory_tmp, logger)

    logger.info('')
    # Get parallel processing pool
    if (int(config_pars['Parallel']['parallel'])==0):
        logger.info('Initialising serial pool.')
        from .parallel import initialize_serial_pool
        Pool = initialize_serial_pool()
    elif (int(config_pars['Parallel']['parallel'])==1):
        logger.info('Initialising multiprocessing processsing pool.')
        from .parallel import initialize_mp_pool, close_pool_mp
        Pool = initialize_mp_pool(int(config_pars['Parallel']['n-processes']))
        close_pool = close_pool_mp
    elif (int(config_pars['Parallel']['parallel'])==2):
        logger.info('Initialising MPI-based processing pool.')
        from .parallel import initialize_mpi_pool, close_pool_mpi
        Pool = initialize_mpi_pool()
        close_pool = close_pool_mpi
    else:
        raise ValueError("Unable to initialise parallelisation method. Use parallel=0 for serial, parallel=1 for multiprocessing or parallel=2 for MPI.")
    logger.info('')

    # Set random seed
    if (int(config_pars['Parallel']['parallel'])<2):
        logger.info('Setting random seed to {}'.format(config_pars['I/O']['random-seed']))
        np.random.seed(int(config_pars['I/O']['random-seed']))
    else:
        # Avoid generation of identical random numbers in different processes
        if Pool.is_master():
            np.random.seed(int(config_pars['I/O']['random-seed']))
            logger.info('Setting random seed to {}'.format(config_pars['I/O']['random-seed']))
        else:
            np.random.seed(int(config_pars['I/O']['random-seed'])+Pool.rank)
            logger.info('Setting random seed to {}'.format(int(config_pars['I/O']['random-seed'])+Pool.rank))
    logger.info('')

    # Open pool
    with Pool as pool:
    
        # If MPI, set workers in wait for commands from master
        if (int(config_pars['Parallel']['parallel'])==2):
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        # Initialise ROQ parameters and structures.
        from . import JenpyROQ as roq
        jenpyroq = roq.jenpyroq(config_pars, params_ranges, pool=pool)
        freq  = jenpyroq.freq
        np.save(jenpyroq.outputdir+'/ROQ_data/full_frequencies.npy', freq)

        data = {}

        for run_type in config_pars['I/O']['run-types']:

            term = run_type[0:3]
            if not(config_pars['I/O']['post-processing-only']):
                # Create the basis and save ROQ.
                data[run_type] = jenpyroq.run(term)
            
                # These data are not saved in output, so plot them now.
                post_processing.plot_preselection_residual_modula(data[run_type]['{}_pre_res_mod'.format(term)], term, jenpyroq.outputdir)
                post_processing.plot_maximum_empirical_interpolation_error(data[run_type]['{}_max_eies'.format(term)], term, jenpyroq.outputdir)
                post_processing.plot_number_of_outliers(data[run_type]['{}_n_outliers'.format(term)], term, jenpyroq.outputdir)

            else:
                # Read ROQ from previous run.
                data[run_type]                                = {}
                data[run_type]['{}_f'.format(term)]           = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/empirical_frequencies_{type}.npy'.format(type=run_type)))
                # For the moment, preserve backwards compatibility with initial runs that did not store empirical nodes.
                try:    data[run_type]['{}_emp_nodes'.format(term)] = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/empirical_nodes_{type}.npy'.format(type=run_type)))
                except: data[run_type]['{}_emp_nodes'.format(term)] = np.searchsorted(freq, data[run_type]['{}_f'.format(term)])
                data[run_type]['{}_interpolant'.format(term)] = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/basis_interpolant_{type}.npy'.format(type=run_type)))
                data[run_type]['{}_params'.format(term)]      = np.load(os.path.join(config_pars['I/O']['output'],'ROQ_data/{type}/basis_waveform_params_{type}.npy'.format(type=run_type)))

            # Store ROQ metadata
            outFile_ROQ_metadata = open(os.path.join(config_pars['I/O']['output'],'ROQ_data/ROQ_metadata.txt'), 'w')
            outFile_ROQ_metadata.write('f-min \t f-max \t seglen\n')
            outFile_ROQ_metadata.write('{} \t {} \t {} \n'.format(jenpyroq.f_min, jenpyroq.f_max, jenpyroq.seglen))
            outFile_ROQ_metadata.close()

            # Output the basis reduction factor.
            logger.info('')
            logger.info('')
            logger.info('#########################')
            logger.info('# \u001b[\u001b[38;5;39mResults {} iteration\u001b[0m #'.format(term))
            logger.info('#########################')
            logger.info('')
            logger.info('{} basis reduction factor: (Original freqs [{}]) / (New freqs [{}]) = {}'.format(run_type, len(freq), len(data[run_type]['{}_f'.format(term)]), len(freq)/len(data[run_type]['{}_f'.format(term)])))
            logger.info('')

            # Plot the basis parameters corresponding to the selected basis (only the first N elements determined during the interpolant construction procedure).
            post_processing.histogram_basis_params(data[run_type]['{}_params'.format(term)][:len(data[run_type]['{}_f'.format(term)])], jenpyroq.outputdir, jenpyroq.i2n, term)
            post_processing.histogram_frequencies(data[run_type]['{}_f'.format(term)], jenpyroq.outputdir, term)

            # Plot the representation error for a random waveform, using the interpolant built from the constructed basis. Useful for visual diagnostics.
            logger.info('Testing the waveform using the parameters:')
            logger.info('')
            parampoint_test = []
            logger.info('name    | value | index')
            for name, val in test_values.items():
                logger.info('{} | {}   | {} '.format(name.ljust(len('nrpmw-tcoll')), val, jenpyroq.n2i[name]))
                parampoint_test.append(val)
            parampoint_test = np.array(parampoint_test)

            post_processing.plot_representation_error(data[run_type]['{}_interpolant'.format(term)], data[run_type]['{}_emp_nodes'.format(term)], parampoint_test, term, jenpyroq.outputdir, freq, jenpyroq.paramspoint_to_wave)

            # Validation tests.
            post_processing.test_roq_error(data[run_type]['{}_interpolant'.format(term)], data[run_type]['{}_emp_nodes'.format(term)], term, jenpyroq, pool)

    # Show plots, if requested.
    if(config_pars['I/O']['show-plots']): plt.show()
