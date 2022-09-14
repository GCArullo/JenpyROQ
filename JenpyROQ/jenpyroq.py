## -*- coding: utf8 -*-
#!/usr/bin/env python

# General python imports
import numpy as np, os, sys, random, time, warnings
from itertools import repeat

# Package internal imports
from .waveform_wrappers import *
from .parallel import eval_func_tuple
from . import initialise, linear_algebra, post_processing

# Initialize logger
import logging
logger = logging.getLogger(__name__)

# Inizialize error handlers
TermError    = ValueError('Unknown basis term requested.')
VersionError = ValueError('Unknown version requested.')
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# ROQ main
class JenpyROQ:
    """
        JenpyROQ Class
    
        * Works with a list of waveform wrappers provided in `waveform_wrappers.py`.
    """

    def __init__(self,
                 config_pars                      ,
                 params_ranges                    ,
                 distance                   = 10  , # [Mpc]. Dummy value, distance does not enter the interpolants construction
                 additional_waveform_params = {}  , # Dictionary with any parameter needed for the waveform approximant
                 pool                       = None, # Parallel processing pool
                 ):

        self.distance                   = distance
        self.additional_waveform_params = additional_waveform_params
        self.params_ranges              = params_ranges
        
        # Read input params
        self.approximant                = config_pars['Waveform_and_parametrisation']['approximant']

        self.mc_q_par                   = config_pars['Waveform_and_parametrisation']['mc-q-par']
        self.m_q_par                    = config_pars['Waveform_and_parametrisation']['m-q-par']
        self.spin_sph                   = config_pars['Waveform_and_parametrisation']['spin-sph']

        self.f_min                      = config_pars['Waveform_and_parametrisation']['f-min']
        self.f_max                      = config_pars['Waveform_and_parametrisation']['f-max']
        self.seglen                     = config_pars['Waveform_and_parametrisation']['seglen']
        self.deltaF                     = 1./self.seglen
        
        self.gram_schmidt               = config_pars['ROQ']['gram-schmidt']
        
        self.start_values               = config_pars['ROQ']['pre-basis']
        self.tolerance_pre_basis_lin    = config_pars['ROQ']['tolerance-pre-basis-lin']
        self.tolerance_pre_basis_qua    = config_pars['ROQ']['tolerance-pre-basis-qua']
        self.n_pre_basis_lin            = config_pars['ROQ']['n-pre-basis-lin']
        self.n_pre_basis_qua            = config_pars['ROQ']['n-pre-basis-qua']
        self.n_pre_basis_search_iter    = config_pars['ROQ']['n-pre-basis-search-iter']

        self.n_training_set_cycles      = config_pars['ROQ']['n-training-set-cycles']
        self.training_set_sizes         = config_pars['ROQ']['training-set-sizes']
        self.training_set_n_outliers    = config_pars['ROQ']['training-set-n-outliers']
        self.training_set_rel_tol       = config_pars['ROQ']['training-set-rel-tol']
        self.tolerance_lin              = config_pars['ROQ']['tolerance-lin']
        self.tolerance_qua              = config_pars['ROQ']['tolerance-qua']

        self.n_tests_post               = config_pars['ROQ']['n-tests-post']
        self.minumum_speedup            = config_pars['ROQ']['minimum-speedup']

        self.parallel                   = config_pars['Parallel']['parallel']
        self.n_processes                = config_pars['Parallel']['n-processes']
        
        self.outputdir                  = config_pars['I/O']['output']
        self.timing                     = config_pars['I/O']['timing']
        self.debug                      = config_pars['I/O']['debug']

        # Set global pool object
        global Pool
        Pool = pool

        # Convert to LAL identification number, if passing a LAL approximant, and choose waveform
        from .waveform_wrappers import __non_lal_approx_names__
        if(not(config_pars['Waveform_and_parametrisation']['approximant'] in __non_lal_approx_names__)):
            self.approximant = lalsimulation.SimInspiralGetApproximantFromString(self.approximant)
        
        if self.approximant in WfWrapper.keys(): self.wvf = WfWrapper[self.approximant](self.approximant, self.additional_waveform_params)
        else:                                    raise ValueError('Unknown approximant requested.')

        # Build the map between params names and indexes
        self.map_params_indexs()  # Declares: self.i2n, self.n2i, self.nparams
        
        # Initial basis
        self.freq = np.arange(self.f_min, self.f_max+self.deltaF, self.deltaF)
        self.set_training_range() # Declares: self.params_low, self.params_hig

        logger.info('Initial number of frequency points: {}'.format(len(self.freq)))

    ## Parameters transformations utils

    def spherical_to_cartesian(self, sph):
        
        x = sph[0]*np.sin(sph[1])*np.cos(sph[2])
        y = sph[0]*np.sin(sph[1])*np.sin(sph[2])
        z = sph[0]*np.cos(sph[1])
        
        return [x,y,z]

    def get_m1m2_from_mcq(self, mc,q):
        
        m2 = mc * q ** (-0.6) * (1+q)**0.2
        m1 = m2 * q
        
        return np.array([m1,m2])

    def get_m1m2_from_mq(self, m,q):
        
        m1 = m*q/(1+q)
        m2 = m/(1+q)
        
        return np.array([m1,m2])

    def mass_range(self, mc_low, mc_high, q_low, q_high):
        
        mmin = self.get_m1m2_from_mcq(mc_low,q_high)[1]
        mmax = self.get_m1m2_from_mcq(mc_high,q_high)[0]
        
        return [mmin, mmax]

    ## Parameters handling functions

    def map_params_indexs(self):
        
        """
            Build a map between the parameters names and the indexes of the parameter arrays, and its inverse.
        """
        names = self.params_ranges.keys()
        self.nparams = len(names)
        self.n2i = dict(zip(names,range(self.nparams)))
        self.i2n = {i: n for n, i in self.n2i.items()}
        
        return

    def update_waveform_params(self, paramspoint):
        
        """
            Update the waveform parameters (dictionary) with those in paramspoint (np.array).
        """
        p = {}
        for i,k in self.i2n.items():
            p[k] = paramspoint[i]

        return p
         
    def generate_params_points(self, npts, round_to_digits=6):
        
        """
            Uniformly sample the parameter arrays
        """
        paramspoints = np.random.uniform(self.params_low,
                                         self.params_hig,
                                         size=(npts, self.nparams))
                                         
        return paramspoints.round(decimals=round_to_digits)
    
    def paramspoint_to_wave(self, paramspoint, term):
        
        """
            Generate a waveform given a paramspoint
            By default, if paramspoint contains the spherical spin, then updates the cartesian accordingly.
        """
        p = self.update_waveform_params(paramspoint)

        if   self.mc_q_par: p['m1'],p['m2'] = self.get_m1m2_from_mcq(p['mc'],p['q'])
        elif self.m_q_par : p['m1'],p['m2'] = self.get_m1m2_from_mq( p['m'] ,p['q'])

        if self.spin_sph:
            s1sphere_tmp               = p['s1s1'],p['s1s2'],p['s1s3']
            p['s1x'],p['s1y'],p['s1z'] = self.spherical_to_cartesian(s1sphere_tmp)
            s2sphere_tmp               = p['s2s1'],p['s2s2'],p['s2s3']
            p['s2x'],p['s2y'],p['s2z'] = self.spherical_to_cartesian(s2sphere_tmp)
    
        # We build a linear basis only for hp, since empirically the same basis accurately works to represent hc too (see [arXiv:1604.08253]).
        hp, hc = self.wvf.generate_waveform(p, self.deltaF, self.f_min, self.f_max, self.distance)
        
        if   term == 'lin': pass
        elif term == 'qua': hp, hc = (np.absolute(hp))**2, (np.absolute(hc))**2
        else              : raise TermError
        
        return hp, hc

    ## Basis construction functions

    def add_new_element_to_basis(self, new_basis_param_point, known_basis, known_params, term):
        
        # Create new basis element.
        hp_new, _ = self.paramspoint_to_wave(new_basis_param_point, term)

        # Orthogonalise, i.e. extract the linearly independent part of the waveform, and normalise the new element, which constitutes a new basis element. Note: when gram-schidtting, the new basis element is not a 'waveform', since subtraction of two waveforms does not generate a waveform.
        if(self.gram_schmidt): basis_new = linear_algebra.gram_schmidt(known_basis, hp_new, self.deltaF)
        else:                  basis_new = linear_algebra.normalise_vector(hp_new, self.deltaF)
        
        # Append to basis.
        known_basis  = np.append(known_basis,  np.array([basis_new]),             axis=0)
        known_params = np.append(known_params, np.array([new_basis_param_point]), axis=0)

        return known_basis, known_params

    def compute_new_element_residual_modulus_from_basis(self, paramspoint, known_basis, term):

        # Create and normalise element to be projected and initialise residual
        h_to_proj, _      = self.paramspoint_to_wave(paramspoint, term)
        h_to_proj         = linear_algebra.normalise_vector(h_to_proj, self.deltaF)
        
        # Compute the normalised projection of the element onto the basis.
        basis_combination = np.zeros(len(h_to_proj),dtype=complex)
        for k in np.arange(0,len(known_basis)):
            basis_combination += linear_algebra.projection(known_basis[k],h_to_proj)
        basis_combination = linear_algebra.normalise_vector(basis_combination, self.deltaF)

        # Evaluate the residual.
        residual = h_to_proj - basis_combination

        return linear_algebra.scalar_product(residual, residual, self.deltaF)
        
    def search_new_basis_element(self, paramspoints, known_basis, term):

        """
           Given an array of new random points in the parameter space (paramspoints) and the known basis elements, this function searches and constructs a new basis element. The new element is constructed by:
           
           1) Projecting the waveforms corresponding to parampoints on the known basis;
           2) Selecting the waveform with the largest residual (modulus) after projection;
           3) Computing the normalised residual projection of the selected waveform on the known basis.
        """

        # Generate len(paramspoints) random waveforms corresponding to parampoints.
        modula           = list(Pool.map(eval_func_tuple, zip(repeat(self.compute_new_element_residual_modulus_from_basis),
                                                              paramspoints,
                                                              repeat(known_basis),
                                                              repeat(term))))

        # Select the worst represented waveform (in terms of the previous known basis).
        arg_newbasis = np.argmax(modula) 
       
        return paramspoints[arg_newbasis], modula[arg_newbasis]
            
    def construct_preselection_basis(self, known_basis, params, residual_modula, term):
        
        if term == 'lin':
            file_basis    = self.outputdir+'/ROQ_data/linear/preselection_linear_basis.npy'
            file_params   = self.outputdir+'/ROQ_data/linear/preselection_linear_basis_waveform_params.npy'
            file_res_mod  = self.outputdir+'/ROQ_data/linear/preselection_linear_basis_residual_modula.npy'
            tolerance_pre = self.tolerance_pre_basis_lin
            pre_basis_n   = self.n_pre_basis_lin
        elif term=='qua':
            file_basis    = self.outputdir+'/ROQ_data/quadratic/preselection_quadratic_basis.npy'
            file_params   = self.outputdir+'/ROQ_data/quadratic/preselection_quadratic_basis_waveform_params.npy'
            file_res_mod  = self.outputdir+'/ROQ_data/quadratic/preselection_quadratic_basis_residual_modula.npy'
            tolerance_pre = self.tolerance_pre_basis_qua
            pre_basis_n   = self.n_pre_basis_qua
        else:
            raise TermError
    
        # This block generates a basis of dimension n_pre_basis.
        logger.info('')
        logger.info('#############################')
        logger.info('# \u001b[38;5;\u001b[38;5;39mStarting {} preselection\u001b[0m #'.format(term))
        logger.info('#############################')
        logger.info('')
        logger.info('N points per iter  : {}'.format(self.n_pre_basis_search_iter))
        logger.info('Tolerance          : {}'.format(tolerance_pre))
        logger.info('Maximum iterations : {}'.format(pre_basis_n-2)) # The -2 comes from the fact that the corner basis is composed by two elements.
        logger.info('')

        if not(pre_basis_n==2):
            k = 0
            while(residual_modula[-1] > tolerance_pre):
                
                # Generate n_pre_basis_search_iter random points.
                paramspoints = self.generate_params_points(self.n_pre_basis_search_iter)
                
                if(self.timing): execution_time_new_pre_basis_element = time.time()
                # From the n_pre_basis_search_iter randomly generated points, select the worst represented waveform corresponding to that point (i.e. with the largest residuals after basis projection).
                params_new, rm_new = self.search_new_basis_element(paramspoints, known_basis, term)
                if(self.timing): logger.info('Timing: pre-selection basis {} iteration, generating {} waveforms with parallel={} [minutes]: {}'.format(k+1, self.n_pre_basis_search_iter, self.parallel, (time.time() - execution_time_new_pre_basis_element)/60.0))
                logger.info('Pre-selection iteration: {}'.format(k+1) + ' -- Largest projection error: {}'.format(rm_new))

                # The worst represented waveform becomes the new basis element.
                known_basis, params = self.add_new_element_to_basis(params_new, known_basis, params, term)
                residual_modula     = np.append(residual_modula, rm_new)

                # Store the pre-selected basis.
                np.save(file_basis,   known_basis    )
                np.save(file_params,  params         )
                np.save(file_res_mod, residual_modula)

                # If a maximum number of iterations was given, stop at that number, otherwise continue until tolerance is reached.
                if(len(known_basis[:,0]) >= pre_basis_n): break
                else                                    : k = k+1

        # Store the pre-selected basis.
        np.save(file_basis,   known_basis    )
        np.save(file_params,  params         )
        np.save(file_res_mod, residual_modula)

        return known_basis, params, residual_modula
    
    ## Initial basis functions
    
    def set_training_range(self):
        """
            Initialize parameter ranges and basis.
        """
        
        logger.info('######################')
        logger.info('# \u001b[\u001b[38;5;39mInitialising basis\u001b[0m #')
        logger.info('######################')
        logger.info('')
        logger.info('nparams = {}'.format(self.nparams))
        logger.info('')
        logger.info('index | name        | ( min - max )           ')

        self.params_low, self.params_hig = [], []
        # Set bounds
        for i,n in self.i2n.items():
            self.params_low.append(self.params_ranges[n][0])
            self.params_hig.append(self.params_ranges[n][1])
            
            logger.info('{}    | {} | ( {:.6f} - {:.6f} ) '.format(str(i).ljust(2), n.ljust(len('nrpmw-tcoll')), self.params_low[i], self.params_hig[i]))
        logger.info('')
        
        return 

    def construct_corner_basis(self, term):

        hp_low, _ = self.paramspoint_to_wave(self.params_low, term)
        
        # Initialise the base with the lowest corner.
        known_basis_start     = np.array([linear_algebra.normalise_vector(hp_low, self.deltaF)])
        params_ini            = np.array([self.params_low])
        residual_modula_start = np.array([1.0])

        # Add the highest corner.
        known_basis_start, params_ini = self.add_new_element_to_basis(self.params_hig, known_basis_start, params_ini, term)
        residual_modula_start         = np.append(residual_modula_start, np.array([1.0]))

        return known_basis_start, params_ini, residual_modula_start

    ## Interpolant building functions

    def compute_empirical_interpolation_error(self, training_point, basis_interpolant, emp_nodes, term):

        # Create and normalise benchmark waveform.
        hp, _ = self.paramspoint_to_wave(training_point, term)
        hp    = linear_algebra.normalise_vector(hp, self.deltaF)

        # Compute the empirical interpolation error.
        hp_interp = np.dot(basis_interpolant,hp[emp_nodes])
        dh        = hp - hp_interp

        return linear_algebra.scalar_product(dh, dh, self.deltaF)

    def search_worst_represented_point(self, outliers, basis_interpolant, emp_nodes, training_set_tol, term):
        
        execution_time_search_worst_point = time.time()
        # Loop over test points.
        eies = list(Pool.map(eval_func_tuple, zip(repeat(self.compute_empirical_interpolation_error),
                                                  outliers,
                                                  repeat(basis_interpolant),
                                                  repeat(emp_nodes),
                                                  repeat(term))))
                                                  
        if(self.timing):
            execution_time_search_worst_point = (time.time() - execution_time_search_worst_point)/60.0
            logger.info('Timing: worst point search, computing {} interpolation errors with parallel={} [minutes]: {}'.format(len(outliers), self.parallel, execution_time_search_worst_point))

        # Select the worst represented point.
        arg_worst                     = np.argmax(eies)
        worst_represented_param_point = outliers[arg_worst]
        
        # Update outliers.
        outliers = outliers[np.array(eies) > training_set_tol]

        return worst_represented_param_point, eies[arg_worst], outliers

    def interpolant_and_empirical_nodes(self, known_basis, emp_nodes):
        
        """
            Generate the empirical interpolation nodes from a given basis.
            Follows the algorithm detailed in Ref. Phys. Rev. X 4, 031006, Sec.III.B and Appendix B.
            See also arXiv:1210.0577 Appendix A.2 for a discretised version and arXiv:1712.08772v2 for a description.
        """
        
        basis_len = len(known_basis)
        
        # If it's the first time the empirical nodes are computed after the pre-selection loop, initialise and use the full basis available. Otherwise, only add the new point.
        if(len(emp_nodes)==0):
            # The first point is chosen to maximise the first basis vector.
            emp_nodes    = np.array([np.argmax(np.absolute(known_basis[0]))])
            start_idx    = 1
        else:
            start_idx    = basis_len-1

        # Iterate the above for all the other nodes.
        for k in np.arange(start_idx, basis_len):
            
            # Build the coefficients, C = V^{-1} * e_k.
            # This choice of coefficients ensures rescaling of the first basis element, such that the interpolant I_1 (I_1[i] = c1*known_basis[0,i]) computed at the first empirical interpolation node (I_1[emp_nodes[0]] = c1*known_basis[0, emp_nodes[0]]), is exactly equal to the second element computed at that point (known_basis[1,emp_nodes[0]]).
            Vtmp         = np.transpose(known_basis[0:k,emp_nodes])
            inverse_Vtmp = np.linalg.pinv(Vtmp)
            
            C = np.dot(inverse_Vtmp, known_basis[k,emp_nodes])

            # Build the interpolant.
            interpolant = np.dot(np.transpose(known_basis[0:k,:]), C)
            
            # Compute the new empirical interpolation node and make sure frequencies are ordered.
            # Ideally, the points are unique, but numerical error in matrix inversion can result in duplicates, especially for long segment lenghts.
            r = np.absolute(interpolant - known_basis[k])
            if(np.argmax(r) in emp_nodes): logger.info("WARNING: Duplicate empirical frequency point found. This won't prevent the construction of the basis, but is a sign of ill-conditioning of the basis matrix inversion. Probably implies sub-optimality in basis construction.")
            # Quick and dirty fix: manually set to zero residuals at previous nodes, to avoid selecting duplicate points.
            # FIXME: fix the matrix inversion.
            for previous_node in emp_nodes: r[previous_node] = 0.0
            new_emp_node = np.argmax(r)

            emp_nodes = np.append(emp_nodes, new_emp_node)
            emp_nodes = sorted(emp_nodes)

            if(self.debug):
                id          = np.dot(Vtmp, inverse_Vtmp)
                id_minus_id = id - np.identity(len(Vtmp[0]))
                cond_num    = np.linalg.cond(Vtmp)
                logger.info('DEBUGGING: Basis matrix maximum inversion error (should be O(1e-16) ): {}'.format(np.max(np.absolute(id_minus_id))))
                logger.info('DEBUGGING: Basis matrix conditioning number     (should be O(1)     ): {}'.format(cond_num))

        # There should be no repetitions, otherwise duplicates on the frequency axis will bias likelihood computation during parameter estimation. Check for them as a consistency check, since previous PyROQ implementations had them.
        if not(len(np.unique(emp_nodes))==len(emp_nodes)): raise ValueError("Repeated empirical interpolation node. The implementation of the algorithm is not correct?")

        V                 = np.transpose(known_basis[0:len(emp_nodes), emp_nodes])
        inverse_V         = np.linalg.pinv(V)
        basis_interpolant = np.dot(np.transpose(known_basis[0:len(emp_nodes)]),inverse_V)

        return basis_interpolant, emp_nodes
    
    def roqs(self, known_basis, known_params, term):

        # Initialise iteration and create paths in which to store the output.
        if term == 'lin':
            tol                  = self.tolerance_lin
            file_interpolant     = self.outputdir+'/ROQ_data/linear/basis_interpolant_linear.npy'
            file_empirical_freqs = self.outputdir+'/ROQ_data/linear/empirical_frequencies_linear.npy'
            file_empirical_nodes = self.outputdir+'/ROQ_data/linear/empirical_nodes_linear.npy'
            file_basis           = self.outputdir+'/ROQ_data/linear/basis_linear.npy'
            file_params          = self.outputdir+'/ROQ_data/linear/basis_waveform_params_linear.npy'
        elif term == 'qua':
            tol                  = self.tolerance_qua
            file_interpolant     = self.outputdir+'/ROQ_data/quadratic/basis_interpolant_quadratic.npy'
            file_empirical_freqs = self.outputdir+'/ROQ_data/quadratic/empirical_frequencies_quadratic.npy'
            file_empirical_nodes = self.outputdir+'/ROQ_data/quadratic/empirical_nodes_quadratic.npy'
            file_basis           = self.outputdir+'/ROQ_data/quadratic/basis_quadratic.npy'
            file_params          = self.outputdir+'/ROQ_data/quadratic/basis_waveform_params_quadratic.npy'
        else:
            raise TermError

        maximum_eies, n_outliers, emp_nodes = np.array([]), np.array([]), np.array([])
        # Start a loop over training cycles with varying training size, tolerance and number of allowed outliers.
        for n_cycle in range(self.n_training_set_cycles):
            
            training_set_size      = self.training_set_sizes[n_cycle]
            training_set_n_outlier = self.training_set_n_outliers[n_cycle]
            training_set_tol       = self.training_set_rel_tol[n_cycle] * tol
        
            logger.info('')
            logger.info('')
            logger.info('################################')
            logger.info('# \u001b[\u001b[38;5;39mStarting {}/{} enrichment loop\u001b[0m #'.format(n_cycle+1, self.n_training_set_cycles))
            logger.info('################################')
            logger.info('')
            logger.info('Training set size  : {}'.format(training_set_size))
            logger.info('Tolerance          : {}'.format(training_set_tol))
            logger.info('Tolerated outliers : {}'.format(training_set_n_outlier))
            logger.info('')

            # Generate the parameters of this training cycle.
            paramspoints = self.generate_params_points(npts=training_set_size)
            outliers     = paramspoints[:training_set_size]
            
            
            # From the basis constructed above, extract: the empirical interpolation nodes (i.e. the subset of frequencies on which the ROQ rule is evaluated); the basis interpolant, which allows to construct an arbitrary waveform at an arbitrary frequency point from the constructed basis.
            
            while(len(outliers) > training_set_n_outlier):

                # Store the current basis and parameters at each step, as a backup.
                np.save(file_basis,  known_basis)
                np.save(file_params, known_params)
                
                # If no outliers were found at the previous iteration, hence no new basis elements have been added, there is no new empirical point to be constructed.
                if not(len(emp_nodes)==len(known_basis)): basis_interpolant, emp_nodes = self.interpolant_and_empirical_nodes(known_basis, emp_nodes)

                # Out of the remaining outliers, select the worst represented point.
                worst_represented_param_point, maximum_eie, outliers = self.search_worst_represented_point(outliers, basis_interpolant, emp_nodes, training_set_tol, term)

                # Update the user on how many outliers remain.
                logger.info('{}'.format(len(emp_nodes))+' basis elements gave {} outliers with interpolation error > {} out of {} training points.'.format(len(outliers), training_set_tol, training_set_size))
                logger.info('Largest interpolation error: {}'.format(maximum_eie))

                # Enrich the basis with the worst outlier. Also store the maximum empirical interpolation error, to monitor the improvement in the interpolation.
                if(len(outliers) > training_set_n_outlier):
                    known_basis, known_params = self.add_new_element_to_basis(worst_represented_param_point, known_basis, known_params, term)
                    maximum_eies              = np.append(maximum_eies, maximum_eie)
                    n_outliers                = np.append(n_outliers  , len(outliers))

                # Check if basis construction became pointless.
                if((len(self.freq)/len(known_basis[:,0])) < self.minumum_speedup): raise Exception('Basis dimension is larger than the minimum speedup requested. Aborting the interpolants construction.')

        # Finalise and store the output.
        frequencies = self.freq[emp_nodes]
        np.save(file_interpolant,     basis_interpolant)
        np.save(file_empirical_freqs, frequencies)
        np.save(file_empirical_nodes, emp_nodes)

        return frequencies, emp_nodes, basis_interpolant, known_params, maximum_eies, n_outliers

    ## Main function handling the ROQ construction.

    def run(self, term):
        
        # Initialise data.
        d = {}

        # Initialise basis, either using a previously constructed one or pre-selecting one from corners of the parameter space plus a user-determined number of iterations.
        execution_time_presel_basis = time.time()
        if(self.start_values=='corners'):
            # We choose the first elements of the basis to correspond to the lower and upper values of the parameters range. Note that corner does not mean the N-D corners of the parameter space N-cube, but simply upper-lower bounds.
            initial_basis, initial_params, initial_residual_modula = self.construct_corner_basis(term)
            # Run a first pre-selection loop, building a basis of dimension `n_pre_basis`.
            preselection_basis, preselection_params, preselection_residual_modula = self.construct_preselection_basis(initial_basis, initial_params, initial_residual_modula, term)

        elif('pre-selected-basis' in self.start_values):

            logger.info('Loading previously computed pre-selected basis.')

            # Load a previously computed pre-selected basis.
            if term == 'lin':
                file_basis_stored   = self.outputdir+'/ROQ_data/linear/preselection_linear_basis.npy'
                file_params_stored  = self.outputdir+'/ROQ_data/linear/preselection_linear_basis_waveform_params.npy'
                file_res_mod_stored = self.outputdir+'/ROQ_data/linear/preselection_linear_basis_residual_modula.npy'
            elif term=='qua':
                file_basis_stored   = self.outputdir+'/ROQ_data/quadratic/preselection_quadratic_basis.npy'
                file_params_stored  = self.outputdir+'/ROQ_data/quadratic/preselection_quadratic_basis_waveform_params.npy'
                file_res_mod_stored = self.outputdir+'/ROQ_data/quadratic/preselection_quadratic_basis_residual_modula.npy'
            else:
                raise TermError
            
            # FIXME: store and load residual modula too.
            preselection_basis, preselection_params, preselection_residual_modula = np.load(file_basis_stored), np.load(file_params_stored), np.load(file_res_mod_stored)

            logger.info('')
            logger.info('################################################')
            logger.info('# \u001b[\u001b[38;5;39mLoaded input pre-computed basis ({} elements)\u001b[0m #'.format(len(preselection_params)))
            logger.info('################################################')
            logger.info('')
            
            if(self.start_values=='partial-pre-selected-basis'):
                preselection_basis, preselection_params, preselection_residual_modula = self.construct_preselection_basis(preselection_basis, preselection_params, preselection_residual_modula, term)
            
        elif(self.start_values=='pre-enriched-basis'):

            logger.info('Loading previously computed enriched basis.')

            # Load a previously computed pre-enriched basis.
            if term == 'lin':
                file_basis_stored  = self.outputdir+'/ROQ_data/linear/basis_linear.npy'
                file_params_stored = self.outputdir+'/ROQ_data/linear/basis_waveform_params_linear.npy'
            elif term=='qua':
                file_basis_stored  = self.outputdir+'/ROQ_data/quadratic/basis_quadratic.npy'
                file_params_stored = self.outputdir+'/ROQ_data/quadratic/basis_waveform_params_quadratic.npy'
            else:
                raise TermError

            # FIXME: store and load residual modula too
            preselection_basis, preselection_params, preselection_residual_modula = np.load(file_basis_stored), np.load(file_params_stored), None

            logger.info('')
            logger.info('################################################')
            logger.info('# \u001b[\u001b[38;5;39mLoaded input pre-enriched basis ({} elements)\u001b[0m #'.format(len(preselection_params)))
            logger.info('################################################')
            logger.info('')
        else:
            raise ValueError('Pre-basis option not recognised.')
        
        if(self.timing):
            execution_time_presel_basis = (time.time() - execution_time_presel_basis)/60.0
            logger.info('Timing: pre-selection basis with parallel={} [minutes]: {}'.format(self.parallel, execution_time_presel_basis))

        # Internally store the output data for later testing.
        d['{}_pre_basis'.format(term)]   = preselection_basis
        d['{}_pre_params'.format(term)]  = preselection_params
        d['{}_pre_res_mod'.format(term)] = preselection_residual_modula

        # Start the series of loops in which the pre-selected basis is enriched by the outliers found on ever increasing training sets.
        frequencies, emp_nodes, basis_interpolant, basis_parameters, maximum_eies, n_outliers = self.roqs(preselection_basis, preselection_params, term)

        # Internally store the output data for later testing.
        d['{}_interpolant'.format(term)] = basis_interpolant
        d['{}_f'.format(term)]           = frequencies
        d['{}_emp_nodes'.format(term)]   = emp_nodes
        d['{}_params'.format(term)]      = basis_parameters
        d['{}_max_eies'.format(term)]    = maximum_eies
        d['{}_n_outliers'.format(term)]  = n_outliers

        return d
