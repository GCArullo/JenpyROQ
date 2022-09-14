## -*- coding: utf8 -*-
#!/usr/bin/env python

# General python imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, os, seaborn as sns, time
import logging
from itertools import repeat

# Package internal imports
from .         import linear_algebra
from .parallel import eval_func_tuple

# Global plotting settings
plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
labels_fontsize = 16

logger = logging.getLogger(__name__)

## Functions to test the performance of the waveform representation, using the interpolant built from the selected basis.

def compute_mismatch_of_all_terms(paramspoint, b, emp_nodes, term, jenpyroq):
    
    # Generate test waveform
    hp, hc = jenpyroq.paramspoint_to_wave(paramspoint, 'lin')

    # Compute quadratic terms and interpolant representations
    if term == 'qua':
#        hphc     = np.real(hp * np.conj(hc))
#        hphc     = linear_algebra.normalise_vector(hphc, jenpyroq.deltaF)
#        hphc_emp = hphc[emp_nodes]
#        hphc_rep = np.dot(b,hphc_emp)

        hp, hc   = (np.absolute(hp))**2, (np.absolute(hc))**2
    
    hp, hc = linear_algebra.normalise_vector(hp, jenpyroq.deltaF), linear_algebra.normalise_vector(hc, jenpyroq.deltaF)

    hp_emp    = hp[emp_nodes]
    hp_rep    = np.dot(b,hp_emp)
    hc_emp    = hc[emp_nodes]
    hc_rep    = np.dot(b,hc_emp)

    # Compute the representation error. This is related to the same measure employed to stop adding elements to the basis.

    eie_hp = 2. * (1 - linear_algebra.scalar_product(hp, hp_rep, jenpyroq.deltaF))
    eie_hc = 2. * (1 - linear_algebra.scalar_product(hc, hc_rep, jenpyroq.deltaF))
#        if term == 'qua':
#            eie_hphc = 2. * (1 - linear_algebra.scalar_product(hphc, hphc_rep, jenpyroq.deltaF))

    return eie_hp, eie_hc

#FIXME: either specialise these plots, or define a single function for all of them

def plot_preselection_residual_modula(pre_residual_modula, term, outputdir):

    # Skip, in case an enriched basis was passed by the user, and no pre-residual modula are available.
    try:
        if not (pre_residual_modula.all()==None):
            plt.figure()
            plt.semilogy(pre_residual_modula)
            plt.xlabel('N basis elements')
            plt.ylabel('Residual modulus')
            plt.savefig(os.path.join(outputdir,'Plots/Preselection_residual_modulus_{}.pdf'.format(term)), bbox_inches='tight')
    except: pass
    
    return

def plot_maximum_empirical_interpolation_error(eies, term, outputdir):

    plt.figure()
    plt.semilogy(eies)
    plt.xlabel('N basis elements')
    plt.ylabel('Maximum empirical interpolation error')
    plt.savefig(os.path.join(outputdir,'Plots/Empirical_interpolation_error_{}.pdf'.format(term)), bbox_inches='tight')

    return

def plot_number_of_outliers(eies, term, outputdir):

    plt.figure()
    plt.plot(eies)
    plt.xlabel('N basis elements')
    plt.ylabel('Number of outliers')
    plt.savefig(os.path.join(outputdir,'Plots/Number_of_outliers_{}.pdf'.format(term)), bbox_inches='tight')

    return

def plot_representation_error(b, emp_nodes, paramspoint, term, outputdir, freq, paramspoint_to_wave):
    
    hp, hc = paramspoint_to_wave(paramspoint, 'lin')
    
    if   term == 'lin':
        pass
    elif term == 'qua':
#        hphc = np.real(hp * np.conj(hc))
        hp   = (np.absolute(hp))**2
        hc   = (np.absolute(hc))**2
    else              :
        raise TermError
    
    freq           = freq
    hp_emp, hc_emp = hp[emp_nodes], hc[emp_nodes]
    hp_rep, hc_rep = np.dot(b,hp_emp), np.dot(b,hc_emp)
    diff_hp        = hp_rep - hp
    diff_hc        = hc_rep - hc
    rep_error_hp   = diff_hp/np.sqrt(np.vdot(hp,hp))
    rep_error_hc   = diff_hc/np.sqrt(np.vdot(hc,hc))
#    if term == 'qua':
#        hphc_emp       = hphc[emp_nodes]
#        hphc_rep       = np.dot(b,hphc_emp)
#        diff_hphc      = hphc_rep - hphc
#        rep_error_hphc = diff_hphc/np.sqrt(np.vdot(hphc,hphc))

    plt.figure(figsize=(8,5))
    if term == 'lin':
        plt.plot(    freq, np.real(hp),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(    freq, np.real(hp_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    else:
        plt.semilogy(freq, np.real(hp),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$mathrm{Full}$')
        plt.semilogy(freq, np.real(hp_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    plt.scatter(freq[emp_nodes], np.real(hp)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{\Re[h_+]}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparisons/Waveform_comparison_hp_real_{}.pdf'.format(term)), bbox_inches='tight')

    plt.figure(figsize=(8,5))
    if term == 'lin':
        plt.plot(freq, np.real(hc),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(freq, np.real(hc_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    else:
        plt.semilogy(freq, np.real(hc),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.semilogy(freq, np.real(hc_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
    plt.scatter(freq[emp_nodes], np.real(hc)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{\Re[h_{\\times}]}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparisons/Waveform_comparison_hc_real_{}.pdf'.format(term)), bbox_inches='tight')

    if term == 'lin':
        plt.figure(figsize=(8,5))
        plt.plot(freq, np.imag(hp),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(freq, np.imag(hp_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
        plt.scatter(freq[emp_nodes], np.imag(hp)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
        plt.ylabel('$\Im[h_+]$', fontsize=labels_fontsize)
        plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparisons/Waveform_comparison_hp_imag_{}.pdf'.format(term)), bbox_inches='tight')

        plt.figure(figsize=(8,5))
        plt.plot(freq, np.imag(hc),     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
        plt.plot(freq, np.imag(hc_rep), color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
        plt.scatter(freq[emp_nodes], np.imag(hc)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
        plt.ylabel('$\Im[h_{\\times}]$', fontsize=labels_fontsize)
        plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparisons/Waveform_comparison_hc_imag_{}.pdf'.format(term)), bbox_inches='tight')

#    else:
#        plt.figure(figsize=(8,5))
#        plt.plot(freq, hphc,     color='orangered', lw=1.3, alpha=0.8, ls='-',  label='$\mathrm{Full}$')
#        plt.plot(freq, hphc_rep, color='black',     lw=0.8, alpha=1.0, ls='--', label='$\mathrm{ROQ}$' )
#        plt.scatter(freq[emp_nodes], hphc[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
#        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
#        plt.ylabel('$\Re[h_+ \, {h}^*_{\\times}]$', fontsize=labels_fontsize)
#        plt.title('$\mathrm{Waveform \,\, comparison \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
#        plt.legend(loc='best')
#        plt.savefig(os.path.join(outputdir,'Plots/Waveform_comparisons/Waveform_comparison_hphc.pdf'), bbox_inches='tight')

#        plt.figure(figsize=(8,5))
#        plt.plot(   freq,            rep_error_hphc,            color='dodgerblue', lw=1.3, alpha=1.0, ls='-', label='$\Re[h_+ \, {h}^*_{\\times}]$')
#        plt.scatter(freq[emp_nodes], rep_error_hphc[emp_nodes], color='dodgerblue', marker='o', s=10,          label='$\mathrm{Empirical \,\, nodes}$')
#        plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
#        plt.ylabel('$\mathrm{Fractional Representation Error}$', fontsize=labels_fontsize)
#        plt.title('$\mathrm{Representation \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
#        plt.legend(loc='best')
#        plt.savefig(os.path.join(outputdir,'Plots/Representation_error_hphc.pdf'), bbox_inches='tight')

    plt.figure(figsize=(8,5))
    plt.plot(freq, np.real(rep_error_hp), color='dodgerblue', lw=1.3, alpha=1.0, ls='-', label='$\Re[h_+]$')
    if term == 'lin':
        plt.plot(freq, np.imag(rep_error_hp), color='darkred',    lw=1.3, alpha=0.8, ls='-', label='$\Im[h_+]$')
    plt.scatter(freq[emp_nodes], np.real(rep_error_hp)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.scatter(freq[emp_nodes], np.imag(rep_error_hp)[emp_nodes], marker='o', c='dodgerblue', s=10)
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{Fractional \,\, Representation \,\, Error}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Representation \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Representation_error_hp_{}.pdf'.format(term)), bbox_inches='tight')

    plt.figure(figsize=(8,5))
    plt.plot(freq, np.real(rep_error_hc), color='dodgerblue', lw=1.3, alpha=1.0, ls='-', label='$\Re[h_{\\times}]$')
    if term == 'lin':
        plt.plot(freq, np.imag(rep_error_hc), color='darkred',    lw=1.3, alpha=0.8, ls='-', label='$\Im[h_{\\times}]$')
    plt.scatter(freq[emp_nodes], np.real(rep_error_hc)[emp_nodes], marker='o', c='dodgerblue', s=10, label='$\mathrm{Empirical \,\, nodes}$')
    plt.scatter(freq[emp_nodes], np.imag(rep_error_hc)[emp_nodes], marker='o', c='dodgerblue', s=10)
    plt.xlabel('$\mathrm{Frequency}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{Fractional \,\, Representation \,\, Error}$', fontsize=labels_fontsize)
    plt.title('$\mathrm{Representation \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(outputdir,'Plots/Representation_error_hc_{}.pdf'.format(term)), bbox_inches='tight')

    return

def test_roq_error(b, emp_nodes, term, jenpyroq, Pool):
    
    # Initialise structures
    nsamples = jenpyroq.n_tests_post
    eies_hp   = np.zeros(nsamples)
    eies_hc   = np.zeros(nsamples)
    
    # Draw random test points
    paramspoints = jenpyroq.generate_params_points(npts=nsamples)
    
    # Select tolerance
    if   term == 'lin':
        tol = jenpyroq.tolerance_lin
        pass
    elif term == 'qua':
        # FIXME: currently unused
        eies_hphc = np.zeros(nsamples)
        tol       = jenpyroq.tolerance_qua
    else:
        raise TermError
    
    # Start looping over test points
    logger.info('')
    logger.info('')
    logger.info('###########################################')
    logger.info('# \u001b[\u001b[38;5;39mStarting validation tests {} iteration\u001b[0m #'.format(term))
    logger.info('###########################################')
    logger.info('')
    logger.info('Validation set size : {}'.format(nsamples))
    logger.info('Tolerance           : {}'.format(tol))
    logger.info('')

    if(jenpyroq.timing): execution_time_validation_tests = time.time()
    xy = Pool.map(eval_func_tuple, zip(repeat(compute_mismatch_of_all_terms),
                                                          paramspoints,
                                                          repeat(b),
                                                          repeat(emp_nodes),
                                                          repeat(term),
                                                          repeat(jenpyroq)
                                                          ))
    if(jenpyroq.timing): logger.info('Timing: validation tests, generating {} waveforms with parallel={} [minutes]: {}'.format(nsamples, jenpyroq.parallel, (time.time() - execution_time_validation_tests)/60.0))


    eies_hp = [x[0] for x in xy]
    eies_hc = [x[1] for x in xy]

        # If a test case exceeds the error, let the user know. Using <dh|dh> = 2(1-<h|h_ROQ>), where dh = h - h_ROQ. Also, print typical test result every 100 steps.
#        np.set_printoptions(suppress=True)
#        if (eies_hp[i] > tol):
#            logger.info('h_+     above tolerance: Iter: {}'.format(i)+' Interpolation error: {}'.format(eies_hp[i])+' Parameters: {}'.format(paramspoints[i]))
#        if (eie_hc[i] > tol):
#            logger.info('h_x     above tolerance: Iter: {}'.format(i)+' Interpolation error: {}'.format(eie_hc[i])+' Parameters: {}'.format(paramspoints[i]))
#                if ((term == 'qua') and (eies_hphc[i] > tol)):
#                    print("h_+ h_x above tolerance: Iter: ", i, "Interpolation error: ", eies_hphc[i], "Parameters: ", paramspoints[i])
#        if i%100==0:
#            logger.info('h_+     rolling check (every 100 steps): Iter: {}'.format(i)+' Interpolation error: {}'.format(eies_hp[i]))
#            logger.info('h_x     rolling check (every 100 steps): Iter: {}'.format(i)+' Interpolation error: {}'.format(eie_hc[i]))
#                    if (term == 'qua'):
#                        print("h_+ h_x rolling check (every 100 steps): Iter: ",             i, "Interpolation error: ", eies_hphc[i])

    # Plot the test results
    plt.figure(figsize=(8,5))
    plt.semilogy(eies_hp, 'x', color='darkred',    label='$\Re[h_+]$')
    plt.semilogy(eies_hc, 'x', color='dodgerblue', label='$\Re[h_{\\times}]$')
    plt.axhline(tol, label='$\mathrm{Tolerance}$', c='k', ls='dashed', lw=0.9)
    plt.xlabel('$\mathrm{Number \,\, of \,\, Random \,\, Test \,\, Points}$', fontsize=labels_fontsize)
    plt.ylabel('$\mathrm{Interpolation \,\, Error \,\, (%s \,\, basis)}$'%(term), fontsize=labels_fontsize)
    plt.legend(loc='best')
    plt.savefig(os.path.join(jenpyroq.outputdir,'Plots/Interpolation_errors_random_test_points_{}.pdf'.format(term)), bbox_inches='tight')

    return

def histogram_basis_params(params_basis, outputdir, i2n, term):

    p = {}
    for i,k in i2n.items():
        p[k] = []
        for j in range(len(params_basis)):
            p[k].append(params_basis[j][i])
        
        plt.figure()
        sns.displot(p[k], color='darkred')
        plt.xlabel(k, fontsize=labels_fontsize)
        plt.savefig(os.path.join(outputdir,'Plots/Basis_parameters/Basis_parameters_{}_{}.pdf'.format(term, k)), bbox_inches='tight')
        plt.close()

def histogram_frequencies(frequencies, outputdir, term):
    
    plt.figure()
    sns.displot(frequencies, color='darkred')
    plt.xlabel('$\mathrm{f\,[Hz]}$', fontsize=labels_fontsize)
    plt.savefig(os.path.join(outputdir,'Plots/Frequencies_{}.pdf'.format(term)), bbox_inches='tight')
    plt.close()
