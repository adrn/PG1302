# coding: utf-8

""" Fit an RV curve to the light curve of  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import time

# Third-party
from astropy.constants import G,c
from astropy import log as logger
import astropy.units as u
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

# Custom
from gary.util import get_pool

usys = (u.day, u.Msun, u.au)
G = G.decompose(usys).value
c = c.decompose(usys).value

def eccentric_anomaly_func(ecc_anom, t, ecc, t0, Tbin):
    return np.abs(2.*np.pi/Tbin*(t-t0) - ecc_anom + ecc*np.sin(ecc_anom))

def eccentric_anomaly(t, ecc, t0, Tbin):
    ecc_anomalies = []
    for t_i in t:
        minEA = fmin(eccentric_anomaly_func, 0., args=(t_i,ecc,t0,Tbin), disp=False)
        ecc_anomalies.append(minEA[0])
    return np.array(ecc_anomalies) % (2*np.pi)

def model(t, ecc, ww, t0, KK, Tbin):
    # e = 0.0
    # ww= pi/3.
    incl = np.pi/2.
    # Mbin = Msun*10**(9.4)
    # Tbin = 5.2*yr
    # T0 = 2.2*yr
    # q=0.01
    # KK = 0.06*c
    vmean = 0.0  # vmean*c

    # Solve for Eccentric Anamoly and then f(t)
    ecc_anom = eccentric_anomaly(t, ecc, t0, Tbin)
    f_t = 2. * np.arctan2(np.sqrt(1.+ecc) * np.tan(ecc_anom/2.), np.sqrt(1. - ecc))

    # Now get radial velocity from Kepler problem
    # KK        = q/(1.+q) * nn*sep * sin(incl)/sqrt(1.-e*e)
    vsec = vmean + KK*(np.cos(ww + f_t) + ecc*np.cos(ww))  # div by q to get seconday vel
    # vpr        = q*vsec

    # NOW COMPUTE REL. BEAMING FORM RAD VEL
    GamS = 1. / np.sqrt(1. - (vsec/c)**2)
    DopS = 1. / (GamS * (1. - vsec/c * np.cos(incl - np.pi/2.)))

    DopLum = DopS**(3. - 1.1)
    mags = 5./2. * np.log10(DopLum)  # mag - mag0= -2.5 * log10(F(t)/F_0)  =  -2.5 * log10(DopLum)

    return mags

def ln_likelihood(p, t, y, dy):
    return -0.5 * (y - model(t,*p))**2 / dy**2

def ln_prior(p):
    ecc, ww, t0, KK, Tbin = p

    if ecc < 0. or ecc >= 1.:
        return -np.inf

    if KK/c >= 1.:
        return -np.inf

    return 0.

def ln_posterior(p, *args):
    lnp = ln_prior(p)
    if np.any(np.isinf(lnp)):
        return -np.inf

    lnl = ln_likelihood(p, *args)
    if np.any(~np.isfinite(lnl)):
        return -np.inf

    return lnp + lnl.sum()

def test_model():
    t = np.linspace(0,3000.,500)
    m = model(t, 0., np.pi/3., 0., 0.06*c, (5.2*u.year).to(u.day).value)
    plt.plot(t, m, linestyle='none')
    plt.show()

def main(mpi=False):

    pool = get_pool(mpi=mpi)

    # read data
    t,lum,err = np.loadtxt("Lums_PG1302.dat").T
    ix = t.argsort()

    # sort on time, subtract min time, subtract mean magnitude
    t = t[ix] - t.min()
    lum = lum[ix] - lum.mean()
    err = err[ix]

    # initial guess at params
    pinit = [0.25,  # eccentricity
             0.0,  # ww
             600,  # t0
             0.08 * c,  # KK
             (5.2*u.year).decompose(usys).value]  # binary period

    # plot data with initial guess
    # plt.errorbar(t, lum, err, marker='o', ecolor='#888888', linestyle='none')
    # plt.plot(t, model(t, *pinit), linestyle='none')
    # plt.show()

    nwalkers = len(pinit) * 4
    nburn = 25
    nsteps = 1000
    sampler = emcee.EnsembleSampler(nwalkers, dim=len(pinit),
                                    lnpostfn=ln_posterior,
                                    args=(t, lum, err),
                                    pool=pool)

    logger.debug("Sampling initial conditions for walkers")
    p0 = emcee.utils.sample_ball(pinit,
                                 std=[0.01,0.01, 10., 0.01*c,
                                      (0.05*u.year).decompose(usys).value],
                                 size=nwalkers)

    logger.info("Burning in MCMC sampler ({0} walkers) for {1} steps".format(nwalkers, nburn))
    timer0 = time.time()
    pos,prob,state = sampler.run_mcmc(p0, nburn)
    logger.debug("Took {:.2f} seconds to run for {} steps.".format(time.time()-timer0, nburn))

    # sampler.reset()
    # pos,prob,state = sampler.run_mcmc(pos, 1000)

    pool.close()

    plt.clf()
    for i in range(nwalkers):
        plt.plot(sampler.chain[i,:,0], drawstyle='steps', marker=None)
    plt.savefig("rv-fit-mcmc-test.png")

    sys.exit(0)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(mpi=args.mpi)
