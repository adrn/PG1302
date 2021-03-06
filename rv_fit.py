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

# PGPATH = "/vega/astro/users/amp2217/projects/PG1302"
PGPATH = "/Users/adrian/projects/PG1302"

def eccentric_anomaly_func(ecc_anom, t, ecc, t0, Tbin):
    return (2.*np.pi/Tbin*(t-t0) - ecc_anom + ecc*np.sin(ecc_anom))**2.

def eccentric_anomaly(t, ecc, t0, Tbin):
    ecc_anomalies = [fmin(eccentric_anomaly_func, 0., args=(t_i,ecc,t0,Tbin), disp=False)[0]
                        for t_i in t]
    return np.array(ecc_anomalies) % (2*np.pi)

def model(t, ecc, cosw, t0, KK, Tbin):
    incl = np.pi/2.
    vmean = 0.0  # vmean*c
    # Tbin = 1899.3

    # Solve for Eccentric Anamoly and then f(t)
    ecc_anom = eccentric_anomaly(t, ecc, t0, Tbin)
    f_t = 2. * np.arctan2(np.sqrt(1.+ecc) * np.tan(ecc_anom/2.), np.sqrt(1. - ecc))

    # Now get radial velocity from Kepler problem
    # KK        = q/(1.+q) * nn*sep * sin(incl)/sqrt(1.-e*e)
    a = -1.
    vsec = vmean + c*KK*(cosw*np.cos(f_t) - a*np.sqrt(1-cosw**2)*np.sin(f_t) + ecc*cosw)  # div by q to get seconday vel
    # vsec = vmean + c*KK*(np.cos(ww + f_t) + ecc*np.cos(ww))
    # vpr        = q*vsec

    # NOW COMPUTE REL. BEAMING FORM RAD VEL
    GamS = 1. / np.sqrt(1. - (vsec/c)**2)
    DopS = 1. / (GamS * (1. - vsec/c * np.cos(incl - np.pi/2.)))

    DopLum = DopS**(3.0 - 1.1)
    mags = 5./2. * np.log10(DopLum)  # mag - mag0= -2.5 * log10(F(t)/F_0)  =  -2.5 * log10(DopLum)

    return mags

def ln_likelihood(p, t, y, dy):
    # V = pp[-1]
    # p = pp[:-1]
    V = 0.
    return -0.5 * (y - model(t,*p))**2 / (dy**2 + V)

def ln_prior(p):
    ecc, cosw, t0, KK, Tbin = p

    lnp = 0.

    if ecc < 0. or ecc >= 1.:
        return -np.inf

    if KK >= 1. or KK < 0.:
        return -np.inf

    # if t0 < 300 or t0 > 750:
    #     return -np.inf

    if cosw < -1 or cosw > 1:
        return -np.inf

    # if V < 0.:
    #     return -np.inf
    # lnp -= np.log(V)

    return lnp

def ln_posterior(p, *args):
    lnp = ln_prior(p)
    if np.any(np.isinf(lnp)):
        return -np.inf

    lnl = ln_likelihood(p, *args)
    if np.any(~np.isfinite(lnl)):
        return -np.inf

    return lnp + lnl.sum()

def read_data():
    # read data
    t,lum,err = np.loadtxt(os.path.join(PGPATH, "data/Lums_PG1302.dat")).T
    ix = t.argsort()

    # sort on time, subtract min time, subtract mean magnitude
    t = t[ix] - t.min()
    lum = lum[ix] - lum.mean()
    err = err[ix]

    return t,lum,err

def test_model():

    t,y,dy = read_data()
    plt.errorbar(t, y, dy, marker='o', ecolor='#888888', linestyle='none')

    m = model(t, 0.0, 0.2, 1100., 0.057)
    plt.plot(t, m, linestyle='-', marker=None)
    plt.show()

def main(mpi=False):

    pool = get_pool(mpi=mpi)

    # file cleanup
    if os.path.exists(os.path.join(PGPATH,"burn_in_done")):
        os.remove(os.path.join(PGPATH,"burn_in_done"))

    t,y,dy = read_data()

    # initial guess at params
    pinit = [0.05,  # eccentricity
             0.0,  # cosw
             1000,  # t0
             0.08,  # KK
             (5.2*u.year).decompose(usys).value]  # binary period
    pstd = [0.01, 0.01, 10., 0.01,
            (0.05*u.year).decompose(usys).value]
    # 0.01]

    # plot data with initial guess
    # plt.errorbar(t, lum, err, marker='o', ecolor='#888888', linestyle='none')
    # plt.plot(t, model(t, *pinit), linestyle='none')
    # plt.show()

    nwalkers = 64  # len(pinit) * 4
    nburn = 250
    nsteps = 1000

    sampler = emcee.EnsembleSampler(nwalkers, dim=len(pinit),
                                    lnpostfn=ln_posterior,
                                    args=(t, y, dy),
                                    pool=pool)

    logger.debug("Sampling initial conditions for walkers")
    p0 = emcee.utils.sample_ball(pinit,
                                 std=pstd,
                                 size=nwalkers)

    logger.info("Burning in MCMC sampler ({0} walkers) for {1} steps".format(nwalkers, nburn))
    timer0 = time.time()
    pos,prob,state = sampler.run_mcmc(p0, nburn)
    logger.debug("Took {:.2f} seconds to run for {} steps.".format(time.time()-timer0, nburn))

    with open(os.path.join(PGPATH,"burn_in_done"), "w") as f:
        f.write("yup")

    sampler.reset()

    timer0 = time.time()
    logger.info("Running main sampling ({0} walkers) for {1} steps".format(nwalkers, nsteps))
    pos,prob,state = sampler.run_mcmc(pos, nsteps)

    chain = sampler.chain
    np.save(os.path.join(PGPATH,"chain2.npy"), chain)
    np.save(os.path.join(PGPATH,"flatlnprob2.npy"), sampler.flatlnprobability)
    logger.debug("Took {:.2f} seconds to run for {} steps.".format(time.time()-timer0, nsteps))

    pool.close()

    for j in range(len(pinit)):
        plt.clf()
        for i in range(nwalkers):
            plt.plot(chain[i,:,j], drawstyle='steps', marker=None)
        plt.savefig(os.path.join(PGPATH,"plots/rv-fit-mcmc-test-{0}.png".format(j)))

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
