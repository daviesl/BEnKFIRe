# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division)
from filterpy.common import dot3
import numpy as np
from numpy import dot, zeros


class IMMEstimator(object):
    """ Implements an Interacting Multiple-Model (IMM) estimator.

    References
    ----------

    Bar-Shalom, Y., Li, X-R., and Kirubarajan, T. "Estimation with
    Application to Tracking and Navigation". Wiley-Interscience, 2001.

    Crassidis, J and Junkins, J. "Optimal Estimation of
    Dynamic Systems". CRC Press, second edition. 2012.

    Labbe, R. "Kalman and Bayesian Filters in Python".
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, filters, mu, M):
        """"
        Parameters
        ----------

        filters : (N,) array_like of KalmanFilter objects
            List of N filters. filters[i] is the ith Kalman filter in the
            IMM estimator.

        mu : (N,) ndarray of float
            mode probability: mu[i] is the probability that
            filter i is the correct one.

        M : (N,N) ndarray of float
            Markov chain transition matrix. M[i,j] is the probability of
            switching from filter j to filter i.
        """

        assert len(filters) > 1

        self.filters = filters
        self.mu = mu
        self.M = M

        # compute # random variables in the state
        x_shape = filters[0].x.shape
        try:
            n_states = x_shape[0]
        except:
            n_states = x_shape

        self.x = np.zeros(x_shape)
        self.P = np.zeros((n_states, n_states))

        self.N = len(filters) # number of filters


    def update(self, z, u=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        u : np.array, optional
            u[i] contains the control input for the ith filter
        """

        # run update on each filter, and save the likelihood in L
        L = zeros(len(self.filters))
        for i, f in enumerate(self.filters):
            f.update(z)
            L[i] = f.likelihood

        # initial condition IMM state, covariance
        xs, Ps = [], []
        # each element j = sum M_ij * mu_i
        
        # cbar is the total probability, after interaction, 
        # that the target is in state j. We use it as the
        # normalization constant.
        self.cbar = dot(self.mu, self.M) 

        # compute mixing probabilities
        omega = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                omega[i, j] = (self.M[i, j] * self.mu[i]) / self.cbar[j]

        # compute mixed initial conditions
        for i, (f, w) in enumerate(zip(self.filters, omega.T)):
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
            Ps.append(P)

        # perform predict step using the mixed initial conditions
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.x = dot(f.F, xs[i])
            if u is not None:
                f.x += dot(f.B, u[i])
            f.P = dot3(f.F, Ps[i], f.F.T) + f.Q


        # compute mixed IMM state and covariance
        self.x.fill(0.)
        self.P.fill(0.)

        for f, w in zip(self.filters, self.mu):
            self.x += f.x * w

        for f, w in zip(self.filters, self.mu):
            y = f.x - self.x
            self.P += w * (np.outer(y, y) + f.P)

        # update mode probabilities from total probability * likelihood
        self.mu =  self.cbar * L
        self.mu /= sum(self.mu) # normalize
