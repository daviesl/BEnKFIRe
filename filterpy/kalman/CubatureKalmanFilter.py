# -*- coding: utf-8 -*-
"""Copyright 2016 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from filterpy.common import dot3
import numpy as np
from math import sqrt
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import inv, cholesky




def spherical_radial_sigmas(x, P):
    r""" Creates cubature points for the the specified state and covariance
    according to [1].

    Parameters
    ----------

    x: ndarray (column vector)
        examples:  np.array([[1.], [2.]])

    P : scalar, or np.array
       Covariance of the filter.

    References
    ----------

    .. [1] Arasaratnam, I, Haykin, S. "Cubature Kalman Filters,"
       IEEE Transactions on Automatic Control, 2009, pp 1254-1269, vol 54, No 6
    """

    n, _ = P.shape
    x = x.flatten()

    sigmas = np.empty((2*n, n))
    U = cholesky(P) * sqrt(n)
    for k in range(n):
        sigmas[k]   = x + U[k]
        sigmas[n+k] = x - U[k]

    return sigmas


def ckf_transform(Xs, Q):

    m, n = Xs.shape

    x = sum(Xs, 0)[:, None] / m
    P = np.zeros((n, n))
    xf = x.flatten()
    for k in range(m):
        P += np.outer(Xs[k], Xs[k]) - np.outer(xf, xf)

    P *= 1 / m
    P += Q

    return x, P


class CubatureKalmanFilter(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=C0103

    r""" Implements the Cubuture Kalman filter (UKF) as defined by
    Ienkaran Arasaratnam and Simon Haykin in [1]


    You will have to set the following attributes after constructing this
    object for the filter to perform properly.

    Attributes
    ----------

    x : numpy.array(dim_x)
        state estimate vector

    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix

    R : numpy.array(dim_z, dim_z)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix


    You may read the following attributes.

    Readable Attributes
    -------------------

    K : numpy.array
        Kalman gain

    y : numpy.array
        innovation residual


    References
    ----------

    .. [1] Arasaratnam, I, Haykin, S. "Cubature Kalman Filters,"
       IEEE Transactions on Automatic Control, 2009, pp 1254-1269, vol 54, No 6
    """


    def __init__(self, dim_x, dim_z, dt, hx, fx,
                 x_mean_fn=None,
                 z_mean_fn=None,
                 residual_x=None,
                 residual_z=None):

        r""" Create a Cubature Kalman filter. You are responsible for setting
        the various state variables to reasonable values; the defaults will
        not give you a functional filter.

        Parameters
        ----------

        dim_x : int
            Number of state variables for the filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.


        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        dt : float
            Time between steps in seconds.

        hx : function(x)
            Measurement function. Converts state vector x into a measurement
            vector of shape (dim_z).

        fx : function(x, dt)
            function that returns the state x transformed by the
            state transistion function. dt is the time step in seconds.

        x_mean_fn : callable  (sigma_points, weights), optional
            Function that computes the mean of the provided sigma points
            and weights. Use this if your state variable contains nonlinear
            values such as angles which cannot be summed.

            .. code-block:: Python

                def state_mean(sigmas, Wm):
                    x = np.zeros(3)
                    sum_sin, sum_cos = 0., 0.

                    for i in range(len(sigmas)):
                        s = sigmas[i]
                        x[0] += s[0] * Wm[i]
                        x[1] += s[1] * Wm[i]
                        sum_sin += sin(s[2])*Wm[i]
                        sum_cos += cos(s[2])*Wm[i]
                    x[2] = atan2(sum_sin, sum_cos)
                    return x

        z_mean_fn : callable  (sigma_points, weights), optional
            Same as x_mean_fn, except it is called for sigma points which
            form the measurements after being passed through hx().

        residual_x : callable (x, y), optional
        residual_z : callable (x, y), optional
            Function that computes the residual (difference) between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles (359-1 degreees is 2, not 358). x and y
            are state vectors, not scalars. One is for the state variable,
            the other is for the measurement state.

            .. code-block:: Python

                def residual(a, b):
                    y = a[0] - b[0]
                    if y > np.pi:
                        y -= 2*np.pi
                    if y < -np.pi:
                        y = 2*np.pi
                    return y
        """

        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2*dim_x
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn


        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.sigmas_f = zeros((2*self._dim_x, self._dim_x))
        self.sigmas_h = zeros((2*self._dim_x, self._dim_z))



    def predict(self, dt=None,  fx_args=()):
        r""" Performs the predict step of the CKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P).

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.

        fx_args : tuple, optional, default (,)
            optional arguments to be passed into fx() after the required state
            variable.
        """

        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        sigmas = spherical_radial_sigmas(self.x, self.P)

        # evaluate cubature points
        for k in range(self._num_sigmas):
            self.sigmas_f[k] = self.fx(sigmas[k], dt, *fx_args)


        self.x, self.P = ckf_transform(self.sigmas_f, self.Q)


    def update(self, z, R=None, hx_args=()):
        """ Update the CKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.
        """

        if z is None:
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        for k in range(self._num_sigmas):
            self.sigmas_h[k] = self.hx(self.sigmas_f[k], *hx_args)

        # mean and covariance of prediction passed through unscented transform
        #zp, Pz = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        zp, Pz = ckf_transform(self.sigmas_h, R)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        m = self._num_sigmas  # literaure uses m for scaling factor
        xf = self.x.flatten()
        zpf = zp.flatten()
        for k in range(m):
            dx = self.sigmas_f[k] - xf
            dz =  self.sigmas_h[k] - zpf
            Pxz += outer(dx, dz)

        Pxz /= m

        self.K = dot(Pxz, inv(Pz))        # Kalman gain
        self.y = self.residual_z(z, zp)   #residual

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot3(self.K, Pz, self.K.T)
