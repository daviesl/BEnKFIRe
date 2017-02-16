cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False

def cy_laplacian(np.ndarray[double, ndim=2] u, np.ndarray[double, ndim=2] o, double dx2, double constfill):
	""" Laplacian central finite difference operator. Input: u, dx2,dy2,constfill, Output: o (same dimensions as u) """
	cdef unsigned int i, j
	for i in xrange(1,u.shape[0]-1):
		# inner square
		for j in xrange(1, u.shape[1]-1):
			o[i,j] = (u[i+1, j] + u[i-1, j] +
			          u[i, j+1] + u[i, j-1] -
			         (u[i, j] * 4)) / dx2
		# left junction line
		j = 0
		o[i,j] = (u[i+1, j] + u[i-1, j] +
		          u[i, j+1] + constfill -
		         (u[i, j] * 4)) / dx2
		# right junction line
		j = u.shape[1]-1
		o[i,j] = (u[i+1, j] + u[i-1, j] +
		          constfill + u[i, j-1] -
		         (u[i, j] * 4)) / dx2
	# horizontal junction lines
	for j in xrange(1, u.shape[1]-1):
		# top
		i = 0
		o[i,j] = (u[i+1, j] + constfill +
		          u[i, j+1] + u[i, j-1] -
		         (u[i, j] * 4)) / dx2
		# bottom
		i = u.shape[0]-1
		o[i,j] = (constfill + u[i-1, j] +
		          u[i, j+1] + u[i, j-1] -
		         (u[i, j] * 4)) / dx2
	# corners
	# top left
	i = 0
	j = 0
	o[i,j] = (u[i+1, j] + constfill +
	          u[i, j+1] + constfill -
	         (u[i, j] * 4)) / dx2
	# top right
	i = 0
	j = u.shape[1]-1
	o[i,j] = (u[i+1, j] + constfill +
	          constfill + u[i, j-1] -
	         (u[i, j] * 4)) / dx2
	# bottom left
	i = u.shape[0]-1
	j = 0
	o[i,j] = (constfill + u[i-1, j] +
	          u[i, j+1] + constfill -
	         (u[i, j] * 4)) / dx2
	# bottom right
	i = u.shape[0]-1
	j = u.shape[1]-1
	o[i,j] = (constfill + u[i-1, j] +
	          constfill + u[i, j-1] -
	         (u[i, j] * 4)) / dx2
