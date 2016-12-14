cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False

# skewed 9 point laplacian
def cy_laplacian(np.ndarray[double, ndim=2] u, np.ndarray[double, ndim=2] o, double h2, double constfill):
	""" Laplacian central finite difference operator. Input: u, h2,constfill, Output: o (same dimensions as u) """
	cdef unsigned int i, j
	cdef double h2_3 = 6 * h2
	for i in xrange(1,u.shape[0]-1):
		# inner square
		for j in xrange(1, u.shape[1]-1):
			o[i,j] = (4 * u[i+1, j] + 4 * u[i-1, j] + u[i+1, j+1] + u[i-1, j-1] +
			          4 * u[i, j+1] + 4 * u[i, j-1] + u[i+1, j-1] + u[i-1, j+1] -
			          (u[i, j] * 20)) / h2_3
		# left junction line
		j = 0
		o[i,j] = (4 * u[i+1, j] + 4 * u[i-1, j] + u[i+1, j+1] + constfill   +
		          4 * u[i, j+1] + 4 * constfill + constfill   + u[i-1, j+1] -
		          (u[i, j] * 20)) / h2_3
		# right junction line
		j = u.shape[1]-1
		o[i,j] = (4 * u[i+1, j] + 4 * u[i-1, j] + constfill   + u[i-1, j-1] +
		          4 * constfill + 4 * u[i, j-1] + u[i+1, j-1] + constfill   -
		          (u[i, j] * 20)) / h2_3
	# horizontal junction lines
	for j in xrange(1, u.shape[1]-1):
		# top
		i = 0
		o[i,j] = (4 * u[i+1, j] + 4 * constfill + u[i+1, j+1] + constfill +
		          4 * u[i, j+1] + 4 * u[i, j-1] + u[i+1, j-1] + constfill -
		          (u[i, j] * 20)) / h2_3
		# bottom
		i = u.shape[0]-1
		o[i,j] = (4 * constfill + 4 * u[i-1, j] + constfill + u[i-1, j-1] +
		          4 * u[i, j+1] + 4 * u[i, j-1] + constfill + u[i-1, j+1] -
		          (u[i, j] * 20)) / h2_3
	# corners
	# top left
	i = 0
	j = 0
	o[i,j] = (4 * u[i+1, j] + 4 * constfill + u[i+1, j+1] + constfill +
	          4 * u[i, j+1] + 4 * constfill + constfill   + constfill -
	          (u[i, j] * 20)) / h2_3
	# top right
	i = 0
	j = u.shape[1]-1
	o[i,j] = (4 * u[i+1, j] + 4 * constfill + constfill   + constfill +
	          4 * constfill + 4 * u[i, j-1] + u[i+1, j-1] + constfill -
	          (u[i, j] * 20)) / h2_3
	# bottom left
	i = u.shape[0]-1
	j = 0
	o[i,j] = (4 * constfill + 4 * u[i-1, j] + constfill   + constfill   +
	          4 * u[i, j+1] + 4 * constfill + constfill   + u[i-1, j+1] -
	          (u[i, j] * 20)) / h2_3
	# bottom right
	i = u.shape[0]-1
	j = u.shape[1]-1
	o[i,j] = (4 * constfill + 4 * u[i-1, j] + constfill + u[i-1, j-1] +
	          4 * constfill + 4 * u[i, j-1] + constfill + constfill    -
	          (u[i, j] * 20)) / h2_3
