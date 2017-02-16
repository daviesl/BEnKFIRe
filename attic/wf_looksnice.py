import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from matplotlib import colors

# Displacements from a cell to its eight nearest neighbours
neighbourhood = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
EMPTY, TREE, FIRE = 0, 1, 2
# Colours for visualization: brown for EMPTY, dark green for TREE and orange
# for FIRE. Note that for the colormap to work, this list and the bounds list
# must be one larger than the number of different values in the array.
colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm = colors.BoundaryNorm(bounds, cmap.N)


_k = 2.1360e-1 # m^2 s^-1 K^-3
_A = 1.8793e3 * 1.5 # K s^-1 # was e2
_B = 5.5849e2 # K
#_C = 4.8372e-5 # K^-1
_C = 4.8372e-5 # K^-1
_Cs = 0.1625 # s^-1
_lambda = 0.027
_beta = 0.4829
_Ta = 298.15 #373.15
_Tal = _Ta - 0.01
_dx = 1 # metres
_dy = 1 # metres
_dt = 600 # 10 minutes
_kd = _k/(_dx*_dy)
LIGHTNING = 450 # Kelvin

def iterate(T,S):
	"""Iterate the forest according to the forest-fire rules."""
	# The boundary of the forest is always empty, so only consider cells
	# indexed from 1 to nx-2, 1 to ny-2
	(ny, nx) = S.shape
	T1 = np.ones((ny, nx)) * _Ta
	S1 = np.zeros((ny, nx))

	for ix in range(3,nx-3):
		for iy in range(3,ny-3):
			if np.random.random() <= f:
				T[iy,ix] = LIGHTNING
				T[iy-1,ix] = LIGHTNING
				T[iy+1,ix] = LIGHTNING
				T[iy,ix-1] = LIGHTNING
				T[iy,ix+1] = LIGHTNING
	#laplace_T = (T[1:ny-1,2:nx] - 2 * T[1:ny-1,1:nx-1] + T[1:ny-1,0:nx-2]) / (_dx ** 2) + (T[2:ny,1:nx-1] - 2 * T[1:ny-1,1:nx-1] + T[0:ny-2,1:nx-1]) / (_dy ** 2)
	laplace_T = np.zeros((ny-2,nx-2))
	ndimage.filters.laplace(T[1:ny-1,1:nx-1],output=laplace_T,mode='constant',cval=_Ta)
	# assume no wind for now
	# dT_dt has the 1-pixel border removed. Account for this later.
	# account for _dt?
	dT_dt = _kd * laplace_T + _A * S[1:ny-1,1:nx-1] * np.exp(-_B / (T[1:ny-1,1:nx-1] - _Tal)) - _A * _C * (T[1:ny-1,1:nx-1] - _Tal)

	print ("Max dT/dt = " + str(np.max(dT_dt)))
	print ("Max T = " + str(np.max(T)))


	T1 = T
	T1[1:ny-1,1:nx-1] = T1[1:ny-1,1:nx-1] + dT_dt

	dS_dt = - _Cs * S[1:ny-1,1:nx-1] * np.exp(-_B / (T[1:ny-1,1:nx-1] - _Tal))

	S1 = S
	S1[1:ny-1,1:nx-1] = S1[1:ny-1,1:nx-1] + dS_dt
	
	return (T1,S1)

# The initial fraction of the forest occupied by trees.
forest_fraction = 0.4
# Probability of new tree growth per empty cell, and of lightning strike.
p, f = 0.05, 0.00001
# Forest size (number of cells in x and y directions).
nx, ny = 200, 200
# Initialize the forest grid.
S  = np.zeros((ny, nx))
S[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
S[1:ny-1, 1:nx-1] = np.random.random(size=(ny-2, nx-2)) < forest_fraction
# Initialize the ambient temperature grid.
T  = np.ones((ny, nx)) * _Ta

fig = plt.figure(figsize=(50/3, 12.5))
gs = gridspec.GridSpec(1,2, width_ratios=[1,1])
axS = fig.add_subplot(gs[0,0])
axS.set_axis_off()
#imS = axS.imshow(S, cmap=cmap, norm=norm)#, interpolation='nearest')
imS = axS.imshow(S, cmap='gray', vmin=0, vmax=1)#, interpolation='nearest')
axT = fig.add_subplot(gs[0,1])
axT.set_axis_off()
#imT = axT.imshow(T, cmap=cmap, norm=norm)#, interpolation='nearest')
imT = axT.imshow(T, cmap='jet', vmin=_Tal, vmax=1000)#, interpolation='nearest')


# The animation function: called to produce a frame for each generation.
def animate(i):
	imT.set_data(animate.T)
	imS.set_data(animate.S)
	(animate.T,animate.S) = iterate(animate.T,animate.S)
# Bind our grid to the identifier X in the animate function's namespace.
animate.S = S
animate.T = T

# Interval between frames (ms).
interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, animate, interval=interval)
anim.save('sim.mp4',writer=writer)
plt.show()


def iterateold(T,S):
	"""Iterate the forest according to the forest-fire rules."""

	# The boundary of the forest is always empty, so only consider cells
	# indexed from 1 to nx-2, 1 to ny-2
	(ny, nx) = S.shape
	T1 = np.zeros((ny, nx))
	S1 = np.zeros((ny, nx))
	for ix in range(1,nx-1):
		for iy in range(1,ny-1):
			#if X[iy,ix] == EMPTY and np.random.random() <= p:
			#	X1[iy,ix] = TREE
			if S[iy,ix] == TREE:
				S1[iy,ix] = TREE
				for dx,dy in neighbourhood:
					if T[iy+dy,ix+dx] == FIRE:
						T1[iy,ix] = FIRE
						S1[iy,ix] = EMPTY
						break
				else:
					if np.random.random() <= f:
						T1[iy,ix] = FIRE
	return (T1,S1)
