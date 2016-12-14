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
_A = 1.8793e2 # K s^-1 # was e2
_B = 5.5849e2 # K
#_C = 4.8372e-5 # K^-1
_C = 4.8372e-5 # K^-1
_Cs = 0.1625 # s^-1
_lambda = 0.027
_beta = 0.4829
_Ta = 25 + 273.15 #373.15
_Tal = _Ta - 0.01
_dx = 1 # metres
_dy = 1 # metres
_dt = 1 # 10 minutes
_kd = _dt *_k/(_dx*_dy)
LIGHTNING = 800 + 273.15 # Kelvin
# The initial fraction of the forest occupied by trees.
forest_fraction = 0.95
# Probability of new tree growth per empty cell, and of lightning strike.
p, f = 0.05, 0.000001
# Forest size (number of cells in x and y directions).
nx, ny = 6000, 6000
# Interval between frames (ms).
interval = 0.1

gausskernel5x5 = np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
		[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
		[0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
		[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
		[0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])

gausskernel7x7 = np.array([
		[0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036],
		[0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
		[0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
		[0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291],
		[0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
		[0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
		[0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036]])

def iterate(T,S):
	"""Iterate the forest according to the forest-fire rules."""
	# The boundary of the forest is always empty, so only consider cells
	# indexed from 1 to nx-2, 1 to ny-2
	(ny, nx) = S.shape
	T1 = np.ones((ny, nx)) * _Ta
	S1 = np.zeros((ny, nx))

	#for ix in range(3,nx-3):
	#	for iy in range(3,ny-3):
	#		if np.random.random() <= f:
	#			T[iy-3:iy+4,ix-3:ix+4] += LIGHTNING * gausskernel7x7 / 0.146768
				#T[iy,ix] += LIGHTNING
				#T[iy-1,ix] += LIGHTNING * 
				#T[iy+1,ix] += LIGHTNING
				#T[iy,ix-1] += LIGHTNING
				#T[iy,ix+1] += LIGHTNING
				#T[iy-2,ix] += LIGHTNING
				#T[iy+2,ix] += LIGHTNING
				#T[iy,ix-2] += LIGHTNING
				#T[iy,ix+2] += LIGHTNING
				#T[iy-1,ix-1] += LIGHTNING
				#T[iy+1,ix-1] += LIGHTNING
				#T[iy+1,ix+1] += LIGHTNING
				#T[iy-1,ix+1] += LIGHTNING

	ks = 3 # kernel size = 2 * ks + 1
	if np.random.random() <= f * (nx - 2*ks) * (ny  - 2*ks):
		ix = np.random.randint(ks,nx-ks)
		iy = np.random.randint(ks,ny-ks)
		T[iy-3:iy+4,ix-3:ix+4] += LIGHTNING * gausskernel7x7 / 0.146768
		

	#laplace_T = (T[1:ny-1,2:nx] - 2 * T[1:ny-1,1:nx-1] + T[1:ny-1,0:nx-2]) / (_dx ** 2) + (T[2:ny,1:nx-1] - 2 * T[1:ny-1,1:nx-1] + T[0:ny-2,1:nx-1]) / (_dy ** 2)
	laplace_T = np.zeros((ny-2,nx-2))
	ndimage.filters.laplace(T[1:ny-1,1:nx-1],output=laplace_T,mode='constant',cval=_Ta)
	# assume no wind for now
	# dT_dt has the 1-pixel border removed. Account for this later.
	# account for _dt?
	dT_dt = _kd * laplace_T + _dt * _A * S[1:ny-1,1:nx-1] * np.exp(-_B / (T[1:ny-1,1:nx-1] - _Tal)) - _dt * _A * _C * (T[1:ny-1,1:nx-1] - _Tal)

	#print ("Max dT/dt = " + str(np.max(dT_dt)))
	#print ("Max T = " + str(np.max(T)))


	T1 = T
	T1[1:ny-1,1:nx-1] = T1[1:ny-1,1:nx-1] + dT_dt

	dS_dt = - _dt * _Cs * S[1:ny-1,1:nx-1] * np.exp(-_B / (T[1:ny-1,1:nx-1] - _Tal))

	S1 = S
	S1[1:ny-1,1:nx-1] = S1[1:ny-1,1:nx-1] + dS_dt
	
	return (T1,S1)

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
ttl = axS.text(.5, 1.05, '', transform = axS.transAxes, va='center')


# The animation function: called to produce a frame for each generation.
def animate(i):
	imT.set_data(animate.T)
	imS.set_data(animate.S)
	ttl.set_text('Iteration: %d (1 simulated second per iteration)'%(animate.Iter))
	animate.Iter += 1
	(animate.T,animate.S) = iterate(animate.T,animate.S)
# Bind our grid to the identifier X in the animate function's namespace.
animate.S = S
animate.T = T
animate.Iter = 0
plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
anim = animation.FuncAnimation(fig, animate, interval=interval)
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
