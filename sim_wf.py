import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from cylaplace_9pts import cy_laplacian
import numexpr as ne
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
_dx = 5 # metres
_dy = 5 # metres
_dt = 15 # seconds 
#_kd = _dt *_k/(_dx*_dy)
_kd = _dt *_k 
LIGHTNING = 175 # + 273.15 # Kelvin
# The initial fraction of the forest occupied by trees.
forest_fraction = 0.7 
# Probability of new tree growth per empty cell, and of lightning strike.
p, f = 0.05, 0.000001
# Forest size (number of cells in x and y directions).
nx, ny = 1200, 1200
# Interval between frames (ms).
interval = 1

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

gausskernel25x25 = np.loadtxt('gk25.txt')
gausskernel13x13 = np.loadtxt('gk13.txt')
gausskernel43x43 = np.loadtxt('gk43.txt')


def iterate(T,S):
	"""Iterate the forest according to the forest-fire rules."""
	(ny, nx) = S.shape
	#laplace_T = (T[1:ny-1,2:nx] - 2 * T[1:ny-1,1:nx-1] + T[1:ny-1,0:nx-2]) / (_dx ** 2) + (T[2:ny,1:nx-1] - 2 * T[1:ny-1,1:nx-1] + T[0:ny-2,1:nx-1]) / (_dy ** 2)
	#[grad_x_T,grad_y_T] = np.gradient(T)
	# assume no wind for now
	#SexpB_T = S * np.exp(-_B / (T - _Tal))
	#laplace_T *= (1/(_dx**2))
	#dT_dt = _kd * laplace_T + _dt * _A * SexpB_T - _dt * _A * _C * (T - _Tal)
	#T_dT_dt = ne.evaluate('T + (_kd * laplace_T + 0.02 * grad_x_T + 0.01 * grad_y_T + _dt * _A * SexpB_T - _dt * _A * _C * (T - _Tal))')
	#print ("Max dT/dt = " + str(np.max(dT_dt)))
	#print ("Max T = " + str(np.max(T)))
	#dS_dt = - _dt * _Cs * SexpB_T

	laplace_T = np.zeros((ny,nx))
	#ndimage.filters.laplace(T,output=laplace_T,mode='constant',cval=_Ta)
	#laplace_T /= _dx**2
	cy_laplacian(T,laplace_T,_dx**2,_Ta)
	#cy_laplacian(T,laplace_T,_dx,_Ta)

	SexpB_T = ne.evaluate('S * exp(-_B / (T - _Tal))')
	T_dT_dt = ne.evaluate('T + (_kd * laplace_T +  _dt * _A * SexpB_T - _dt * _A * _C * (T - _Tal))')
	S_dS_dt = ne.evaluate('S - _dt * _Cs * SexpB_T')

	return (T_dT_dt,S_dS_dt)

# Initialize the forest grid.
S  = np.zeros((ny, nx))
S[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
S[1:ny-1, 1:nx-1] = np.random.random(size=(ny-2, nx-2)) < forest_fraction
# Initialize the ambient temperature grid.
T  = np.ones((ny, nx)) * _Ta

ks = 21 # kernel size = 2 * ks + 1
#if np.random.random() <= f * (nx - 2*ks) * (ny  - 2*ks):
ix = np.random.randint(ks,nx-ks)
iy = np.random.randint(ks,ny-ks)
T[iy-ks:iy+ks+1,ix-ks:ix+ks+1] += LIGHTNING * gausskernel43x43 / np.amax(gausskernel43x43)
		

fig = plt.figure(figsize=(50/3, 12.5))
gs = gridspec.GridSpec(1,2, width_ratios=[1,1])
axS = fig.add_subplot(gs[0,0])
axS.set_axis_off()
#imS = axS.imshow(S, cmap=cmap, norm=norm)#, interpolation='nearest')
imS = axS.imshow(S, cmap='gray', vmin=0, vmax=1)#, interpolation='nearest')
axT = fig.add_subplot(gs[0,1])
axT.set_axis_off()
#imT = axT.imshow(T, cmap=cmap, norm=norm)#, interpolation='nearest')
imT = axT.imshow(T, cmap='jet', vmin=_Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
ttl = axS.text(.5, 1.05, '', transform = axS.transAxes, va='center')


# The animation function: called to produce a frame for each generation.
def animate(i):
	imT.set_data(animate.T)
	imS.set_data(animate.S)
	ttl.set_text('Iteration: %d (%d simulated second per iteration)'%(animate.Iter,_dt))
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
