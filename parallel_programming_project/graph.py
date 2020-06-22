from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import re, math

def loadDataFile(fileName, m="\{'TILE_WIDTH': ([0-9]*), 'BLOCK_SIZE': ([0-9]*)\}"):
	out = []
	with open(fileName) as fp:
		for line in fp:
			lineVals = re.findall(f"{m}, \(\['([0-9]*\.[0-9]*)'",line)
			lineVals2 = re.findall(f"{m}, \(\['([0-9]*\.[0-9]*)', '([0-9]*\.[0-9]*)'",line)
			if len(lineVals2) > 0:
				out.append(lineVals2[0])
			elif len(lineVals) > 0:
				out.append(lineVals[0] + (np.nan,))
	return out

def graph3D(fileName):
	out = loadDataFile(fileName)
	data = np.array(out)
	tile_widths = data[:,0]
	block_sizes = data[:,1]

	# x = np.geomspace(1, 128, num=8, dtype=int)
	# y = np.geomspace(1, 4096, num=13, dtype=int)
	x = np.linspace(0, 7, num=8)
	y = np.linspace(0, 13, num=14)
	X,Y = np.meshgrid(x, y)

	data_optime0 = np.ones((len(x),len(y)))*np.nan
	data_optime1 = np.ones((len(x),len(y)))*np.nan
	for d in data:
		tw = int(math.log2(float(d[0])))
		bs = int(math.log2(float(d[1])))
		data_optime0[tw, bs] = float(d[2])
		data_optime1[tw, bs] = float(d[3])
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	data_optime0 = np.log2(data_optime0)
	data_optime1 = np.log2(data_optime1)

	surf = ax.plot_surface(X, Y, data_optime1.T, rstride=1, cstride=1,
	            cmap='viridis', edgecolor='grey')
	ax.set_xlabel('log2(TILE_WIDTH)')
	# ax.xaxis._set_scale('log')
	ax.set_ylabel('log2(BLOCK_SIZE)')
	# ax.yaxis._set_scale('log')
	ax.set_zlabel('log2(optime1)')

	fig.colorbar(surf, shrink=0.5, aspect=5)
	return plt

# def graph2D_multi(fileName):
# 	out = loadDataFile(fileName)
# 	data = np.array(out)
# 	tile_widths = data[:,0]
# 	block_sizes = data[:,1]

# 	# x = np.geomspace(1, 128, num=8, dtype=int)
# 	# y = np.geomspace(1, 4096, num=13, dtype=int)
# 	x = np.linspace(0, 7, num=8)
# 	y = np.linspace(0, 13, num=14)
# 	X,Y = np.meshgrid(x, y)

# 	data_optime0 = np.ones((len(x),len(y)))*np.nan
# 	data_optime1 = np.ones((len(x),len(y)))*np.nan
# 	for d in data:
# 		tw = int(math.log2(float(d[0])))
# 		bs = int(math.log2(float(d[1])))
# 		data_optime0[tw, bs] = float(d[2])
# 		data_optime1[tw, bs] = float(d[3])
# 	fig = plt.figure()
# 	ax = plt.axes(projection='3d')

# 	data_optime0 = np.log2(data_optime0)
# 	data_optime1 = np.log2(data_optime1)

# 	for i in x:
# 		plt.plot(y, data_optime1[x,:])
# 	plt.ylabel('time')
# 	plt.xlabel('tile_width')

# 	return plt

def graph2D(fileName):
	out = loadDataFile(fileName, m="\{'TILE_WIDTH': ([0-9]*)\}")
	data = np.array(out)
	tile_widths = data[:,0]

	# x = np.geomspace(1, 128, num=8, dtype=int)
	# y = np.geomspace(1, 4096, num=13, dtype=int)

	data_optime0 = np.ones(len(tile_widths))*np.nan
	data_optime1 = np.ones(len(tile_widths))*np.nan
	for d in data:
		tw = int(math.log2(float(d[0])))
		data_optime0[tw] = float(d[1])
		data_optime1[tw] = float(d[2])
	fig = plt.figure()

	# data_optime0 = np.log2(data_optime0)
	# data_optime1 = np.log2(data_optime1)

	plt.plot(tile_widths, data_optime1)
	plt.ylabel('time')
	plt.xlabel('tile_width')

	return plt
plt = graph3D('./optimization1.txt')
# plt = graph2D('./optimization2.txt')
plt.show()
# x = np.linspace(0, 9, 10)
# y = np.linspace(0, 9, 10)
# X, Y = np.meshgrid(x, y)

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, s, rstride=1, cstride=1,
#             cmap='viridis', edgecolor='none')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# return plt