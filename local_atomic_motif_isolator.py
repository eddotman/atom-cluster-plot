#Written by Edward Kim

#Script for plotting clusters of atoms
#Intended to work with output from RMC++

#Import necessary libraries
from numpy import *
from local_atomic_motif import LAM
from periodic_kdtree import PeriodicCKDTree


class LAMisolator(): 

	def __init__():
		self.LAMs = empty()

	#Imports a text (data) file
	def load_file (file_name, start_row = 0):
		return loadtxt(file_name, skiprows = start_row)
		
	#Saves a files
	def save_file (location, file, format):
		savetxt(location, file, fmt=format)
		
	#Saves nearest neighbours
	def compute_nn_list (file_name, start_row, num_nn):
		f1 = load_file(file_name, start_row)
		bounds = array([1.999999999, 1.999999999, 1.999999999]) #Can't do exactly 2.0 because rounding
		atom_list = PeriodicCKDTree(bounds, f1)
		nn = atom_list.query(f1, k=num_nn)
		save_file("nn_list", nn[1], "%.6d")

	#Default settings are for a-Si WWW model
	def plot_LAMs(cfg_file, cfg_start_row, num_atoms, cluster_coord=4, nn_file = "nn_list", box_length=62.024362):

		#Load file
		f_cfg = load_file(cfg_file, cfg_start_row)
		
		#Resize box and reorient
		f_cfg = dot(box_length, f_cfg) 
		f_cfg = vstack((arange(num_atoms), transpose(f_cfg)))
		f_cfg = transpose(f_cfg)

		#Plot clusters within model using NNs
		f_nn = load_file(nn_file)

		#Holds cluster coords
		clusters = empty((0,4))

		#Plot single-clusters
		for x in arange(f_nn.shape[0]):
			
			cluster = empty((cluster_coord + 1,4))
			
			for y in arange(cluster_coord + 1):
				cluster[y] = f_cfg[f_nn[x][y]]
				
			#Plot anisotropic clusters if desired
			plot_aniso = False

			if plot_aniso is True:
				#Look for NNs of the NN-1 (which will be fixed to z-axis)
				nn_list = f_nn[cluster[1][0]]
				for y in arange(1, 2*cluster_coord):
					if (nn_list[y] != cluster[0][0]) and (nn_list[y] != cluster[1][0]) and (nn_list[y] != cluster[2][0]) and (nn_list[y] != cluster[3][0]) and (nn_list[y] != cluster[4][0]) and (cluster.shape[0] < 2*cluster_coord):
						cluster = vstack((cluster, f_cfg[nn_list[y]]))
								
					#number of atoms in cluster
					cluster_num = 2 * cluster_coord
				else:
					cluster_num = cluster_coord + 1
					
				#check for PBC
				L = box_length #half-dimension of box
				for y in arange(1, cluster_num):
					for z in arange(1, 4):
						if abs(cluster[y][z]-cluster[0][z]) > L:
							cluster[y][z] -= sign(cluster[y][z]-cluster[0][z])*L*2