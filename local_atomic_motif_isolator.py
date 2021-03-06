#Written by Edward Kim
#Script for managing local atomic motifs of atoms

#Import necessary libraries
from numpy import *
from local_atomic_motif import LAM
from periodic_kdtree import PeriodicCKDTree


class LAMisolator(): 

	def __init__(self, name=""):
		self.LAMs = empty(0)
		self.LAMatoms = empty((0,4))
		self.name = name
		print "LAMisolator built with name: " + self.name

	#Imports a text (data) file
	def load_file (self, file_name, start_row = 0):
		return loadtxt(file_name, skiprows = start_row)
		
	#Saves a files
	def save_file (self, location, data):
		savetxt(location, data)
		
	#Saves nearest neighbours
	def compute_nn_list (self, file_name, start_row, num_nn):
		f1 = self.load_file(file_name, start_row)

		print "Loaded file: " + file_name
		print "Computing nearest-neighbour list..."

		bounds = array([1.999999999, 1.999999999, 1.999999999]) #Can't do exactly 2.0 because rounding
		atom_list = PeriodicCKDTree(bounds, f1)
		nn = atom_list.query(f1, k=num_nn)
		self.save_file("build/" + file_name + "_nn", nn[1])

		print "Nearest-neighbour list successfully computed!"

	#Default settings are for a-Si WWW model
	def read_LAMs(self, cfg_file, cfg_start_row, num_atoms, cluster_coord, nn_file, box_length=62.024362):

		#Load file
		f_cfg = self.load_file(cfg_file, cfg_start_row)
		
		#Resize box and reorient
		f_cfg = dot(box_length, f_cfg) 
		f_cfg = vstack((arange(num_atoms), transpose(f_cfg)))
		f_cfg = transpose(f_cfg)

		#Plot clusters within model using NNs
		f_nn = self.load_file(nn_file)

		#Holds LAMs
		LAMs = empty(0)

		print "Reading LAMs from file..."

		#Plot LAMs
		for x in arange(f_nn.shape[0]):
			
			cluster = empty((cluster_coord + 1, 4))
			
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

			#Store LAM
			LAM_temp = LAM(cluster_coord, cluster)
			LAMs = append(LAMs, LAM_temp)

		self.LAMs = LAMs
		print "LAMs successfully read from file!"

	def orient_LAMs (self):
		print "Orienting LAMs..."
		for i in arange(self.LAMs.shape[0]):
			self.LAMatoms = vstack((self.LAMatoms, self.LAMs[i].orient_LAM(self.LAMs[i].atoms)))
		print "LAMs successfully oriented!"

	def save_LAMs (self, savefile):
		print "Saving LAMs to file..."
		self.save_file("build/" + savefile + "_LAMs", self.LAMatoms)

		print "LAMs successfully saved!"

	def load_LAMs (self, loadfile, coord_num):
		print "Loading LAMs directly from LAM file..."
		LAM_file = self.load_file(loadfile)
		LAM_temp = empty((0,4))

		c = 0

		for row in LAM_file:
			if c != coord_num:
				LAM_temp = vstack((LAM_temp, row))
			else:
				self.LAMs = append(self.LAMs, LAM(coord_num, LAM_temp))
				LAM_temp = empty((0,4))
				c=-1
			c+=1

	def save_LAMdata (self, savefile):
		LAMdata = empty((0,2))

		print "Analyzing LAMs..."

		for LAM in self.LAMs:
			LAMdata = vstack((LAMdata, LAM.analyze_LAM(LAM.atoms)))

		self.save_file("build/" + savefile + "_LAMdata", LAMdata)

		print "LAM data successfully analyzed and saved!"

	def full_compute (self):
		self.compute_nn_list(self.name, 0, 18)
		self.read_LAMs(self.name, 0, 100000, 16, "build/" + self.name + "_nn") # box_length=65.9893 for germanium
		self.orient_LAMs()
		#self.load_LAMs("build/" + self.name + "_LAMs", 16)
		#self.save_LAMdata(self.name)
		self.save_LAMs(self.name)


if __name__ == "__main__":
	
	iso1 = LAMisolator("rmc_expr_coords")
	iso1.full_compute()
	