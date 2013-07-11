#Written by Edward Kim
#Script for plotting local atomic motifs of atoms

#Import necessary libraries
from numpy import *
from local_atomic_motif import LAM
from mayavi import mlab
import scipy.ndimage as ndi

class LAMplotter:

	def __init__(self, name):
		self.name = name
		self.LAMgauss = None
		print "Plotter created with name: " + self.name

	#Imports a text (data) file
	def load_file (self, file_name, start_row = 0):
		return loadtxt(file_name, skiprows = start_row)

	#Loads array data
	def load_data (self, readfile):
		return load(readfile + ".npy")
		
	#Saves an array file
	def save_file (self, location, data):
		save(location, data)

	def grid_density_gaussian_filter(self, x0, y0, z0, x1, y1, z1, w, h, l, data):
		#Modified from: http://stackoverflow.com/questions/6652671/efficient-method-of-calculating-density-of-irregularly-spaced-points
	    kx = (w - 1) / (x1 - x0)
	    ky = (h - 1) / (y1 - y0)
	    kz = (l - 1) / (z1 - z0)
	    r = 2.5
	    border = r
	    imgw = (w + 2 * border)
	    imgh = (h + 2 * border)
	    imgl = (l + 2 * border)
	    img = zeros((imgh,imgw,imgl))
	    for x, y, z in data:
	        ix = int((x - x0) * kx) + border
	        iy = int((y - y0) * ky) + border
	        iz = int((z - z0) * kz) + border
	        if 0 <= ix < imgw and 0 <= iy < imgh and 0 <= iz < imgl:
	            img[ix][iy][iz] += 1
	    return ndi.gaussian_filter(img, (r,r,r))  #gaussian convolution

	def load_LAMs(self, readfile):
		print "Loading LAMs from file..."
		self.LAMraw = self.load_file(readfile)

	def save_LAM_density(self):
		print "Computing gaussian filter..."
		self.LAMgauss = self.grid_density_gaussian_filter(-10, -10, -10, 10, 10, 10, 256, 256, 256, self.LAMraw[:,1:4])

		self.save_file("build/" + self.name + "_gauss_f", self.LAMgauss)
		print "Gaussian filter saved!"

	def plot_LAM_contours(self, readfile):
		if self.LAMgauss == None:
	 		print "Reading file..."
	 		self.LAMgauss = self.load_data(readfile)

	 	print "Plotting..."
		mlab.contour3d(self.LAMgauss, contours = 75, opacity=0.10)
		mlab.show()

	#Only include LAMs which have atom #<atom_id> in a specified radial range 
	def radial_filter(self, r_min, r_max, atom_id, num_atoms):
		new_LAMraw = empty((0, 4))

		print "Radially filtering LAMs..."

		i = atom_id

		while i < self.LAMraw.shape[0]:
			if r_min < linalg.norm(self.LAMraw[i,1:4]) < r_max:
				new_LAMraw = vstack((new_LAMraw, self.LAMraw[i-(atom_id):i+(num_atoms-atom_id), :]))
			i += num_atoms

		self.LAMraw = new_LAMraw

	def full_compute(self):
		self.load_LAMs("build/" + self.name + "_LAMs")
		self.radial_filter(2.7, 10, 4, 17)
		self.save_LAM_density()
		self.plot_LAM_contours("build/" + self.name + "_gauss_f")


if __name__ == "__main__":
	plot1 = LAMplotter("rmc_expr_coords")
	plot1.full_compute()