#Written by Edward Kim
#Script for plotting local atomic motifs of atoms

#Import necessary libraries
from numpy import *
from local_atomic_motif import LAM
from mayavi import mlab
import scipy.ndimage as ndi

class LAMplotter:

	def __init__(self, name):
		print "Plotter created!"
		self.name = name

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

	def save_LAM_density(self, readfile):
		print "Loading LAMs from file..."
		LAMraw = self.load_file(readfile)

		print "Computing gaussian filter..."
		self.LAMgauss = self.grid_density_gaussian_filter(-10, -10, -10, 10, 10, 10, 256, 256, 256, LAMraw[:,1:4])

		self.save_file("build/" + self.name + "_gauss_f", self.LAMgauss)
		print "Gaussian filter saved!"

	def plot_LAM_contours(self, readfile):
	 	print "Reading file..."
	 	cluster_gauss = self.load_data(readfile)

	 	print "Plotting..."
	 	print cluster_gauss.shape
		mlab.contour3d(cluster_gauss, contours = 70, opacity=0.10)
		mlab.show()

	def full_compute(self):
		self.save_LAM_density("build/" + self.name + "_LAMs")
		self.plot_LAM_contours("build/" + self.name + "_gauss_f")


if __name__ == "__main__":
	plot1 = LAMplotter("asimp")
	plot1.full_compute()