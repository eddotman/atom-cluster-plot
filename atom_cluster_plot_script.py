#Written by Edward Kim

#Script for plotting clusters of atoms
#Intended to work with output from RMC++


#Import necessary libraries
from numpy import *
from scipy import stats
import scipy.ndimage as ndi
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from periodic_kdtree import PeriodicCKDTree


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

#Makes a 3D scatterplot
def plot_3d_scatter (x, y, z, x_lbl, y_lbl, z_lbl, title, xlb = -1, xub = 1, ylb = -1, yub = 1, zlb = -1, zub = 1, sz=1):
	fig = plt.figure()
	axes = Axes3D(fig)
	
	axes.set_autoscale_on(False)
	axes.set_xbound(xlb, xub)
	axes.set_ybound(ylb, yub)
	axes.set_zbound(zlb, zub)
	
	axes.scatter(x, y, z, s=sz)
	axes.set_xlabel(x_lbl)
	axes.set_ylabel(y_lbl)
	axes.set_zlabel(z_lbl)
	axes.set_title(title)
	fig.add_axes(axes)
	plt.show()

#Plots larger 3d points	
def plot_3d_glyph (x, y, z, f=0, op = .7, sz = 1.0, m="glyph", c=(1,1,1)):
	if m is "glyph":
		mlab.points3d(x, y, z, opacity=op, scale_factor=sz)
	elif m is "quiver":
		mlab.quiver3d(x, y, z, f, opacity=op, scale_factor=sz)
	elif m is "point":
		mlab.points3d(x, y, z, opacity=op, scale_factor=sz, mode="point", color=c)
	
	mlab.show()
	
#Makes a 2D scatterplot
def plot_2d_scatter (x, y, x_lbl, y_lbl, title):
	fig = plt.figure()
	plt.plot(x, y, "g")
	plt.xlabel(x_lbl)
	plt.ylabel(y_lbl)
	plt.title(title)
	plt.show()

#returns a cartesian flattened 2D slice of a 3D wedge
def compute_3d_wedge (data, r_range, theta_range, phi_range):
	wedge_points = empty((0, 2))

	for x in data:
		x = sph_coords(x)

		if x[0] > r_range[0] and x[0] < r_range[1] and x[1] > theta_range[0] and x[1] < theta_range[1] and x[2] > phi_range[0] and x[2] < phi_range[1]:
			x_crt = (x[0]*cos(x[2]), x[0]*sin(x[2])) #force to xy-plane
			wedge_points = vstack((wedge_points, x_crt))

	return wedge_points

def plot_histogram (x, b, x_lbl, y_lbl, title):
	fig = plt.figure()
	plt.hist(x, bins = b, histtype="stepfilled", color="k")
	plt.xlabel(x_lbl)
	plt.ylabel(y_lbl)
	plt.title(title)
	plt.show()
	
def save_histogram (filename, data, r=None, b=100):
	hist_dat = histogram(data, bins=b, range=r)
	hist_dat = transpose(vstack((hist_dat[1], append(hist_dat[0], 0))))
	savetxt(filename, hist_dat)

def compute_3d_kde (data):
	kde = stats.gaussian_kde(data)
	return kde

def grid_density_gaussian_filter(x0, y0, z0, x1, y1, z1, w, h, l, data):
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
            img[iy][ix][iz] += 1
    return ndi.gaussian_filter(img, (r,r,r))  #gaussian convolution
	
def plot_2d_histogram (x, y, b):
	return histogram2d(x, y, bins = b, range=[[-4,4],[-4,4]])
	
#Plots two lines on same graph
def plot_2d_line_comparison (x1, y1, x2, y2, x_lbl, y_lbl, lbl1, lbl2, title):
	fig = plt.figure()
	plt.plot(x1, y1, "b", label = lbl1)
	plt.plot(x2, y2, "r", label = lbl2)
	
	#Compute and plot error using root(N)
	err1top = empty(y1.shape[0])
	err1bot = empty(y1.shape[0])
	err2top = empty(y2.shape[0])
	err2bot = empty(y2.shape[0])
	
	for x in arange(y1.shape[0]):
		err1top[x] = y1[x] + sqrt(y1[x])
		err1bot[x] = y1[x] - sqrt(y1[x])
	
	for x in arange(y2.shape[0]):
		err2top[x] = y2[x] + sqrt(y2[x])
		err2bot[x] = y2[x] - sqrt(y2[x])
	
	plt.plot(x1, err1top, "k:")
	plt.plot(x1, err1bot, "k:")
	plt.plot(x2, err2top, "k:")
	plt.plot(x2, err2bot, "k:", label = "Error")
	
	plt.xlabel(x_lbl)
	plt.ylabel(y_lbl)
	plt.title(title)
	plt.legend() 

	plt.show()	
	
#Plots three lines on same graph
def plot_three_line_comparison (x1, y1, x2, y2, x3, y3, x_lbl, y_lbl, lbl1, lbl2, lbl3, title):
	fig = plt.figure()
	p1 = plt.plot(x1, y1, "b", label = lbl1)
	p2 = plt.plot(x2, y2, "r", label = lbl2)
	p2 = plt.plot(x3, y3, "g", label = lbl3)
	plt.xlabel(x_lbl)
	plt.ylabel(y_lbl)
	plt.title(title)
	plt.legend() 
	
	plt.show()	

#Visualizes 2D array data
def plot_2d_image (arr2d, title):
	plt.imshow (arr2d, interpolation='nearest') #optional grayscale: cmap=plt.cm.gray
	plt.title(title)
	plt.colorbar()
	plt.show()

#Quaternion
def rotation_matrix (axis,theta):
    axis = axis/sqrt(dot(axis,axis))
    a = cos(theta/2)
    b,c,d = -axis*sin(theta/2)
    return array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

#turns cartesian to spherical			
def sph_coords (coords): 
	rad = sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2)
	if rad != 0:
		theta = arccos(coords[2]/rad)
	else:
		theta = 0

	phi = arctan2(coords[1],coords[0]) #handles infinity cases
	if phi < 0:
		phi += 2*pi #return in [0,2pi] range
	
	return (rad, theta, phi)

#turns spherical into cartesian	
def crt_coords (coords):
	crd_x = coords[0]*sin(coords[1])*cos(coords[2])
	crd_y = coords[0]*sin(coords[1])*sin(coords[2])
	crd_z = coords[0]*cos(coords[1])
	
	return (crd_x, crd_y, crd_z)

#Computes angle (in degrees) between two cartesian points, using p2 as centre
def compute_angle (p1, p2, p3):
	v1 = p1-p2
	v2 = p3-p2
	
	if linalg.norm(v1) == 0 or linalg.norm(v2) == 0:
		return 0
	
	return (180.0/pi)*arccos(dot(v1, v2)/(linalg.norm(v1)*linalg.norm(v2)))

#computes vector from a point and origin
def compute_vector (x, y, z, o=(0,0,0)):
	return (x-o[0], y-o[1], z-o[2])

#computes the radial distribution function of a set of xyz points	
def compute_rdf (centres, points, min_dist=0):
	dists = empty(0)
	for x in centres:
		for y in points:
			dist = linalg.norm(x - y)
			if dist > min_dist:
				dists = append(dists, dist)
	return dists
	
#Computes density histograms
def compute_grid (grid_size, grid_file, start_row):
	grid2d = zeros((grid_size, grid_size)) #grid size; filled with zeros
	grid3d = zeros((grid_size, grid_size, grid_size))
	f4 = load_file (grid_file, start_row)

	for x in f4:
	    grid2d[x[0]][x[1]] += 1	
	    grid3d[x[0]][x[1]][x[2]] += 1

	plot_2d_image (grid2d, "Z-Compressed Density Map")
	#mlab.contour3d(grid3d, "3D Density Isosurface Map")
	#mlab.show()
	mlab.contour_surf(grid2d, warp_scale = 0.05)
	mlab.show()
	
#Compute coordination number from g(r)
def compute_coord_num (data, rho, lb, ub):
	return 4.0*pi*trapz(multiply(data[lb:ub,1],multiply(data[lb:ub,0],data[lb:ub,0]))*rho,x=data[lb:ub,0],dx=0.001)
	
#Compute 3D clusters of atoms in an overlay and save to file
def compute_clusters (cfg_file, cfg_start_row, num_atoms, cluster_coord=4, nn_file = "nn_list", box_length=62.024362):
	
	#Load file
	f_cfg = load_file(cfg_file, cfg_start_row)
	
	#Resize box and reorient
	f_cfg = dot(box_length, f_cfg) 
	f_cfg = vstack((arange(num_atoms), transpose(f_cfg)))
	f_cfg = transpose(f_cfg)

	#Plot clusters within model using NNs
	f_nn = load_file(nn_file)

	#Max clusters to plot
	cluster_count = 0
	cluster_max = num_atoms

	#Holds cluster coords
	clusters = empty((0,4))

	#Holds single clusters
	sing_clusters = empty((0,4))

	#Plot single-clusters
	for x in arange(f_nn.shape[0]):
		
		cluster_count += 1
		
		if cluster_count is cluster_max: #max clusters to plot
			break
			
		cluster = empty((cluster_coord + 1,4))
		
		for y in arange(cluster_coord + 1):
			cluster[y] = f_cfg[f_nn[x][y]]
			
		#Plot double-clusters if desired
		plot_double = False
		
		if plot_double is True:
			#Look for NNs of the NN-1 (which will be fixed to z-axis)
			nn_list = f_nn[cluster[1][0]]
			for y in arange(1, 2*cluster_coord):
				if (nn_list[y] != cluster[0][0]) and (nn_list[y] != cluster[1][0]) and (nn_list[y] != cluster[2][0]) and (nn_list[y] != cluster[3][0]) and (nn_list[y] != cluster[4][0]) and (cluster.shape[0] < 2*cluster_coord):
					cluster = vstack((cluster, f_cfg[nn_list[y]]))
					
			#number of atoms in cluster
			cluster_num = 2*cluster_coord
		else:
			cluster_num = cluster_coord+1
			
		#check for PBC
		L = box_length #half-dimension of box
		for y in arange(1, cluster_num):
			for z in arange(1, 4):
				if abs(cluster[y][z]-cluster[0][z]) > L:
					cluster[y][z] -= sign(cluster[y][z]-cluster[0][z])*L*2

		#translate cluster to origin
		origin = cluster[0]
		
		#shift to origin
		for y in arange(1, cluster_num):
			cluster[y][1] -= origin[1]
			cluster[y][2] -= origin[2]
			cluster[y][3] -= origin[3]
		
		cluster[0] = [origin[0], 0, 0, 0]
	
		#spherical transformation
		for y in arange(cluster_num):
			cluster[y][1],cluster[y][2],cluster[y][3] = sph_coords(cluster[y, 1:4])
		
		#fix one of the atoms to the z-axis
		theta_adjust = cluster[1][2]
		
		#transform back to cartesian
		for y in arange(cluster_num):
			cluster[y][1],cluster[y][2],cluster[y][3] = crt_coords(cluster[y, 1:4])
			
		#Find rotation axis using cross product
		rot_axis = cross([0,0,1], cluster[1, 1:4])
		
		#Apply quaternion transform
		for y in arange(cluster_num):
			cluster[y][1],cluster[y][2],cluster[y][3] = dot(rotation_matrix(rot_axis,theta_adjust),cluster[y, 1:4])
		
		#spherical transformation
		for y in arange(cluster_num):
			cluster[y][1],cluster[y][2],cluster[y][3] = sph_coords(cluster[y, 1:4])
		
		#fix one atom to the xz plane
		phi_adjust = cluster[2][3]
		
		#rotate cluster
		for y in arange(cluster_num):
			cluster[y][3] -= phi_adjust
		
		#transform back to cartesian
		for y in arange(cluster_num):
			cluster[y][1],cluster[y][2],cluster[y][3] = crt_coords(cluster[y, 1:4])
		
		#save clusters information
		clusters = vstack((clusters, cluster))
		sing_clusters = vstack((sing_clusters, cluster[0:5]))
			
	#Save clusters to files
	savetxt("single_clusters", sing_clusters)
	savetxt("double_clusters", clusters)
	
#Analyzes clusters
def analyze_clusters (cluster_num = 8, file="double_clusters"):

	#load complete clusters
	dbl_file = load_file(file)
	#sing_file = load_file("single_clusters")

	#Holds z-atom distances
	z_dists = empty(0)

	#Holds xz-atom angles
	xz_angles = empty(0)

	#Dihedral angle from file
	dih_angles = empty(0)

	#plane angles
	plane_angles = empty(0)

	#dihedral atoms
	dih_atoms = empty((0,4))

	#lower atoms
	lower_atoms = empty((0,4))

	#atom2 distances
	atom2_dists = empty(0)
	
	#atom4 distances
	atom4_dists = empty(0)
	
	#dihedral distances
	dih_dists = empty(0)
	
	#1-2, 1-3, and 1-4 distances
	atom1_dists = empty(0)
	
	#2-567, 3-567, and 4-567 distances
	atom234_dists = empty(0)

	counter = 0
	cluster = empty((0,4))

	for x in dbl_file:
		if counter == cluster_num:
			
			#Compute dihedral atoms
			dih_atoms = vstack((dih_atoms, cluster[5], cluster[6], cluster[7]))
			
			#compute dihedral distances
			dih_dists = append(dih_dists, (linalg.norm(cluster[5,1:4]), linalg.norm(cluster[6,1:4]), linalg.norm(cluster[7,1:4])))
			
			#Compute dihedral angles in spherical
			dih_angles = append(dih_angles, ((180/pi)*sph_coords(cluster[5,1:4])[2], (180/pi)*sph_coords(cluster[6,1:4])[2], (180/pi)*sph_coords(cluster[7,1:4])[2]))
			
			#Compute xz angle
			xz_angles =  append(xz_angles, compute_angle(cluster[1, 1:4], cluster[0, 1:4], cluster[2, 1:4]))
			
			#compute distribution of atom-1 distance
			z_dists = append(z_dists, cluster[1][3])
			
			#compute atom 2 distance
			atom2_dists = append(atom2_dists, linalg.norm(cluster[2][1:4]))
			
			#compute atom 4 distance
			atom4_dists = append(atom4_dists, linalg.norm(cluster[4][1:4]))
			
			#compute 1-2, 1-3, and 1-4 distances
			atom1_dists = append(atom1_dists, (linalg.norm(compute_vector(cluster[1][1], cluster[1][2], cluster[1][3], o=cluster[2,1:4])), linalg.norm(compute_vector(cluster[1][1], cluster[1][2], cluster[1][3], o=cluster[3,1:4])), linalg.norm(compute_vector(cluster[1][1], cluster[1][2], cluster[1][3], o=cluster[4,1:4]))))
			
			#compute 2-567, 3-567, and 4-567 distances
			atom234_dists = append(atom234_dists, (linalg.norm(compute_vector(cluster[2][1], cluster[2][2], cluster[2][3], o=cluster[5,1:4])), linalg.norm(compute_vector(cluster[2][1], cluster[2][2], cluster[2][3], o=cluster[6,1:4])), linalg.norm(compute_vector(cluster[2][1], cluster[2][2], cluster[2][3], o=cluster[7,1:4])), linalg.norm(compute_vector(cluster[3][1], cluster[3][2], cluster[3][3], o=cluster[5,1:4])), linalg.norm(compute_vector(cluster[3][1], cluster[3][2], cluster[3][3], o=cluster[6,1:4])), linalg.norm(compute_vector(cluster[3][1], cluster[3][2], cluster[3][3], o=cluster[7,1:4])), linalg.norm(compute_vector(cluster[4][1], cluster[4][2], cluster[4][3], o=cluster[5,1:4])), linalg.norm(compute_vector(cluster[4][1], cluster[4][2], cluster[4][3], o=cluster[6,1:4])), linalg.norm(compute_vector(cluster[4][1], cluster[4][2], cluster[4][3], o=cluster[7,1:4]))))
			
			#compute the plane normal angle
			basis1a = compute_vector(cluster[2][1], cluster[2][2], cluster[2][3], o=cluster[3, 1:4])
			basis1b = compute_vector(cluster[4][1], cluster[4][2], cluster[4][3], o=cluster[3, 1:4])
			axis1 = cross(basis1a, basis1b)
			
			basis2a = compute_vector(cluster[5][1], cluster[5][2], cluster[5][3], o=cluster[6, 1:4])
			basis2b = compute_vector(cluster[7][1], cluster[7][2], cluster[7][3], o=cluster[6, 1:4])
			axis2 = cross(basis2a, basis2b)
			
			axis_angle = compute_angle(axis1, (0,0,0), axis2)
			
			if axis_angle > 90.0:
				axis_angle = 180.0 - axis_angle
			
			plane_angles = append(plane_angles, axis_angle)
			
			cluster = empty((0,4))	
			counter = 0
			
		cluster = vstack((cluster, x))	
		counter += 1
		
	return (dih_atoms, dih_angles, xz_angles, z_dists, atom1_dists, atom2_dists, atom4_dists, atom234_dists, plane_angles, dih_dists)
