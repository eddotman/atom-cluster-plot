#Written by Edward Kim
#Contains the Local Atomic Motif (LAM) class definitions

from numpy import *

class LAM:

	def __init__(self, num_neighbours, atoms):
		
		self.num_atoms = 1 + num_neighbours;
		self.atoms = atoms

	#Quaternion
	def rotation_matrix (self, axis,theta):
	    axis = axis/sqrt(dot(axis,axis))
	    a = cos(theta/2)
	    b,c,d = -axis*sin(theta/2)
	    return array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
	                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
	                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

	#turns cartesian to spherical			
	def sph_coords (self, coords): 
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
	def crt_coords (self, coords):
		crd_x = coords[0]*sin(coords[1])*cos(coords[2])
		crd_y = coords[0]*sin(coords[1])*sin(coords[2])
		crd_z = coords[0]*cos(coords[1])
		
		return (crd_x, crd_y, crd_z)

	#Computes angle (in degrees) between two cartesian points, using p2 as centre
	def compute_angle (self, p1, p2, p3):
		v1 = p1-p2
		v2 = p3-p2
		
		if linalg.norm(v1) == 0 or linalg.norm(v2) == 0:
			return 0
		
		return (180.0/pi)*arccos(dot(v1, v2)/(linalg.norm(v1)*linalg.norm(v2)))

	#computes vector from a point and origin
	def compute_vector (self, (x, y, z), o=(0,0,0)):
		return (x-o[0], y-o[1], z-o[2])
		
	#Orients the LAM at the origin by pinning 3 atoms
	#Defaults set for WWW model a-Si
	def orient_LAM (self, cluster):

		#translate cluster to origin
		origin = cluster[0]
		
		#shift to origin
		for y in arange(1, self.num_atoms):
			cluster[y][1] -= origin[1]
			cluster[y][2] -= origin[2]
			cluster[y][3] -= origin[3]
		
		cluster[0] = [origin[0], 0, 0, 0]
	
		#spherical transformation
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = self.sph_coords(cluster[y, 1:4])
		
		#fix one of the atoms to the z-axis
		theta_adjust = cluster[4][2]
		
		#transform back to cartesian
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = self.crt_coords(cluster[y, 1:4])
			
		#Find rotation axis using cross product
		rot_axis = cross([0,0,1], cluster[4, 1:4])
		
		#Apply quaternion transform
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = dot(self.rotation_matrix(rot_axis,theta_adjust),cluster[y, 1:4])
		
		#spherical transformation
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = self.sph_coords(cluster[y, 1:4])
		
		#fix one atom to the xz plane
		phi_adjust = cluster[1][3]
		
		#rotate cluster
		for y in arange(self.num_atoms):
			cluster[y][3] -= phi_adjust
		
		#transform back to cartesian
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = self.crt_coords(cluster[y, 1:4])

		###Do plane alignment###

		#Find plane normal
		plane_leg1 = self.compute_vector(cluster[2][1:4], cluster[1, 1:4])
		plane_leg2 = self.compute_vector(cluster[3][1:4], cluster[1, 1:4])
		plane_normal = cross(plane_leg1, plane_leg2)

		#Get angle of rotation
		rot_angle = (pi/180.0)*self.compute_angle(plane_normal, array((0,0,0)), array((0,0,1)))
		if (rot_angle > pi/2.0):
			plane_normal = (-1)*plane_normal
			rot_angle = (pi/180.0)*self.compute_angle(plane_normal, array((0,0,0)), array((0,0,1)))

		#Get rotation axis using z-axis and plane normal
		rot_axis = cross([0,0,1], plane_normal)

		#Do rotation
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = dot(self.rotation_matrix(rot_axis,rot_angle),cluster[y, 1:4])

		#spherical transformation
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = self.sph_coords(cluster[y, 1:4])
		
		#fix one atom to the xz plane
		phi_adjust = cluster[1][3]

		#rotate cluster
		for y in arange(self.num_atoms):
			cluster[y][3] -= phi_adjust
		
		#transform back to cartesian
		for y in arange(self.num_atoms):
			cluster[y][1],cluster[y][2],cluster[y][3] = self.crt_coords(cluster[y, 1:4])

		#save clusters information
		self.atoms = cluster
		return self.atoms
		
	#Analyzes cluster
	def analyze_LAM (self, cluster):
		
		#Compute xz angle
		xz_angle =  self.compute_angle(cluster[1, 1:4], cluster[0, 1:4], cluster[2, 1:4])
		
		#compute atom 3 distance
		atom3_dist =  linalg.norm(cluster[3][1:4])
		
		'''
		#compute the plane normal angle
		basis1a = self.compute_vector(cluster[2,1:4], o=cluster[3, 1:4])
		basis1b = self.compute_vector(cluster[4,1:4], o=cluster[3, 1:4])
		axis1 = cross(basis1a, basis1b)
		
		basis2a = self.compute_vector(cluster[5,1:4], o=cluster[6, 1:4])
		basis2b = self.compute_vector(cluster[7,1:4], o=cluster[6, 1:4])
		axis2 = cross(basis2a, basis2b)
		
		axis_angle = self.compute_angle(axis1, (0,0,0), axis2)
		
		if axis_angle > 90.0:
			axis_angle = 180.0 - axis_angle
		plane_angle = axis_angle
		'''
			
		return (xz_angle, atom3_dist)


if __name__ == "__main__":
	print "Build successful!"