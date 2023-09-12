import numpy as np
from Incompact3D_params import SolverParams

class ProbeLayout():
	
	def __init__(self, probe_params, solver_params):
		self.s_params = solver_params
		self.layout_type = probe_params['type']
		self.n_probes = probe_params['n_probes']

	def generate_probe_layout_file(self, output_file):
		probe_coords = None
		print(self.layout_type)
		margin = 0.2 # Distance from the edge of the thing we're aligning with
		if self.layout_type == 'uniform_wake': # Generates 200 uniform probes in the wake of the body
			#print('indicator1')
			back_face_mesh_x = np.ceil(self.s_params.mesh_x * (self.s_params.x_centre + self.s_params.ar/2 + margin) / self.s_params.domain_x)
			last_probe_x = min(back_face_mesh_x + 0.25 * self.s_params.mesh_x, self.s_params.mesh_x) # Take the min to ensure last_probe is still part of mesh
			top_face_y = self.s_params.y_centre + 1/2
			bottom_face_y = self.s_params.y_centre - 1/2

			# Find the closest refined meshpoints to the top face and bottom face y coordinates
			top_face_mesh_point_y = np.argmin(abs(top_face_y - self.s_params.refined_mesh_y_pts))
			bottom_face_mesh_point_y = np.argmin(abs(bottom_face_y - self.s_params.refined_mesh_y_pts))

			x_probe_space = np.linspace(back_face_mesh_x, last_probe_x, 20)
			y_probe_space = np.linspace(bottom_face_mesh_point_y, top_face_mesh_point_y, 10, dtype=int)
			probe_coords = np.array(np.meshgrid(x_probe_space, y_probe_space), dtype=int) # TODO: We dont need meshgrid, just iterate over two linspaces
			self.n_probes = 200
		
		elif self.layout_type == 'uniform_flap_wake': # Generates 200 uniform probes after the flaps
			#print('indicator2')
			back_flap_mesh_x = np.ceil(self.s_params.mesh_x * (self.s_params.x_centre + self.s_params.ar/2 + self.s_params.flap_length + margin) / self.s_params.domain_x)
			last_probe_x = min(back_flap_mesh_x + 0.25 * self.s_params.mesh_x, self.s_params.mesh_x) # Take the min to ensure last_probe is still part of mesh
			top_face_y = self.s_params.y_centre + 1/2
			bottom_face_y = self.s_params.y_centre - 1/2

			# Find the closest refined meshpoints to the top face and bottom face y coordinates
			top_face_mesh_point_y = np.argmin(abs(top_face_y - self.s_params.refined_mesh_y_pts))
			bottom_face_mesh_point_y = np.argmin(abs(bottom_face_y - self.s_params.refined_mesh_y_pts))

			x_probe_space = np.linspace(back_flap_mesh_x, last_probe_x, 20)
			y_probe_space = np.linspace(bottom_face_mesh_point_y, top_face_mesh_point_y, 10, dtype=int)
			probe_coords = np.array(np.meshgrid(x_probe_space, y_probe_space), dtype=int) # TODO: We dont need meshgrid, just iterate over two linspaces
			self.n_probes = 200
			

		elif self.layout_type == 'uniform_base':
			#print('indicator3')
			back_face_mesh_x = np.ceil(self.s_params.mesh_x * (self.s_params.x_centre + self.s_params.ar/2 + margin) / self.s_params.domain_x)
			top_face_y = self.s_params.y_centre + 1/2
			bottom_face_y = self.s_params.y_centre - 1/2

			top_face_mesh_point_y = np.argmin(abs(top_face_y - self.s_params.refined_mesh_y_pts))
			bottom_face_mesh_point_y = np.argmin(abs(bottom_face_y - self.s_params.refined_mesh_y_pts))

			x_probe_location = int(142)
			y_probe_location = np.linspace(bottom_face_mesh_point_y, top_face_mesh_point_y, 29, dtype=int)
			

		else:
			raise NotImplementedError

		if probe_coords is not None:
			out = open(output_file, 'w')
			if self.layout_type == 'uniform_base':

				#print('indicator4')
				for i in range(y_probe_location.shape[0]):
					out.write(f'{x_probe_location} {y_probe_location[i]} 1\n')
			else:
				#print('indicator5')
				for i in range(probe_coords.shape[1]): # Loop over probe_coords or flatten to obtain pairs of (x, y) coordinates then write into a file to generate probes
					for j in range(probe_coords.shape[2]):
						out.write(f'{probe_coords[0, i, j]} {probe_coords[1, i, j]} 1\n')
			out.close()
	
	def get_n_probes(self):
		assert self.n_probes is not None, 'n_probes was not set'
		
		return self.n_probes
