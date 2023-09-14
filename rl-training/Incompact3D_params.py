import re
import numpy as np
import os
import copy

class LineParamMap():

	param_map = { # Maps parameter names to line numbers in incompact3d.prm
		"domain_x": 4,
		"domain_y": 5,
		"domain_z": 6,
		"re": 7,
		"sch": 8,
		"u1": 9,
		"u2": 10,
		"turbint": 11,
		"dt": 12,
		"last_iter": 22,
		"iord": 31,
		"read_restart": 36,
		"first_snapshot": 38,
		"snapshot_iter": 39,
		"x_centre": 48,
		"y_centre": 49,
		"flap_length": 76,
		"flap_thickness": 77,
		"body_ar": 79,
		"body_type": 81
	}

	def __init__(self, lines):
		self.lines = lines
	
	def __getitem__(self, key):
		return self.lines[self.param_map[key] - 1]
	
	def get_split(self, key, idx=0, dtype=float):
		return dtype(self.__getitem__(key).split()[idx])

class SolverParams():

	def __init__(self, prm_dir_path):
		params = open(os.path.join(prm_dir_path, 'incompact3d.prm'), 'r')
		lines = params.readlines()
		line_map = LineParamMap(lines)

		self.step_iter = line_map.get_split('last_iter', dtype=int) # How many iterations does each solver run do
		
		# Domain lengths in all three directions
		self.domain_x = line_map.get_split('domain_x', dtype=float)
		self.domain_y = line_map.get_split('domain_y', dtype=float)
		self.domain_z = line_map.get_split('domain_z', dtype=float)

		self.re = line_map.get_split('re', dtype=float) # Reynolds number
		self.sch = line_map.get_split('sch', dtype=float) # Schmidt number
		self.u1 = line_map.get_split('u1', dtype=float)
		self.u2 = line_map.get_split('u2', dtype=float)
		self.turbulence_intensity = line_map.get_split('turbint', dtype=float)
		self.dt = line_map.get_split('dt', dtype=float)

		self.read_restart = line_map.get_split('read_restart', dtype=bool)
		self.iord = line_map.get_split('iord', dtype=int)
		
		# Body x and y centres
		self.x_centre = line_map.get_split('x_centre', dtype=float)
		self.y_centre = line_map.get_split('y_centre', dtype=float)
		
		self.flap_length = line_map.get_split('flap_length', dtype=float)
		self.flap_thickness = line_map.get_split('flap_thickness', dtype=float)
		self.ar = line_map.get_split('body_ar', dtype=float) # Body aspect ratio (width always = 1)
		self.ahmed_body = line_map.get_split('body_type', dtype=bool) # (Square: 0, Ahmed Body: 1) 

		params.close()
		try:
			module_params = open(os.path.join(prm_dir_path, 'src/module_param.f90'), 'r')
			valid_lines = [line.strip().split() for line in module_params if not line.startswith('!')]
			mesh_line = valid_lines[1][2]

			mesh_point_sizes = re.findall("[0-9]*=([0-9]+)(?:\,|\n)", mesh_line, re.M) # Extract the meshpoints using regex which looks for numbers inbeteween = and , or \n
			
			self.mesh_x = float(mesh_point_sizes[0])
			self.mesh_y = float(mesh_point_sizes[1])
		except:
			print('Unable to open src/module_params.f90. Ignoring mesh sizes.')

		refined_y_file = open(os.path.join(prm_dir_path, 'yp.dat'), 'r')
		self.refined_mesh_y_pts = np.asfarray(refined_y_file.readlines())

	def compare_to_json(self, dict_params):
		geom_params = dict_params['geom']
		if not (dict_params['domain_x'] == self.domain_x):
			return False, 'domain_x'
		elif not (dict_params['domain_y'] == self.domain_y):
			return False, 'domain_y'
		elif not (dict_params['re'] == self.re):
			return False, 're'
		elif not (dict_params['dt'] == self.dt):
			return False, 'dt'
		elif not (dict_params['iters'] == self.step_iter):
			return False, 'iters'
		elif not (dict_params['iord'] == self.iord):
			return False, 'iord'
		elif not (geom_params['flap_length'] == self.flap_length):
			return False, 'flap_length'
		elif not (geom_params['flap_thickness'] == self.flap_thickness):
			return False, 'flap_thickness'
		elif not (bool(geom_params['body_type']) == self.ahmed_body):
			return False, 'body_type'
		elif not (geom_params['body_ar'] == self.ar):
			return False, 'body_ar'
		elif not (geom_params['x_centre'] == self.x_centre):
			return False, 'x_centre'
		elif not (geom_params['y_centre'] == self.y_centre):
			return False, 'y_centre'
		else:
			return True, ''

	@classmethod
	def compare(cls, inst1, inst2):
		if isinstance(inst2, inst1.__class__):
			
			# We don't want to compare all the parameters, delete the ones we want to ignore
			inst1_dict_copy = copy.deepcopy(inst1.__dict__) # First make a deepcopy in order to not mutate the original classes
			inst2_dict_copy = copy.deepcopy(inst2.__dict__)

			ignored_attributes = ['step_iter', 'read_restart', 'mesh_x', 'mesh_y', 'refined_mesh_y_pts']
			for key in ignored_attributes:
				if key in inst1_dict_copy:
					del inst1_dict_copy[key]
				if key in inst2_dict_copy:
					del inst2_dict_copy[key]
			return inst1_dict_copy == inst2_dict_copy, set(inst1_dict_copy.items()) ^ set(inst2_dict_copy.items()) # Compare the dicts and get the keys that are different
		else:
			return False
