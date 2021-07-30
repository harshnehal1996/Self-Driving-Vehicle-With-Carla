import numpy as np
import cv2
import matplotlib.pyplot as plt

width = 1280
height = 720
K = np.array([[ width / 2, 0.0, width / 2],\
	          [ 0.0, width / 2, height / 2],\
			  [ 1.0, 1.0, 1.0]], dtype=np.float64)
proj = np.zeros((3,3))
proj[0, 1] = -1
proj[1, 2] = -1
proj[2, 0] = 1
K = K.dot(proj)
Kinv = np.linalg.inv(K)

class Perception:
	def __init__(self):
		self.current_location = None
		self.current_orientation = None
		self.inverse_orientation = None
		self.max_dynamic_objects = 30
		self.dynamic_objects = [None for _ in range(self.max_dynamic_objects)]
		self.availabilty_queue = [i for i in range(self.max_dynamic_objects)]
		self.update_rate = 0.5
		self.object_distance_threshold = 30
		self.trigger = None
		self.pcl_voxel = None
		self.camera_height = None
		self.path = []

	# right now just take return the actual location
	# will have to find a way to communicate with the localization module
	def perform_localization_update(self, location):
		return x

	def __fill_extra_pts(x, y, voxel, dist_voxel):
		if dist_voxel.__contains__((x, y)):
			return

		if voxel.__contains__((x+1, y)) and voxel.__contains__((x-1, y)) and voxel.__contains__((x, y+1)) and voxel.__contains__((x, y-1)):
			dist = (voxel[(x + 1, y)][0] + voxel[(x-1, y)][0] + voxel[(x, y+1)][0] + voxel[((x, y-1))][0]) / \
				   (voxel[(x + 1, y)][1] + voxel[(x-1, y)][1] + voxel[(x, y+1)][1] + voxel[((x, y-1))][1])
			dist_voxel[(x, y)] = dist


	def fill_voxel(self, pcl_data, dist_voxel):
		voxel = {}
		for i in range(len(pcl_data)):
			x = math.floor(pcl_data[i][0])
			y = math.floor(pcl_data[i][1])
			z = pcl_data[i][2]

			if voxel.__contains__((x, y)):
				voxel[(x, y)][0] += z
				voxel[(x, y)][1] += 1
			else:
				voxel[(int(x), int(y))] = [z, 1]

		for key in voxel.keys():
			if not voxel.__contains__((key[0] + 1, key[1])):
				self.__fill_extra_pts(key[0] + 1, key[1], voxel, dist_voxel)
			if not voxel.__contains__((key[0] - 1, key[1])):
				self.__fill_extra_pts(key[0] - 1, key[1], voxel, dist_voxel)
			if not voxel.__contains__((key[0], key[1] - 1)):
				self.__fill_extra_pts(key[0], key[1] - 1, voxel, dist_voxel)
			if not voxel.__contains__((key[0], key[1] + 1)):
				self.__fill_extra_pts(key[0], key[1] + 1, voxel, dist_voxel)

			dist_voxel[key] = voxel[key][0] / voxel[key][1]

	# perform constant motion update to dynamic objects
	def perform_constant_update(self, timestamp):
		for i in range(self.max_dynamic_objects):
			if not dynamic_objects[i]:
				continue

			paths = dynamic_objects[i]
			delta_t = timestamp - paths[-1][-2]
			paths.append([paths[-1][0] + paths[-1][1] * delta_t, paths[-1][1], 0, paths[-1][3], timestamp, paths[-1][-1]])
			paths[-1][2] = np.linalg.norm(paths[-1][0] - self.current_location)

	
	# only consider vehicles and pedestrian as dynamic objects
	# current assumption is that the ground to body connection is visible for the object
	# To eliminate the assumption one way is to autocomplete the object with the nonvisible part
	# or atleast give rough estimation of tight bounding box
	def update_dynamic_object_state(self, ins_segmented_mask, seg_class, object_id, timestamp, patience=3):
		for i in range(len(ins_segmented_mask)):
			if seg_class[i] == 'Vehicle' or seg_class[i] == 'Pedestrian':
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(ins_segmented_mask[i], connectivity=25)
				if not len(centroids):
					continue

				refined_mask = output[0]
				left = centroids[0][0]
				right = height

				while left <= right:
					mid = (left + right) // 2 
					x = np.sum(output[mid])
					if x:
						left = mid + 1
					else:
						right = mid - 1

				x = np.ones(3)
				x = np.argmax(refined_mask[left - 1]), left - 1, 1
				X = Kinv.dot(x.reshape(3, 1))
				d = np.linalg.norm(X)
				X = self.current_orientation.dot(X) / d
				Y = None
				steps = 1
				wait = 0
				found = False
				ground = (-self.current_orientation[:, 2]).reshape(3, 1)

				while steps < self.object_distance_threshold and wait < 10:
					X = ((1 + steps) / steps) * X
					Y = (X + self.current_location).astype(int)
					if self.road_voxel.__contains__((Y[0][0], Y[1][0])):
						elevation = self.road_voxel[(Y[0][0], Y[1][0])]
						if Y[2][0] - elevation < 0:
							found = True
							break
					elif wait > 0 or X.T.dot(ground)[0][0] > self.camera_height:
						wait += 1
					steps += 1

				if found:
					Y[2][0] = self.road_voxel[(Y[0][0], Y[1][0])]
					distance = np.linalg.norm(self.inverse_orientation.dot(Y - self.current_orientation))
					if dynamic_objects[object_id[i]]:
						paths = dynamic_objects[object_id[i]]
						paths.append([Y, np.zeros((3, 1)), distance, 0, timestamp, len(paths)])
						paths[-1][0] = self.update_rate * ((paths[-1][0] - paths[paths[-2][-1]][0]) / (paths[-1][-2] - paths[paths[-2][-1]][-2])) + \
									   (1 - self.update_rate) * paths[-2][1]
					else:
						dynamic_objects[object_id[i]] = [Y, np.zeros((3, 1)), distance, 0, timestamp, 0]

				else:
					print('unable to track position for object %d, at %f type %s' % (object_id[i], timestamp, seg_class[i]))
		
		
		for i in range(self.max_dynamic_objects):
			if not dynamic_objects[i]:
				continue
			if dynamic_objects[i][-2] != timestamp:
				if dynamic_objects[i][3] >= patience:
					dynamic_objects[i] = None
					self.availabilty_queue.append(i)
				else:
					dynamic_objects[i][3] += 1


class State:
	# VEHICLE in stop vehicle waiting for new mission
	STOPPED = 0

	# follow the current lane speed
	TRACK_SPEED = 1

	# set a stop point for the motion planner
	DEACCELERATE_TO_STOP = 2
	
	# maintain speed with lead vehicle upfront
	FOLLOW_LEAD_VEHICLE = 3
	
	# check traffic light state, dynamic vehicle state
    # execute TRACK_SPEED when safe to start
	AT_STOP = 4


"""

	Waypoints : directed points that align to the direction of lane
	

"""



class Behavior(object):
	def __init__(self):
		self.state = STATE.STOPPED
		self.current_goal_location = None
		self.lane_id = None
		self.road_id = None
		self.current_mission_plan = []

	def transition(self, waypoint_object):
		if self.state == STATE.TRACK_SPEED:
			waypoints = waypoint_object.next(2)


def update_vehicle_state(image_rgb, waypoint, timestamp):
	return True

