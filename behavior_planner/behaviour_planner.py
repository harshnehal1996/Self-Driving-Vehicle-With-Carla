import numpy as np
import carla

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


class Behavior(object):
	def __init__(self):
		self.state = STATE.STOPPED
		self.current_goal_location = None
		self.current_mission_plan = None
		self.mission_point = None
		self.perception_module = None
		self.next_road_waypoint = None
		self.isconnecting = None
		self.max_plan_path_length = 15
		self.max_approaching_distance = None
		self.local_planner = None
	
	def perform_traffic_check(self):
		return self.perception_module.trafic_light_status()

	def get_next_road_in_mission(self, waypoint):
		if self.mission_point + 1 >= len(self.current_mission_plan):
			return None

		current_road = waypoint.road_id
		waypt = waypoint.next_until_lane_end(2)[-1]
		jw = waypt.next(1.5)
		next_id = self.current_mission_plan[self.mission_point + 1]
		
		for jwaypt in jw:
			if jwaypt.road_id == next_id:
				return jwaypt

		if jwaypt.is_junction:
			confusion = jwaypt.get_waypoints()
			for x, y in confusion:
				U = x.previous(1)
        		for i in range(len(U)):
        			if U[i].road_id == current_road:
        				return x
		
		max_it = 1000
		for i in range(max_it):
			w = waypoint.next(1)
			if not len(w) or w[0] == waypoint:
				return None
			end = True
			for v in w:
				if v.road_id == current_road:
					end = False
					wayp = v
					break
				if v.road_id == next_id:
					return w
			if end:
				return None
		return None

	def get_distance(self, u, v):
		u = u.transform.location
        v = v.transform.location
        l = (u.x - v.x) ** 2
        l += (u.y - v.y) ** 2
        l += (u.z - v.z) ** 2
        return l ** 0.5

    def is_approaching_junction(self, current_waypoint, next_waypoint):
		if not next_waypoint or not next_waypoint.is_junction or current_waypoint.is_junction:
			return False
		return self.get_distance(current_waypoint, next_waypoint) <= self.max_approaching_distance

	def __resolve_multiple(self, waypoints, road_id):
		if not type(road_id) is list:
			road_id = [road_id]
		for w in waypoints:
			if w.road_id in road_id:
				return w
		return None

	def get_possible_stops_on_junction(self, waypoint):
		target = self.next_road_waypoint
		possibilities = [self.__resolve_multiple(target.previous(1), waypoint.road_id)]
		if possibilities[0] == None:
			return None
		targetL = target.get_left_lane()
		while targetL:
			previous = self.__resolve_previous(targetL.previous(1), waypoint.road_id)
			if not previous:
				break
			possibilities.append(previous)
			targetLn = targetL.get_left_lane()
			if targetLn == targetL:
				break
			targetL = targetLn

		targetR = target.get_right_lane()
		while targetR:
			previous = self.__resolve_previous(targetR.previous(1), waypoint.road_id)
			if not previous:
				break
			possibilities.append(previous)
			targetRn = targetR.get_right_lane()
			if targetRn == targetR:
				break
			targetR = targetRn

		return possibilities

	def get_target_waypoint(self, waypoint):
		isLast = len(current_mission_plan) <= mission_point + 1
		next_id = self.next_road_waypoint.road_id if not isLast else -1
		max_depth = self.max_plan_path_length / 2
		current_road = [waypoint.road_id, next_id]
		for i in range(max_depth):
			new_waypt = self.__resolve_multiple(waypoint.next(2), current_road)
			if not new_waypt or new_waypt == waypoint:
				return waypoint
			waypoint = new_waypt
		return waypoint

	# ....................
	######################
	# ....................
	######################
	# ....................

	# input lane profile 
	# target lane profile

	def transition(self, current_waypoint, **kwargs):
		if not self.next_road_waypoint:
			self.next_road_waypoint = self.get_next_road_in_mission(current_waypoint)
				if self.next_road_waypoint == None:
					print('Cant find road ', self.current_mission_plan[self.mission_point + 1])
					return -1
		if self.state == STATE.TRACK_SPEED:
			if self.is_approaching_junction(waypoint, self.next_road_waypoint):
				self.state = STATE.DEACCELERATE_TO_STOP
				self.possibilities = self.get_possible_stops_on_junction()
				if self.possibilities == None:
					print('Cant find connection road from ', current_waypoint.road_id)
					return -1
				return self.transition(current_waypoint)
			else:
				target_waypoint = self.get_target_waypoint(current_waypoint)
				if target_waypoint == current_waypoint:
					print('Cant find any new target to update for road_id ', current_waypoint.road_id, mission_point)
					return -1
				motion_parameters = self.local_planner(self.current_waypoint, self.target_waypoint)
				execute(motion_parameters)
		elif self.state == STATE.DEACCELERATE_TO_STOP:
			motion_parameters = self.local_planner(self.current_waypoint, self.target_waypoint)
			execute(motion_parameters)
		elif self.state == STATE.AT_STOP:
			self.state = STATE.TRACK_SPEED
		return 0


def update_vehicle_state(image_rgb, waypoint, timestamp):
	return True

