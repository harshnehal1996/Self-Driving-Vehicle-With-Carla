import glob
import os
import sys
import carla
import pygame
import numpy as np
import queue
import random
import networkx as nx
import matplotlib.pyplot as plt

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
m = world.get_map()

# split each road into 4 nodes

start_pose = random.choice(m.get_spawn_points())
waypoint = m.get_waypoint(start_pose.location)
print(waypoint)

def find_next_road(waypoint, _next=True):
    if _next:
        x = waypoint.next(2)[0]
        if x.road_id != waypoint.road_id:
            return x
        x = waypoint.next_until_lane_end(2)[-1]
        y = x.next(2)
    else:
        x = waypoint.previous(2)[0]
        if x.road_id != waypoint.road_id:
            return x
        x = waypoint.previous_until_lane_start(2)[-1]
        y = x.previous(2)
    if len(y):
        if len(y) > 1:
            for i in range(len(y)):
                if not y[i].is_junction:
                    print(y[i].road_id)
        return y[0]
    
    return None

class Node:
    def __init__(self, road_id, node_type, road_type):
        self.road_id = road_id
        self.node_type = node_type
        self.road_type = road_type
        self.jn_id = None
        self.to = []

def get_length(waypoint):
    waypoint_list = waypoint.previous_until_lane_start(2) + waypoint.next_until_lane_end(2)
    length = 0
    this_loc = waypoint_list[0].transform.location
    for i in range(1, len(waypoint_list)):
        prev_loc = this_loc
        this_loc = waypoint_list[i].transform.location
        l = (this_loc.x - prev_loc.x) ** 2
        l += (this_loc.y - prev_loc.y) ** 2
        l += (this_loc.z - prev_loc.z) ** 2
        length += l**0.5
    
    return length

def build_graph(waypoint, vertex, flag):
    current_road = waypoint.road_id
    
    if flag == -1:
        y_next = find_next_road(waypoint, _next=False)
        y_prev = find_next_road(waypoint, _next=True)
    else:
        y_next = find_next_road(waypoint, _next=True)
        y_prev = find_next_road(waypoint, _next=False)
    
    out_1 = Node(current_road, 1, 0)
    in_2 = Node(current_road, 0, 0)
    out_2 = Node(current_road, 1, 0)
    
    vlist = vertex[current_road]
    in_1 = vlist[0]
    out_1.jn_id = in_1.jn_id
    vlist[1] = in_2
    vlist[2] = out_1
    vlist[3] = out_2
    road_length = get_length(waypoint)
    in_1.to.append((out_1, road_length))
    in_2.to.append((out_2, road_length))
    
    if y_next:
        jnn = y_next.get_junction()
        if jnn:
            out_2.jn_id = jnn.id
            in_2.jn_id = jnn.id
            way_confusion = jnn.get_waypoints(carla.LaneType.Driving)
            conn = {}
            for tup in way_confusion:
                U = tup[0].previous(1)
                V = tup[1].next(1)
                if len(U) > 1:
                    print('more than one previous')
                    for i in range(len(U)):
                        print(U[i].road_id, tup[0].road_id)
                if len(V) > 1:
                    print('more than one next')
                    for i in range(len(V)):
                        print(V[i].road_id, tup[0].road_id)
                u = U[0].road_id
                v = V[0].road_id
                c = tup[0].road_id
                if u == current_road and not conn.__contains__((v, c)):
                    conn[(v, c)] = True
                    cnr = Node(c, -1, 1)
                    if vertex.__contains__(v):
                        tlist = vertex[v]
                        if tlist[0].jn_id == jnn.id:
                            cnr.to.append((tlist[0], 0))
                        elif tlist[1].jn_id == jnn.id:
                            cnr.to.append((tlist[1], 0))
                        else:
                            print('jnn_id=',jnn.id,'does not match ',tlist[0].jn_id,' or ',tlist[1].jn_id)
                            print(u, c, v)
                        out_1.to.append((cnr, 0))
                    else:
                        target_node = Node(v, 0, 0)
                        cnr.to.append((target_node, 0))
                        out_1.to.append((cnr, 0))
                        target_node.jn_id = jnn.id
                        vertex[v] = [target_node, 0, 0, 0]
                        build_graph(V[0], vertex, 1)
        else:
            v = y_next.road_id
            if vertex.__contains__(v):
                tlist = vertex[v]
                if tlist[0].jn_id == None:
                    if tlist[1].jn_id == None:
                        if current_road != tlist[2].to[0][0].road_id:
                            out_1.to.append((tlist[0], 0))
                        else:
                            out_1.to.append((tlist[1], 0))
                    else:
                        out_1.to.append((tlist[0], 0))
                else:
                    out_1.to.append((tlist[1], 0))
            else:
                target_node = Node(v, 0, 0)
                out_1.to.append((target_node, 0))
                vertex[v] = [target_node, 0, 0, 0]
                build_graph(y_next, vertex, flag)
                
    if y_prev:
        jnp = y_prev.get_junction()
        if jnp:
            way_confusion = jnp.get_waypoints(carla.LaneType.Driving)
            conn = {}
            for tup in way_confusion:
                U = tup[0].previous(1)
                V = tup[1].next(1)
                if len(U) > 1:
                    print('more than one previous')
                    for i in range(len(U)):
                        print(U[i].road_id, tup[0].road_id)
                if len(V) > 1:
                    print('more than one next')
                    for i in range(len(V)):
                        print(V[i].road_id, tup[0].road_id)
                u = U[0].road_id
                v = V[0].road_id
                c = tup[0].road_id
                if u == current_road and not conn.__contains__((v, c)):
                    conn[(v, c)] = True
                    cnr = Node(c, -1, 1)
                    if vertex.__contains__(v):
                        tlist = vertex[v]
                        if tlist[0].jn_id == jnp.id:
                            cnr.to.append((tlist[0], 0))
                        elif tlist[1].jn_id == jnp.id:
                            cnr.to.append((tlist[1], 0))
                        else:
                            print('jnp_id=',jnp.id,'does not match ',tlist[0].jn_id,' or ',tlist[1].jn_id)
                            print(u, c, v)
                        out_1.to.append((cnr, 0))
                    else:
                        target_node = Node(v, 0, 0)
                        cnr.to.append((target_node, 0))
                        out_2.to.append((cnr, 0))
                        target_node.jn_id = jnp.id
                        vertex[v] = [target_node, 0, 0, 0]
                        build_graph(V[0], vertex, 1)
        else:
            v = y_prev.road_id
            if vertex.__contains__(v):
                tlist = vertex[v]
                if tlist[0].jn_id == None:
                    if tlist[1].jn_id == None:
                        if current_road != tlist[2].to[0][0].road_id:
                            out_2.to.append((tlist[0], 0))
                        else:
                            out_2.to.append((tlist[1], 0))
                    else:
                        out_2.to.append((tlist[0], 0))
                else:
                    out_2.to.append((tlist[1], 0))
            else:
                target_node = Node(v, 0, 0)
                out_2.to.append((target_node, 0))
                vertex[v] = [target_node, 0, 0, 0]
                build_graph(y_prev, vertex, -flag)

vertex = {}
def create_road_network(waypoint, vertex):
    start_id = waypoint.road_id
    v = Node(start_id, 0, 0)
    w = find_next_road(waypoint, _next=False)
    jn = w.get_junction()
    v.jn_id = jn.id if jn else None
    vertex[start_id] = [v, 0, 0, 0]
    build_graph(waypoint, vertex, 1)

create_road_network(waypoint, vertex)

