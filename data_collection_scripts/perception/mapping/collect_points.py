import glob
import os
import sys

sys.path.append(glob.glob(sys.argv[1])[0])

import carla
from carla import ColorConverter as cc

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from numpy import random
import matplotlib.pyplot as plt
import traceback
import math
import time
from carla import VehicleLightState as vls
import pickle
import shutil
import json


class CarlaSyncMode(object):
    def __init__(self, client, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.client = client
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def initialize_queues(self, sensors):
        self.sensors = sensors
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def make_queue(self, register_event):
        q = queue.Queue()
        register_event(q.put)
        self._queues.append(q)

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class ActorManager(object):
    def __init__(self, client, world, tm, ego_actor='vehicle.bmw.isetta'):
        self.client = client
        self.world = world
        self.traffic_manager = tm
        self.ego_type = ego_actor
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

    def spawn_all_actor(self, number_of_vehicles, number_of_walkers, num_ego_vehicle, safe=True, car_lights_on=False):
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprintsWalkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')

        if safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            print('requested %d vehicles, but could only find %d spawn points' % (number_of_vehicles, number_of_spawn_points))
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            if n < num_ego_vehicle:
                blueprint = random.choice(self.world.get_blueprint_library().filter(self.ego_type))
            else:
                blueprint = random.choice(blueprints)
            
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot') if n >= num_ego_vehicle else blueprint.set_attribute('role_name', 'hero_' + str(n))

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                print(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.2      # how many pedestrians will run
        percentagePedestriansCrossing = 0.2     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))
        self.traffic_manager.global_percentage_speed_difference(30.0)

    def destroy_all_actors(self):
        if not len(self.vehicles_list):
            return

        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(1)
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

recording = False

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
            elif event.key == pygame.K_r:
                global recording
                recording = True
                print(recording)
    return False


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        self.elements = []
        world = self._parent.get_world()
        temp = carla.Transform(carla.Location(x=1.6, y=-0.3, z=1.7))
        self.camera_2_imu = np.array(temp.get_matrix())
        bp = world.get_blueprint_library().find('sensor.other.imu')
        
        # bp.set_attribute('noise_accel_stddev_x', str(0.05))
        # bp.set_attribute('noise_accel_stddev_y', str(0.05))
        # bp.set_attribute('noise_accel_stddev_z', str(0.05))
        # bp.set_attribute('noise_gyro_stddev_x', str(0.02))
        # bp.set_attribute('noise_gyro_stddev_y', str(0.02))
        # bp.set_attribute('noise_gyro_stddev_z', str(0.02))
        
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        self.listen = self.sensor.listen

    def parse_imu(self, sensor_data):
        limits = (-99.9, 99.9)
        
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        
        self.compass = math.degrees(sensor_data.compass)

        accelerometer = (sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z)
        gyroscope = (sensor_data.gyroscope.x, sensor_data.gyroscope.y, sensor_data.gyroscope.z)
        this_transform = sensor_data.transform
        imu_2_world = np.array(this_transform.get_matrix())
        camera_2_world = imu_2_world.dot(self.camera_2_imu)

        this_element = {'timestamp' : sensor_data.timestamp,\
                        'accelerometer' : accelerometer,\
                        'gyroscope' : gyroscope,\
                        'compass' : self.compass,\
                        'transform' : camera_2_world.tolist()}
        
        return this_element


def main():
    actor_list = []
    # div = int(sys.argv[1])
    pygame.init()
    initial_delay = 0

    display = pygame.display.set_mode(
        (820, 820),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    port = 8000
    # max_trajectory_per_spawn = int(sys.argv[2])
    seed = None
    hybrid = True
    random.seed(seed if seed is not None else int(time.time()))
    traffic_manager = client.get_trafficmanager(port)
    ac = None
    no_rendering = True
    camera_elements = []
    imu_elements = []
    lidar_elements = []
    seg_elements = []

    try:
        if hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if seed is not None:
            traffic_manager.set_random_device_seed(seed)
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()
        Attachment = carla.AttachmentType

        # Create a synchronous mode context.
        with CarlaSyncMode(client, world, fps=10) as sync_mode:
            ac = ActorManager(client, world, traffic_manager)
            ac.spawn_all_actor(1, 0, 1)

            dynamic_actors = [actor for actor in world.get_actors() if 'vehicle' in actor.type_id or 'walker.pedestrian' in actor.type_id]
            ego_vehicles = [v for v in dynamic_actors if 'hero' in v.attributes['role_name']]
            assert len(ego_vehicles) == 1
            vehicle = ego_vehicles[0]
            traffic_manager.ignore_lights_percentage(vehicle, 100)
            bp = blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '820')
            bp.set_attribute('image_size_y', '820')
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', '2.2')

            camera_rgb = world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=1.6, y=-0.3, z=1.7)),
                attach_to=vehicle,
                attachment_type=Attachment.Rigid)
            actor_list.append(camera_rgb)
            bp = blueprint_library.find('sensor.camera.semantic_segmentation')
            bp.set_attribute('image_size_x', '820')
            bp.set_attribute('image_size_y', '820')
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', '2.2')

            camera_semseg = world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=1.6, y=-0.3, z=1.7)),
                attach_to=vehicle,
                attachment_type=Attachment.Rigid)
            actor_list.append(camera_semseg)

            item = ['sensor.lidar.ray_cast', \
                   {'range': '65', 'channels' : '64', 'rotation_frequency' : '25', 'points_per_second' : '140000',\
                    'dropoff_general_rate' : '0.', 'dropoff_intensity_limit' : '0.1', \
                    'upper_fov' : '20.0'}]

            lidar_range = 50
            bp = blueprint_library.find(item[0])
            for attr_name, attr_value in item[1].items():
                bp.set_attribute(attr_name, attr_value)
                if attr_name == 'range':
                    lidar_range = float(attr_value)

            lidar = world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=1.6, z=2.5)),
                attach_to=vehicle,
                attachment_type=Attachment.Rigid)
            actor_list.append(lidar)

            imu_sensor = IMUSensor(vehicle)
            actor_list.append(imu_sensor.sensor)
            counter = 0

            sync_mode.initialize_queues(actor_list)

            while True:
                if should_quit():
                    return
                clock.tick()

                snapshot, image_rgb, image_semseg, lidar_data, imu_data = sync_mode.tick(timeout=2.0)
                image_rgb.convert(cc.Raw)

                if counter >= 25 and counter % 10 == 0 and recording:
                    imu_data = imu_sensor.parse_imu(imu_data)
                    image_semseg.convert(cc.Raw)

                    dir_path = 'out/cam_out/'
                    camera_elements.append({'timestamp' : image_rgb.timestamp})
                    image_rgb.save_to_disk(dir_path + 'original_images/%08d' % image_rgb.frame)
                    
                    dir_path = 'out/seg_out/'
                    seg_elements.append({'timestamp' : image_semseg.timestamp})
                    image_semseg.save_to_disk(dir_path + '_out/%08d' % image_semseg.frame)
                    
                    dir_path = 'out/cast_out/'
                    lidar_elements.append({'timestamp' : lidar_data.timestamp})
                    lidar_data.save_to_disk(dir_path + '_out/%08d' % lidar_data.frame)
                    
                    imu_elements.append(imu_data)

                counter += 1

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                draw_image(display, image_rgb)
                
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                
                pygame.display.flip()

    except:
        traceback.print_exception(*sys.exc_info())

    finally:

        if ac:
            ac.destroy_all_actors()

        if len(camera_elements):
            data = {'elements' : camera_elements}
            json_object = json.dumps(data, indent=4)
            with open("out/cam_out/cam_data.json", "w") as outfile:
                outfile.write(json_object)

        if len(seg_elements):
            data = {'elements' : seg_elements}
            json_object = json.dumps(data, indent=4)
            with open("out/seg_out/seg_data.json", "w") as outfile:
                outfile.write(json_object)

        if len(lidar_elements):
            data = {'elements' : lidar_elements}
            json_object = json.dumps(data, indent=4)
            with open("out/cast_out/lidar_data.json", "w") as outfile:
                outfile.write(json_object)

        if len(imu_elements):
            data = {'elements' : imu_elements}
            json_object = json.dumps(data, indent=4)
            with open("out/imu_data.json", "w") as outfile:
                outfile.write(json_object)

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
