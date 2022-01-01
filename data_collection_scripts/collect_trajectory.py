import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

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
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

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


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
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

recording = False

def parse_inputs(controls):
    if should_quit():
        return None, None, None

    keys = pygame.key.get_pressed()
    milliseconds = 1 / 15

    _steer_cache = controls[0]
    throttle = controls[1]
    brake = controls[2]

    if keys[pygame.K_r]:
        global recording
        recording = True
        print(recording)

    if keys[pygame.K_UP] or keys[pygame.K_w]:
        throttle = min(throttle + 0.01, 0.8)
    else:
        throttle = 0.0

    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        brake = min(brake + 0.2, 1)
    else:
        brake = 0

    steer_increment =  milliseconds
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        if _steer_cache > 0:
            _steer_cache = 0
        else:
            _steer_cache -= steer_increment
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        if _steer_cache < 0:
            _steer_cache = 0
        else:
            _steer_cache += steer_increment
    else:
        _steer_cache = 0.0
    
    steer = round(_steer_cache, 4)

    print(steer, throttle, brake)
    return float(steer), float(throttle), float(brake)


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
    # traffic_manager = client.get_trafficmanager(port)
    ac = None
    no_rendering = True
    camera_elements = []
    imu_elements = []
    velocity_elements = []

    try:

        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.bmw.isetta')),
            start_pose)
        actor_list.append(vehicle)

        # if hybrid:
        #     traffic_manager.set_hybrid_physics_mode(True)
        # if seed is not None:
        #     traffic_manager.set_random_device_seed(seed)
        # settings = world.get_settings()
        # traffic_manager.set_synchronous_mode(True)

        # blueprint_library = world.get_blueprint_library()
        Attachment = carla.AttachmentType

        # Create a synchronous mode context.
        with CarlaSyncMode(client, world, fps=15) as sync_mode:
            # ac = ActorManager(client, world, traffic_manager)
            # ac.spawn_all_actor(1, 0, 1)

            # dynamic_actors = [actor for actor in world.get_actors() if 'vehicle' in actor.type_id or 'walker.pedestrian' in actor.type_id]
            # ego_vehicles = [v for v in dynamic_actors if 'hero' in v.attributes['role_name']]
            # assert len(ego_vehicles) == 1
            # vehicle = ego_vehicles[0]
            # traffic_manager.ignore_lights_percentage(vehicle, 100)
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

            imu_sensor = IMUSensor(vehicle)
            actor_list.append(imu_sensor.sensor)

            counter = 0
            sync_mode.initialize_queues(actor_list[1:])

            while True:
                # if should_quit():
                #     return
                clock.tick()

                control = vehicle.get_control()
                s,t,b = parse_inputs([control.steer, control.throttle, control.brake])
                if s is None:
                    return

                control = carla.VehicleControl()
                control.steer = float(s)
                control.throttle = float(t)
                control.brake = float(b)
                vehicle.apply_control(control)

                snapshot, image_rgb, imu_data = sync_mode.tick(timeout=2.0)
                image_rgb.convert(cc.Raw)

                if counter >= 25 and counter % 10 == 0 and recording:
                    imu_data = imu_sensor.parse_imu(imu_data)

                    dir_path = 'cam_out/'
                    camera_elements.append({'timestamp' : image_rgb.timestamp})
                    image_rgb.save_to_disk(dir_path + '_out/%08d' % image_rgb.frame)
                    
                    imu_elements.append(imu_data)

                    v = vehicle.get_velocity()
                    velocity_elements.append({'timestamp' : image_rgb.timestamp, 'velocity' : math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)})

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

        if len(camera_elements):
            data = {'elements' : camera_elements}
            json_object = json.dumps(data, indent=4)
            with open("cam_out/cam_data.json", "w") as outfile:
                outfile.write(json_object)

        if len(imu_elements):
            data = {'elements' : imu_elements}
            json_object = json.dumps(data, indent=4)
            with open("imu_data.json", "w") as outfile:
                outfile.write(json_object)

        if len(velocity_elements):
            data = {'elements' : velocity_elements}
            json_object = json.dumps(data, indent=4)
            with open("velocity_data.json", "w") as outfile:
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
