import glob
import os
import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

sys.path.append(glob.glob(sys.argv[1])[0])

import carla

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


class CarlaSyncMode(object):
    def __init__(self, client, world, no_rendering, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.no_rendering = no_rendering
        self.num_renderer = 1

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=self.no_rendering,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))
        return self

    def initialize_queues(self, sensors, num_renderer):
        self.sensors = sensors
        self.num_renderer += num_renderer
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def remove_queues(self):
        self.num_renderer = 1
        for i in range(len(self.sensors)):
            self.sensors[i].destroy()
        self.sensors = []
        self._queues = []

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(self._queues[i], timeout) for i in range(self.num_renderer)]
        assert all(x.frame == self.frame for x in data)
        data = data + [self._retrieve_data(self._queues[i], None, block=False) for i in range(self.num_renderer, len(self._queues))]
        return data

    def __exit__(self, *args, **kwargs):
        self._settings.no_rendering_mode = False
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout, block=True):
        while True:
            if not block and sensor_queue.empty():
                return None
            data = sensor_queue.get(block=block, timeout=timeout)
            if not block or data.frame == self.frame:
                return data

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

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

class ActorManager(object):
    def __init__(self, client, world, tm, ego_actor='vehicle.audi.a2'):
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

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

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

            light_state = vls.NONE
            if car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                print(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        
        percentagePedestriansRunning = 0.2      # how many pedestrians will run
        percentagePedestriansCrossing = 0.2     # how many pedestrians will walk through the road
        
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
           
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
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
        
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        self.world.tick()

        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].start()
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))
        self.traffic_manager.global_percentage_speed_difference(30.0)

    def destroy_all_actors(self):
        if not len(self.vehicles_list):
            return

        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(1)
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []


class data_iter(object):
    def __init__(self, id, ego_vehicle):
        self.id = id
        self.ego_vehicle = ego_vehicle
        self.ego_commands = [[]]
        self.is_moving = False
        self.SPEED_ON_STOP_TH = 0.2
        self.on_trajectory = False

    def step(self, snapshot, col_event):
        if col_event:
            actor_type = get_actor_display_name(col_event.other_actor)
            impulse = col_event.normal_impulse
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            print('\n...............\nCollision with %r at intensity %f\n...............\n' % (actor_type, intensity))
            return 0, True
        
        ret = 0
        ego_speed = self.ego_vehicle.get_velocity()
        m_ego_speed = math.sqrt(ego_speed.x**2 + ego_speed.y**2 + ego_speed.z**2)

        if self.is_moving:
            if m_ego_speed < self.SPEED_ON_STOP_TH:
                self.is_moving = False
                if self.ego_vehicle.is_at_traffic_light():
                    print('trajectory end for vehicle %s, finish=%d' % (self.ego_vehicle.attributes['role_name'], len(self.ego_commands)))
                    self.on_trajectory = False
                    ret = 1
                    control = self.ego_vehicle.get_control()
                    # print(len(self.ego_commands), control, m_ego_speed)
                    ego_acc = self.ego_vehicle.get_acceleration()
                    self.ego_commands[-1].append([len(self.ego_commands) - 1, snapshot.elapsed_seconds, control.throttle,\
                                                  control.steer, control.brake, ego_speed.x, ego_speed.y,\
                                                  ego_acc.x, ego_acc.y])
                    self.ego_commands.append([])
        elif m_ego_speed > self.SPEED_ON_STOP_TH:
            self.is_moving = True
            self.on_trajectory = True
        
        if self.on_trajectory:
            control = self.ego_vehicle.get_control()
            # print(len(self.ego_commands), control, m_ego_speed)
            ego_acc = self.ego_vehicle.get_acceleration()

            self.ego_commands[-1].append([len(self.ego_commands) - 1, snapshot.elapsed_seconds, control.throttle,\
                                          control.steer, control.brake, ego_speed.x, ego_speed.y,\
                                          ego_acc.x, ego_acc.y])

        return ret, False


def main():
    actor_list = []
    div = int(sys.argv[2])
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    save_path = 'collected_trajectories/'

    world = client.get_world()
    port = 8000
    max_trajectory_per_spawn = int(sys.argv[3])
    seed = None
    hybrid = True
    random.seed(seed if seed is not None else int(time.time()))
    traffic_manager = client.get_trafficmanager(port)
    ac = None
    no_rendering = True

    try:
        if hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if seed is not None:
            traffic_manager.set_random_device_seed(seed)
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        
        # traffic_manager.set_respawn_dormant_vehicles(True)
        # traffic_manager.set_boundaries_respawn_dormant_vehicles(25,100)
        
        blueprint_library = world.get_blueprint_library()

        with CarlaSyncMode(client, world, no_rendering, fps=15) as sync_mode:
            ac = ActorManager(client, world, traffic_manager)
            ac.spawn_all_actor(20, 0, 1)
            
            dynamic_actors = [actor for actor in world.get_actors() if 'vehicle' in actor.type_id or 'walker.pedestrian' in actor.type_id]
            ego_vehicles = [v for v in dynamic_actors if 'hero' in v.attributes['role_name']]
            assert len(ego_vehicles) > 0
            num_data_actor = len(ego_vehicles)
            print(num_data_actor, len(dynamic_actors))
            measurements = [[] for _ in range(len(dynamic_actors))]
            orientation_vector = [[] for _ in range(len(dynamic_actors))]
            type_actor = []
            for i in range(len(dynamic_actors)):
                vertex = dynamic_actors[i].bounding_box.get_local_vertices()
                for v in vertex:
                    measurements[i].append([v.x, v.y, v.z])

            j = 0
            data_actor = []
            for i in range(len(dynamic_actors)):
                if 'hero' in dynamic_actors[i].attributes['role_name']:
                    data_actor.append(data_iter(j, dynamic_actors[i]))
                    j += 1
                if 'vehicle' in dynamic_actors[i].type_id:
                    type_actor.append(1)
                if 'walker.pedestrian' in dynamic_actors[i].type_id:
                    type_actor.append(0)

            if not no_rendering:
                display_vehicle = ego_vehicles[0]
                camera_semseg = world.spawn_actor(
                    blueprint_library.find('sensor.camera.semantic_segmentation'),
                    carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                    attach_to=display_vehicle)
                actor_list.append(camera_semseg)

            for i in range(num_data_actor):
                collision_sensor = world.spawn_actor(
                    blueprint_library.find('sensor.other.collision'),
                    carla.Transform(), attach_to=ego_vehicles[i])
                actor_list.append(collision_sensor)
            
            sync_mode.initialize_queues(actor_list, not no_rendering)
            tn = 0

            while tn < max_trajectory_per_spawn:
                if should_quit():
                    return
                
                clock.tick()
                sensor_data = sync_mode.tick(timeout=2.0)
                snapshot = sensor_data[0]
                
                if not no_rendering:
                    image_semseg = sensor_data[1]
                    col_events = sensor_data[2:]
                else:
                    col_events = sensor_data[1:]
                
                j = 0
                for i in range(len(dynamic_actors)):
                    carla_transform = dynamic_actors[i].get_transform()
                    loc = carla_transform.location
                    rot = carla_transform.rotation
                    orientation_vector[i].append([snapshot.elapsed_seconds, rot.yaw, rot.pitch, rot.roll, loc.x, loc.y, loc.z])

                    if 'hero' in dynamic_actors[i].attributes['role_name']:
                        complete, done = data_actor[j].step(snapshot, col_events[j])
                        if done:
                            break
                        j += 1
                        tn += complete
                if done:
                    break

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                if not no_rendering:
                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    draw_image(display, image_semseg)

                print('collected %d trajectories, data collection speed %f fps' % (tn, num_data_actor * clock.get_fps()))
                
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

            save_dir = os.path.join(save_path, str(div))
            os.makedirs(save_dir)
            dump = {'measurements' : measurements, 'orientation' : orientation_vector, 'type_actor' : type_actor}
            
            for actor in data_actor:
                dump[str(actor.id)] = actor.ego_commands

            with open(os.path.join(save_dir, 'trajectories'), 'wb') as dbfile:
                pickle.dump(dump, dbfile)

            div += 1
            sync_mode.remove_queues()
            actor_list = []
            ac.destroy_all_actors()

    except:
        traceback.print_exception(*sys.exc_info())
    
    finally:
        
        if ac:
            ac.destroy_all_actors()

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

