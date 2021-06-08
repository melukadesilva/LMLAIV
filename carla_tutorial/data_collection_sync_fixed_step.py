#https://www.geeksforgeeks.org/context-manager-in-python/
# https://docs.python.org/3/library/queue.html
# https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py

#1- Define Context manager

#2- Define your sensors and data aquisition methods

#3- Consumption of data
#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import glob
import os
import sys
import argparse
import logging
import cv2
import open3d as o3d

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

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

#1- Implementation of the contex manager
class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, delta_seconds = 0.1):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = delta_seconds
        #print(kwargs.get('fps', 20))
        #print(self.delta_seconds)
        self._queues = []
        self._settings = None

    def __enter__(self):
        ## take the initial settings of the world
        self._settings = self.world.get_settings()
        ## At entrance put sync mode + fix step
        ## init the self.frame
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        ## if we want a world snapshot
        for sensor in self.sensors:
            make_queue(sensor.listen)
        
        return self

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        '''
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        '''

    def _retrieve_data(self, _queue, timeout):
        while True:
            data = _queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
    
    def tick(self, timeout):
        ## sync mode send a tick to the server for simulation frame compute
        ## id of the frame computed
        self._sync(self.world.tick())
        #self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        #print(data)
        assert all(x.frame == self.frame for x in data)
        return data
    # https://gist.github.com/nsubiron/b6effce218e10a855a1674fb74bc135c
    def _sync(self, frame):
        while frame > self.world.get_snapshot().timestamp.frame:
            pass
        assert frame == self.world.get_snapshot().timestamp.frame
        self.frame = frame

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

##2- Implement the sensors and data collection methods
## implementation of rgb draw + save
def draw_image(surface, image, save_path, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    cv2.imwrite(os.path.join(save_path, str(image.frame)+'.jpg'),
                array)
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def save_lidar(point_cloud, save_path):
    out = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    out = np.reshape(out, (int(out.shape[0] / 4), 4)) # -> nx4
    #print(out.shape)

    ## unpack xzy
    xyz_cloud = np.zeros((out.shape[0], 3)) # -> nx3
    xyz_cloud[:, 0] = out[:, 0]
    xyz_cloud[:, 1] = out[:, 1]
    xyz_cloud[:, 2] = out[:, 2]
    #print(xyz_cloud.shape)

    ## make open3d pointcloud object and store the data in the object
    pcd = o3d.geometry.PointCloud()
    ## convert and assign the points
    pcd.points = o3d.utility.Vector3dVector(xyz_cloud)
    #print(np.asarray(pcd.points))
    o3d.io.write_point_cloud(os.path.join(save_path, str(point_cloud.frame) +
                                          '.pcd'), pcd)

def draw_dvs(surface, image, save_path, blend=False):
    array = list()
    for e in image.to_image():
        array.append([e.r, e.g, e.b, e.a])
    #print(array[1])
    array = np.array(array)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    cv2.imwrite(os.path.join(save_path, str(image.frame)+'.jpg'),
                array)
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    #client = carla.Client('localhost', 2000)
    #client.set_timeout(2.0)


    try:
        world = client.get_world()
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)
        #print(waypoint)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            blueprint_library.find('vehicle.tesla.model3'), start_pose)
        #vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        #vehicle.set_simulate_physics(False)
        #vehicle.set_autopilot(True)
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        #cam_bp.set_attribute("image_size_x",str(1920))
        #cam_bp.set_attribute("image_size_y",str(1080))
        #cam_bp.set_attribute("fov",str(105))
        #cam_bp.set_attribute("motion_blur_intensity",str(1.0))
        #cam_bp.set_attribute("motion_blur_max_distortion",str(0.8))
        #cam_bp.set_attribute("chromatic_aberration_intensity",str(10))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        camera_rgb = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        actor_list.append(camera_rgb) 
        # --------------
        # Add a new LIDAR sensor to my ego
        # --------------
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(32))
        lidar_bp.set_attribute('points_per_second',str(120000))
        lidar_bp.set_attribute('rotation_frequency',str(10))
        lidar_bp.set_attribute('range',str(40))
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=vehicle)

        actor_list.append(lidar_sen)

        '''
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        '''
        # --------------
        # Add a new DVS sensor to my ego
        # --------------
        dvs_bp = world.get_blueprint_library().find('sensor.camera.dvs')
        #dvs_bp.set_attribute('positive_threshold', str(1.0))
        dvs_location = carla.Location(2,0,1)
        dvs_rotation = carla.Rotation(0,0,0)
        dvs_transform = carla.Transform(dvs_location,dvs_rotation)
        #print(dvs_transform.location)
        dvs_sen = world.spawn_actor(dvs_bp, dvs_transform,
                        attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        actor_list.append(dvs_sen)
        
        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=2, y=0, z=1),
                            carla.Rotation(0, 0, 0)),
                            attach_to=vehicle)

        actor_list.append(camera_semseg)

        spectator = world.get_spectator()
        spectator.set_transform(vehicle.get_transform())
        # Create a synchronous mode context.
        with CarlaSyncMode(world, *actor_list[1:], delta_seconds=0.1) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                image_rgb, lidar_out, image_dvs, image_semseg = \
                        sync_mode.tick(timeout=2.0)

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)


                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = 10# round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb, 'output/camera_rgb')
                #draw_image(display, image_semseg, blend=True)
                #draw_image(display, image_semseg, 'output/camera_seg',
                #           blend=True)
                save_lidar(lidar_out, 'output/lidar')
                #draw_dvs(display, image_dvs, 'output/camera_dvs', blend=True)
                #save_dvs(image_dvs)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()        

                

    finally:

        print('destroying actors.')
        for actor in actor_list:
            print(actor)
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
