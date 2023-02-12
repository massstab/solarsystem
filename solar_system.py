#!/usr/bin/env python
# -*- coding: utf-8 -*-

### 13.10.19, Linder Dave, Simulating the solar system ###

import sys
import PIL
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D



plt.style.use('bmh')
plt.rc('axes', facecolor='k')
default_cycler = (cycler(color=['grey', 'goldenrod', 'cyan', 'darkred']))
plt.rc('axes', prop_cycle=default_cycler)


class Body:
    '''
    instanciate a planet with this class
    '''

    def __init__(self, planet, mass, x, y, z, vx, vy, vz, radius, color):
        self.name = planet
        self.mass = mass
        self.r = np.array([x, y, z])
        self.v = np.array([vx, vy, vz])
        self.radius = radius/ 150e6
        self.color = color


def acceleration(position, bodies):
    '''
    computes the acceleration on each planet
    :param position: the momentary position of the planet
    :param bodies: all planets of class Body
    :return: a numpy array of all the accelerations
    '''
    k = 0.01720209895
    accel = len(bodies) * [0]
    length = range(len(bodies))
    for i in length:
        for j in length:
            if j != i:
                delta_r = (position[j] - position[i])
                norm_r = np.linalg.norm(delta_r)
                accel[i] += k ** 2 * bodies[j].mass * norm_r ** -3 * delta_r
    return np.array(accel)


def leap_frog(bodies, h):
    """
    Numerical integration for a N-body problem with only gravitational acceleration due to the mass of the bodies
    :param bodies: Must be a list of instances of the class Body
    :param h: The size of each timestep delta t.
    :return: Returns a list of position vectors (x, y, z) for one timestep h for all bodies
    """
    # generating position vectors for all bodies
    length = range(len(bodies))
    position = []
    for m in length:
        position.append(bodies[m].r)

    # generating velocity vectors for all bodies
    velocity = []
    for i in length:
        velocity.append(bodies[i].v)

    # performing the fist half drift
    drift_half_ri = []
    for i in length:
        drift_half_i = position[i] + 0.5 * h * velocity[i]
        drift_half_ri.append(drift_half_i)

    # kick with calling the function acceleration. Also in this for loop the last kick is performed and the
    # intance variables v (velocity) and r (position) of each body is updated
    a = acceleration(drift_half_ri, bodies)
    kick_ri = []
    for i in length:
        kick_i = velocity[i] + h * a[i]
        bodies[i].v = kick_i
        drift_i = drift_half_ri[i] + 0.5 * h * kick_i
        bodies[i].r = drift_i
        kick_ri.append(drift_i)

    return kick_ri


def setup_solarsystem():
    """
    read the initial condition file and sets up the planets
    :return: all planets as instances of the class Body
    """
    # File to read in
    data_folder = Path("data/")
    file = data_folder / "SolSystData.dat"

    # Loading text files with NumPy
    data1 = np.loadtxt(file, delimiter=',', unpack=False,
                       dtype={'names': ('planet', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radius', 'color'),
                              'formats': (
                                  'U10', float, float, float, float, float, float, float, float,
                                  'U10')})

    # instantiating body class for planets and sun
    sun = Body(*data1[0])
    mercury = Body(*data1[1])
    venus = Body(*data1[2])
    earth = Body(*data1[3])
    mars = Body(*data1[4])
    jupiter = Body(*data1[5])
    saturn = Body(*data1[6])
    uranus = Body(*data1[7])
    neptun = Body(*data1[8])
    bodies = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptun]

    return bodies


def get_coordinates(o_days):
    """
    computes the position vector r for every planet
    :param o_days: how many days to simulate the outer planets (inner planets are sliced form outer planets)
    :return: position vector r and a list of instances (planets) of the Body class
    """
    r = []
    planets = setup_solarsystem()
    printcounter = 0
    progresscounter = 0
    print('calculating planet positions for {} days ...'.format(o_days))
    for day in range(o_days):
        coordinates = (leap_frog(planets, 1))
        r.append(coordinates)
        if printcounter == 1000:
            # write animation progress
            sys.stdout.write('\r')
            sys.stdout.write('{0:.0f}%'.format(100 * progresscounter / o_days))
            sys.stdout.flush()
            printcounter = 0
        printcounter += 1
        progresscounter += 1
    r = np.array(r)
    return r, planets


def two_d_plot(i_days, o_days):
    """
    plots a 2x2 images with a top and a side view of the solar system
    :param i_days: days to simulate the inner planets
    :param o_days: days to simulate the outer planets
    :return: figure, the 4 axes and the position vector r of all planets in a numpy array
    """

    data_folder = Path("data/")
    media_folder = Path("media/")
    file_sim = data_folder / "r_simulated.npy"
    file_planets = data_folder / "planets.npy"

    ### Uncomment if saved location data ###
    # r = np.load(file_sim)
    # planets = np.load(file_planets, allow_pickle=True)

    ### Uncomment if no saved location data or new ones needed ###
    r, planets = get_coordinates(o_days)
    np.save(file_sim, r)
    np.save(file_planets, planets)

    x_inner, y_inner, z_inner = r[:i_days, 0:5, 0], r[:i_days, 0:5, 1], 2 * r[:i_days, 0:5, 2]
    x_outer, y_outer, z_outer = r[:o_days, 5:9, 0], r[:o_days, 5:9, 1], 2 * r[:o_days, 5:9, 2]

    labels = []
    for planet in planets:
        labels.append(planet.name.strip('\''))

    fig, axs = plt.subplots(2, 2, figsize=(12, 7.2), facecolor='white')
    img = plt.imread('media/milkyway_old.jpg')
    fig.figimage(img, alpha=.45)

    axs[0, 0].axis('equal')
    axs[0, 0].plot(x_inner, y_inner)
    axs[0, 0].plot(x_inner[-1], y_inner[-1], '.', color='grey')
    axs[0, 0].plot(x_inner[-1][0], y_inner[-1][0], 'o', color='yellow')
    axs[0, 0].set_title('x, y - Plane of the 4 Inner Planets')
    legend = axs[0, 0].legend(labels[0:5], loc=1)
    axs[0, 0].text(0.1, 0.9, 'simulating\n{} days'.format(i_days),
                   color='w', ha='center', va='center', transform=axs[0, 0].transAxes)
    plt.setp(legend.get_texts(), color='w')

    axs[0, 1].axis('equal')
    axs[0, 1].plot(x_outer, y_outer)
    axs[0, 1].plot(x_outer[-1], y_outer[-1], '.', color='grey')
    axs[0, 1].plot(x_inner[-1][0], y_inner[-1][0], 'o', color='yellow')
    axs[0, 1].set_title('x, y - Plane of the 4 Outer Planets')
    legend = axs[0, 1].legend(labels[5:], loc=1)
    axs[0, 1].text(0.1, 0.9, 'simulating\n{} days'.format(o_days),
                   color='w', ha='center', va='center', transform=axs[0, 1].transAxes)
    plt.setp(legend.get_texts(), color='w')

    axs[1, 0].axis('equal')
    axs[1, 0].plot(x_inner, z_inner)
    axs[1, 0].plot(x_inner[-1], z_inner[-1], '.', color='grey')
    axs[1, 0].plot(x_inner[-1][0], z_inner[-1][0], 'o', color='yellow')
    axs[1, 0].set_xlabel('AU', color='w')
    axs[1, 0].xaxis.set_label_coords(0.5, 0.1)
    axs[1, 0].set_title('x, z - Plane of the 4 Inner Planets')

    axs[1, 1].axis('equal')
    axs[1, 1].plot(x_outer, z_outer, label='jupiter')
    axs[1, 1].plot(x_outer[-1], z_outer[-1], '.', color='grey')
    axs[1, 1].plot(x_inner[-1][0], z_inner[-1][0], 'o', color='yellow')
    axs[1, 1].set_xlabel('AU', color='w')
    axs[1, 1].xaxis.set_label_coords(0.5, 0.1)
    axs[1, 1].set_title('x, z - Plane of the 4 Outer Planets')

    plt.tight_layout()
    file_plot = media_folder / "solarsystem_simulation"
    plt.savefig(file_plot)
    plt.show()

    return fig, axs, r


def two_d_animation(i_days, o_days):
    """
    saves a mp4 animation in the media folder
    :param i_days: days to simulate the inner planets
    :param o_days: days to simulate the outer planes
    :return: print 'animation done'
    """
    frames = i_days

    data_folder = Path("data/")
    file_sim = data_folder / "r_simulated.npy"
    file_planets = data_folder / "planets.npy"
    
    ### Uncomment if saved location data ###
    r = np.load(file_sim)
    planets = np.load(file_planets, allow_pickle=True)

    ### Uncomment if no saved location data or new ones needed ###
    # r, planets = get_coordinates(o_days)
    # np.save(file_sim, r)
    # np.save(file_planets, planets)

    x_sun, y_sun, z_sun = r[:i_days, 0, 0], r[:i_days, 0, 1], r[:i_days, 0, 2]
    x_inner, y_inner, z_inner = r[:i_days, 1:5, 0], r[:i_days, 1:5, 1], r[:i_days, 1:5, 2]
    x_outer, y_outer, z_outer = r[:o_days, 5:9, 0], r[:o_days, 5:9, 1], r[:o_days, 5:9, 2]

    labels = []
    for planet in planets[1:]:
        labels.append(planet.name.strip('\''))

    fig, axs = plt.subplots(2, 2, figsize=(3840/240, 2160/240), dpi=240, facecolor='white')
    img = PIL.Image.open('media/milkyway-2.jpg')
    fig.figimage(img, alpha=.45)

    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].axis('equal')
    axs[0, 0].plot(x_inner, y_inner)
    axs[0, 0].plot(x_sun[-1], z_sun[-1], 'o', color='yellow')
    axs[0, 0].set_title('x, y - Plane of the 4 Inner Planets')
    legend = axs[0, 0].legend(labels[0:4], loc=1)

    plt.setp(legend.get_texts(), color='w')

    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].axis('equal')
    axs[0, 1].plot(x_outer, y_outer)
    axs[0, 1].plot(x_sun[-1], z_sun[-1], 'o', color='yellow')
    axs[0, 1].set_title('x, y - Plane of the 4 Outer Planets')
    legend = axs[0, 1].legend(labels[4:], loc=1)

    plt.setp(legend.get_texts(), color='w')

    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].axis('equal')
    axs[1, 0].plot(x_inner, z_inner)
    axs[1, 0].plot(x_sun[-1], z_sun[-1], 'o', color='yellow')
    axs[1, 0].set_xlabel('AU', color='w')
    axs[1, 0].xaxis.set_label_coords(0.5, 0.1)
    axs[1, 0].set_title('x, z - Plane of the 4 Inner Planets')

    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].axis('equal')
    axs[1, 1].plot(x_outer, z_outer)
    axs[1, 1].plot(x_sun[-1], z_sun[-1], 'o', color='yellow')
    axs[1, 1].set_xlabel('AU', color='w')
    axs[1, 1].xaxis.set_label_coords(0.5, 0.1)
    axs[1, 1].set_title('x, z - Plane of the 4 Outer Planets')

    plt.tight_layout()

    line_y_inner, = axs[0, 0].plot([], [], '.', markersize='6', color='w')
    day_text_inner = axs[0, 0].text(0.1, 0.9, '', color='w', ha='center', va='center', transform=axs[0, 0].transAxes)
    line_y_outer, = axs[0, 1].plot([], [], '.', markersize='6', color='w')
    day_text_outer = axs[0, 1].text(0.1, 0.9, '', color='w', ha='center', va='center', transform=axs[0, 1].transAxes)
    line_z_inner, = axs[1, 0].plot([], [], '.', markersize='6', color='w')
    line_z_outer, = axs[1, 1].plot([], [], '.', markersize='6', color='w')

    print('animate frames:')

    def animate(i):
        x_inner, y_inner, z_inner = r[:i_days, 1:5, 0][i], r[:i_days, 1:5, 1][i], r[:i_days, 1:5, 2][i]
        x_outer, y_outer, z_outer = r[:o_days, 5:9, 0][87 * i], r[:o_days, 5:9, 1][87 * i], r[:o_days, 5:9, 2][87 * i]
        line_y_inner.set_data(x_inner, y_inner)
        day_text_inner.set_text('simulating\nday {:03d}'.format(i))
        line_y_outer.set_data(x_outer, y_outer)
        day_text_outer.set_text('simulating\nday {:03d}'.format(87 * i))
        line_z_inner.set_data(x_inner, z_inner)
        line_z_outer.set_data(x_outer, z_outer)
        # write animation progress
        sys.stdout.write('\r')
        sys.stdout.write("{0:.0f}%".format(i * 100 / frames))
        sys.stdout.flush()

        return line_y_inner, line_y_outer, line_z_inner, line_z_outer

    anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False)
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Dave Linder'), bitrate=8000, codec='h264')
    anim.save('media/solarsystem_anim_4k.mp4', writer=writer)

    return print('animation done')


def sun_wobbling(i_days, o_days):
    frames = i_days

    data_folder = Path("data/")
    media_folder = Path("media/")
    file_sim = data_folder / "r_simulated.npy"
    file_planets = data_folder / "planets.npy"
    
    ### Uncomment if saved location data ###
    # r = np.load(file_sim)
    # planets = np.load(file_planets, allow_pickle=True)

    ### Uncomment if no saved location data or new ones needed ###
    r, planets = get_coordinates(o_days)
    np.save(file_sim, r)
    np.save(file_planets, planets)

    sun = planets[0]

    x_sun, y_sun, z_sun = r[:o_days, 0:1, 0], r[:o_days, 0:1, 1], r[:o_days, 0:1, 2]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7.2), facecolor='white')
    img = plt.imread('media/milkyway_old.jpg')
    img_sun = plt.imread('media/sun.png')
    fig.figimage(img, alpha=.45)
    ax.imshow(img_sun, extent=[-img_sun.shape[1] / 2. + 200, img_sun.shape[1] / 2. + 200, -img_sun.shape[0] / 2.,
                               img_sun.shape[0] / 2.])

    ax.axis('equal')
    ax.plot(x_sun, y_sun, ls='')
    ax.set_title('x, y - plane with sun wobbling mainly because of jupiter?')

    plt.tight_layout()

    line_sun, = ax.plot([], [], 'o', color='yellow', markersize=20)
    line_jupiter, = ax.plot([], [])
    day_text = ax.text(0.1, 0.9, '', color='w', ha='center', va='center', transform=ax.transAxes)

    print('animate frames:')

    def animate(i):
        ax.clear()
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        x_sun, y_sun = r[:o_days, :1, 0][50 * i], r[:o_days, :1, 1][50 * i]
        ax.imshow(img_sun,
                  extent=[-sun.radius + x_sun[0], sun.radius + x_sun[0], -sun.radius + y_sun[0],
                          sun.radius + y_sun[0]])
        day_text.set_text('simulating\nday {:03d}'.format(50 * i))

        # write animation progress
        sys.stdout.write('\r')
        sys.stdout.write("{0:.0f}%".format(i * 100 / frames))
        sys.stdout.flush()

        return line_sun, line_jupiter

    anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
    file_anim = media_folder / "sun_wobbling.mp4"
    anim.save(file_anim, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
    return print('animation done')


def sun_wobbling_3d(i_days, o_days):
    ### Work in Progress. Animation has error: ValueError: too many values to unpack (expected 2) ###
    frames = i_days

    data_folder = Path("data/")
    media_folder = Path("media/")
    file_sim = data_folder / "r_simulated.npy"
    file_planets = data_folder / "planets.npy"
    
    ### Uncomment if saved location data ###
    # r = np.load(file_sim)
    # planets = np.load(file_planets, allow_pickle=True)

    ### Uncomment if no saved location data or new ones needed ###
    r, planets = get_coordinates(o_days)
    np.save(file_sim, r)
    np.save(file_planets, planets)

    sun = planets[0]

    x_sun, y_sun, z_sun = r[:o_days, 0, 0], r[:o_days, 0, 1], r[:o_days, 0, 2]
    # X,Y = np.meshgrid(x_all, y_all)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    img = plt.imread('media/milkyway_old.jpg')
    img_sun = plt.imread('media/sun.png')
    # fig.figimage(img, alpha=.5)

    sun, = ax.plot(x_sun, y_sun, z_sun)
    # ax.plot(x_jupiter, y_jupiter, ls='')
    # ax.set_title('x, y - plane with sun wobbling mainly because of jupiter?')

    # line_sun, = ax.plot([], [], 'o', color='yellow', markersize=20)
    # line_jupiter, = ax.plot([], [])
    # ax.imshow(img_sun, extent=[-img_sun.shape[1]/200., img_sun.shape[1]/200., -img_sun.shape[0]/200., img_sun.shape[0]/200. ])
    # day_text = ax.text(0.1, 0.9, 0.1, '', color='w', ha='center', va='center', transform=ax.transAxes)

    plt.show()

    print('animate frames:')

    def animate(i):
        ax.clear()
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        x_sun, y_sun, z_sun = r[:o_days, 0, 0][50 * i], r[:o_days, :0, 1][50 * i], r[:o_days, :0, 2][50 * i]
        # ax.imshow(img_sun,
        #           extent=[-sun.radius + x_sun[0], sun.radius + x_sun[0], -sun.radius + y_sun[0],
        #                   sun.radius + y_sun[0]])
        sun.set_data(x_sun, y_sun, z_sun)
        day_text.set_text('simulating\nday {:03d}'.format(50 * i))

        # write animation progress
        sys.stdout.write('\r')
        sys.stdout.write("{0:.0f}%".format(i * 100 / frames))
        sys.stdout.flush()

        return line_sun, line_jupiter

    anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
    file_anim = media_folder / "sun_wobbling_3d.mp4"
    anim.save(file_anim, fps=30, extra_args=['-vcodec', 'libx264'])

    # plt.show()
    return print('animation done')


def main():
    ### i_days, o_days: how many days to simulate the inner and outer planets respectively ###
    i_days = 6  # 687 days is one mars-year
    o_days = 601  # 60182 days is one neptune year
    two_d_plot(i_days, o_days)

    ### uncomment this lines to save an animation in the media folder ###
    # two_d_animation(i_days, o_days)
    # sun_wobbling(i_days, o_days)
    # sun_wobbling_3d(i_days, o_days)


if __name__ == "__main__":
    main()

    """
    # 3D plot of the wobbling of the sun
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = np.load('data/r_simulated.npy')
    x = r[:, 0, 0]
    y = r[:, 0, 1]
    z = r[:, 0, 2]
    plt.plot(x, y, z)
    plt.show()
    """