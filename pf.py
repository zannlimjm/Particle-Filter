import numpy as np
import scipy.stats
import random

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits
    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles


def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]
    
    # standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans
    
    # generate new particle set after motion update
    new_particles = []

    for particle in particles:
        new_particle = dict()
        #sample noisy motions
        noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
        noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
        noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)
        
        #calculate new particle pose
        new_particle['x'] = particle['x'] + \
            noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
        new_particle['y'] = particle['y'] + \
            noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
        new_particle['theta'] = particle['theta'] + \
            noisy_delta_rot1 + noisy_delta_rot2
        new_particles.append(new_particle)
    return new_particles


def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []
    
    '''my code here'''
    landmarks = list(landmarks.values())
    for num in range(len(particles)):
        for i in range(len(landmarks)):
            particleWeight = 1
            distance = np.power((particles[num]['x'] - landmarks[i][0])**2 +(particles[num]['y'] - landmarks[i][1])**2,0.5)
            #multiply for every landmark
            particleWeight *= int(scipy.stats.norm(distance, ((sigma_r)**2)).pdf(ranges[i]))
            weights = [i + 1.e-300 for i in weights] #prevents rounding off weights to 0
        weights.append(particleWeight)

    #higher weight value means particles is closer to the landmark 
    #normalize weights
    normalizer = sum(weights)
    newWeights = [i / normalizer for i in weights]
    
    #Following given code not working because: unsupported operand type(s) for /: 'list' and 'int'
    # weights = weights / normalizer

    return newWeights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []
    new_particles_weights = []

    '''your code here'''
    
    #systematic sampling
    # N = len(weights)
    # if 1./sum(np.power(weights,2)) < N/2.:
    #     positions = (np.arange(N) + np.random.random()) / N
    #     indexes = np.zeros(N, 'i')
    #     cumulative_sum = np.cumsum(weights)
    #     #sorting out the ones with high weights
    #     i, j = 0, 0
    #     while i < N and j<N:
    #         if positions[i] < cumulative_sum[j]:
    #             indexes[i] = j
    #             i += 1
    #         else:
    #             j += 1

    #residual sampling
    n = len(weights)
    indexes = np.zeros(n, np.uint32)
    # take int(N*w) copies of each weight
    num_copies = (n * np.asarray(weights)).astype(np.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indexes[k] = i
            k += 1
    # use multinormial resample on the residual to fill up the rest.
    residual = weights - num_copies  # get fractional part
    residual /= np.sum(residual)
    cumsum = np.cumsum(residual)
    cumsum[-1] = 1
    # array of index of particles chosen
    indexes[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))

    #Code for -> particles[:] = particles[indexes]
    for i in range(len(indexes)):
        new_particles.append(particles[indexes[i]])
        #Code for -> weights[:] = weights[indexes]
        new_particles_weights.append(weights[indexes[i]]) #not needed since only returning sample of new particles

    # particles[:] = particles[indexes]
    # weights[:] = weights[indexes]
    # weights /= np.sum(weights)

    return new_particles

