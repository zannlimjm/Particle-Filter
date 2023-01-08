import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import *

from pf import initialize_particles, mean_pose, sample_motion_model, eval_sensor_model, resample_particles

#add random seed for generating comparable pseudo random numbers
np.random.seed(1)

#plot preferences, interactive plotting mode
plt.axis([0, 10, 0, 10])
plt.ion()
plt.show()

def plot_state(particles, landmarks, actual_robot_pos, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.
    
    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(actual_robot_pos[0], actual_robot_pos[1], 'ko', markersize=5)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy',color='g')
    plt.quiver(actual_robot_pos[0], actual_robot_pos[1], np.cos(actual_robot_pos[2]), np.sin(actual_robot_pos[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)


def update_robot_pos(actual_robot_pos, odometry, map_limits):
    # Move robot to new position, based on old positions, the odometry, the motion noise 
    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]
    
    # standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans
        
    noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
    noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
    noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)

    #calculate new particle pose
    x = actual_robot_pos[0] + \
        noisy_delta_trans * np.cos(actual_robot_pos[2] + noisy_delta_rot1)
    y = actual_robot_pos[1] + \
        noisy_delta_trans * np.sin(actual_robot_pos[2] + noisy_delta_rot1)
    theta = actual_robot_pos[2] + \
        noisy_delta_rot1 + noisy_delta_rot2
 
    new_robot_pos = [x,y,theta]    
    
    return new_robot_pos

def update_sensor_readings(actual_robot_pos, landmarks):
    # Computes the sensor readings
    # The employed sensor model is range only.
    sigma_r = 0.2
    
    lm_ids=[]
    ranges = []
    bearings = []
    
    for lm_id in landmarks.keys():

        lx = landmarks[lm_id][0]
        ly = landmarks[lm_id][1]
        px = actual_robot_pos[0]
        py = actual_robot_pos[1]
        
        #calculate range measurement with added noise
        meas_range = np.sqrt( (lx - px)**2 + (ly - py)**2 ) + np.random.normal(loc=0.0, scale=sigma_r)
        meas_bearing = 0 # bearing is not computed
        
        lm_ids.append(int(lm_id))    
        ranges.append(meas_range)
        bearings.append(meas_bearing)
        
        sensor_readings = {'id':lm_ids,'range':ranges,'bearing':bearings} 
        
    return sensor_readings

def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("./world.dat")

    #initialize the particles
    map_limits = [0, 10, 0, 10]
    particles = initialize_particles(1000, map_limits)

    actual_robot_pos = read_pos("./pos.dat")

    accumulated = []
    num_iterations = []
    #run particle filter
    for iteration in range(100):
        num_iterations.append(iteration)        
        sensor_readings = 1

        #plot the current state
        plot_state(particles, landmarks, actual_robot_pos, map_limits)

        #move actual robot and sense state
        odometry = {'r1':0.1,'t':0.6,'r2':0.15} # constant motion
        actual_robot_pos = update_robot_pos(actual_robot_pos, odometry, map_limits)
        sensor_readings = update_sensor_readings(actual_robot_pos, landmarks)

        #predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(odometry, particles)

        #calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings, new_particles, landmarks)

        #resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)
        
        predicted_pos = mean_pose(particles)
        pred_error = np.sqrt( (predicted_pos[0] - actual_robot_pos[0])**2 + (predicted_pos[1] - actual_robot_pos[1])**2 ) 
        accumulated.append(pred_error)
        print('iter: %d, localization error: %.3f'%(iteration, pred_error))    
    
    print(num_iterations)
    print(accumulated)

    plt.show('hold')

if __name__ == "__main__":
    main()