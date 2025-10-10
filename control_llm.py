import carla
import numpy as np
import llm_feedback
import math
import time
import socket
import json

def main():
    # Connect to the Carla server
    print("Establing connection to Carla...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(300.0)
    print("...connected")

    # Start a TCP server to share data
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 5000))
    server_socket.listen(1)

    print("Waiting for connection...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # Load the Carla world
    world = client.load_world('circle_V4')
    # world = client.load_world('Circle_R50')
    world = client.get_world()
    print("...done")

    # Set the fixed time step to avoid the influcence of different computer performance
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10

    settings.synchronous_mode = True # Enables synchronous mode
    world.apply_settings(settings)

    # Select a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.mercedes.coupe_2020')[0]

    # Spawn the vehicle
    vehicle = None
    spawn_points = world.get_map().get_spawn_points()
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            # Vehicle could be spawned
            break

    
    physics_control = vehicle.get_physics_control()

    mass = physics_control.mass
    print("mass")
    print(mass)
    # Center of mass in absolute coordinates

    center_of_mass = physics_control.center_of_mass + vehicle.get_location()
    print("vehicle_location")
    print(vehicle.get_location())
    print("center_of_mass")
    print(center_of_mass)

    position = vehicle.get_location()
    print("vehicle_position")
    print(position)

    # WARNING: Wheel position unit is centimeters!

    wheel_front_left = physics_control.wheels[0]
    print("wheel_front_left_position")
    print(wheel_front_left.position)


    wheel_front_right = physics_control.wheels[1]
    print("wheel_front_right_position")
    print(wheel_front_right.position)

    wheel_back_left = physics_control.wheels[2]

    wheel_back_right = physics_control.wheels[3]

    wheel_stiffness_front_left = wheel_front_left.lat_stiff_value
    print("wheel_stiffness_front_left")
    print(wheel_stiffness_front_left)

    wheel_stiffness_back_left = wheel_back_left.lat_stiff_value
    print("wheel_stiffness_back_left")
    print(wheel_stiffness_back_left)

    wheel_stiffness_front_right = wheel_front_right.lat_stiff_value
    print("wheel_stiffness_front_right")
    print(wheel_stiffness_front_right)

    wheel_stiffness_back_right = wheel_back_right.lat_stiff_value
    print("wheel_stiffness_back_right")
    print(wheel_stiffness_back_right)

    max_steer_angle = wheel_front_left.max_steer_angle

    wheel_base = wheel_back_left.position.distance(wheel_front_left.position) / 100
    print("wheel_base")
    print(wheel_base)

    # The center point between the two front tires

    front_middle = 0.01 * (0.5 * (wheel_front_left.position + wheel_front_right.position))
    print("front middle")
    print(front_middle)

    # center_to_front = abs(center_of_mass.x - front_middle.x)
    center_to_front = 0.48 * wheel_base
    print("center_to_front")
    print(center_to_front)

    center_to_back = wheel_base - center_to_front
    print("center_to_back")
    print(center_to_back)

    ## Initialize the class of LLM
    # llm_control = llm_feedback.llm("Qwen/Qwen2.5-0.5B-Instruct", mass=mass, length=wheel_base, length_center_to_front=center_to_front,
    #     length_center_to_back=center_to_back, tire_stiffness_front=wheel_stiffness_front_left + wheel_stiffness_back_right,
    #     tire_stiffness_back=wheel_stiffness_back_left + wheel_stiffness_back_right)
    
    # Get vehicle control (the initialization of control algorithm with only acceleration)
    vehicle_control = carla.VehicleControl(throttle=0.4, steer=0.0, brake=0.0, hand_brake=False, reverse=False)


    positions = []
    waypoints_plot = []
    maximum_lateral_error = 0.0 # in meters
    maximum_orientation_erorr = 0.0 # in degrees
    orientation_error = []
    accumlated_lateral_error = 0.0
    accumlated_orientation_error = 0.0
    total_time = 0
    velocity_target = []
    velocity_real = []


    # Simulation loop

    # Parameters
    num_waypoints = 20 # Number of waypoints used in the trajectory

    try:

        print("Simulating the environment...")

        initial_time = world.get_snapshot().timestamp.elapsed_seconds
        initial_time_real = time.time()

        accumlated_error = 0.0
        last_error = 0.0
        last_time = world.get_snapshot().timestamp.elapsed_seconds - initial_time

        accumalted_error_long = 0.0
        last_error_long = 0.0
        time_array = []
        distance_real_array = []
        steering_angle_plot = []

        while True:

            # Simulate world

            world.tick()

            # world.wait_for_tick()

            now_time = world.get_snapshot().timestamp.elapsed_seconds - initial_time
            time_array.append(now_time)

            print(f"Second: {time.time() - initial_time}")

            time_elapsed = now_time - last_time
            
            # pygame.event.pump()
            # pygame.display.flip()


            # Get state

            velocity = vehicle.get_velocity()
            angular_velocity = vehicle.get_angular_velocity()         
            transform = vehicle.get_transform()
            position = transform.location
            orientation = transform.rotation # in degrees
            center_of_mass = physics_control.center_of_mass + vehicle.get_location()
            center_of_geometry = vehicle.get_location()

            current_control = vehicle.get_control()
            current_steer = current_control.steer

            waypoint = world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * 0)
            waypoints_plot.append(waypoint)
            waypoints_pid = []
            waypoints_mpc = []
            for i in range(num_waypoints):
                #  waypoints.append(world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * i)) # for MPC
                waypoints_pid.append(world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * i)) # for PID
            waypoints_array_pid = np.array(waypoints_pid)

            for i in range(num_waypoints):
                waypoints_mpc.append(world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * i)) # for MPC
                #  waypoints_pid.append(world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * (i - 5))) # for PID
            waypoints_array_mpc = np.array(waypoints_mpc)
            # waypoints = np.random.rand(num_waypoints)
            # for i in range (num_waypoints):
            #     waypoints[i] = world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * (i + 5))

            world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                    persistent_lines=True)
            
            # print(now_time * 10 % len(velocity_profile_target))
            # target_velocity = velocity_profile_target[int(now_time * 10 % len(velocity_profile_target))]
            # print('target velocity:')
            # print(target_velocity)
            # velocity_target.append(target_velocity)
            
            # # Calculate the deviation of the speed
            # print('acutal velocity:')
            # print(math.sqrt(velocity.x**2 + velocity.y**2))
            # velocity_real.append(math.sqrt(velocity.x**2 + velocity.y**2))
            # speed_error = target_velocity - math.sqrt(velocity.x**2 + velocity.y**2)
            # print(speed_error)

            # ## Calculate the Longitudinal Control using PID
            # # Calcualte the propotional part
            # propotional_error = speed_error
            # # Calculate the integrator part of PID (Euler-Forward method)
            # accumlated_error_long = accumalted_error_long + time_elapsed * speed_error
            # print('accumalted error')
            # print(accumlated_error_long)
            # # Calculate the derivator part of PID (increment method)
            # derivation_long = (speed_error - last_error_long) / time_elapsed
            # last_error_long = speed_error

            # control_signal_long = set_cp_long * propotional_error + set_ci_long * accumlated_error_long + set_cd_long * derivation_long
            
            ## Comment out the following code if you do prefer a constant speed of vehicle
            ## =========================================================================================================
            ## =========================================================================================================
            # if control_signal_long >= 0:
            #     if control_signal_long < 1:
            #         vehicle_control.throttle = control_signal_long
            #     else:
            #         vehicle_control.throttle = 1
            #     vehicle_control.brake = 0
            # else:
            #     if control_signal_long > -1:
            #         vehicle_control.brake = -control_signal_long
            #     else:
            #         vehicle_control.brake = 1
            #     vehicle_control.throttle = 0
            ## =========================================================================================================
            ## =========================================================================================================

            # print('Throttle position:')
            # print(vehicle_control.throttle)
            # print('Brake position:')
            # print(vehicle_control.brake)
            # Calculate control

            # steering_angle = simple_control.calculate_steering(

            #     d_y_req=0, psi_state=orientation.yaw+180,

            #     psi_dot_state=angular_velocity.z, x_waypoint=waypoint.transform.location.x,

            #     y_waypoint=waypoint.transform.location.y, x_state=center_of_geometry.x, y_state=center_of_geometry.y, 

            #     v_array=velocity, curv=0, beta_state=0, csteer = current_steer

            # )

            # steering_angle_mpc = mpc_control.calculate_steering(
                
            #     d_y_req=0, psi_state=orientation.yaw,

            #     psi_dot_state=angular_velocity.z, x_waypoint=waypoint.transform.location.x,

            #     y_waypoint=waypoint.transform.location.y, x_state=center_of_geometry.x, y_state=center_of_geometry.y, 

            #     v_array=velocity, curv=0, beta_state=0, csteer = current_steer, waypoints = waypoints_array_mpc, forward = orientation.get_forward_vector().make_unit_vector(), n_horizon = i_horizon
            # )

            # (steering_angle_pid, acc_error, lst_error) = pid_control.calculate_steering(
            #     waypoints=waypoints_array_pid, x_state=center_of_geometry.x, y_state=center_of_geometry.y, psi_state=orientation.yaw, aerror=accumlated_error, lerror=last_error, delta_time=time_elapsed, n_waypoints = num_waypoints
            # )
            # accumlated_error = acc_error
            # last_error = lst_error
            send_data = {
                "psi_state": orientation.yaw,
                "x_state": center_of_geometry.x,
                "y_state": center_of_geometry.y,
                "waypoints": waypoints_array_mpc,
                "n_waypoints": num_waypoints
            }

            json_data = json.dumps(send_data)
            conn.sendall(json_data.encode('utf-8'))
            print("Data sent to LLM")

            print("Waiting for data from LLM...")
            try:
                received_data = conn.recv(4096)
            
                if not received_data:
                    print("No data received, connection might be closed")
                    break

                # Process the received data
                qwen_output = float(received_data.decode('utf-8').strip())
                print("Received data from Qwen2.5:", qwen_output)

            except socket.timeout:
                print("Timeout occurred while waiting for data from Qwen2.5")
                break

            # steering_angle_llm = llm_control.calculate_steering(psi_state=orientation.yaw, x_state=center_of_geometry.x, y_state=center_of_geometry.y, waypoints = waypoints_array_mpc, n_waypoints=num_waypoints)

            # Combine the steering angle calculated from MPC und PID

            # steering_angle = steering_angle_mpc + steering_angle_pid
            steering_angle = qwen_output
            # print("MPC Steering")
            # print(steering_angle_mpc)
            # print("PID Steering")
            # print(steering_angle_pid)

            # # Apply low-pass filter to smooth out steering angle changes
            # alpha = time_elapsed / (time_constant + time_elapsed)
            # filtered_steering_angle = alpha * steering_angle + (1.0 - alpha) * filtered_steering_angle

            # Normalize steering angle between [-1; 1]

            # limited_steer_angle = min(max(steering_angle, -max_steer_angle), max_steer_angle)
            # limited_steer_angle = min(max(filtered_steering_angle, -max_steer_angle), max_steer_angle)
            limited_steer_angle = min(max(steering_angle, -max_steer_angle), max_steer_angle)
            vehicle_control.steer = limited_steer_angle / max_steer_angle
            steering_angle_plot.append(limited_steer_angle / max_steer_angle)
        

            # Apply control
        
            # print("Current position:")

            # print(position)

            positions.append(position)

            # print("Applying control:")

            # print(vehicle_control)

            ## Calculate the vehicle lateral error
            distance = (position.x - waypoint.transform.location.x)**2 + (position.y - waypoint.transform.location.y)**2
            accumlated_lateral_error += distance * time_elapsed


            ## Calculate the vehicle orientation error
            
            if len(waypoints_plot) > 30:   # Waypoints at the beginning of the trajectory leads to huge uncertainties in the calculation
                trajectory_orientation = math.atan2(
                    waypoints_plot[-1].transform.location.y - waypoints_plot[-2].transform.location.y,
                    waypoints_plot[-1].transform.location.x - waypoints_plot[-2].transform.location.x)
                orientation_forward = orientation.get_forward_vector()
                orientation_normalized = math.atan2(orientation_forward.y, orientation_forward.x)
                orientation_difference = math.degrees(trajectory_orientation) - math.degrees(orientation_normalized)
                orientation_difference = orientation_difference % 360
    
                # If the angle is greater than 180, shift it to be within [-180, 180)
                if orientation_difference > 180:
                    orientation_difference -= 360

                orientation_error.append(orientation_difference)
                accumlated_orientation_error += (orientation_difference)**2 * time_elapsed
                print('trajectory_orientation')
                print(math.degrees(trajectory_orientation))
                print('vehicle_orientation')
                print(math.degrees(orientation_normalized))
            else:
                orientation_error.append(0)

            if len(waypoints_plot) > 30:
                print("trajectory orientation:")
                print(math.degrees(trajectory_orientation))
                print("waypoint orientation:")
                print(orientation.yaw)
            

            vehicle.apply_control(vehicle_control)


            # Calculate the distance to the trajectory
            waypoint_single = world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * 0)
            waypoint_last = world.get_map().get_waypoint(vehicle.get_location() + orientation.get_forward_vector().make_unit_vector() * (-1))
            numerator = abs((waypoint_last.transform.location.y - waypoint_single.transform.location.y) * position.x - (waypoint_last.transform.location.x - waypoint_single.transform.location.x) * position.y + 
                            waypoint_last.transform.location.x * waypoint_single.transform.location.y - waypoint_last.transform.location.y * waypoint_single.transform.location.x)
            denominator = math.sqrt((waypoint_last.transform.location.y - waypoint_single.transform.location.x)**2 + (waypoint_last.transform.location.x - waypoint_single.transform.location.x)**2)
            distance_real = numerator / denominator

            distance_real_array.append(distance_real)

            # Stop the simulation after 10 seconds

            last_time = now_time

            # if time.time() - initial_time_real > 100:
            #     total_time = world.get_snapshot().timestamp.elapsed_seconds - initial_time
            #     print("...done")

            #     break

            if time.time() - initial_time_real > 50 and abs(waypoints_plot[0].transform.location.x - waypoints_plot[-1].transform.location.x) < 2 and abs(waypoints_plot[0].transform.location.y - waypoints_plot[-1].transform.location.y) < 2:
                total_time = world.get_snapshot().timestamp.elapsed_seconds - initial_time
                print("The elapsed time is:")
                print(time.time() - initial_time_real)
                print("...done")

                break



    finally:

        # Destroy actors and cleanup
        print("Succeed in continuing to next step")
        vehicle.destroy()
        settings.synchronous_mode = False # Disables synchronous mode
        world.apply_settings(settings)
        # camera.destroy()

