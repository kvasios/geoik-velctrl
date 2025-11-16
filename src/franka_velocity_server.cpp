// Geometric IK-based Velocity Control Server for Franka Robot
// Adapted from franka-vr-teleop (https://github.com/wengmister/franka-vr-teleop)
// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE-APACHE-2.0 and LICENSE-FRANKA-VR-TELEOP
#include <cmath>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <array>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#include <franka/exception.h>
#include <franka/robot.h>
#include <Eigen/Dense>

#include "examples_common.h"
#include "weighted_ik.h"
#include <ruckig/ruckig.hpp>

struct PoseCommand
{
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
    double quat_x = 0.0, quat_y = 0.0, quat_z = 0.0, quat_w = 1.0;
    bool has_valid_data = false;
};

class VelocityServer
{
private:
    std::atomic<bool> running_{true};
    PoseCommand current_pose_command_;
    std::mutex command_mutex_;

    int server_socket_;
    const int PORT = 8888;

    // Control parameters
    struct ControlParams
    {
        double smoothing = 0.05;       // Less for more responsive control (VR mode)

        // Deadzones to prevent drift from small sensor noise
        double position_deadzone = 0.001;   // 1mm
        double orientation_deadzone = 0.03; // ~1.7 degrees

        // Workspace limits to keep the robot in a safe area
        double max_position_offset = 0.75;   // 75cm from initial position
        
        // Visual servoing specific parameters
        double vs_position_gain = 0.6;           // Position gain (0.1-1.0, how fast to follow position)
        double vs_orientation_gain = 0.3;        // Orientation gain (0.01-0.5, how fast to follow rotation)
        double vs_smoothing = 0.75;              // EMA smoothing for VS target (0.0-0.95, higher = smoother)
        double vs_position_deadband = 0.002;     // 2mm position deadband (ignore small position errors)
        double vs_orientation_deadband = 0.02;   // ~1.15° orientation deadband (ignore small rotation errors)
    } params_;

    // Target Pose (in robot base frame)
    Eigen::Vector3d target_position_;
    Eigen::Quaterniond target_orientation_;

    // Filtering state for smooth control
    Eigen::Vector3d filtered_position_{0, 0, 0};
    Eigen::Quaterniond filtered_orientation_{1, 0, 0, 0};

    // Initial poses used as a reference frame
    Eigen::Affine3d initial_robot_pose_;
    Eigen::Vector3d initial_command_position_{0, 0, 0};
    Eigen::Quaterniond initial_command_orientation_{1, 0, 0, 0};
    bool initialized_ = false;

    // Joint space tracking
    std::array<double, 7> current_joint_angles_;
    std::array<double, 7> neutral_joint_pose_;
    std::unique_ptr<WeightedIKSolver> ik_solver_;
    
    // Q7 limits
    double Q7_MIN;
    double Q7_MAX;
    bool bidexhand_;
    static constexpr double Q7_SEARCH_RANGE = 0.5; // look for q7 angle candidates in +/- this value in the current joint range 
    static constexpr double Q7_OPTIMIZATION_TOLERANCE = 1e-6; // Tolerance for optimization
    static constexpr int Q7_MAX_ITERATIONS = 100; // Max iterations for optimization

    // Ruckig trajectory generator for smooth joint space motion
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7> ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool ruckig_initialized_ = false;
    
    // Gradual activation to prevent sudden movements
    std::chrono::steady_clock::time_point control_start_time_;
    static constexpr double ACTIVATION_TIME_SEC = 0.5; // Faster activation
    
    // Franka joint limits for responsive teleoperation 
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY = {1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0};     // Increase for responsiveness
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION = {4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0}; // Increase for snappier response
    static constexpr std::array<double, 7> MAX_JOINT_JERK = {8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0};  // Higher jerk for snappier response
    static constexpr double CONTROL_CYCLE_TIME = 0.001;  // 1 kHz

    // Visual servoing (eye-in-hand) mode flag
    bool vs_mode_ = false;          // true: interpret incoming data as EE-relative marker pose
    bool vs_has_ref_ = false;       // have we locked the reference EE→marker transform?
    bool vs_has_meas_ = false;      // do we have at least one measurement?
    Eigen::Affine3d T_base_marker_ref_;  // (optional) BASE→marker at lock, for debug only
    Eigen::Affine3d T_ee_marker_ref_;    // reference EE→marker (relative pose we want to keep)
    Eigen::Affine3d T_ee_marker_meas_;   // latest measured EE→marker
    
    // Visual servo filtered state
    Eigen::Vector3d vs_filtered_position_{0, 0, 0};
    Eigen::Quaterniond vs_filtered_orientation_{1, 0, 0, 0};
    
    // Measurement timeout (safety: stop if marker lost)
    std::chrono::steady_clock::time_point last_measurement_time_;
    static constexpr double VS_MEASUREMENT_TIMEOUT_SEC = 0.2;  // 200ms (generous for 30Hz camera)

public:
    VelocityServer(bool bidexhand = true, bool vs_mode = false)
        : Q7_MIN(bidexhand ? -0.2 : -2.89),
          Q7_MAX(bidexhand ? 1.9 : 2.89),
          bidexhand_(bidexhand),
          vs_mode_(vs_mode)
    {
        setupNetworking();
    }

    ~VelocityServer()
    {
        running_ = false;
        close(server_socket_);
    }

    void setupNetworking()
    {
        server_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (server_socket_ < 0)
        {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(PORT);

        if (bind(server_socket_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            throw std::runtime_error("Failed to bind socket");
        }

        int flags = fcntl(server_socket_, F_GETFL, 0);
        fcntl(server_socket_, F_SETFL, flags | O_NONBLOCK);

        std::cout << "UDP server listening on port " << PORT << " for end-effector pose commands" << std::endl;
    }

    void networkThread()
    {
        char buffer[1024];
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        while (running_)
        {
            ssize_t bytes_received = recvfrom(server_socket_, buffer, sizeof(buffer), 0,
                                              (struct sockaddr *)&client_addr, &client_len);

            if (bytes_received > 0)
            {
                buffer[bytes_received] = '\0';
                std::string message(buffer);

                if (vs_mode_)
                {
                    // Visual-servo mode: interpret incoming data as EE→marker measurement
                    double px, py, pz, qx, qy, qz, qw;
                    int parsed = std::sscanf(message.c_str(), "%lf %lf %lf %lf %lf %lf %lf",
                                             &px, &py, &pz, &qx, &qy, &qz, &qw);
                    if (parsed == 7)
                    {
                        Eigen::Vector3d p(px, py, pz);
                        Eigen::Quaterniond q(qw, qx, qy, qz);
                        q.normalize();

                        Eigen::Affine3d T;
                        T.linear() = q.toRotationMatrix();
                        T.translation() = p;

                        {
                            std::lock_guard<std::mutex> lock(command_mutex_);
                            T_ee_marker_meas_ = T;
                            vs_has_meas_ = true;
                            last_measurement_time_ = std::chrono::steady_clock::now();  // Update timeout timer

                            // Note: reference locking happens in control loop (needs robot_state)

                            if (!initialized_)
                            {
                                initialized_ = true;
                                std::cout << "Visual servo: first measurement received, starting control." << std::endl;
                            }
                        }
                    }
                }
                else
                {
                    // VR / base-frame mode: interpret incoming data as base-frame pose command
                    PoseCommand cmd;
                    int parsed_count = std::sscanf(message.c_str(), "%lf %lf %lf %lf %lf %lf %lf",
                                                   &cmd.pos_x, &cmd.pos_y, &cmd.pos_z,
                                                   &cmd.quat_x, &cmd.quat_y, &cmd.quat_z, &cmd.quat_w);

                    if (parsed_count == 7)
                    {
                        cmd.has_valid_data = true;
                    }

                    if (cmd.has_valid_data)
                    {
                        std::lock_guard<std::mutex> lock(command_mutex_);
                        current_pose_command_ = cmd;

                        if (!initialized_)
                        {
                            initial_command_position_ = Eigen::Vector3d(cmd.pos_x, cmd.pos_y, cmd.pos_z);
                            initial_command_orientation_ = Eigen::Quaterniond(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z).normalized();

                            filtered_position_ = initial_command_position_;
                            filtered_orientation_ = initial_command_orientation_;

                            initialized_ = true;
                            std::cout << "Initial base-frame pose command received (VR mode)!" << std::endl;
                        }
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    // Calculate the desired target pose from incoming base-frame pose commands (VR mode)
    void updateTargetPose(const PoseCommand &cmd)
    {
        if (!cmd.has_valid_data || !initialized_)
        {
            return;
        }

        // Current commanded pose in BASE frame
        Eigen::Vector3d cmd_pos(cmd.pos_x, cmd.pos_y, cmd.pos_z);
        Eigen::Quaterniond cmd_quat(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z);
        cmd_quat.normalize();

        // Smooth incoming data to reduce jitter
        double alpha = 1.0 - params_.smoothing;
        filtered_position_ = params_.smoothing * filtered_position_ + alpha * cmd_pos;
        filtered_orientation_ = filtered_orientation_.slerp(alpha, cmd_quat);

        // Calculate deltas from the initial commanded pose
        Eigen::Vector3d pos_delta = filtered_position_ - initial_command_position_;
        Eigen::Quaterniond quat_delta = filtered_orientation_ * initial_command_orientation_.inverse();

        // Apply deadzones to prevent drift
        if (pos_delta.norm() < params_.position_deadzone)
        {
            pos_delta.setZero();
        }
        double rotation_angle = 2.0 * acos(std::abs(quat_delta.w()));
        if (rotation_angle < params_.orientation_deadzone)
        {
            quat_delta.setIdentity();
        }

        // Apply workspace limits
        if (pos_delta.norm() > params_.max_position_offset)
        {
            pos_delta = pos_delta.normalized() * params_.max_position_offset;
        }

        // The final calculation updates the target pose
        target_position_ = initial_robot_pose_.translation() + pos_delta;
        target_orientation_ = quat_delta * Eigen::Quaterniond(initial_robot_pose_.rotation());
        target_orientation_.normalize();
    }

    // Visual servo: compute EE target pose to keep marker fixed in BASE frame
    void updateVisualServoTarget(const franka::RobotState &robot_state)
    {
        if (!vs_mode_ || !vs_has_meas_ || !initialized_)
        {
            return;
        }

        // Check measurement timeout (marker lost = safety stop)
        auto current_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - last_measurement_time_).count();
        
        if (elapsed > VS_MEASUREMENT_TIMEOUT_SEC)
        {
            if (vs_has_ref_)
            {
                std::cout << "Visual servo: marker lost (timeout " << elapsed*1000 << "ms), resetting reference." << std::endl;
                vs_has_ref_ = false;  // Require new lock when measurements resume
            }
            // Keep last target (freeze in place) - don't update target_position/orientation
            return;
        }

        // Current base→ee
        Eigen::Affine3d T_base_ee_current(
            Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

        // Lock reference on first call: store EE→marker reference (relative pose)
        if (!vs_has_ref_)
        {
            // Reference: keep this EE→marker transform constant while marker moves in world
            T_ee_marker_ref_ = T_ee_marker_meas_;
            vs_has_ref_ = true;

            // Optionally store BASE→marker at lock for debugging
            T_base_marker_ref_ = T_base_ee_current * T_ee_marker_meas_;

            // Initialize filtered state to current pose
            vs_filtered_position_ = T_base_ee_current.translation();
            vs_filtered_orientation_ = Eigen::Quaterniond(T_base_ee_current.rotation());

            // Initialize target to current pose (no motion initially)
            target_position_ = vs_filtered_position_;
            target_orientation_ = vs_filtered_orientation_;

            std::cout << "Visual servo: locked reference EE→marker." << std::endl;
            std::cout << "  Position gain: " << params_.vs_position_gain 
                      << ", Orientation gain: " << params_.vs_orientation_gain << std::endl;
            std::cout << "  Smoothing: " << params_.vs_smoothing << std::endl;
            std::cout << "  Deadbands: position " << params_.vs_position_deadband*1000 << "mm"
                      << ", orientation " << params_.vs_orientation_deadband*180/M_PI << "°" << std::endl;
            return;
        }

        // Servo law: Track marker position and orientation
        
        // ===== POSITION SERVOING =====
        // Compute marker position offset from reference in EE frame
        Eigen::Vector3d marker_pos_ref = T_ee_marker_ref_.translation();
        Eigen::Vector3d marker_pos_meas = T_ee_marker_meas_.translation();
        Eigen::Vector3d marker_delta_ee = marker_pos_meas - marker_pos_ref;
        
        // Transform delta to base frame
        Eigen::Vector3d marker_delta_base = T_base_ee_current.rotation() * marker_delta_ee;
        
        // Move EE in same direction as marker moved to maintain relative offset
        Eigen::Vector3d raw_target_pos = T_base_ee_current.translation() + marker_delta_base;
        
        // ===== ORIENTATION SERVOING =====
        // Goal: Maintain EE→marker relative orientation
        // R_base_ee_target * R_ee_marker_ref = R_base_ee_current * R_ee_marker_meas
        // Therefore: R_base_ee_target = R_base_ee_current * R_ee_marker_meas * R_ee_marker_ref^T
        
        Eigen::Matrix3d R_base_ee_current = T_base_ee_current.rotation();
        Eigen::Matrix3d R_ee_marker_ref = T_ee_marker_ref_.rotation();
        Eigen::Matrix3d R_ee_marker_meas = T_ee_marker_meas_.rotation();
        
        Eigen::Matrix3d R_base_ee_target = R_base_ee_current * R_ee_marker_meas * R_ee_marker_ref.transpose();
        
        Eigen::Quaterniond raw_target_quat(R_base_ee_target);
        Eigen::Quaterniond current_quat(R_base_ee_current);
        raw_target_quat.normalize();
        current_quat.normalize();
        
        // Handle quaternion double-cover: ensure we take the shortest path
        if (raw_target_quat.dot(current_quat) < 0.0) {
            raw_target_quat.coeffs() = -raw_target_quat.coeffs();
        }
        
        // ===== POSITION GAIN & DEADBAND =====
        Eigen::Vector3d current_pos = T_base_ee_current.translation();
        Eigen::Vector3d pos_delta = raw_target_pos - current_pos;
        
        // Apply proportional gain to position (damping)
        pos_delta *= params_.vs_position_gain;
        
        // Apply per-axis deadband to prevent micro-oscillations
        for (int i = 0; i < 3; ++i)
        {
            if (std::abs(pos_delta[i]) < params_.vs_position_deadband)
            {
                pos_delta[i] = 0.0;
            }
        }
        
        // Compute damped target position
        Eigen::Vector3d damped_target_pos = current_pos + pos_delta;
        
        // ===== ORIENTATION GAIN & DEADBAND =====
        // Compute angular distance between current and target orientation
        double angular_distance = 2.0 * std::acos(std::abs(current_quat.dot(raw_target_quat)));
        
        Eigen::Quaterniond damped_target_quat;
        if (angular_distance < params_.vs_orientation_deadband)
        {
            // Within deadband: keep current orientation (no rotation command)
            damped_target_quat = current_quat;
        }
        else
        {
            // Apply gain through SLERP (move only a fraction towards target)
            damped_target_quat = current_quat.slerp(params_.vs_orientation_gain, raw_target_quat);
        }
        damped_target_quat.normalize();
        
        // ===== SMOOTHING (EMA filter for both position and orientation) =====
        double alpha = 1.0 - params_.vs_smoothing;
        vs_filtered_position_ = params_.vs_smoothing * vs_filtered_position_ + alpha * damped_target_pos;
        vs_filtered_orientation_ = vs_filtered_orientation_.slerp(alpha, damped_target_quat);
        vs_filtered_orientation_.normalize();
        
        // Set final target
        target_position_ = vs_filtered_position_;
        target_orientation_ = vs_filtered_orientation_;
        
        // Debug output (every 100 cycles)
        static int debug_counter = 0;
        if (++debug_counter % 100 == 0)
        {
            std::cout << "VS: delta_norm=" << pos_delta.norm() 
                      << " target=[" << target_position_.x() << ", "
                      << target_position_.y() << ", " << target_position_.z() << "]" << std::endl;
        }
    }

    // Helper function to clamp q7 within limits
    double clampQ7(double q7) const {
        return std::max(Q7_MIN, std::min(Q7_MAX, q7));
    }
    
    // Convert Eigen types to arrays for geofik interface
    std::array<double, 3> eigenToArray3(const Eigen::Vector3d& vec) const {
        return {vec.x(), vec.y(), vec.z()};
    }
    
    std::array<double, 9> quaternionToRotationArray(const Eigen::Quaterniond& quat) const {
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        return {rot(0,0), rot(0,1), rot(0,2),
                rot(1,0), rot(1,1), rot(1,2), 
                rot(2,0), rot(2,1), rot(2,2)};
    }

public:
    void run(const std::string &robot_ip)
    {
        try
        {
            franka::Robot robot(robot_ip);
            setDefaultBehavior(robot);

            // Move to a suitable starting joint configuration
            std::array<double, 7> q_goal = {{0.0, -M_PI/4, 0.0, -3*M_PI/4, 0.0, M_PI/2, M_PI/4}};
            MotionGenerator motion_generator(0.5, q_goal);
            std::cout << "WARNING: This example will move the robot! "
                      << "Please make sure to have the user stop button at hand!" << std::endl
                      << "Press Enter to continue..." << std::endl;
            std::cin.ignore();
            robot.control(motion_generator);
            std::cout << "Finished moving to initial joint configuration." << std::endl;

            // Collision behavior
            robot.setCollisionBehavior(
                {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}}, {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
                {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}}, {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
                {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
                {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}});

            // Joint impedance for smooth motion (instead of Cartesian)
            robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

            // Initialize poses from the robot's current state
            franka::RobotState state = robot.readOnce();
            initial_robot_pose_ = Eigen::Affine3d(Eigen::Matrix4d::Map(state.O_T_EE.data()));
            
            // Initialize joint angles
            for (int i = 0; i < 7; i++) {
                current_joint_angles_[i] = state.q[i];
                neutral_joint_pose_[i] = q_goal[i];  // Use the initial joint configuration as neutral
            }
            
            // Create IK solver with neutral pose and weights
            // Joint weights for base stabilization: higher weights for base joints (0,1)
            std::array<double, 7> base_joint_weights = {{
                3.0,  // Joint 0 (base rotation) - high penalty for stability
                6.0,  // Joint 1 (base shoulder) - high penalty for stability  
                1.5,  // Joint 2 (elbow) - normal penalty
                1.5,  // Joint 3 (forearm) - normal penalty
                1.0,  // Joint 4 (wrist) - normal penalty
                1.0,  // Joint 5 (wrist) - normal penalty
                1.0   // Joint 6 (hand) - normal penalty
            }};
            
            ik_solver_ = std::make_unique<WeightedIKSolver>(
                neutral_joint_pose_,
                1.0,  // manipulability weight
                2.0,  // neutral distance weight  
                2.0,  // current distance weight
                base_joint_weights,  // per-joint weights for base stabilization
                false // verbose = false for production use
            );
            
            // Initialize Ruckig trajectory generator (but don't set initial state yet)
            trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
            trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
            
            // Set up joint limits for safe teleoperation (but don't set positions yet)
            for (size_t i = 0; i < 7; ++i) {
                ruckig_input_.max_velocity[i] = MAX_JOINT_VELOCITY[i];
                ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
                ruckig_input_.max_jerk[i] = MAX_JOINT_JERK[i];
                ruckig_input_.target_velocity[i] = 0.0;
                ruckig_input_.target_acceleration[i] = 0.0;
            }
            
            std::cout << "Ruckig trajectory generator configured with 7 DOFs" << std::endl;

            // Initialize target pose to the robot's starting pose
            target_position_ = initial_robot_pose_.translation();
            target_orientation_ = Eigen::Quaterniond(initial_robot_pose_.rotation());

            std::thread network_thread(&VelocityServer::networkThread, this);

            std::cout << "Waiting for pose commands..." << std::endl;
            while (!initialized_ && running_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (initialized_)
            {
                std::cout << "Server initialized! Starting real-time velocity control." << std::endl;
                this->runControl(robot);
            }

            running_ = false;
            if (network_thread.joinable())
                network_thread.join();
        }
        catch (const franka::Exception &e)
        {
            std::cerr << "Franka exception: " << e.what() << std::endl;
            running_ = false;
        }
    }

private:
    void runControl(franka::Robot &robot)
    {
        auto control_callback = [this](
                                       const franka::RobotState &robot_state,
                                       franka::Duration period) -> franka::JointVelocities
        {
            // Update target pose from latest command / visual servoing
            {
                std::lock_guard<std::mutex> lock(command_mutex_);
                if (vs_mode_)
                {
                    updateVisualServoTarget(robot_state);
                }
                else
                {
                    PoseCommand cmd = current_pose_command_;
                    updateTargetPose(cmd);
                }
            }

            // Initialize Ruckig with actual robot state on first call
            if (!ruckig_initialized_) {
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i] = robot_state.q[i];
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = 0.0; // Start with zero velocity command
                    ruckig_input_.current_acceleration[i] = 0.0; // Start with zero acceleration
                    ruckig_input_.target_position[i] = robot_state.q[i]; // Start with current position as target
                    ruckig_input_.target_velocity[i] = 0.0; // Start with zero target velocity
                }
                control_start_time_ = std::chrono::steady_clock::now();
                ruckig_initialized_ = true;
                std::cout << "Ruckig initialized for velocity control!" << std::endl;
                std::cout << "Starting with zero velocity commands to smoothly take over control" << std::endl;
            } else {
                // Update current joint state for Ruckig using previous Ruckig output for continuity
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i] = robot_state.q[i];
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = ruckig_output_.new_velocity[i]; // Use our own velocity command for continuity
                    ruckig_input_.current_acceleration[i] = ruckig_output_.new_acceleration[i]; // Use Ruckig's acceleration
                }
            }
            
            // Calculate activation factor for gradual activation
            auto current_time = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(current_time - control_start_time_).count();
            double activation_factor = std::min(1.0, elapsed_sec / ACTIVATION_TIME_SEC);
            
            // Solve IK for target pose to get target joint angles
            std::array<double, 3> target_pos = eigenToArray3(target_position_);
            std::array<double, 9> target_rot = quaternionToRotationArray(target_orientation_);
            
            // Calculate q7 search range around current value
            double current_q7 = current_joint_angles_[6];
            // Use full Franka Q7 range for IK solving, not bidexhand limits
            double q7_start = std::max(-2.89, current_q7 - Q7_SEARCH_RANGE);
            double q7_end = std::min(2.89, current_q7 + Q7_SEARCH_RANGE);
            
            // Solve IK with weighted optimization
            WeightedIKResult ik_result = ik_solver_->solve_q7_optimized(
                target_pos, target_rot, current_joint_angles_,
                q7_start, q7_end, Q7_OPTIMIZATION_TOLERANCE, Q7_MAX_ITERATIONS
            );
            
            // Debug output for velocity control
            static int debug_counter = 0;
            debug_counter++;
            
            if (debug_counter % 100 == 0) {
                std::cout << "IK: " << (ik_result.success ? "\033[32msuccess\033[0m" : "\033[31mfail\033[0m") << " | Joints: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << std::fixed << std::setprecision(2) << current_joint_angles_[i];
                    if (i < 6) std::cout << " ";
                }
                std::cout << std::endl;
            }
            
            // Set Ruckig targets based on IK solution and gradual activation
            if (ruckig_initialized_) {
                if (ik_result.success) {
                    // Gradually blend from current position to IK solution for target position
                    for (int i = 0; i < 7; i++) {
                        double current_pos = current_joint_angles_[i];
                        double ik_target_pos = ik_result.joint_angles[i];
                        ruckig_input_.target_position[i] = current_pos + activation_factor * (ik_target_pos - current_pos);
                        // Always target zero velocity for smooth stops
                        ruckig_input_.target_velocity[i] = 0.0;
                    }
                    // Enforce q7 limits
                    ruckig_input_.target_position[6] = clampQ7(ruckig_input_.target_position[6]);
                }
                // If IK fails, keep previous targets (don't change target_position/velocity)
            }
            
            // Always run Ruckig to generate smooth velocity commands
            ruckig::Result ruckig_result = trajectory_generator_->update(ruckig_input_, ruckig_output_);
            
            std::array<double, 7> target_joint_velocities;
            
            if (ruckig_result == ruckig::Result::Working || ruckig_result == ruckig::Result::Finished) {
                // Use Ruckig's smooth velocity output
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = ruckig_output_.new_velocity[i];
                }
            } else {
                // Emergency fallback: zero velocity to stop smoothly
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = 0.0;
                }
                if (debug_counter % 100 == 0) {
                    std::cout << "Ruckig error, using zero velocity for safety" << std::endl;
                }
            }
            
            // Debug output for the first few commands
            // if (debug_counter <= 10 || debug_counter % 100 == 0) {
            //     std::cout << "Target vel: ";
            //     for (int i = 0; i < 7; i++) std::cout << std::fixed << std::setprecision(4) << target_joint_velocities[i] << " ";
            //     std::cout << " [activation: " << std::setprecision(3) << activation_factor << "]" << std::endl;
            // }

            if (!running_)
            {
                return franka::MotionFinished(franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
            }
            return franka::JointVelocities(target_joint_velocities);
        };

        try
        {
            robot.control(control_callback);
        }
        catch (const franka::ControlException &e)
        {
            std::cerr << "Control exception: " << e.what() << std::endl;
        }
    }
};

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <robot-hostname> [bidexhand] [mode]" << std::endl;
        std::cerr << "  bidexhand: true (default) for BiDexHand limits, false for full range" << std::endl;
        std::cerr << "  mode: vr (default) for base-frame VR control, vs for eye-in-hand visual servo" << std::endl;
        return -1;
    }

    bool bidexhand = false;
    if (argc >= 3)
    {
        std::string bidexhand_arg = argv[2];
        bidexhand = (bidexhand_arg == "true" || bidexhand_arg == "1");
    }

    bool vs_mode = false;
    if (argc == 4)
    {
        std::string mode_arg = argv[3];
        if (mode_arg == "vs" || mode_arg == "visual_servo")
        {
            vs_mode = true;
        }
    }

    try
    {
        VelocityServer server(bidexhand, vs_mode);
        // Add a signal handler to gracefully shut down on Ctrl+C
        server.run(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}