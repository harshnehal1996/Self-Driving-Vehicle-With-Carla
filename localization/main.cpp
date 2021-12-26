#include "filter.h"
#include "data_loader.h"
#include <jsoncpp/json/json.h> 
#include <jsoncpp/json/writer.h>

// look at opencv inliers
const int width = 820;
const int height = 820;

int main(int argc, char **argv){
	std::string lidar_keyframes_path(argv[1]);
	std::string trajectory_data_path(argv[2]);

	// axis_mapping : maps {x, y, z} axis of the camera frame(opencv convention) to {axis_mapping[0], axis_mapping[1], axis_mapping[2]} axis in world frame
	// 1,2,3 represents x, y, and z axis of world frame. Similarly -1,-2,-3 are -x, -y and -z axis of the world frame
	int axis_mapping[3] = {-2, -3, 1};
	float f_x = float(width) / 2.0f;
	float f_y = float(width) / 2.0f;
	float c_x = float(width) / 2.0f;
	float c_y = float(height) / 2.0f;
	ukf_tracker::initialize_intrinsic_matrix(f_x, f_y, c_x, c_y, axis_mapping);

	load_recorded_lidar_keyframes(lidar_keyframes_path);

	trajectoryFrame** trajectory_data;
	int num_frames = load_current_trajectory_data(trajectory_data_path, &trajectory_data);
	printf("Frames : %d\n", num_frames);

	ukf_tracker::State* state = new ukf_tracker::State();
	// Values taken from paper with dt close to 1 / 30 fps
	state->sqrt_process_noise.setZero();
	state->sqrt_process_noise.diagonal() << std::sqrt(0.8274), std::sqrt(0.8274), std::sqrt(0.8274),\
											std::sqrt(0.2500), std::sqrt(2.0000), std::sqrt(0.0085),\
											std::sqrt(0.0085), std::sqrt(0.0085), std::sqrt(0.0003),\
											std::sqrt(0.0003), std::sqrt(0.0003);
	// slice to run only on subpart
	int start_index = 0;
	if(argc > 3){
		start_index = atoi(argv[3]);
		printf("start_index %d\n", start_index);
	}

	int end_index = num_frames;
	if(argc > 4){
		end_index = atoi(argv[4]);
		printf("end_index %d\n", end_index);
	}
	
	// Eigen::Vector3f angles = trajectory_data[0]->ground_truth.toRotationMatrix().eulerAngles(0, 1, 2);
	Eigen::Vector3f angles = trajectory_data[start_index]->_ground_truth.topLeftCorner<3,3>().eulerAngles(0, 1, 2);
	// Initialize state with the ground truth at t=0 value 
	state->state_mean_vector(0) = trajectory_data[start_index]->_ground_truth(0, 3);
	state->state_mean_vector(1) = trajectory_data[start_index]->_ground_truth(1, 3);
	state->state_mean_vector(2) = trajectory_data[start_index]->_ground_truth(2, 3);
	state->state_mean_vector(3) = trajectory_data[start_index]->imu_data(6); // v
	state->state_mean_vector(4) = trajectory_data[start_index]->imu_data(0);// - g_acc * trajectory_data[0]->ground_truth(0, 2); // a_x - g * cos(theta)
	state->state_mean_vector(5) = angles(0); // roll
	state->state_mean_vector(6) = angles(1); // pitch
	state->state_mean_vector(7) = angles(2); // yaw
	state->state_mean_vector(8) = trajectory_data[start_index]->imu_data(3); // roll_rate
	state->state_mean_vector(9) = trajectory_data[start_index]->imu_data(4); // pitch_rate
	state->state_mean_vector(10) = trajectory_data[start_index]->imu_data(5); // yaw_rate
	// Initialize covariance matrix
	state->sqrt_state_covariance.setZero();
	state->sqrt_state_covariance.diagonal() << 1e-3, 1e-3, 1e-3, 1, 1, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3;

	ukf_tracker::Measurement* measurement = new ukf_tracker::Measurement(state);
	measurement->acc_std = 1;
	measurement->vel_std = 1;
	measurement->roll_rate_std = 1e-3;
	measurement->pitch_rate_std = 1e-3;
	measurement->yaw_rate_std = 1e-3;
	measurement->utm_longitude_std = 5;
	measurement->utm_latitude_std = 5;
	measurement->utm_altitude_std = 5;

	ukf_tracker::sqrtUkfFilter ukf(state);

	// testing by adding random noise
	auto dist = std::bind(std::normal_distribution<float>{0, 5},\
                      std::mt19937(std::random_device{}()));
	
	int i=start_index+1;
	std::ofstream ofs_file("errors.json");
    Json::StyledWriter writer;
    Json::Value main_obj;

	while(i < end_index){
		// update measurement
		measurement->speed = trajectory_data[i]->imu_data(6);
		measurement->acceleration = trajectory_data[i]->imu_data(0);
		measurement->roll_rate = trajectory_data[i]->imu_data(3);
		measurement->pitch_rate = trajectory_data[i]->imu_data(4);
		measurement->yaw_rate = trajectory_data[i]->imu_data(5);
		measurement->time_delta = trajectory_data[i]->timestamp - trajectory_data[i-1]->timestamp;
		measurement->utm_longitude =  trajectory_data[i]->_ground_truth(0, 3) + dist();
		measurement->utm_latitude = trajectory_data[i]->_ground_truth(1, 3) + dist();
		measurement->utm_altitude = trajectory_data[i]->_ground_truth(2, 3) + dist();
		measurement->set_camera_observation(&trajectory_data[i]->obs);

		ukf.filter_step(measurement);
		
		float x_new = (float)state->state_mean_vector(0);
		float y_new = (float)state->state_mean_vector(1);
		float z_new = (float)state->state_mean_vector(2);
		float v_new = (float)state->state_mean_vector(3);
		float a_new = (float)state->state_mean_vector(4);
		float w_new = (float)state->state_mean_vector(5);
		float p_new = (float)state->state_mean_vector(6);
		float k_new = (float)state->state_mean_vector(7);
		
		float true_x = trajectory_data[i]->location(0);
		float true_y = trajectory_data[i]->location(1);
		float true_z = trajectory_data[i]->location(2);
		float true_v = (float)trajectory_data[i]->imu_data(6);
		float true_a = (float)trajectory_data[i]->imu_data(0);
		Eigen::Vector3f angles = trajectory_data[i]->_ground_truth.topLeftCorner<3,3>().eulerAngles(0, 1, 2);
		float true_w = angles(0);
		float true_p = angles(1);
		float true_k = angles(2);

		printf("Frame %d\n", i);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "x", x_new, "x", true_x, x_new - true_x);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "y", y_new, "y", true_y, y_new - true_y);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "z", z_new, "z", true_z, z_new - true_z);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "v", v_new, "v", true_v, v_new - true_v);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "a", a_new, "a", true_a, a_new - true_a);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "roll", w_new, "roll", true_w, w_new - true_w);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "pitch", p_new, "pitch", true_p, p_new - true_p);
		printf("predicted %s = %f  True %s = %f  , diff = %f\n", "yaw", k_new, "yaw", true_k, k_new - true_k);
		printf("\n");

		auto key = std::to_string(i);
		main_obj["frame_no"][key]["t_x"] = true_x;
	    main_obj["frame_no"][key]["t_y"] = true_y;
	    main_obj["frame_no"][key]["t_z"] = true_z;
	    main_obj["frame_no"][key]["t_v"] = true_v;
	    main_obj["frame_no"][key]["t_a"] = true_a;
	    main_obj["frame_no"][key]["t_roll"] = true_w;
	    main_obj["frame_no"][key]["t_pitch"] = true_p;
	    main_obj["frame_no"][key]["t_yaw"] = true_k;
	    main_obj["frame_no"][key]["p_x"] = x_new;
	    main_obj["frame_no"][key]["p_y"] = y_new;
	    main_obj["frame_no"][key]["p_z"] = z_new;
	    main_obj["frame_no"][key]["p_v"] = v_new;
	    main_obj["frame_no"][key]["p_a"] = a_new;
	    main_obj["frame_no"][key]["p_roll"] = w_new;
	    main_obj["frame_no"][key]["p_pitch"] = p_new;
	    main_obj["frame_no"][key]["p_yaw"] = k_new;

		i++;
	}
	ofs_file << writer.write(main_obj);
	ofs_file.close();


	return 0;
}

