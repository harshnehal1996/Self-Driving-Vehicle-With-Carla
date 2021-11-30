#ifndef MEASUREMENT_H
#define MEASUREMENT_H

namespace ukf_tracker{

class Measurement{

	observation* obs;
	State* state;

public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	scaler_t acceleration;
	scaler_t speed;
	scaler_t roll_rate;
	scaler_t pitch_rate;
	scaler_t yaw_rate;
	scaler_t acc_std;
	scaler_t vel_std;
	scaler_t yaw_rate_std;
	scaler_t pitch_rate_std;
	scaler_t roll_rate_std;
	scaler_t utm_latitude;
	scaler_t utm_longitude;
	scaler_t utm_altitude;
	scaler_t utm_latitude_std;
	scaler_t utm_longitude_std;
	scaler_t utm_altitude_std;
	scaler_t time_delta;
	
	Measurement(State* _state){
		state = _state;
		obs = new observation();
	}

	void set_camera_observation(cameraObservation* cam_observation){
		obs->update_camera_frame(cam_observation);
	}

	void capture_observation(MatrixS2S1& sigma_points, MatrixX& H_x, VectorX& observation, VectorX& root_variance_vector){
		Quaternion rot_frame_2_world = AngleAxis(state->state_mean_vector(5), Vector3::UnitX()) *\
									   AngleAxis(state->state_mean_vector(6), Vector3::UnitY()) *\
		    						   AngleAxis(state->state_mean_vector(7), Vector3::UnitZ());
		Vector3 location = state->state_mean_vector.block<3,1>(0,0);
		Quaternion u_x(0, 1, 0, 0);
		acceleration = acceleration - g_acc * (rot_frame_2_world * u_x * rot_frame_2_world.inverse()).vec()(2);

		// printf("%s\n", "before_new_observation");
		if(obs->register_new_observation(rot_frame_2_world, location, observation, root_variance_vector, H_x, sigma_points, true)){
			int num_camera_observation = observation.rows() - const_measurement;
			observation(num_camera_observation) = speed;
			observation(num_camera_observation + 1) = acceleration;
			observation(num_camera_observation + 2) = roll_rate;
			observation(num_camera_observation + 3) = pitch_rate;
			observation(num_camera_observation + 4) = yaw_rate;
			observation(num_camera_observation + 5) = utm_longitude;
			observation(num_camera_observation + 6) = utm_latitude;
			observation(num_camera_observation + 7) = utm_altitude;
			root_variance_vector(num_camera_observation) = vel_std;
			root_variance_vector(num_camera_observation + 1) = acc_std;
			root_variance_vector(num_camera_observation + 2) = roll_rate_std;
			root_variance_vector(num_camera_observation + 3) = pitch_rate_std;
			root_variance_vector(num_camera_observation + 4) = yaw_rate_std;
			root_variance_vector(num_camera_observation + 5) = utm_longitude_std;
			root_variance_vector(num_camera_observation + 6) = utm_latitude_std;
			root_variance_vector(num_camera_observation + 7) = utm_altitude_std;

			scaler_t* h_writer = H_x.data();
			for(int i=0; i < 2 * state_length + 1; i++){
				int offset = i * observation.rows();
				h_writer[offset + num_camera_observation] = sigma_points(3, i);
				h_writer[offset + num_camera_observation + 1] = sigma_points(4, i);
				h_writer[offset + num_camera_observation + 2] = sigma_points(8, i);
				h_writer[offset + num_camera_observation + 3] = sigma_points(9, i);
				h_writer[offset + num_camera_observation + 4] = sigma_points(10, i);
				h_writer[offset + num_camera_observation + 5] = sigma_points(0, i);
				h_writer[offset + num_camera_observation + 6] = sigma_points(1, i);
				h_writer[offset + num_camera_observation + 7] = sigma_points(2, i);
			}
		}
		else{
			observation.resize(const_measurement);
			H_x.resize(const_measurement, 2 * state_length + 1);
			root_variance_vector.resize(const_measurement);
			observation(0) = speed;
			observation(1) = acceleration;
			observation(2) = roll_rate;
			observation(3) = pitch_rate;
			observation(4) = yaw_rate;
			observation(5) = utm_longitude;
			observation(6) = utm_latitude;
			observation(7) = utm_altitude;
			root_variance_vector(0) = vel_std;
			root_variance_vector(1) = acc_std;
			root_variance_vector(2) = roll_rate_std;
			root_variance_vector(3) = pitch_rate_std;
			root_variance_vector(4) = yaw_rate_std;
			root_variance_vector(5) = utm_longitude_std;
			root_variance_vector(6) = utm_latitude_std;
			root_variance_vector(7) = utm_altitude_std;

			scaler_t* h_writer = H_x.data();
			for(int i=0; i < 2 * state_length + 1; i++){
				int offset = i * const_measurement;
				h_writer[offset] = sigma_points(3, i);
				h_writer[offset + 1] = sigma_points(4, i);
				h_writer[offset + 2] = sigma_points(8, i);
				h_writer[offset + 3] = sigma_points(9, i);
				h_writer[offset + 4] = sigma_points(10, i);
				h_writer[offset + 5] = sigma_points(0, i);
				h_writer[offset + 6] = sigma_points(1, i);
				h_writer[offset + 7] = sigma_points(2, i);
			}
		}
	}
};
}

#endif
