#ifndef STATE_H
#define STATE_H

#include "observation.h"

namespace ukf_tracker{

struct outData
{
	float timestamp;
	Quaternion R;
	Vector3 T;	
	cameraObservation* obs;
};

class State{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	VectorS state_mean_vector;
	MatrixSS sqrt_state_covariance;
	MatrixSS sqrt_process_noise;
	std::vector<outData*> processed_states;

	// assumed roll = 0 for all t
	// state_vector = [x, y, z, v, a, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, lon, lat, altitude]
	// takes an input state(sigma points) and maps it predicted step points
	void motion_update(MatrixS2S1& sigma_point, scaler_t dt){
		scaler_t* ptr = sigma_point.data();
		
		for(int i=0; i < 2*state_length+1; i++){
			int offset = i * state_length;
			scaler_t v_0 = ptr[offset + 3];
			scaler_t a = ptr[offset + 4];
			scaler_t roll_0 = ptr[offset + 5];
			scaler_t pitch_0 = ptr[offset + 6];
			scaler_t yaw_0 = ptr[offset + 7];
			scaler_t roll_rate = ptr[offset + 8];
			scaler_t pitch_rate = ptr[offset + 9];
			scaler_t yaw_rate = ptr[offset + 10];
			scaler_t v_t = v_0 + a * dt;
			scaler_t roll_t = roll_0 + roll_rate * dt;
			scaler_t yaw_t = yaw_0 + yaw_rate * dt;
			scaler_t pitch_t = pitch_0 + pitch_rate * dt;

			scaler_t temp_x = 0, temp_y = 0;
			scaler_t cosp_t = std::cos(pitch_t + yaw_t);
			// scaler_t sinp_t = (((int)std::floor((pitch_t + yaw_t) / pi)) % 2) ? - std::sqrt(1 - cosp_t * cosp_t) : std::sqrt(1 - cosp_t * cosp_t);
			scaler_t sinp_t = std::sin(pitch_t + yaw_t);
			scaler_t cosn_t = std::cos(yaw_t - pitch_t);
			// scaler_t sinn_t = (((int)std::floor((yaw_t - pitch_t) / pi)) % 2) ? - std::sqrt(1 - cosn_t * cosn_t) : std::sqrt(1 - cosn_t * cosn_t);
			scaler_t sinn_t = std::sin(yaw_t - pitch_t);
			scaler_t cosp_0 = std::cos(pitch_0 + yaw_0);
			// scaler_t sinp_0 = (((int)std::floor((yaw_0 + pitch_0) / pi)) % 2) ? - std::sqrt(1 - cosp_0 * cosp_0) : std::sqrt(1 - cosp_0 * cosp_0);
			scaler_t sinp_0 = std::sin(pitch_0 + yaw_0);
			scaler_t cosn_0 = std::cos(yaw_0 - pitch_0);
			// scaler_t sinn_0 = (((int)std::floor((yaw_0 - pitch_0) / pi)) % 2) ? - std::sqrt(1 - cosn_0 * cosn_0) : std::sqrt(1 - cosn_0 * cosn_0);
			scaler_t sinn_0 = std::sin(yaw_0 - pitch_0);

			if(std::abs(pitch_rate + yaw_rate) < epsilon){
				scaler_t t = (v_t - (a * dt) / 2.0) * dt;
				temp_x += t * cosp_t;
				temp_y += t * sinp_t;
			}
			else{
				scaler_t add_inv = 1.0 / (pitch_rate + yaw_rate);
				temp_x += ((v_t * sinp_t - v_0 * sinp_0) * add_inv);
				temp_x += ((cosp_t - cosp_0) * a * add_inv * add_inv);
				temp_y += ((-v_t * cosp_t + v_0 * cosp_0) * add_inv);
				temp_y += ((sinp_t - sinp_0) * a * add_inv * add_inv);
			}

			if(std::abs(pitch_rate - yaw_rate) < epsilon){
				scaler_t t = (v_t - (a * dt) / 2.0) * dt;
				temp_x += t * cosn_t;
				temp_y += t * sinn_t;
			}
			else{
				scaler_t sub_inv = 1.0 / (yaw_rate - pitch_rate);
				temp_x += ((v_t * sinn_t - v_0 * sinn_0) * sub_inv);
				temp_x += ((cosn_t - cosn_0) * a * sub_inv * sub_inv);
				temp_y += ((-v_t * cosn_t + v_0 * cosn_0) * sub_inv);
				temp_y += ((sinn_t - sinn_0) * a * sub_inv * sub_inv);
			}

			ptr[offset] += (temp_x / 2.0);
			ptr[offset + 1] += (temp_y / 2.0);

			if(std::abs(pitch_rate) < epsilon)
				ptr[offset + 2] += std::sin(pitch_t) * (v_t - (a * dt) / 2) * dt;
			else
				ptr[offset + 2] += ((-v_t * std::cos(pitch_t) + v_0 * std::cos(pitch_0)) / pitch_rate) + ((std::sin(pitch_t) - std::sin(pitch_0)) * a) / (pitch_rate * pitch_rate);
			
			ptr[offset + 3] = v_t;
			ptr[offset + 5] = roll_t;
			ptr[offset + 6] = pitch_t;
			ptr[offset + 7] = yaw_t;
		}
	}
};

}
#endif
