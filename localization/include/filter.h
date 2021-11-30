#ifndef FILTER_H
#define FILTER_H

#include "state.h"
#include "measurement.h"

namespace ukf_tracker{

class sqrtUkfFilter{
private:

	scaler_t W1m;
	scaler_t W0m;
	scaler_t W1c;
	scaler_t root_W1c;
	scaler_t W0c;
	scaler_t gamma;
	scaler_t lambda;
	State* state;
	MatrixS2S1 sigma_points;
	MatrixS3S1 delta;
	VectorS sigma_mean;
	VectorX raw_measurement;
	VectorX root_measurement_variance;
	MatrixX H_x;
	VectorX measurement_mean;
	MatrixX measurement_delta;
	MatrixX measurement_root_covariance;
	MatrixX cross_covariance;

	inline void calculate_mean_from_sigma(MatrixS2S1& sigma_points, VectorS& mean_vector){
		mean_vector = W0m * sigma_points.col(0) + W1m * (sigma_points.topRightCorner<state_length, 2*state_length>().rowwise().sum());
	}

	inline void calculate_mean_from_dynamic(MatrixX& H_sigma, VectorX& mean_vector){
		mean_vector = W0m * H_sigma.col(0) + W1m * (H_sigma.rightCols(2 * state_length).rowwise().sum());
	}

	inline void calculate_sigma_matrix(VectorS& mean, MatrixSS& sqrt_covariance, MatrixS2S1& sigma_points){
		sigma_points.block<state_length, 1>(0, 0) = mean;
		sigma_points.block<state_length, state_length>(0, 1) = (gamma * sqrt_covariance).colwise() + mean;
		sigma_points.block<state_length, state_length>(0, state_length + 1) = (-gamma * sqrt_covariance).colwise() + mean;
	}

	inline void update_target_delta(MatrixS2S1& sigma_points, VectorS& mean, MatrixSS& root_noise, MatrixS3S1& target){
		target.block<state_length, 2 * state_length>(0, 1) = root_W1c * (sigma_points.block<state_length, 2 * state_length>(0, 1).colwise() - mean);
		target.block<state_length, state_length>(0, 2 * state_length + 1) = root_noise;
		target.col(0) = (sigma_points.col(0) - mean);
	}

	inline void update_target_delta_dynamic(MatrixX& H_points, VectorX& mean, VectorX& root_noise, MatrixX& target){
		int observation_length = H_points.rows();
		target.resize(observation_length, 2 * state_length + 1 + observation_length);
		target.block(0, 1, observation_length, 2 * state_length) = root_W1c * (H_points.rightCols(2 * state_length).colwise() - mean);
		target.rightCols(observation_length) = root_noise.asDiagonal();
		target.col(0) = (H_points.col(0) - mean);
	}

	void prediction_step(scaler_t time_delta){
		calculate_sigma_matrix(state->state_mean_vector, state->sqrt_state_covariance, sigma_points);
		state->motion_update(sigma_points, time_delta);
		calculate_mean_from_sigma(sigma_points, state->state_mean_vector);
		update_target_delta(sigma_points, state->state_mean_vector, state->sqrt_process_noise, delta);
		state->sqrt_state_covariance = delta.topRightCorner<state_length,3*state_length>().transpose().householderQr().matrixQR().topLeftCorner<state_length, state_length>().triangularView<Eigen::Upper>();
		Eigen::internal::llt_inplace<scaler_t, Eigen::Upper>::rankUpdate(state->sqrt_state_covariance, delta.col(0), W0c);
        state->sqrt_state_covariance.transposeInPlace();
	}

	void observation_step(Measurement* new_measurement){
		calculate_sigma_matrix(state->state_mean_vector, state->sqrt_state_covariance, sigma_points);
		new_measurement->capture_observation(sigma_points, H_x, raw_measurement, root_measurement_variance);
		calculate_mean_from_dynamic(H_x, measurement_mean);
		update_target_delta_dynamic(H_x, measurement_mean, root_measurement_variance, measurement_delta);
		measurement_root_covariance = measurement_delta.rightCols(measurement_delta.cols()-1).transpose().householderQr().matrixQR().topLeftCorner(H_x.rows(), H_x.rows()).triangularView<Eigen::Upper>();
		Eigen::internal::llt_inplace<scaler_t, Eigen::Upper>::rankUpdate(measurement_root_covariance, measurement_delta.col(0), W0c);
		measurement_root_covariance.transposeInPlace();
		cross_covariance = W0c * (sigma_points.col(0) - state->state_mean_vector) * (H_x.col(0) - measurement_mean).transpose() +\
		W1c * (sigma_points.rightCols(2 * state_length).colwise() - state->state_mean_vector) * ((H_x.rightCols(2 * state_length).colwise() - measurement_mean).transpose());
	}

	void update_step(){
		MatrixX kalman_gain = measurement_root_covariance.transpose().fullPivHouseholderQr().solve(measurement_root_covariance.fullPivHouseholderQr().solve(cross_covariance.transpose()));
		kalman_gain.transposeInPlace();
		state->state_mean_vector = state->state_mean_vector + kalman_gain * (raw_measurement - measurement_mean);
		cross_covariance.noalias() = kalman_gain * measurement_root_covariance;
		for(int i=0; i < cross_covariance.cols(); i++)
			Eigen::internal::llt_inplace<scaler_t, Eigen::Lower>::rankUpdate(state->sqrt_state_covariance, cross_covariance.col(i), -1);
	}

public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	sqrtUkfFilter(State* _state){
		state = _state;
		lambda = alpha * alpha * (state_length + kappa) - state_length;
		gamma = std::sqrt(state_length + lambda);
		W1m = 1.0 / (2 * (state_length + lambda));
		W1c = W1m;
		root_W1c = std::sqrt(W1m);
		W0m = lambda / (state_length + lambda);
		W0c = W0m + 1 - (alpha * alpha) + beta;
	}

	void filter_step(Measurement* new_measurement){
		prediction_step(new_measurement->time_delta);
		observation_step(new_measurement);
		update_step();
	}
};

}

#endif
