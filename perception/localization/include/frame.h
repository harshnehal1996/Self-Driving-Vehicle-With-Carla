#ifndef FRAME_H
#define FRAME_H

#include "common.h"

namespace ukf_tracker{

cv::Mat _K;
Matrix3 K;
Matrix3 Kinv;
Eigen::Matrix3d Kinv_double;
bool isIntialized = false;

inline int signum(int val){
	return (0 < val) - (val < 0);
}

void initialize_intrinsic_matrix(float f_x, float f_y, float c_x, float c_y, int axis_switch[3]){
	K = Matrix3::Identity();
	K(0, 0) = (scaler_t)f_x;
	K(1, 1) = (scaler_t)f_y;
	K(0, 2) = (scaler_t)c_x;
	K(1, 2) = (scaler_t)c_y;
	_K = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
	_K.at<double>(0,0) = (double)f_x;
	_K.at<double>(1,1) = (double)f_y;
	_K.at<double>(0,2) = (double)c_x;
	_K.at<double>(1,2) = (double)c_y;
	_K.at<double>(2,2) = 1.0;
	//std::cout << _K << std::endl;

	Matrix3 switch_mat = Matrix3::Zero();
	switch_mat(0, std::abs(axis_switch[0]) - 1) = signum(axis_switch[0]);
	switch_mat(1, std::abs(axis_switch[1]) - 1) = signum(axis_switch[1]);
	switch_mat(2, std::abs(axis_switch[2]) - 1) = signum(axis_switch[2]);
	
	K = K * switch_mat;
	Kinv = K.inverse();
	// Vector3 X;
	// X << 225, 310, 1;
	// Vector3 Y = Kinv * X;
	std::cout << "camera_matrix : " << K << std::endl;
	// std::cout << Kinv << std::endl;
	// std::cout << Y << std::endl;
	// std::cout << K * Y << std::endl;
	// std::cout << Kinv * (K * Y) << std::endl;
	// std::cout << K * Kinv << std::endl;
	Kinv_double = Kinv.cast<double>();
	isIntialized = true;
}

class Frame;

std::map<int, std::map<int, std::vector<Frame*>*>*> Voxel2D;

class Frame{
private:
	
	void add_frame_to_voxel(){
		int x = (int)trans_frame_2_world(0);
		int y = (int)trans_frame_2_world(1);
		int grid_x = x / grid_size;
		int grid_y = y / grid_size;
		std::map<int, std::vector<Frame*>*>* temp_x = Voxel2D[grid_x];
		if(!temp_x){
			temp_x = new std::map<int, std::vector<Frame*>*>();
			Voxel2D[grid_x] = temp_x;
			std::vector<Frame*>* temp_y = new std::vector<Frame*>();
			(*temp_x)[grid_y] = temp_y;
			(*temp_y).push_back(this);
		}
		else{
			std::vector<Frame*>* temp_y = (*temp_x)[grid_y];
			if(!temp_y){
				temp_y = new std::vector<Frame*>();
				(*temp_x)[grid_y] = temp_y;	
			}
			(*temp_y).push_back(this);
		}
	}

public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	int keyframe_id;
	Matrix3 Rmat_frame_2_world;
	Vector3 trans_frame_2_world;
	Quaternion rot_frame_2_world;
	Matrix4 g_frame_2_world;

	scaler_t** keypoints;
	double** points3D; 
	int num_keypoints;
	Eigen::MatrixXf descriptors;
	Frame* next_keyframe;
	Frame* previous_keyframe;
	cv::Mat image;

	Frame(int id, float* keypt, float* descriptor, int num_keypts, float* transform, Frame* previous, cv::Mat& img){
		keyframe_id = id;
		num_keypoints = num_keypts;
		descriptors.resize(descriptor_size, num_keypoints);
		float* writer = descriptors.data();

		for(int i=0; i < num_keypoints; i++){
			for(int j=0; j < descriptor_size; j++){
				writer[i * descriptor_size + j] = descriptor[i * descriptor_size + j];
			}
		}

		keypoints = new scaler_t*[num_keypoints];
		for(int i=0; i < num_keypoints; i++){
			keypoints[i] = new scaler_t[4];
			for(int j=0; j < 4; j++){
				keypoints[i][j] = (scaler_t)keypt[i * 4 + j];
			}
		}

		for(int i=0; i < 3; i++){
			for(int j=0; j < 3; j++){
				Rmat_frame_2_world(i, j) = (scaler_t)transform[i * 4 + j];
			}
		}

		for(int i=0; i < 4; i++){
			for(int j=0; j < 4; j++){
				g_frame_2_world(i, j) = (scaler_t)transform[i * 4 + j];
			}
		}

		rot_frame_2_world = Rmat_frame_2_world;
		trans_frame_2_world(0) = (scaler_t)transform[3];
		trans_frame_2_world(1) = (scaler_t)transform[7];
		trans_frame_2_world(2) = (scaler_t)transform[11];

		if(isIntialized){
			points3D = new double*[num_keypoints];
			for(int i=0; i < num_keypoints; i++){
				points3D[i] = new double[3];
				Eigen::Vector3d point((double)keypoints[i][0], (double)keypoints[i][1], 1);
				Eigen::Vector3d output = ((double)keypoints[i][2]) * Kinv_double * point;
				points3D[i][0] = -output(1);
				points3D[i][1] = -output(2);
				points3D[i][2] = output(0);
			}
		}
		else{
			printf("Error : Intrinsic matrix is not intialized\n");
			exit(0);
		}

		image = img;
		next_keyframe = 0;
		previous_keyframe = previous;
		add_frame_to_voxel();
	}

	~Frame(){
		for(int i=0; i < num_keypoints; i++)
			delete keypoints[i];
		delete keypoints;
	}
};
}

#endif
