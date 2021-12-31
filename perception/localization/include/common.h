#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <experimental/filesystem>
#include <jsoncpp/json/json.h> 
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <random>
#include "cnpy.h"

#define num_features 3000
#define max_perframes 3
#define descriptor_size 128
#define grid_size 30
#define max_assistive_x_degree 50
#define max_assist_altitude_diff 5
#define max_reference_distance 15
#define success_p 0.99
#define sample_size_m2 8
#define sample_size_m1 4
#define outlier_ratio 0.2
#define max_obs_size 96
#define alpha 0.3
#define beta 2.0
#define kappa 0.0
#define state_length 11
#define const_measurement 8
#define epsilon 1e-4
#define pi 3.1415926535
#define g_acc 9.7864103317

namespace ukf_tracker{

#ifndef USE_DOUBLE_PRECISION
typedef Eigen::Matrix<double, state_length, state_length> MatrixSS;
typedef Eigen::Matrix<double, state_length, 1> VectorS;
typedef Eigen::Matrix<double, state_length, 2 * state_length + 1> MatrixS2S1;
typedef Eigen::Matrix<double, state_length, 3 * state_length + 1> MatrixS3S1;
typedef Eigen::Quaterniond Quaternion;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector4d Vector4;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::Matrix4d Matrix4;
typedef Eigen::MatrixXd MatrixX;
typedef Eigen::VectorXd VectorX;
typedef Eigen::AngleAxisd AngleAxis;
typedef double scaler_t;
#else
typedef Eigen::Matrix<float, state_length, state_length> MatrixSS;
typedef Eigen::Matrix<float, state_length, 1> VectorS;
typedef Eigen::Matrix<float, state_length, 2 * state_length + 1> MatrixS2S1;
typedef Eigen::Matrix<float, state_length, 3 * state_length + 1> MatrixS3S1;
typedef Eigen::Quaternionf Quaternion;
typedef Eigen::Vector3f Vector3;
typedef Eigen::Vector4f Vector4;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Matrix4f Matrix4;
typedef Eigen::MatrixXf MatrixX;
typedef Eigen::VectorXf VectorX;
typedef Eigen::AngleAxisf AngleAxis;
typedef float scaler_t;
#endif

}

#endif
