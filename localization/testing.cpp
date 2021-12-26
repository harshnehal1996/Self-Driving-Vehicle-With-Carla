#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include "cnpy.h"

int main(){

	int n = 250;
	std::vector<cv::Point2d> points1(n);
	std::vector<cv::Point2d> points2(n);

	std::string camera_path_1 = "/home/harsh/Documents/recorded_data/cam_out/keypoints/00001086.npy";
	std::string camera_path_2 = "/home/harsh/Documents/recorded_data/cam_out/keypoints/00001087.npy";
	std::string _path_1 = "/home/harsh/Documents/recorded_data/cam_out/original_images/000001086.png";
	std::string _path_2 = "/home/harsh/Documents/recorded_data/cam_out/original_images/000001087.png";

	cnpy::NpyArray keypt_1 = cnpy::npy_load(camera_path_1);
    float* loaded_keypt_1 = keypt_1.data<float>();

    cnpy::NpyArray keypt_2 = cnpy::npy_load(camera_path_2);
    float* loaded_keypt_2 = keypt_2.data<float>();

    // cnpy::NpyArray keypt_3 = cnpy::npy_load(_path_1);
    // float* loaded_desc_1 = keypt_3.data<float>();

    // cnpy::NpyArray keypt_4 = cnpy::npy_load(_path_2);
    // float* loaded_desc_2 = keypt_4.data<float>();
    // std::vector<std::vector<cv::DMatch>> sym_matches;

    for(int i=0; i < n; i++){
    	points1[i] = cv::Point2d((double)loaded_keypt_1[3*i], (double)loaded_keypt_1[3*i + 1]);
    	points2[i] = cv::Point2d((double)loaded_keypt_2[3*i], (double)loaded_keypt_2[3*i + 1]);
    	// std::cout << points1[i] << " " << points2[i] << std::endl;
    	// sym_matches[i] = cv::DMatch(i,i,5.0f);
    }

    // cv::Mat img_1 = cv::imread(_path_1, cv::IMREAD_COLOR);
    // cv::Mat img_2 = cv::imread(_path_2, cv::IMREAD_COLOR);

    double max_pixel_distance = 1;
    double success_p = 0.99;
    std::vector<uchar> inliers(n);
    std::cout << cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, max_pixel_distance, success_p, inliers) << std::endl;
    for(int i=0;i<n;i++){
    	if(inliers[i])
    		std::cout << "in " << (int)inliers[i] << std::endl;
    	else
    		std::cout << "out " << (int)inliers[i] << std::endl;
    }

    // std::vector<cv::KeyPoint> points_1;
    // std::vector<cv::KeyPoint> points_2;
    // for(int i=0;i<n;i++){
    	// points_1[i] = cv::KeyPoint(points1[i], 1);
    	// points_2[i] = cv::KeyPoint(points2[i], 1);
    // }
    
 //    std::vector<char> inliers_c(inliers.begin(), inliers.end());
 //    cv::Mat matched_points_RANSAC;
	// cv::drawMatches(img_1, points_1, img_2, points_2, sym_matches, matched_points_RANSAC, cv::Scalar::all(-1), cv::Scalar::all(-1), inliers_c, 0);

	// cv::imshow("Window Name", matched_points_RANSAC);
 //    cv::waitKey(0);
    // for(int i=0; i < n ; i++){
    // 	std::cout << (int)inliers[i] << std::endl;
    // }

	// for(int i=0; i < n; i++){
	// 	int left = matches[index][i].first;
	// 	int right = matches[index][i].second;
	// 			points1[i] = cv::Point2d((double)current_frame->keypoints[left][0], (double)current_frame->keypoints[left][1]);
	// 			points2[i] = cv::Point2d((double)reference_list[index]->keypoints[right][0], (double)reference_list[index]->keypoints[right][1]);
	// 		}

	// 		inliers[j].resize(n);
	
	return 0;
}
