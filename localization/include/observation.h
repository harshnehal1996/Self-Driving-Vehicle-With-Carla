#ifndef OBSERVATION_H
#define OBSERVATION_H

#include "frame.h"

namespace ukf_tracker{

struct cameraObservation{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	#ifndef NO_IMAGE_OUT
	Matrix4 g_frame_to_world;
	Vector3 true_angles;
	Vector3 true_location;
	#endif
	
	cv::Mat image;
	
	#if num_features > 500
	Eigen::MatrixXf descriptors;
	#else
	Eigen::Matrix<float, num_features, descriptor_size> descriptors;
	#endif
	scaler_t keypoints[num_features][3];
	
	float scores[num_features];
};

const scaler_t min_view_cosorientation = std::cos(max_assistive_x_degree * pi / 180.0);

class observation{
private:

	bool location_assistance_status;
	Frame *reference;
	cameraObservation* current_frame; 
	int num_ransac_iteration_m1;
	int num_ransac_iteration_m2;

	// uses Ransac PNP method to eliminate outlier
	int ransac_elimination_m1(std::vector<Frame*>& reference_list, std::vector<std::pair<int, int>>* matches, int* priority_index,\
							  std::vector<uchar>* inliers, int length, float max_pixel_distance=7.0f){
		for(int j=0; j < length; j++){
			int index = priority_index[j];
			int n = matches[index].size();
			if(n < 5)
				return j;

			printf("%d\n", n);
			std::vector<cv::Point2d> points2D(n);
			std::vector<cv::Point3d> points3D(n);

			for(int i=0; i < n; i++){
				int left = matches[index][i].first;
				int right = matches[index][i].second;
				points2D[i] = cv::Point2d((double)current_frame->keypoints[left][0], (double)current_frame->keypoints[left][1]);
				points3D[i] = cv::Point3d(reference_list[index]->points3D[right][0], reference_list[index]->points3D[right][1], reference_list[index]->points3D[right][2]);
			}
			cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);    
    		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);          
    		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
			cv::Mat inlier;

			cv::Mat r_vec, t_vec;
			inliers[j].resize(n);
			
			for(int k=0; k < n; k++)
				inliers[j][k] = 0;

			int ret = cv::solvePnPRansac(points3D, points2D, _K, distCoeffs, r_vec, t_vec, false, 100, max_pixel_distance, (double)success_p, inlier, cv::SOLVEPNP_ITERATIVE);
			for(int i=0; i < inlier.rows; i++){
				inliers[j][inlier.at<int>(i)] = 1;
			}

			//std::cout << "ransac success = "  << ret << std::endl;

			// std::vector<cv::Point2d> points1;
			// std::vector<cv::Point2d> points2;

			// for(int i=0; i < n; i++){
			// 	if(inliers[j][i]){
			// 		int left = matches[index][i].first;
			// 		int right = matches[index][i].second;
			// 		points1.push_back(cv::Point2d((double)current_frame->keypoints[left][0], (double)current_frame->keypoints[left][1]));
			// 		points2.push_back(cv::Point2d((double)reference_list[index]->keypoints[right][0], (double)reference_list[index]->keypoints[right][1]));
			// 	}
			// }
			// cv::Mat R, t, mask;

			// cv::Mat E = cv::findEssentialMat(points1, points2, _K, cv::RANSAC, (double)success_p, 1.0, mask);
			// cv::recoverPose(E, points1, points2, _K, R, t, mask);

			// std::cout << "Rotation matrix : " << R << std::endl;
			// std::cout << "Translation : " << t << std::endl;
		}

		return length;
	}

	// uses Fundamental matrix as the way to eliminate outlier
	int ransac_elimination_m2(std::vector<Frame*>& reference_list, std::vector<std::pair<int, int>>* matches, int* priority_index,\
							  std::vector<uchar>* inliers, int length, double max_pixel_distance=3){
		printf("in m2\n");
		for(int j=0; j < length; j++){
			int index = priority_index[j];
			int n = matches[index].size();
			if(n < 8)
				return j;

			printf("%d\n", n);
			std::vector<cv::Point2d> points1(n);
			std::vector<cv::Point2d> points2(n);

			for(int i=0; i < n; i++){
				int left = matches[index][i].first;
				int right = matches[index][i].second;
				points1[i] = cv::Point2d((double)current_frame->keypoints[left][0], (double)current_frame->keypoints[left][1]);
				points2[i] = cv::Point2d((double)reference_list[index]->keypoints[right][0], (double)reference_list[index]->keypoints[right][1]);
			}

			inliers[j].resize(n);
			// cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, max_pixel_distance, (double)success_p, num_ransac_iteration_m2, inliers[j]);
			cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, max_pixel_distance, (double)success_p, inliers[j]);
		}
		
		return length;
	}

	Frame* find_frame_assistance(Quaternion& current_rotation, Vector3& this_location){
		int x = (int)this_location(0);
		int y = (int)this_location(1);
		int grid_x = x / grid_size;
		int grid_y = y / grid_size;
		int loc_x = x % grid_size;
		int loc_y = y % grid_size;
		int neighbor_x=-1, neighbor_y=-1;

		if(loc_x < 0)
			loc_x = grid_size - loc_x;

		if(loc_y < 0)
			loc_y = grid_size - loc_y;

		if(loc_x <= grid_size / 4)
			neighbor_x = grid_x - 1;
		else if(loc_x >= 3 * (grid_size / 4))
			neighbor_x = grid_x + 1;

		if(loc_y <= grid_size / 4)
			neighbor_y = grid_y - 1;
		else if(loc_y >= 3 * (grid_size / 4))
			neighbor_y = grid_y + 1;

		scaler_t closest = 1e7;
		Frame* best_assistance = 0;
		Quaternion unitX(0,1,0,0);
		Quaternion unitZ(0,0,0,1);
		Vector3 z_current_vec = (current_rotation * unitZ * current_rotation.inverse()).vec();
		Vector3 view_vec = (current_rotation * unitX * current_rotation.inverse()).vec();

		auto temp = Voxel2D[grid_x];
		if(temp){
			auto candidate = (*temp)[grid_y];
			if(candidate){
				std::vector<Frame*>& frame_list = *candidate;
				for(int i=0; i < frame_list.size(); i++){
					Vector3 frame_location = frame_list[i]->trans_frame_2_world;
					Vector3 x_frame = (frame_list[i]->rot_frame_2_world * unitX * frame_list[i]->rot_frame_2_world.inverse()).vec();
					scaler_t view_angle_disparity = view_vec.dot(x_frame);
					if(view_angle_disparity >= min_view_cosorientation && std::abs((this_location - frame_location).dot(z_current_vec)) <= max_assist_altitude_diff){
						scaler_t distance = (this_location - frame_location).norm();
						if(distance < closest){
							closest = distance;
							best_assistance = frame_list[i];
						}
					}
				}
			}
		}

		if(neighbor_x != -1){
			temp = Voxel2D[neighbor_x];
			if(temp){
				auto candidate = (*temp)[grid_y];
				if(candidate){
					std::vector<Frame*>& frame_list = *candidate;
					for(int i=0; i < frame_list.size(); i++){
						Vector3 frame_location = frame_list[i]->trans_frame_2_world;
						Vector3 x_frame = (frame_list[i]->rot_frame_2_world * unitX * frame_list[i]->rot_frame_2_world.inverse()).vec();
						scaler_t view_angle_disparity = view_vec.dot(x_frame);
						if(view_angle_disparity >= min_view_cosorientation && std::abs((this_location - frame_location).dot(z_current_vec)) <= max_assist_altitude_diff){
							scaler_t distance = (this_location - frame_location).norm();
							if(distance < closest){
								closest = distance;
								best_assistance = frame_list[i];
							}
						}
					}
				}
			}
		}

		if(neighbor_y != -1){
			temp = Voxel2D[grid_x];
			if(temp){
				auto candidate = (*temp)[neighbor_y];
				if(candidate){
					std::vector<Frame*>& frame_list = *candidate;
					for(int i=0; i < frame_list.size(); i++){
						Vector3 frame_location = frame_list[i]->trans_frame_2_world;
						Vector3 x_frame = (frame_list[i]->rot_frame_2_world * unitX * frame_list[i]->rot_frame_2_world.inverse()).vec();
						scaler_t view_angle_disparity = view_vec.dot(x_frame);
						if(view_angle_disparity >= min_view_cosorientation && std::abs((this_location - frame_location).dot(z_current_vec)) <= max_assist_altitude_diff){
							scaler_t distance = (this_location - frame_location).norm();
							if(distance < closest){
								closest = distance;
								best_assistance = frame_list[i];
							}
						}
					}
				}
			}
		}

		return best_assistance;
	}

	void find_frame_candidates(Quaternion& current_rotation, Vector3& current_location, Frame** reference_pointer,\
							   std::vector<Frame*>& candidates, bool mode=true, int max_search_it=50){
		// Eigen::Vector4f ref_vec = reference_location.col(3);
		Frame* reference_frame = *reference_pointer;

		if(!location_assistance_status || !reference_frame){
			if(!mode){
				location_assistance_status = false;
				return;
			}
			reference_frame = find_frame_assistance(current_rotation, current_location);
			if(!reference_frame){
				location_assistance_status = false;
				*reference_pointer = 0;
				return;
			}

			location_assistance_status = true;
		}
		else{
			int k=0;
			scaler_t previous_distance = 1e7;
			while(k++ < max_search_it){  // ignores loop closing or weird motion
				scaler_t distance = (current_location - reference_frame->trans_frame_2_world).norm();
				if(distance > previous_distance)
					break;
				previous_distance = distance;
				if(reference_frame->next_keyframe)
					reference_frame = reference_frame->next_keyframe;
				else
					break;
			}
			if(k != max_search_it + 1 && reference_frame->next_keyframe)
				reference_frame = reference_frame->previous_keyframe;
		}

		Quaternion unitX(0, 1, 0, 0);
		Vector3 view_vec = (current_rotation * unitX * current_rotation.inverse()).vec();
		Frame* up_frame = reference_frame->next_keyframe;
		Frame* down_frame = reference_frame;
		int num_it = max_search_it / 10;

		bool sw = false;
		for(int i=0; i < num_it && candidates.size() < max_perframes; i++){
			if(up_frame){
				Vector3 frame_vec = (up_frame->rot_frame_2_world * unitX * up_frame->rot_frame_2_world.inverse()).vec();
				scaler_t cosine = view_vec.dot(frame_vec);
				scaler_t distance = (current_location - up_frame->trans_frame_2_world).norm();
				if(distance <= max_reference_distance){
					if(cosine >= min_view_cosorientation){
						candidates.push_back(up_frame);
						sw = true;
					}
				}
				up_frame = up_frame->next_keyframe;
				
				if(down_frame){
					Vector3 frame_vec = (down_frame->rot_frame_2_world * unitX * down_frame->rot_frame_2_world.inverse()).vec();
					scaler_t cosine = view_vec.dot(frame_vec);
					scaler_t distance = (current_location - down_frame->trans_frame_2_world).norm();
					if(distance <= max_reference_distance){
						if (cosine >= min_view_cosorientation){
							candidates.push_back(down_frame);
							sw = true;
						}
					}
					down_frame = down_frame->previous_keyframe;
				}
			}
			else if(down_frame){
				Vector3 frame_vec = (down_frame->rot_frame_2_world * unitX * down_frame->rot_frame_2_world.inverse()).vec();
				scaler_t cosine = view_vec.dot(frame_vec);
				scaler_t distance = (current_location - down_frame->trans_frame_2_world).norm();
				if(distance <= max_reference_distance){
					if(cosine >= min_view_cosorientation){
						candidates.push_back(down_frame);
						sw = true;
					}
				}
				down_frame = down_frame->previous_keyframe;
			}
			else
				break;
		}
		location_assistance_status = sw;
		*reference_pointer = reference_frame;
	}


	// Match each with frameA = [frameB, frameC, frameD]
	// keep matching list atmost 4 for lower cost
	// Can use KDTree if num_keypoints are large, fast matmul may bring cost down from cube
	void match_currentframe(std::vector<Frame*> reference_list, std::vector<std::pair<int, int>>* matches, float* match_scores, int min_match_score=0.9){
		float x_max[num_features];
		int x_argmax[num_features];

		for(int i=0; i < reference_list.size(); i++){
			match_scores[i] = 0;
			Eigen::MatrixXf distance_matrix(num_features, reference_list[i]->num_keypoints);
			distance_matrix.noalias() = current_frame->descriptors * reference_list[i]->descriptors;
			float* response = distance_matrix.data(); 	// column major
			int y_argmax[reference_list[i]->num_keypoints];
			
			for(int j=0; j < num_features; j++){
				x_max[j] = min_match_score;
				x_argmax[j] = -1;
			}
			
			for(int k=0; k < reference_list[i]->num_keypoints; k++){
				int offset = k * num_features;
				y_argmax[k] = -1;
				float y_max = min_match_score;
				for(int j=0; j < num_features; j++){
					int index = j + offset;
					if(y_max < response[index]){
						y_max = response[index];
						y_argmax[k] = j;
					}
					if(x_max[j] < response[index]){
						x_max[j] = response[index];
						x_argmax[j] = k;
					}
				}
			}
			if(num_features <= reference_list[i]->num_keypoints){
				for(int j=0; j < num_features; j++){
					if(x_argmax[j] != -1){
						if(y_argmax[x_argmax[j]] == j){
							match_scores[i] += x_max[j];
							matches[i].push_back(std::make_pair(j, x_argmax[j]));
						}
					}
				}
			}
			else{
				for(int j=0; j < reference_list[i]->num_keypoints; j++){
					if(y_argmax[j] != -1){
						if(x_argmax[y_argmax[j]] == j){
							match_scores[i] += x_max[y_argmax[j]];
							matches[i].push_back(std::make_pair(y_argmax[j], j));
						}
					}
				}
			}
		}
	}

	int frame_no = 0;

public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	observation(){
		num_ransac_iteration_m1 = (int)(std::log2(1 - success_p) / std::log2(1 - std::pow(1 - outlier_ratio, sample_size_m1))) + 1;
		num_ransac_iteration_m2 = (int)(std::log2(1 - success_p) / std::log2(1 - std::pow(1 - outlier_ratio, sample_size_m2))) + 1;
		current_frame = 0;
		location_assistance_status = false;
	    reference = 0;
    }

	void update_camera_frame(cameraObservation* frame){
		current_frame = frame;
		frame_no++;
	}

	void drawCircle(cv::Mat& img, cv::Point center, cv::Scalar S, int thickness, int radius)
	{
  		cv::circle(img,
      			   center,
      			   radius,
      			   S,
      			   thickness,
      			   cv::LINE_8);
	}

	bool register_new_observation(Quaternion& current_rotation, Vector3& current_location, VectorX& observation_points,\
								  VectorX& root_variance, MatrixX& H_of_sigma, MatrixS2S1& sigma_points, bool mode){
		
		if(mode){
			std::vector<Frame*> candidate;
			find_frame_candidates(current_rotation, current_location, &reference, candidate);
			if(!candidate.size())
				return false;

			std::vector<std::pair<int, int>> matches[candidate.size()];
			float match_scores[candidate.size()];
			printf("before match %ld\n", candidate.size());
			match_currentframe(candidate, matches, match_scores);
			int priority_index[2] = {0, 0};
			float maximum = 0, second_max = 0;
			
			for(int i=0; i < candidate.size(); i++){
				if(maximum < match_scores[i]){
					second_max = maximum;
					maximum = match_scores[i];
					priority_index[1] = priority_index[0];
					priority_index[0] = i;
				}
				else if(second_max < match_scores[i]){
					second_max = match_scores[i];
					priority_index[1] = i;
				}
			}
			std::vector<uchar> inliers[2];

			#ifndef USE_RANSAC_PNP
			int result = ransac_elimination_m1(candidate, matches, priority_index, inliers, 2); 
			scaler_t sigma_x = 8.0f;
			scaler_t sigma_y = 8.0f;
			#else
			int result = ransac_elimination_m2(candidate, matches, priority_index, inliers, 2);
			scaler_t sigma_x = 10.0f;
			scaler_t sigma_y = 10.0f;
			#endif

			#ifndef NO_IMAGE_OUT
			// ground truth

			for(int i=0; i < 1; i++){
				Matrix3 R_world_2_frame;
				R_world_2_frame = AngleAxis(-current_frame->true_angles(2), Vector3::UnitZ()) *\
								  AngleAxis(-current_frame->true_angles(1), Vector3::UnitY()) *\
								  AngleAxis(-current_frame->true_angles(0), Vector3::UnitX());
				int index = priority_index[i];
				int n = matches[index].size();
				cv::Mat image_1 = current_frame->image.clone();
				cv::Mat image_2 = candidate[index]->image.clone();
				// auto trans = R_world_2_frame * (candidate[index]->trans_frame_2_world - current_frame->true_location);
				// //trans /= trans.norm();
				// Vector3 framed;
				// framed(2) = trans(0);
				// framed(1) = -trans(2);
				// framed(0) = -trans(1);
				// std::cout << "actual translation" << -framed / framed.norm() << std::endl;
				
				for(int j=0; j < n; j++){
					if(!inliers[i][j])
						continue;
					
					int left = matches[index][j].first;
					int right = matches[index][j].second;
					auto centre_1 = cv::Point2f(current_frame->keypoints[left][0], current_frame->keypoints[left][1]);
					auto centre_2 = cv::Point2f(candidate[index]->keypoints[right][0], candidate[index]->keypoints[right][1]);
					int r = rand() % 256;
					int g = rand() % 256;
					int b = rand() % 256;
					auto zz = candidate[index]->keypoints[matches[index][j].second][2];

					for(int k=0; k < 1; k++){
						Vector4 Y;
						Vector3 X;
						X(0) = candidate[index]->keypoints[matches[index][j].second][0];
						X(1) = candidate[index]->keypoints[matches[index][j].second][1];
						X(2) = 1;
						Y.block<3,1>(0,0) = zz * (Kinv * X);
						Y(3) = 1;

						auto Z = current_frame->g_frame_to_world.inverse() * candidate[index]->g_frame_2_world * Y;
						Vector3 projections;
						projections(0) = Z(0);
						projections(1) = Z(1);
						projections(2) = Z(2);
						projections = projections / projections(0);
						projections = K * projections;
						
						auto centre_3 = cv::Point2f((float)projections(0), (float)projections(1));
						if(k == 0)
							drawCircle(image_1, centre_3, cv::Scalar(b, g, r), cv::FILLED, 4);
						else
							drawCircle(image_1, centre_3, cv::Scalar(b, g, r), cv::FILLED, 2);
						break; // to plot a line projection, remove break
						zz += 1;
					}
					// zz = candidate[index]->keypoints[matches[index][j].second][2];
					drawCircle(image_1, centre_1, cv::Scalar(b, g, r), 2, 4);
					drawCircle(image_2, centre_2, cv::Scalar(b, g, r), 2, 4);
				}
				// cv::imshow("recorded", image_2);
				// cv::imshow("current", image_1);
				// cv::waitKey(0);
				// cv::destroyAllWindows();
				std::string path_to_write_1 = "projections/current/" + std::to_string(frame_no) + ".png";
				std::string path_to_write_2 = "projections/recorded/" + std::to_string(frame_no) + ".png";
        		cv::imwrite(path_to_write_1, image_1);
        		cv::imwrite(path_to_write_2, image_2);
			}

			// for(int i=0; i < result; i++){
			// 	int index = priority_index[i];
			// 	int n = matches[index].size();
			// 	cv::Mat image_1 = current_frame->image.clone();
			// 	cv::Mat image_2 = candidate[index]->image.clone();
			// 	for(int j=0; j < n; j++){
			// 		if(!inliers[i][j])
			// 			continue;
			// 		int left = matches[index][j].first;
			// 		int right = matches[index][j].second;
			// 		auto centre_1 = cv::Point2f(current_frame->keypoints[left][0], current_frame->keypoints[left][1]);
			// 		auto centre_2 = cv::Point2f(candidate[index]->keypoints[right][0], candidate[index]->keypoints[right][1]);
			// 		int r = rand() % 256;
			// 		int g = rand() % 256;
			// 		int b = rand() % 256;
			// 		drawCircle(image_1, centre_1, cv::Scalar(b, g, r), 2, 4);
			// 		drawCircle(image_2, centre_2, cv::Scalar(b, g, r), 2, 4);
			// 	}
			// 	cv::imshow("current", image_1);
			// 	cv::imshow("recorded", image_2);
			// 	cv::waitKey(0);
			// 	cv::destroyAllWindows();
			// }

			#endif

			
			if(result == 1){
				int index = priority_index[0];
				int n = matches[index].size();
				int npt=0;
				for(int i=0; i < n; i++)
					if(inliers[0][i])
						npt++;

				npt = std::min(npt, max_obs_size);
				int m = 2 * npt + const_measurement;
				int num_sigma = sigma_points.cols();
				observation_points.resize(m);
				root_variance.resize(m);
				H_of_sigma.resize(m, num_sigma);
				MatrixX points(3, npt);
				scaler_t* h_writer = H_of_sigma.data();
				scaler_t* writer = observation_points.data();
				scaler_t* var_writer = root_variance.data();
				scaler_t* point_writer = points.data();
				int j=0;
				int t=0;

				for(int i=0; i < n && t < npt; i++){
					if(inliers[0][i]){
						writer[2*j] = current_frame->keypoints[matches[index][i].first][0];
						writer[2*j + 1] = current_frame->keypoints[matches[index][i].first][1];
						var_writer[2*j] = sigma_x;
						var_writer[2*j + 1] = sigma_y;
						point_writer[3*j] = candidate[index]->keypoints[matches[index][i].second][2] * candidate[index]->keypoints[matches[index][i].second][0];
						point_writer[3*j+1] = candidate[index]->keypoints[matches[index][i].second][2] * candidate[index]->keypoints[matches[index][i].second][1];
						point_writer[3*j+2] = candidate[index]->keypoints[matches[index][i].second][2];
						j++;
						t++;
					}
				}
				
				for(int i=0; i < num_sigma; i++){
					Matrix3 R_world_2_frame;
					R_world_2_frame = AngleAxis(-sigma_points(7, i), Vector3::UnitZ()) *\
								      AngleAxis(-sigma_points(6, i), Vector3::UnitY()) *\
									  AngleAxis(-sigma_points(5, i), Vector3::UnitX());
					MatrixX projections = ((K * R_world_2_frame * candidate[index]->Rmat_frame_2_world * Kinv) * points).colwise() +\
										   (K * R_world_2_frame * (candidate[index]->trans_frame_2_world - sigma_points.block<3, 1>(0, i)));
					VectorX z_div = projections.row(2);
					projections = projections.array().rowwise() / z_div.transpose().array();
					scaler_t* ptr = projections.data();
					int offset = m * i;
					for(int j=0; j < npt; j++){
						h_writer[offset + 2*j] = ptr[3*j];
						h_writer[offset + 2*j + 1] = ptr[3*j + 1];
					}
				}
			}
			else if(result == 2){
				int index_1 = priority_index[0];
				int n = matches[index_1].size();
				int npt_1=0;
				for(int i=0; i < n; i++)
					if(inliers[0][i])
						npt_1++;

				int index_2 = priority_index[1];
				n = matches[index_2].size();
				int npt_2=0;
				for(int i=0; i < n; i++)
					if(inliers[1][i])
						npt_2++;
				
				if(npt_1 + npt_2 > max_obs_size){
					if(2 * npt_1 <= max_obs_size){
						npt_2 = max_obs_size - npt_1;
					}
					else if(2 * npt_2 <= max_obs_size){
						npt_1 = max_obs_size - npt_2;
					}
					else{
						npt_2 = max_obs_size / 2;
						npt_1 = max_obs_size - npt_2;
					}
				}

				int m = 2 * (npt_1 + npt_2) + const_measurement;
				int num_sigma = sigma_points.cols();
				observation_points.resize(m);
				root_variance.resize(m);
				H_of_sigma.resize(m, num_sigma);
				MatrixX points_1(3, npt_1);
				MatrixX points_2(3, npt_2);
				scaler_t* h_writer = H_of_sigma.data();
				scaler_t* writer = observation_points.data();
				scaler_t* var_writer = root_variance.data();
				scaler_t* point_writer_1 = points_1.data();
				scaler_t* point_writer_2 = points_2.data();
				int j=0;
				int t=0;

				n = matches[index_1].size();
				for(int i=0; i < n && t < npt_1; i++){
					if(inliers[0][i]){
						writer[2*j] = current_frame->keypoints[matches[index_1][i].first][0];
						writer[2*j+1] = current_frame->keypoints[matches[index_1][i].first][1];
						var_writer[2*j] = sigma_x;
						var_writer[2*j+1] = sigma_y;
						point_writer_1[3*j] = candidate[index_1]->keypoints[matches[index_1][i].second][2] * candidate[index_1]->keypoints[matches[index_1][i].second][0];
						point_writer_1[3*j+1] = candidate[index_1]->keypoints[matches[index_1][i].second][2] * candidate[index_1]->keypoints[matches[index_1][i].second][1];
						point_writer_1[3*j+2] = candidate[index_1]->keypoints[matches[index_1][i].second][2];
						j++;
						t++;
					}
				}

				n = matches[index_2].size();
				int k=j;
				t=0;
				for(int i=0; i < n && t < npt_2; i++){
					if(inliers[1][i]){
						writer[2*j] = current_frame->keypoints[matches[index_2][i].first][0];
						writer[2*j+1] = current_frame->keypoints[matches[index_2][i].first][1];
						var_writer[2*j] = sigma_x;
						var_writer[2*j+1] = sigma_y;
						point_writer_2[3*(j-k)] = candidate[index_2]->keypoints[matches[index_2][i].second][2] * candidate[index_2]->keypoints[matches[index_2][i].second][0];
						point_writer_2[3*(j-k)+1] = candidate[index_2]->keypoints[matches[index_2][i].second][2] * candidate[index_2]->keypoints[matches[index_2][i].second][1];
						point_writer_2[3*(j-k)+2] = candidate[index_2]->keypoints[matches[index_2][i].second][2];
						j++;
						t++;
					}
				}
				
				for(int i=0; i < num_sigma; i++){
					Matrix3 R_world_2_frame;
					R_world_2_frame = AngleAxis(-sigma_points(7, i), Vector3::UnitZ()) *\
									  AngleAxis(-sigma_points(6, i), Vector3::UnitY()) *\
									  AngleAxis(-sigma_points(5, i), Vector3::UnitX());
					MatrixX F = K * R_world_2_frame;
					MatrixX projections = ((F * candidate[index_1]->Rmat_frame_2_world * Kinv) * points_1).colwise() +\
										   (F * (candidate[index_1]->trans_frame_2_world - sigma_points.block<3, 1>(0, i)));
					VectorX z_div = projections.row(2);
					projections = projections.array().rowwise() / z_div.transpose().array();
					scaler_t* ptr = projections.data();
					int offset = m * i;
					for(int j=0; j < npt_1; j++){
						h_writer[offset + 2*j] = ptr[3*j];
						h_writer[offset + 2*j + 1] = ptr[3*j + 1];
					}
					
					projections = ((F * candidate[index_2]->Rmat_frame_2_world * Kinv) * points_2).colwise() +\
								   (F * (candidate[index_2]->trans_frame_2_world - sigma_points.block<3, 1>(0, i)));
					z_div = projections.row(2);
					projections = projections.array().rowwise() / z_div.transpose().array();
					ptr = projections.data();
					offset += 2 * npt_1;
					for(int j=0; j < npt_2; j++){
						h_writer[offset + 2*j] = ptr[3*j];
						h_writer[offset + 2*j + 1] = ptr[3*j + 1];
					}
				}
			}
			else
				return false;
		}
		else{
			std::vector<Frame*> candidate;
			find_frame_candidates(current_rotation, current_location, &reference, candidate, false);
			return false;
		}

		// printf("returning\n");
		return true;
	}
};
}

#endif
