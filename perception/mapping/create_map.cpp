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
#include <pcl/visualization/pcl_visualizer.h>
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
#include "omp.h"
#include "cnpy.h"

#define camera_fov 90
#define height 820
#define width 820
#define num_keypoints 3000
#define descriptor_size 128
#define keyframe_trans_threshold 1.5f
#define point_distance_threshold 40.0f
#define max_neighbor_distance 40.0f
#define min_score_for_validity 0.10f
#define lidar_range 40
#define max_reliable_distance 90.0f
#define low_shadow_dist 7.0f
#define voxel_size 10    // even and should divide height and width
#define min_pt_per_voxel 2
#define min_valid_obs 2
#define neighbor_angle_degree 0.1
#define neighbor_max_angle_degree 0.5
#define flip -1.0f     // for left to right coordinate system and vice versa
#define GetCurrentDir getcwd
#define num_classes 5

typedef enum {

BUILDINGS=1,
ROADLINES=6,
ROADS,
SIDEWALKS=8,
TRAFFIC_SIGN=9
} CLASSES;

class Frame;
Frame** frames = 0;
const int x_len = width / voxel_size;
const int y_len = height / voxel_size;
const float min_pt_distance = 0.1;

struct frame_distance_array 
{
	int frame_id;
	float distance;
};

struct globalPoint
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Eigen::Vector4f global_point;
	Frame* frame;
	int id;
	volatile int num_observation;
	volatile float score[num_classes];
	volatile float weight_sum;
};

struct Candidate
{
	globalPoint* point;
	
	#if voxel_size > 16
		unsigned short int pos;
	#else
		uint8_t pos;
	#endif
	
	float projection_score[num_classes];
};

struct tempLocalPoint
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Eigen::Vector4f pt_vec;
	Candidate* candidate_point;
	double distance;
};

struct voxel_grid 
{
	int num_pt;
	float iou;
	omp_lock_t writelock;
	std::vector<Candidate> candidate_points;
};

bool compare_frames(frame_distance_array frameA, frame_distance_array frameB){
	if(frameA.distance < frameB.distance)
		return true;
	else
		return false;
}

bool compare_candidates(tempLocalPoint x, tempLocalPoint y){
	if(x.distance < y.distance)
		return true;
	else
		return false;
}

class Frame {
private:

	inline void push_candidate(std::vector<tempLocalPoint>& block, int x, int y, int i){
		Eigen::Vector4f local_frame_pt = world_2_frame * voxel[y][x].candidate_points[i].point->global_point;
		double distance = std::sqrt(local_frame_pt.squaredNorm() - 1.0f);
		tempLocalPoint T = {local_frame_pt, &voxel[y][x].candidate_points[i], distance};
		block.push_back(T);
	}

	bool threshold_false_points(std::vector<tempLocalPoint>& V, int& start, int& end,\
								const double cos_limit, const double cos_limit_recheck, int direction=1, float min_parallel_dist=1.0f){
		
		float max_cosine = 1.0f, max_cosine_recheck = 1.0f;
		int new_start = start, new_recheck_start = start;
		int new_end = end;

		for(int i = start + direction; i * direction <= end * direction; i += direction){
			double cosine = ((double)(V[start].pt_vec.dot(V[i].pt_vec) - 1.0)) / (V[start].distance * V[i].distance);

			if(cosine > cos_limit){
				if(((V[i].distance * cosine) - V[start].distance) * direction > min_parallel_dist){
					new_end = i;
				}
				else if(cosine < max_cosine){
					max_cosine = cosine;
					new_start = i;
				}
			}
			else if(cosine > cos_limit_recheck){
				if(((V[i].distance * cosine) - V[start].distance) * direction <= min_parallel_dist){
					if(cosine < max_cosine_recheck){
						max_cosine_recheck = cosine;
						new_recheck_start = i;
					}
				}
			}
		}

		if(new_start == start){
			if(new_recheck_start != start){
				start = new_recheck_start;
				end = new_end;
			}
			else
				return false;
		}
		else{
			start = new_start;
			end = new_end;
		}
		
		return true;
	}

	void initialize_keyframe(){
		for(int x=0; x<x_len; x++){
        	for(int y=0; y<y_len; y++){
        		omp_init_lock(&voxel[y][x].writelock);
        	}
        }

        for(int i=0; i < num_keypoints; i++){
        	down_relation[i] = -1;
        	up_relation[i] = -1;
        	estimated_3distance[i][0] = -1;
        	estimated_3distance[i][1] = -1;
        	estimated_3distance[i][2] = -1;
        	estimated_3distance[i][3] = -1;
        	estimated_3distance[i][4] = -1;
        	estimated_2distance[i] = -1;
        }
	}

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // stores closest keyframe if isKeyFrame is False else stores next keyframe in the sequence
    Frame* next_keyframe;
    Frame* previous_keyframe;
    float nearest_keyframe_distance;
	std::vector<frame_distance_array> neighbors;

	// Frame id should correspond to frames position in the global array
	int frame_id;
	Eigen::Matrix4f& frame_2_world;
	Eigen::Matrix4f world_2_frame;
	int cam_index;
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
	cv::Mat p_mask[num_classes];
    cv::Mat img;
    cv::Mat refined_pmask[num_classes];
    
    // scope for memory optimization incase not a keyframe
    bool isKeyFrame;
    voxel_grid voxel[y_len][x_len];
    std::vector<Eigen::Vector4f> granular_projection[height][width];
    std::vector<std::pair<int, float>> final_keypoints;
    float keypoints[num_keypoints][3];
    Eigen::MatrixXf descriptors; // size constraint
    float* descriptor_ptr;
    float scores[num_keypoints];
    int down_relation[num_keypoints];
    int up_relation[num_keypoints];
    float estimated_2distance[num_keypoints];
    float estimated_3distance[num_keypoints][5];
    
	Frame(int id, int camera_index, float* desc, float* score, float* keypt, cv::Mat image[num_classes], cv::Mat transformed_p[num_classes],\
	      pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, Eigen::Matrix4f& frame2world) : frame_2_world(frame2world), descriptors(num_keypoints, descriptor_size){
        
        frame_id = id;
        cam_index = camera_index;
        
        // img = cv::Mat::zeros(height, width, CV_8UC3);
        // int fromTo[] = {0, 0};
        // cv::mixChannels(&image[2], 1, &img, 1, fromTo, 1);

        for(int i=0; i < num_classes; i++){
        	image[i].convertTo(p_mask[i], CV_32F);
        	p_mask[i] = p_mask[i] / 255.0;
        	refined_pmask[i] = transformed_p[i]; // cv uses shared_ptr
        }
        
        next_keyframe = 0;
        previous_keyframe = 0;
        nearest_keyframe_distance = -1;
        isKeyFrame = false;
        world_2_frame = frame_2_world.inverse();
        point_cloud = input_cloud;
        descriptor_ptr = descriptors.data();

        for(int i=0; i < num_keypoints; i++){
        	for(int j=0; j < 3; j++){
        		keypoints[i][j] = keypt[3 * i + j];
        	}

        	for(int j=0; j < descriptor_size; j++){
        		descriptor_ptr[j * num_keypoints + i] = desc[descriptor_size * i + j];
        	}

        	scores[i] = score[i];
        }
    }

	~Frame(){
		neighbors.clear();
	}

	bool try_set_keyframe(Frame* prev_keyframe){
		if(frame_id==0){
			isKeyFrame=true;
			initialize_keyframe();
			return true;
		}

		if(!prev_keyframe->isKeyFrame){
			printf("Error! previous frame not a keyframe\n");
			exit(0);
		}

		Eigen::Vector4f vec1 = prev_keyframe->frame_2_world.col(3);
		Eigen::Vector4f vec2 = frame_2_world.col(3);
		float distance = (vec1 - vec2).norm();
		if(distance >= keyframe_trans_threshold){
			isKeyFrame = true;
			initialize_keyframe();
			prev_keyframe->next_keyframe = this;
			previous_keyframe = prev_keyframe;
			int i = frame_id - 1;
			while(i > prev_keyframe->frame_id){
				if(frames[i]->isKeyFrame){
					printf("Error! Unexpected keyframe\n");
			        exit(0);
				}
				Eigen::Vector4f vec1 = frames[i]->frame_2_world.col(3);
				float distance = (vec1 - vec2).norm();
				if(distance < frames[i]->nearest_keyframe_distance){
					frames[i]->nearest_keyframe_distance = distance;
					frames[i]->next_keyframe = this;
				}
				i--;
			}
		}
		else{
			isKeyFrame = false;
			nearest_keyframe_distance = distance;
			next_keyframe = prev_keyframe;
		}

		return isKeyFrame;
	}

	void fill_neighbor_list(std::vector<int>& frame_ids){
		if(!isKeyFrame)
			return;
		
		int n = frame_ids.size();
		Eigen::Vector4f vec2 = frame_2_world.col(3);
		for(int i=0; i < frame_ids.size(); i++){
			if(frame_id == frame_ids[i])
				continue;

			Eigen::Vector4f vec1 = frames[frame_ids[i]]->frame_2_world.col(3);
			float distance = (vec1 - vec2).norm();
			if(distance < max_neighbor_distance){
				frame_distance_array neighbor_frame = {frame_ids[i], distance};
				neighbors.push_back(neighbor_frame);
			}
		}
		std::sort(neighbors.begin(), neighbors.end(), compare_frames);
	}

	Frame* get_keyframe(){
		if(isKeyFrame)
			return this;
		else
			return next_keyframe;
	}

	// The key frames can be considered as a sample points from a smooth function: dist(trajectory of car, point)
	// This method is twice more constly than directional but covers more edge cases(wierd motion)
	Frame* find_nondirectional_closest_Frame(Eigen::Vector4f& point){
		if(!isKeyFrame)
			return 0;

		Frame* up_frame = this;
		Frame* down_frame = this;
		float current_best = 1e7;
		Frame* closest = this;
		
		for(int i=0; i < lidar_range && (down_frame || up_frame); i++){
			if(up_frame){
				float distance = (up_frame->frame_2_world.col(3) - point).norm();
				if(distance < current_best){
					current_best = distance;
					closest = up_frame;
				}
				up_frame = up_frame->next_keyframe;
			}
			if(down_frame){
				float distance = (down_frame->frame_2_world.col(3) - point).norm();
				if(distance < current_best){
					current_best = distance;
					closest = down_frame;
				}
				down_frame = down_frame->previous_keyframe;
			}
		}
		return closest;
	}

	// searches only in direction of motion
	// use this when lidar_range is higher than 50
	Frame* find_directional_closest_Frame(Eigen::Vector4f& point){
		if(!isKeyFrame)
			return 0;

		Frame* up_frame = this;
		Frame* down_frame = this;
		for(int i=0; i < lidar_range; i++){
			if(up_frame->next_keyframe)
				up_frame = up_frame->next_keyframe;
			if(down_frame->previous_keyframe)
				down_frame = down_frame->previous_keyframe;
		}
		Eigen::Vector4f trajectory_vector = up_frame->frame_2_world.col(3) - down_frame->frame_2_world.col(3);
		Eigen::Vector4f local_point_frame = point - frame_2_world.col(3);
		float component = trajectory_vector.dot(local_point_frame) - 1.0;
		float current_best = 1e7;
		Frame* closest = this;

		// acute angle 
		if(component > 0){
			Frame* current_frame = this;
			for(int i=0; i < lidar_range && current_frame; i++){
				float distance = (current_frame->frame_2_world.col(3) - point).norm();
				if(distance < current_best){
					current_best = distance;
					closest = current_frame;
				}
				current_frame = current_frame->next_keyframe;
			}
		}
		else{
			Frame* current_frame = this;
			for(int i=0; i < lidar_range && current_frame; i++){
				float distance = (current_frame->frame_2_world.col(3) - point).norm();
				if(distance < current_best){
					current_best = distance;
					closest = current_frame;
				}
				current_frame = current_frame->previous_keyframe;
			}
		}

		return closest;
	}

	void match_descriptors(Eigen::MatrixXf& P, Eigen::MatrixXf& Q, std::vector<std::pair<int, int>>& matches, Eigen::MatrixXf& distance_matrix, float min_match_score=0.5){
		float x_max[num_keypoints];
		int x_argmax[num_keypoints];
		distance_matrix.noalias() = P * Q.transpose();
		float* response = distance_matrix.data();
		int y_argmax[num_keypoints];
		for(int j=0; j < num_keypoints; j++){
			x_max[j] = min_match_score;
			x_argmax[j] = -1;
		}
		for(int k=0; k < num_keypoints; k++){
			int offset = k * num_keypoints;
			y_argmax[k] = -1;
			float y_max = min_match_score;
			for(int j=0; j < num_keypoints; j++){
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
		
		for(int j=0; j < num_keypoints; j++){
			if(x_argmax[j] != -1){
				if(y_argmax[x_argmax[j]] == j){
					matches.push_back(std::make_pair(j, x_argmax[j])); // (P,Q)
				}
			}
		}
	}

	// TODO : use reprojection error instead of ransac(temperory soln)
	int match_keypoints(){
		if(!frame_id || !isKeyFrame)
			return 0;

		std::vector<std::pair<int, int>> matches;
		Eigen::MatrixXf distance_matrix(num_keypoints, num_keypoints);
		match_descriptors(descriptors, previous_keyframe->descriptors, matches, distance_matrix);
		int n = matches.size();
		std::vector<cv::Point2d> points1(n);
		std::vector<cv::Point2d> points2(n);
		for(int i=0; i < n; i++){
			int left = matches[i].first;
			int right = matches[i].second;
			points1[i] = cv::Point2d((double)keypoints[left][0], (double)keypoints[left][1]);
			points2[i] = cv::Point2d((double)previous_keyframe->keypoints[right][0], (double)previous_keyframe->keypoints[right][1]);
		}

		std::vector<uchar> inliers(n);
		int match = 0;
		cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99, inliers);
		for(int i=0; i < n; i++){
			if(inliers[i]){
				int left = matches[i].first;
				int right = matches[i].second;
				previous_keyframe->up_relation[right] = left;
				down_relation[left] = right;
				match++;
			}
		}

		return match;
	}

	// descriptors are assumed to be normalized already
	// matching occurs iff both keypoints are each others first preference 
	// int match_keypoints_(float minimum_score=0.96){
	// 	if(!frame_id || !isKeyFrame)
	// 		return 0;

	// 	int match = 0;
	// 	for(int i=0; i < num_keypoints; i++){
	// 		float maximum = 0;
	// 		int index = 0;
	// 		for(int j=0; j < num_keypoints; j++){
	// 			float sum = 0;
	// 			for(int k=0; k < descriptor_size; k++){
	// 				sum += previous_keyframe->descriptors[j][k] * descriptors[i][k];
	// 			}
	// 			if(maximum < sum){
	// 				maximum = sum;
	// 				index = j;
	// 			}
	// 		}
	// 		if(maximum >= minimum_score){
	// 			if(previous_keyframe->up_relation[index] != -1){  // ambiguious case
	// 				float sum = 0;
	// 				for(int k=0; k < descriptor_size; k++){
	// 					sum += descriptors[previous_keyframe->up_relation[index]][k] * previous_keyframe->descriptors[index][k];
	// 			    }
	// 			    if(sum < maximum){
	// 			    	down_relation[previous_keyframe->up_relation[index]] = -1;
	// 				    down_relation[i] = index;
	// 				    previous_keyframe->up_relation[index] = i;
	// 			    }
	// 			}
	// 			else{
	// 				match++;
	// 				previous_keyframe->up_relation[index] = i;
	// 				down_relation[i] = index;
	// 			}
	// 		}
	// 	}

	// 	return match;
	// }
	// int checker=0;

	// refine the previous distance estimation of interest point from epipolar geometry through lidar points
	float estimate_distance_with_lidar(int x, int y, Eigen::Vector4f prior_vector, int threshold, int max_at,\
									   float max_weight, int voxel_length=17, float chunk=0.50){
		// return -1.0f;
		// ignore border points for simplicity
		int half_voxel = voxel_length >> 1;
		int y_start = y - half_voxel;
		int x_start = x - half_voxel;
		int y_end = y + half_voxel;
		int x_end = x + half_voxel;
		if(x_start < 0 || x_end >= width || y_start < 0 || y_end >= height){
			return -1;
		}

		int npt=0;
		for(int i=y_start; i <= y_end; i++)
			for(int j=x_start; j <= x_end; j++)
				npt += granular_projection[i][j].size();
		
		if(npt < threshold){
			// printf("npt=%d\n", npt);
			return -1;
		}
		
		std::vector<tempLocalPoint> near_points(npt);
		int first=0;
		float closest=1e7, distance;
		npt = 0;
		for(int i=y_start; i <= y_end; i++){
			for(int j=x_start; j <= x_end; j++){
				for(int k=0; k < granular_projection[i][j].size(); k++){
					near_points[npt].distance = std::sqrt(granular_projection[i][j][k].squaredNorm()- 1.0f);
					near_points[npt].pt_vec = granular_projection[i][j][k];
					int* this_index = new int;
					*this_index = j + i * width;
					near_points[npt].candidate_point = (Candidate *)(void *)this_index;
					npt++;
				}
			}
		}

		std::sort(near_points.begin(), near_points.end(), compare_candidates);

		for(int i=0; i < npt; i++){
			float temp = (near_points[i].pt_vec - prior_vector).squaredNorm();
			if(temp < closest){
				closest = temp;
				first = i;
			}
		}

		const double cos_limit = 0.99;
		const double cos_limit_recheck = 0.96;

		// point clustering
		int start = first;
		int end = npt - 1;
		while(start < end && threshold_false_points(near_points, start, end, cos_limit, cos_limit_recheck, 1, 0.8));
		int cluster_right = start;
		
		start = first;
		end = 0;
		while(start > end && threshold_false_points(near_points, start, end, cos_limit, cos_limit_recheck, -1, 0.8));
		int cluster_left = start;

		int cluster_size = cluster_right - cluster_left + 1;
		float weight_lidar = max_weight * std::min((float)cluster_size / max_at, 1.0f);
		int k = std::max((int)(chunk * cluster_size), 1);
		std::pair<float, int> pairs[cluster_size];
		for(int i=cluster_left; i <= cluster_right; i++){
			int index = *(int *)(void *)near_points[i].candidate_point;
			pairs[i-cluster_left] = std::make_pair(std::sqrt(((index / width) - y) * ((index / width) - y) + ((index % width) - x) * ((index % width) - x)), i);
		}

		std::sort(pairs, pairs + cluster_size);
		float num=0,den=0;
		for(int i=0; i < k; i++){
			float weight = (1.0 / (float)(1 + pairs[i].first));
			num += weight * near_points[pairs[i].second].distance;
			den += weight;
		}
		
		return weight_lidar * (num / den) + (1 - weight_lidar) * std::sqrt(prior_vector.squaredNorm() - 1.0);
	}

	// heuristic, use it to eliminate false-positive(and false negative with some help of the mask)
	// pts that are projected see-through from an actual object 
	// for points identified as valid observation tries to score them
	void score_projection_points(int min_to_divide=15){	
		if(!isKeyFrame)
			return;

		const double cos_limit = 0.98;
		const double cos_limit_recheck = 0.95; 
		int new_voxel_size = voxel_size >> 1;
		
		for(int y=0; y < y_len; y++){
			for(int x=0; x < x_len; x++){
				int n = voxel[y][x].candidate_points.size();
				if(n > 0){
					std::vector<tempLocalPoint> block[4];
					int num_block;

					if(n > min_to_divide && voxel[y][x].num_pt >= min_pt_per_voxel){
						
						for(int i=0; i<n; i++){
							int r_y = voxel[y][x].candidate_points[i].pos / voxel_size;
							int r_x = voxel[y][x].candidate_points[i].pos % voxel_size;
							
							if(r_x < new_voxel_size){
								if(r_y < new_voxel_size)
									push_candidate(block[0], x, y, i);									
								else
									push_candidate(block[1], x, y, i);
							}
							else{
								if(r_y < new_voxel_size)
									push_candidate(block[2], x, y, i);
								else
									push_candidate(block[3], x, y, i);	
							}
						}
						num_block = 4;
					}
					else{
						for(int i=0; i<n; i++)
							push_candidate(block[0], x, y, i);
						num_block = 1;
					}

					if(voxel[y][x].num_pt >= min_pt_per_voxel){
						for(int i=0; i < num_block; i++){
							int n = block[i].size();
							if(!n)
								continue;
							std::sort(block[i].begin(), block[i].end(), compare_candidates);
						}
					
						for(int b=0; b < num_block; b++){
							int n = block[b].size();
							if(!n)
								continue;
							int start = 0, end = n-1;
							while(start < end && threshold_false_points(block[b], start, end, cos_limit, cos_limit_recheck));
							float dist;
							for(int i=0; i <= start; i++){
								if(block[b][i].distance > min_pt_distance){
									int id = std::min(block[b][i].candidate_point->point->id, y_len * x_len - 1);
									int _x = id % x_len;
									int _y = id / x_len;
									omp_set_lock(&block[b][i].candidate_point->point->frame->voxel[_y][_x].writelock);
									block[b][i].candidate_point->point->num_observation += 1;
									dist = std::pow(block[b][i].distance, 1.0);
									for(int j=0; j < num_classes; j++){
										block[b][i].candidate_point->point->score[j] += (block[b][i].candidate_point->projection_score[j] / dist);
									}
									block[b][i].candidate_point->point->weight_sum += (1.0 / dist);
									omp_unset_lock(&block[b][i].candidate_point->point->frame->voxel[_y][_x].writelock);
								}
							}
						}
					}
				}
			}
		}
	}
};

void drawGreenCircle(cv::Mat& img, cv::Point center)
{
  cv::circle(img,
      center,
      2,
      cv::Scalar(0, 255, 0),
      cv::FILLED,
      cv::LINE_8);
}

void drawRedCircle(cv::Mat& img, cv::Point center)
{
  cv::circle(img,
      center,
      2,
      cv::Scalar(0, 0, 255),
      cv::FILLED,
      cv::LINE_8);
}

void remove_projection_shadows(cv::Mat& image, cv::Mat& destination, int ksize=3, int scale=1, int delta=0){
    cv::Mat grad_x, grad_y, grad;
    cv::Mat abs_grad_p, abs_grad_n;

    cv::Sobel(image, grad_x, CV_16S, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(image, grad_y, CV_16S, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

    cv::Mat grad_comp_p = cv::Mat::zeros(height, width, CV_32S);
    cv::Mat grad_comp_n = cv::Mat::zeros(height, width, CV_32S);

    int w_w = width >> 1;
    for(int i=0; i < height; i++){
        for(int j=0; j < width; j++){
            float distance = std::sqrt((w_w - j) * (w_w - j) + (height - i - 1) + (height - i - 1));
            if(distance > 5){
                float component = ((float)(((int)grad_x.at<int16_t>(i, j)) * (w_w - j) + ((int)grad_y.at<int16_t>(i, j)) * (height - i - 1))) / distance;
                if(component > 0){
                    grad_comp_n.at<int>(i, j) = (int)component;
                }
                else{
                    grad_comp_p.at<int>(i, j) = -(int)component;
                }
            }
        }
    }

    cv::convertScaleAbs(grad_comp_p, abs_grad_p);
    cv::convertScaleAbs(grad_comp_n, abs_grad_n);
    // cv::Mat img = cv::Mat::zeros(heigth, width, CV_8UC3);
    // int fromTo[] = {0, 2, 1, 1};
    // cv::Mat image_arr[] = {abs_grad_n, abs_grad_p};
    // cv::mixChannels(image_arr, 2, &img, 1, fromTo, 2);
    // cv::imshow("Window Name", img);
    // cv::waitKey(0);
    // double min, max;
    // cv::minMaxLoc(abs_grad_n, &min, &max);
    // printf("%f %f \n", min, max);
    destination = cv::Mat::zeros(height, width, CV_32F);
  
    for(int i=0; i < height; i++){
        for(int j=0; j < width; j++){
            float neg = ((float)abs_grad_n.at<uint8_t>(i, j)) / 255.0;
            float pos = ((float)abs_grad_p.at<uint8_t>(i, j)) / 255.0; 
            float probability_object = ((float)image.at<uint8_t>(i, j)) / 255.0;
            if(neg * pos < 0.95){
                float norm = 1 - neg * pos;
                // float probability_pos = (1 - neg) * pos / norm;
                float probability_neg = (1 - pos) * neg / norm;
                probability_object *= (1.0 - probability_neg);
            }
            else{
                probability_object = std::min(probability_object * (1.0 - neg + pos), 1.0);
            }
            destination.at<float>(i, j) = probability_object;
        }
    }
}

std::string find_filename(std::string& image_path){
	bool trigger=false;
	int j = 0;
	for(int i=image_path.size()-1; i >= 0; i--){
		if(trigger){
			if(image_path[i] == '/'){
				return image_path.substr(i+1, j-i-1);
			}
			continue;
		}
		if(image_path[i] == '.'){
			trigger=true;
			j = i;
		}
	}

	return "";
}

int solve(Eigen::MatrixXf& A, Eigen::VectorXf& b, Eigen::VectorXf& x, float left_bound=5.0f, float right_bound=90.0f){
	int size = 3 * (A.cols() - 1), j=1;
	Eigen::Matrix<float, 3, 2> temp_A;
	Eigen::Vector3f temp_b;
	Eigen::Vector2f temp_estimate;
	float this_estimate=0, prev_estimate;
	for(int i=0; i<size; i+=3){
		temp_A.block<3,1>(0, 0) = A.block<3,1>(i, 0);
		temp_A.block<3,1>(0, 1) = A.block<3,1>(i, j);
		temp_b = b.block<3,1>(i, 0);
		temp_estimate = (temp_A.transpose() * temp_A).inverse() * temp_A.transpose() * temp_b;
		prev_estimate = this_estimate;
		this_estimate = temp_estimate(0);
		// Eigen::Vector3f projection = (-this_estimate * temp_A.col(0)) + temp_b;
		// float reprojection_error = ((projection / projection(2)) - temp_A.col(1)).norm();
		// if(reprojection_error > 8) break;
		if(this_estimate < left_bound || this_estimate > right_bound || (i > 0 && std::abs(this_estimate - prev_estimate) > 5.0f))
			break;
		j++;
	}
	if(j > 1){
		Eigen::MatrixXf A_f = A.block(0,0,3*(j-1),j);
		Eigen::VectorXf b_f = b.block(0,0,3*(j-1),1);
		x.block(0,0,j,1) = (A_f.transpose() * A_f).inverse() * A_f.transpose() * b_f;
	}

	return j;
}

std::string get_current_dir() {
   char buff[FILENAME_MAX];
   GetCurrentDir(buff, FILENAME_MAX );
   std::string current_working_dir(buff);
   return current_working_dir;
}

void seperate_mask(cv::Mat& mask, cv::Mat out[num_classes]){
	for(int i=0; i < num_classes; i++){
		out[i] = cv::Mat::zeros(height, width, CV_8U);
	}

	for(int i=0; i < height; i++){
		for(int j=0; j < width; j++){
			int m = (int)mask.at<uint8_t>(i, j);
			switch((CLASSES)m){
				case BUILDINGS:
				{
					out[0].at<uint8_t>(i, j) = 255;
					break;
				}
				case ROADLINES:
				{
					out[1].at<uint8_t>(i, j) = 255;
					break;
				}
				case ROADS:
				{
					out[2].at<uint8_t>(i, j) = 255;
					break;
				}
				case SIDEWALKS:
				{
					out[3].at<uint8_t>(i, j) = 255;
					break;
				}
				case TRAFFIC_SIGN:
				{
					out[4].at<uint8_t>(i, j) = 255;
					break;
				}
				default:
				{
					break;
				}
			}
		}
	}
}

void get_set_intersection(std::vector<float>* inputs, int n, std::vector<int*>& result){
	int current_index[n];
	for(int i=0; i < n; i++){
		current_index[i] = 0;
	}

	while(current_index[0] < inputs[0].size()){
		float maximum = inputs[0][current_index[0]];
		bool val_set = false;
		for(int i=1; i < n; i++){
			if(current_index[i] == inputs[i].size())
				return;
			if(maximum < inputs[i][current_index[i]]){
				maximum = inputs[i][current_index[i]];
			}
			if(!val_set)
				val_set = (inputs[i][current_index[i]] != inputs[i-1][current_index[i-1]]);
		}

		if(val_set){
			for(int i=0; i < n; i++){
				if(maximum > inputs[i][current_index[i]])
					current_index[i]++;
			}
		}
		else{
			int* element = new int[n];
			for(int i=0; i < n; i++)
				element[i] = current_index[i]++;
			result.push_back(element);
		}
	}
}

int main(int argc, char **argv){
	printf("%d\n", argc);

	std::string main_path = argv[1];
    std::string lidar_path = main_path + "/cast_out/";
    std::string seg_path = main_path + "/seg_out/";
    std::string camera_path = main_path + "/cam_out/";

    std::vector<std::string> lidar_files;
    for(const auto &entry : std::experimental::filesystem::directory_iterator(lidar_path + "_out/"))
        lidar_files.push_back(entry.path());

    std::vector<std::string> camera_files;
    for(const auto &entry : std::experimental::filesystem::directory_iterator(seg_path + "_out/"))
        camera_files.push_back(entry.path());

    std::sort(lidar_files.begin(), lidar_files.end());
    std::sort(camera_files.begin(), camera_files.end());

	std::ifstream ifs_lidar(lidar_path + "lidar_data.json");
	Json::Reader reader;
	Json::Value lidar_obj;
	reader.parse(ifs_lidar, lidar_obj);
	const Json::Value& cast = lidar_obj["elements"];
    std::vector<float> lidar_times;
	for(int i=0; i<cast.size(); i++){
    	lidar_times.push_back(cast[i]["timestamp"].asFloat());
    }
    ifs_lidar.close();

    std::ifstream ifs_camera(seg_path + "seg_data.json");
    Json::Value camera_obj;
    reader.parse(ifs_camera, camera_obj);
    const Json::Value& cam = camera_obj["elements"];
    std::vector<float> camera_times;
    for(int i=0; i<cam.size(); i++){
        camera_times.push_back(cam[i]["timestamp"].asFloat());
    }
    ifs_camera.close();

    std::ifstream ifs_camera_original(camera_path + "cam_data.json");
    Json::Value camera_obj_orginal;
    reader.parse(ifs_camera_original, camera_obj_orginal);
    const Json::Value& cam_original = camera_obj_orginal["elements"];
    std::vector<float> camera_times_original;
    for(int i=0; i<cam_original.size(); i++){
        camera_times_original.push_back(cam_original[i]["timestamp"].asFloat());
    }
    ifs_camera_original.close();

    std::ifstream ifs_imu(main_path + "/imu_data.json");
    Json::Value imu_obj;
    reader.parse(ifs_imu, imu_obj);
    const Json::Value& imu = imu_obj["elements"];
    std::vector<float> imu_times;
    std::vector<Eigen::Matrix4f> frame2world_transforms;
    for(int i=0; i<imu.size(); i++){
        imu_times.push_back(imu[i]["timestamp"].asFloat());
        const Json::Value& this_frame_transform = imu[i]["transform"];
        Eigen::Matrix4f frame_transform;
        for(int j=0; j<4; j++)
            for(int k=0; k<4; k++)
                frame_transform(j, k) = this_frame_transform[j][k].asFloat();
        frame2world_transforms.push_back(frame_transform);
    }
    ifs_imu.close();

    std::vector<float> times[] = {camera_times, lidar_times, imu_times, camera_times_original};
    
    int test_size = std::min(times[0].size(), std::min(times[1].size(), std::min(times[2].size(), times[3].size())));
    printf("test_size : %d\n", test_size);
    for(int i=0; i < test_size; i++)
    	printf("%f %f %f %f\n", times[0][i], times[1][i], times[2][i], times[3][i]);
    
    std::vector<int*> matches;
    get_set_intersection(times, 4, matches);

    int match = matches.size();

    printf("match %d out_of %ld\n", match, camera_times_original.size());

    // assert(match == matches.size());
    Eigen::MatrixXf projection_matrix(3, 4);

    Eigen::Matrix4f lidar2camera = Eigen::Matrix4f::Identity();
    lidar2camera(1, 3) = 0.3;
    lidar2camera(2, 3) = 0.8;

    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = float(width) / 2.0; // * tan(fov/2) for general camera
    K(1, 1) = float(width) / 2.0;
    K(0, 2) = float(width) / 2.0;
    K(1, 2) = float(height) / 2.0;

    // [X, Y, Z, 1] -> [Y, -Z, X]
    Eigen::MatrixXf projection = Eigen::MatrixXf::Zero(3, 4);
    projection(0, 1) = -1. * flip;
    projection(1, 2) = -1.;
    projection(2, 0) = 1.;

    projection_matrix = K * projection;
    Eigen::Matrix3f K3 = projection_matrix.topLeftCorner(3, 3);
    Eigen::Matrix3f K3_inv = K3.inverse();
    int test_start = 0;
    match -= test_start;
    frames = new Frame*[match];

    // add outliers removal radius method 
    for(int i=test_start; i<matches.size(); i++){
        int cam_index = matches[i][0];
        int cast_index = matches[i][1];
        int imu_index = matches[i][2];
        
        cv::Mat image = cv::imread(camera_files[cam_index], cv::IMREAD_COLOR);
        cv::Mat img = cv::Mat::zeros(height, width, CV_8U);
    	int fromTo[] = {2, 0};
    	cv::mixChannels(&image, 1, &img, 1, fromTo, 1);
        
        cv::Mat distinct_mask[num_classes];
        seperate_mask(img, distinct_mask);
        // for(int i=0; i < num_classes; i++){
        // 	cv::imshow("window", distinct_mask[i]);
        // 	cv::waitKey(0);
        // }
        cv::Mat transformed_p[num_classes];
        for(int i=0; i < num_classes; i++){
        	remove_projection_shadows(distinct_mask[i], transformed_p[i]);
    	}

        std::string this_filename = find_filename(camera_files[cam_index]);
        if(this_filename == ""){
        	std::cerr << "Error handling filename" << camera_files[cam_index] << std::endl;
        	return -1;
        }
        cnpy::NpyArray desc = cnpy::npy_load(camera_path + "descriptors/" + this_filename + ".npy");
    	float* loaded_desc = desc.data<float>();

    	cnpy::NpyArray keypt = cnpy::npy_load(camera_path + "keypoints/" + this_filename + ".npy");
    	float* loaded_keypt = keypt.data<float>();

    	cnpy::NpyArray score = cnpy::npy_load(camera_path + "scores/" + this_filename + ".npy");
    	float* loaded_score = score.data<float>();
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ> ());
        if(pcl::io::loadPLYFile(lidar_files[cast_index], *source_cloud) < 0){
            std::cerr << "Error loading point cloud " << lidar_files[cast_index] << std::endl << std::endl;
            return -1;
        }
        pcl::transformPointCloud(*source_cloud, *transformed_cloud, lidar2camera);
        frames[i-test_start] = new Frame(i-test_start, cam_index, loaded_desc, loaded_score, loaded_keypt,\
        								 distinct_mask, transformed_p, transformed_cloud, frame2world_transforms[imu_index]);
    }
    printf("io complete\n");

    // Form keyframes that has appreciable translation between frames
    // non-keyframe is associated with the closest keyframe in the chain 
    // Store sorted array of distance to each keyframe for each frame struct
    // For all the frame in the sorted array take frames while distance between the point and frame is less than threshold
    // For all the keyframes(and the middleframes) where x > 0 find the observation score
    // Only define a keyframe as observable if the distance below closest


    // Projection shadow issue
    // project and score normally?
    // calculate distance gradient from all the projection in the image

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_clouds[num_classes + 1];
    for(int i=0; i <= num_classes; i++){
    	pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ> ());
    	point_clouds[i] = temp;
    }

    Eigen::Matrix4f world_2_start = frames[0]->world_2_frame;
    globalPoint** point_collection[match];
    int _size[match];

    int frame_count = 1;
    Frame* prev_keyframe = 0;
    std::vector<int> keyframe_id;
    for(int i=0; i < match; i++){
    	if(frames[i]->try_set_keyframe(prev_keyframe)){
    		prev_keyframe = frames[i];
    		keyframe_id.push_back(i);
    	}
    }

    printf("number keyframes : %ld\n", keyframe_id.size());

    for(int i=0; i < keyframe_id.size(); i++){
    	frames[keyframe_id[i]]->fill_neighbor_list(keyframe_id);
    }

    #pragma omp parallel num_threads(8)
    {
    	int kernel_size;
    	int kernel_radius;
    	
    	#pragma omp for nowait
    	for(int i=0; i < match; i++){
    		Frame* frame = frames[i];
    		int n_point = (*frame->point_cloud).size();
    		point_collection[i] = new globalPoint*[n_point];
    		int n_pt = 0;

    		for(pcl::PointCloud<pcl::PointXYZ>::iterator it = (*frame->point_cloud).begin(); it != (*frame->point_cloud).end(); it++){
    			pcl::PointXYZ& pt = *it;
    			Eigen::Vector4f this_point;
            	this_point << pt.x, pt.y, pt.z, 1.0;
            	// dont register very close point as it may be on self vehicle
            	if(this_point.squaredNorm() < 17){ 
            		continue;
            	}
            	this_point = frame->frame_2_world * this_point;
            	Frame* reference_frame = frame->get_keyframe()->find_nondirectional_closest_Frame(this_point);
            	globalPoint* gp = (globalPoint*)malloc(sizeof(globalPoint)); 
            	gp->global_point = this_point;
            	gp->frame = frame;
            	gp->id = (int)(y_len * x_len * ((float)n_pt / (float)n_point));
            	gp->num_observation = 0;
            	for(int k=0; k < num_classes; k++)
            		gp->score[k] = 0;
            	gp->weight_sum = 0;
            	point_collection[i][n_pt] = gp;
            	n_pt++;
            
            	for(int j=0; j < reference_frame->neighbors.size(); j++){
            		Frame* sample_frame = frames[reference_frame->neighbors[j].frame_id];
            		Eigen::Vector4f local_point = sample_frame->world_2_frame * this_point;
            		float point_distance = std::sqrt(local_point.squaredNorm() - 1);

            		if(local_point(0) > 0 and point_distance < max_reliable_distance){
            			Eigen::Vector3f projection_point = projection_matrix * local_point;
                		projection_point /= projection_point(2);
                		int _j = round(projection_point(0));
                		int _i = round(projection_point(1));
                		bool use_refined;
                		if(point_distance < low_shadow_dist){
                			kernel_size = 1;
                			kernel_radius = kernel_size / 2;
                			use_refined = false;
                		}
                		else{
                			kernel_size = 1;
                			kernel_radius = kernel_size / 2;
                			use_refined = true;
                		}

                		if(_i >= kernel_radius and _i < height - kernel_radius and _j >= kernel_radius and _j < width - kernel_radius){
                			float sum[num_classes];
                			
                			int y = _i / voxel_size;
                        	int x = _j / voxel_size;
                			Candidate candidate_point = {gp, (_j % voxel_size) +  voxel_size * (_i % voxel_size)};
                    		
                    		if(!use_refined){
                    			for(int k=0 ; k < num_classes; k++){
                    				sum[k] = 0;
                    				for(int l=-kernel_radius; l <= kernel_radius; l++)
                        				for(int m=-kernel_radius; m <= kernel_radius; m++)
                            				sum[k] += sample_frame->p_mask[k].at<float>(_i+l, _j+m);
                    				candidate_point.projection_score[k] = sum[k] / (kernel_size * kernel_size);
                    			}
                    		}
                    		else{
                    			for(int k=0 ; k < num_classes; k++){
                    				sum[k] = 0;
                    				for(int l=-kernel_radius; l <= kernel_radius; l++)
                        				for(int m=-kernel_radius; m <= kernel_radius; m++)
                            				sum[k] += sample_frame->refined_pmask[k].at<float>(_i+l, _j+m);
                    				candidate_point.projection_score[k] = sum[k] / (kernel_size * kernel_size);
                    			}
                    		}

                    		cv::Point center =  cv::Point(_j, _i);
                			omp_set_lock(&sample_frame->voxel[y][x].writelock);
                			sample_frame->voxel[y][x].num_pt += 1;
    						sample_frame->voxel[y][x].candidate_points.push_back(candidate_point);
    						sample_frame->granular_projection[_i][_j].push_back(local_point);
    						omp_unset_lock(&sample_frame->voxel[y][x].writelock);
                		}
           	    	}
            	}
    		}
    		_size[i] = n_pt;
    		printf("processing completed frame : %d(unsynced value) thread_id %d\n", frame_count++, omp_get_thread_num());
    	}
    }

    // evaluation can also be parallelized, find a way to update global variable global_point
    frame_count = 0;
    #pragma omp parallel for
    for(int i=0; i < keyframe_id.size(); i++){
    	printf("evaluating frame : %d thread_id %d\n", frame_count++, omp_get_thread_num());
    	frames[keyframe_id[i]]->score_projection_points();
    	// std::string path_to_write = "projections/" + std::to_string(keyframe_id[i] + 1) + ".png";
        // cv::imwrite(path_to_write, frames[keyframe_id[i]]->img);
    }

    #pragma omp parallel for
    for(int i=1; i < keyframe_id.size(); i++){
    	printf("Matched Points for frame %d  is  %d\n", keyframe_id[i], frames[keyframe_id[i]]->match_keypoints());
    }
    int first = 0;

    for(int i=1; i < keyframe_id.size(); i++){
    	Frame *this_keyframe = frames[keyframe_id[i]];
    	Frame *prev_keyframe = frames[keyframe_id[i]]->previous_keyframe;
    	
    	for(int j=0; j < num_keypoints; j++){
    		if(this_keyframe->down_relation[j] != -1){
    			int p = this_keyframe->down_relation[j];
    			Eigen::Matrix4f current_2_previous = prev_keyframe->world_2_frame * this_keyframe->frame_2_world;
    			Eigen::Vector3f X(this_keyframe->keypoints[j][0], this_keyframe->keypoints[j][1], 1.0);
    			Eigen::Vector3f Y(prev_keyframe->keypoints[p][0], prev_keyframe->keypoints[p][1], 1.0);
    			Eigen::Matrix3f R1 = current_2_previous.topLeftCorner(3, 3);
    			Eigen::Vector3f T1 = current_2_previous.block<3, 1>(0, 3);
    			
    			if(this_keyframe->up_relation[j] != -1){
    				int q = this_keyframe->up_relation[j];
    				Eigen::Matrix4f current_2_next = this_keyframe->next_keyframe->world_2_frame * this_keyframe->frame_2_world;
    				Eigen::Vector3f Z(this_keyframe->next_keyframe->keypoints[q][0], this_keyframe->next_keyframe->keypoints[q][1], 1.0);
    				Eigen::Matrix3f R2 = current_2_next.topLeftCorner(3, 3);
    				Eigen::Vector3f T2 = current_2_next.block<3, 1>(0, 3);
    				Frame *wild_card=0;
    				
    				int r;
    				if(this_keyframe->next_keyframe->up_relation[q] != -1){
    					r = this_keyframe->next_keyframe->up_relation[q];
    					wild_card = this_keyframe->next_keyframe->next_keyframe;
    				}
    				else if(prev_keyframe->down_relation[p] != -1){
    					r = prev_keyframe->down_relation[p];
    					wild_card = prev_keyframe->previous_keyframe;
    				}
    				
    				Eigen::MatrixXf A;
    				Eigen::VectorXf b;
    				if(wild_card){
    					A = Eigen::MatrixXf::Zero(9, 4);
    					b = Eigen::VectorXf::Zero(9); 
    					Eigen::Matrix4f current_2_wildcard = wild_card->world_2_frame * this_keyframe->frame_2_world;
    					Eigen::Vector3f W(wild_card->keypoints[r][0], wild_card->keypoints[r][1], 1.0);
    					Eigen::Matrix3f R3 = current_2_wildcard.topLeftCorner(3, 3);
    					Eigen::Vector3f T3 = current_2_wildcard.block<3, 1>(0, 3);
    					A.block<3, 1>(6, 0) = - K3 * R3 * K3_inv * X;
    					A.block<3, 1>(6, 3) = W;
    					b.block<3, 1>(6, 0) = K3 * T3;
    				}
    				else{
    					A = Eigen::MatrixXf::Zero(6, 3);
    					b = Eigen::VectorXf::Zero(6);
    				}
    				
    				A.block<3, 1>(0, 0) = - K3 * R1 * K3_inv * X;
    				A.block<3, 1>(0, 1) = Y;
    				A.block<3, 1>(3, 0) = - K3 * R2 * K3_inv * X;
    				A.block<3, 1>(3, 2) = Z;
    				b.block<3, 1>(0, 0) = K3 * T1;
    				b.block<3, 1>(3, 0) = K3 * T2; 
    				Eigen::VectorXf estimate = Eigen::VectorXf::Zero(4);
    				int res = solve(A, b, estimate);
    				
    				switch(res){
    					case 2 : {
    						this_keyframe->estimated_2distance[j] = estimate(0);
    						prev_keyframe->estimated_2distance[p] = estimate(1);
    						break;
    					}
    					case 3 : {
    						this_keyframe->estimated_3distance[j][0] = estimate(0);
    						prev_keyframe->estimated_3distance[p][1] = estimate(1);
    						this_keyframe->next_keyframe->estimated_3distance[q][2] = estimate(2);
    						break;
    					}
    					case 4: {
    						this_keyframe->estimated_3distance[j][0] = estimate(0);
    						prev_keyframe->estimated_3distance[p][1] = estimate(1);
    						this_keyframe->next_keyframe->estimated_3distance[q][2] = estimate(2);
    						if(wild_card->frame_id > this_keyframe->frame_id)
    							wild_card->estimated_3distance[r][3] = estimate(3);
    						else
    							wild_card->estimated_3distance[r][4] = estimate(3);
    						break;
    					}
    					default : 
    						break;
    				}
    			}
    			else{
    				Eigen::MatrixXf A = Eigen::MatrixXf::Zero(3, 2);
    				Eigen::VectorXf b(3);
    				A.block<3, 1>(0, 0) = - K3 * R1 * K3_inv * X;
    				A.block<3, 1>(0, 1) = Y;
    				b.block<3, 1>(0, 0) = K3 * T1;
    				Eigen::VectorXf estimate(2);
    				if(solve(A, b, estimate) > 1){
    					this_keyframe->estimated_2distance[j] = estimate(0);
    					prev_keyframe->estimated_2distance[p] = estimate(1);
    				}
    			}
    		}
		}
    }

    // re-estimation heuristic
    int max_keypt_distance=85;
    for(int i=0; i < keyframe_id.size(); i++){
    	Frame *this_keyframe = frames[keyframe_id[i]];
    	for(int j=0; j < num_keypoints; j++){
    		if(this_keyframe->estimated_3distance[j][0] != -1 || this_keyframe->estimated_3distance[j][1] != -1 || this_keyframe->estimated_3distance[j][2] != -1){
    			Eigen::Vector4f prior_vector;
    			Eigen::Vector3f X(this_keyframe->keypoints[j][0], this_keyframe->keypoints[j][1], 1.0);
    			float x = this_keyframe->estimated_3distance[j][0];
    			float y = this_keyframe->estimated_3distance[j][1];
    			float z = this_keyframe->estimated_3distance[j][2];
    			float w = this_keyframe->estimated_3distance[j][3];
    			float v = this_keyframe->estimated_3distance[j][4];
    			bool x_t = (x != -1);
    			bool y_t = (y != -1);
    			bool z_t = (z != -1);
    			bool w_t = (w != -1);
    			bool v_t = (v != -1);
    			if(!x_t){
    				float d_estimate = (y * y_t + z * z_t) / (float)(y_t + z_t);
    				if(w_t && std::abs(w - d_estimate) >= 5)
    					w_t = false;
    				if(v_t && std::abs(v - d_estimate) >= 5)
    					v_t = false;
    				d_estimate = (v * v_t + w * w_t + y * y_t + z * z_t) / (float)(w_t + v_t + y_t + z_t);
    				prior_vector.block<3, 1>(0, 0) = d_estimate * K3_inv * X;
    				prior_vector(3) = 1.0f;
    				float prior_dist = std::sqrt(prior_vector.squaredNorm() - 1);
    				if(prior_dist <= max_keypt_distance){
    					float rd_estimate = this_keyframe->estimate_distance_with_lidar(round(X(0)), round(X(1)), prior_vector, 5, 2, 5.0 / 6.0);
    					if(rd_estimate != -1){
    						if(std::abs(rd_estimate - d_estimate) <= 10){
    							float _rd_estimate = rd_estimate / (K3_inv * X).norm();
    							this_keyframe->final_keypoints.push_back(std::make_pair(j, _rd_estimate));
    							// printf("Earlier 3estimate estimate=%f, Renewed estimate=%f \n", prior_dist, rd_estimate);
    						}
    					}
    					else{
    						// this_keyframe->final_keypoints.push_back(std::make_pair(j, d_estimate));
    					}
    				}
    			}
    			else{
    				if(w_t && std::abs(w - x) >= 5)
    					w_t = false;
    				if(v_t && std::abs(v - x) >= 5)
    					v_t = false;
    				float d_estimate = (2 * x + v * v_t + w * w_t + y * y_t + z * z_t) / (float)(2 + w_t + v_t + y_t + z_t);
    				prior_vector.block<3, 1>(0, 0) = d_estimate * K3_inv * X;
    				prior_vector(3) = 1.0f;
    				float prior_dist = std::sqrt(prior_vector.squaredNorm() - 1);
    				if(prior_dist <= max_keypt_distance){
    					float rd_estimate = this_keyframe->estimate_distance_with_lidar(round(X(0)), round(X(1)), prior_vector, 5, 2, 5.0 / 6.0);
    					if(rd_estimate != -1){
    						if(std::abs(rd_estimate - d_estimate) <= 10){
    							float _rd_estimate = rd_estimate / (K3_inv * X).norm();
    							this_keyframe->final_keypoints.push_back(std::make_pair(j, _rd_estimate));
    							// printf("Earlier 3*estimate %f , Renewed estimate %f \n", d_estimate, _rd_estimate);
    						}
    					}
    					else{
    						// this_keyframe->final_keypoints.push_back(std::make_pair(j, d_estimate));
    					}
    				}
    			}
    		}
    		else if(this_keyframe->estimated_2distance[j] != -1){
    			Eigen::Vector4f prior_vector;
    			Eigen::Vector3f X(this_keyframe->keypoints[j][0], this_keyframe->keypoints[j][1], 1.0);
    			float d_estimate = this_keyframe->estimated_2distance[j];
    			prior_vector.block<3, 1>(0, 0) = d_estimate * K3_inv * X;
    			prior_vector(3) = 1.0f;
    			float prior_dist = std::sqrt(prior_vector.squaredNorm() - 1);
    			if(d_estimate <= max_keypt_distance){
    				float rd_estimate = this_keyframe->estimate_distance_with_lidar(round(X(0)), round(X(1)), prior_vector, 5, 2, 5.0 / 6.0);
    				if(rd_estimate != -1){
    					if(std::abs(rd_estimate - d_estimate) <= 20){
    						float _rd_estimate = rd_estimate / (K3_inv * X).norm();
    						this_keyframe->final_keypoints.push_back(std::make_pair(j, _rd_estimate));
    						// printf("Earlier 2estimate %f, Renewed estimate %f\n", prior_dist, rd_estimate);
    					}
    				}
    			}
    		}
    	}
    	// printf("next_keyframe\n");
    }

    for(int i=0; i < match; i++){
    	for(int j=0; j < _size[i]; j++){
    		int num_observation = point_collection[i][j]->num_observation;
    		Eigen::Vector4f point = point_collection[i][j]->global_point;
    		pcl::PointXYZ temp(point(0), flip * point(1), point(2));
    		int index = num_classes;

    		if(num_observation > min_valid_obs){
    			float maximum = min_score_for_validity;
    			float distance = point_collection[i][j]->weight_sum;
    			
    			for(int k=0; k < num_classes; k++){
    				float lane_validity = point_collection[i][j]->score[k] / distance;
    				if(lane_validity > maximum){
    					maximum = lane_validity;
    					index = k;
    				}
    			}

    		}
    		(*point_clouds[index]).push_back(temp);
    	}
    }

    printf("%s\n", "saving outputs");

    std::string out_path = get_current_dir() + "/map_out";
    if(std::experimental::filesystem::is_directory(out_path)){
    	std::experimental::filesystem::remove_all(out_path);
    }

    if(mkdir(out_path.c_str(), 0777) == -1){
        std::cerr << "Error :  " << strerror(errno) << std::endl;
        exit(0);
    }

    for(int i=0; i < keyframe_id.size(); i++){
    	Frame *this_keyframe = frames[keyframe_id[i]];
    	Eigen::Matrix4f frame_2_start = this_keyframe->frame_2_world;
    	frame_2_start.row(1) = flip * frame_2_start.row(1);
    	frame_2_start.col(1) = flip * frame_2_start.col(1);
    	frame_2_start.transposeInPlace();
    	float* data = frame_2_start.data();
    	float keypoint[4 * this_keyframe->final_keypoints.size() + 1];
    	float descriptor[descriptor_size * this_keyframe->final_keypoints.size()];
    	keypoint[0] = (float)this_keyframe->final_keypoints.size();

    	for(int j=0; j < this_keyframe->final_keypoints.size(); j++){
    		int index = this_keyframe->final_keypoints[j].first;
    		float distance = this_keyframe->final_keypoints[j].second;
    		keypoint[4*j + 1] = this_keyframe->keypoints[index][0];
    		keypoint[4*j + 2] = this_keyframe->keypoints[index][1];
    		keypoint[4*j + 3] = distance;
    		keypoint[4*j + 4] = this_keyframe->keypoints[index][2];
    		int offset = descriptor_size * j;
    		for(int k=0; k < descriptor_size; k++){
    			descriptor[offset + k] = this_keyframe->descriptor_ptr[k * num_keypoints + index];
    		}
    	}
    	std::string path_to_dir = out_path + "/keyframe_" + std::to_string(i);
    	if(mkdir(path_to_dir.c_str(), 0777) == -1){
        	std::cerr << "Error :  " << strerror(errno) << std::endl;
        	exit(0);
    	}
    	cnpy::npy_save(path_to_dir + "/transform.npy", data, {16}, "w");
    	cnpy::npy_save(path_to_dir + "/keypoints.npy", keypoint, {4 * this_keyframe->final_keypoints.size() + 1}, "w");
    	cnpy::npy_save(path_to_dir + "/descriptors.npy", descriptor, {descriptor_size * this_keyframe->final_keypoints.size()}, "w");
    	std::string file_path = camera_path + "original_images/" + find_filename(camera_files[this_keyframe->cam_index]) + ".png";
    	cv::Mat temp = cv::imread(file_path, cv::IMREAD_COLOR);
    	cv::imwrite(path_to_dir + "/image.png", temp);
    }

    for(int i=0; i <= num_classes; i++){
    	pcl::io::savePLYFileASCII(out_path + "/point_class_" + std::to_string(i) + ".ply", *point_clouds[i]);
	}

	#ifndef VISUALIZE

    pcl::visualization::PCLVisualizer viewer("simple_ptviewer");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>::Ptr temp_ptr[num_classes + 1];

    int colors[num_classes+1][3];
    colors[0][0] = 255;
    colors[0][1] = 127;
    colors[0][2] = 0;
    colors[1][0] = 0;
    colors[1][1] = 255;
    colors[1][2] = 0;
    colors[2][0] = 255;
    colors[2][1] = 0;
    colors[2][2] = 0;
    colors[3][0] = 139;
    colors[3][1] = 0;
    colors[3][2] = 255;
    colors[4][0] = 255;
    colors[4][1] = 255;
    colors[4][2] = 255;
    colors[5][0] = 0;
    colors[5][1] = 0;
    colors[5][2] = 0;

    for(int i=0; i <= num_classes; i++){
    	// int r = rand() % 256;
    	// int g = rand() % 256;
    	// int b = rand() % 256;
    	// if(r < 128 && g < 128 && b < 128)
    	// 	g += 64;
    	if(point_clouds[i]->size() > 0){
    		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>::Ptr temp(new pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(point_clouds[i], colors[i][0], colors[i][1], colors[i][2]));
    		temp_ptr[i] = temp;
    		viewer.addPointCloud(point_clouds[i], *temp, "point_cloud_" + std::to_string(i));
    	}
    }

    viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    for(int i=0; i <= num_classes; i++){
    	if(point_clouds[i]->size() > 0){
    		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point_cloud_" + std::to_string(i));
    	}
    }

    while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
      viewer.spinOnce ();
    }

    #endif

	return 0;
}

