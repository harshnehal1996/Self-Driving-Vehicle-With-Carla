// to be loaded with the main cpp

class trajectoryFrame{
public:
	int frame_id;
	ukf_tracker::scaler_t timestamp;
	Eigen::Matrix<ukf_tracker::scaler_t, 7, 1> imu_data;
	Eigen::Matrix<ukf_tracker::scaler_t, 3, 1> gnss_data;
    Eigen::Quaternionf ground_truth;
    Eigen::Vector3f location; 
	Eigen::Matrix4f _ground_truth;
	ukf_tracker::cameraObservation obs;

	trajectoryFrame(int id, float time, float* keypts, float* descriptor, float* scores, cv::Mat& image, /*Eigen::Matrix<ukf_tracker::scaler_t, 3, 1>& gnss*/\
					Eigen::Matrix<ukf_tracker::scaler_t, 7, 1>& imu, Eigen::Matrix4f& true_position){
		
		frame_id = id;
        timestamp = (ukf_tracker::scaler_t)time;
        
        #if num_features > 500
        obs.descriptors.resize(num_features, descriptor_size);
		#endif

        float* writer = obs.descriptors.data();

		for(int i=0; i < num_features; i++){
			for(int j=0; j < descriptor_size; j++){
				writer[j * num_features + i] = descriptor[i * descriptor_size + j];
			}
		}

		for(int i=0; i < num_features; i++){
			for(int j=0; j < 3; j++){
				obs.keypoints[i][j] = (ukf_tracker::scaler_t)keypts[i * 3 + j];
			}
		}

		for(int i=0; i < num_features; i++){
			obs.scores[i] = scores[i];
		}

		obs.image = image;
		imu_data = imu;
        _ground_truth = true_position;
		ground_truth = _ground_truth.topLeftCorner<3,3>();
        location = true_position.block<3,1>(0, 3);
        #ifndef NO_IMAGE_OUT
        obs.g_frame_to_world = true_position.cast<ukf_tracker::scaler_t>();
        obs.true_angles = ground_truth.toRotationMatrix().eulerAngles(0,1,2).cast<ukf_tracker::scaler_t>();
        obs.true_location = location.cast<ukf_tracker::scaler_t>();
	    #endif
    }
};

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

// expected file structure
// {main_path}/cam_out/original_images/*.png
// {main_path}/cam_out/cam_data.json
// {main_path}/cam_out/keypoints/*.npy
// {main_path}/cam_out/descriptors/*.npy
// {main_path}/cam_out/scores/*.npy
// {main_path}/imu_data.json
// {main_path}/gnss_data.json
// y_flip to convert left-hand coordinate system to right
int load_current_trajectory_data(std::string main_path, trajectoryFrame*** trajectory, float y_flip=-1.0f){
    std::string camera_path = main_path + "/cam_out/";

    std::vector<std::string> camera_files;
    for(const auto &entry : std::experimental::filesystem::directory_iterator(camera_path + "original_images/"))
        camera_files.push_back(entry.path());

    std::sort(camera_files.begin(), camera_files.end());

    std::ifstream ifs_camera(camera_path + "cam_data.json");
    Json::Value camera_obj;
    Json::Reader reader;
    reader.parse(ifs_camera, camera_obj);
    const Json::Value& cam = camera_obj["elements"];
    std::vector<float> camera_times;
    for(int i=0; i<cam.size(); i++){
        camera_times.push_back(cam[i]["timestamp"].asFloat());
    }
    ifs_camera.close();

    std::ifstream ifs_imu(main_path + "/imu_data.json");
    Json::Value imu_obj;
    reader.parse(ifs_imu, imu_obj);
    const Json::Value& imu = imu_obj["elements"];
    std::vector<float> imu_times;
    std::vector<Eigen::Matrix4f> frame2world_transforms(imu.size());
    std::vector<Eigen::Matrix<ukf_tracker::scaler_t, 7, 1>> imu_recorded(imu.size());
    for(int i=0; i<imu.size(); i++){
        imu_times.push_back(imu[i]["timestamp"].asFloat());
        imu_recorded[i] << (ukf_tracker::scaler_t)imu[i]["accelerometer"][0].asFloat(),\
        (ukf_tracker::scaler_t)(y_flip * imu[i]["accelerometer"][1].asFloat()), (ukf_tracker::scaler_t)imu[i]["accelerometer"][2].asFloat(),\
        (ukf_tracker::scaler_t)imu[i]["gyroscope"][0].asFloat(), (ukf_tracker::scaler_t)(y_flip * imu[i]["gyroscope"][1].asFloat()), (ukf_tracker::scaler_t)imu[i]["gyroscope"][2].asFloat(), 0;
        const Json::Value& this_frame_transform = imu[i]["transform"];
        Eigen::Matrix4f frame_transform;
        for(int j=0; j<4; j++)
            for(int k=0; k<4; k++)
                frame_transform(j, k) = this_frame_transform[j][k].asFloat();
        frame_transform.row(1) = y_flip * frame_transform.row(1);
        frame_transform.col(1) = y_flip * frame_transform.col(1);
        frame2world_transforms[i] = frame_transform;
    }
    ifs_imu.close();

    std::ifstream ifs_vel(main_path + "/velocity_data.json");
    Json::Value vel_obj;
    reader.parse(ifs_vel, vel_obj);
    const Json::Value& vel = vel_obj["elements"];
    std::vector<float> vel_times;
    std::vector<ukf_tracker::scaler_t> vel_data(vel.size());
    for(int i=0; i<vel.size(); i++){
        vel_times.push_back(vel[i]["timestamp"].asFloat());
        vel_data[i] = (ukf_tracker::scaler_t)vel[i]["velocity"].asFloat();
    }
    ifs_vel.close();

    // currently gnss data is artificially added
    // std::ifstream ifs_gnss(main_path + "/gnss_data.json");
    // Json::Value gnss_obj;
    // reader.parse(ifs_gnss, gnss_obj);
    // const Json::Value& gnss = gnss_obj["elements"];
    // std::vector<float> gnss_times;
    // std::vector<Eigen::Matrix<ukf_tracker::scaler_t, 3, 1>> gnss_data(gnss.size());
    // for(int i=0; i<gnss.size(); i++){
    //     gnss_times.push_back(gnss[i]["timestamp"].asFloat());
    //     gnss_data[i] << (ukf_tracker::scaler_t)gnss[i]["latitude"].asFloat(), (ukf_tracker::scaler_t)gnss[i]["longitude"].asFloat(), (ukf_tracker::scaler_t)gnss[i]["altitude"].asFloat();
    // }
    // ifs_gnss.close();

    int i=0,j=0;
    std::vector<int*> matches;
    while(i < imu_times.size() and j < camera_times.size()){
        if(camera_times[j] == imu_times[i]){
            int* this_match = new int[3];
            this_match[0] = j;
            this_match[1] = i;
            this_match[2] = -1;
            matches.push_back(this_match);
            j++;
            i++;
        }
        else if(camera_times[j] > imu_times[i])
            i++;
        else
            j++;
    }

    i=0,j=0;
    int match = 0;
    while(i < matches.size() and j < vel_times.size()){
        if(camera_times[matches[i][0]] == vel_times[j]){
            matches[i][2] = j;
            match++;
            i++;
            j++;
        }
        else if(camera_times[matches[i][0]] > vel_times[j])
            j++;
        else{
            printf("%f sec is unmatched at velocity_data\n", camera_times[matches[i][0]]);
            i++;
        }
    }

    printf("match %d size %ld total_size %ld\n", match, matches.size(), camera_times.size());
    *trajectory = new trajectoryFrame*[match];
    int offset = 0;
    
    for(i=0; i<matches.size(); i++){
        int cam_index = matches[i][0];
        int imu_index = matches[i][1];
        int vel_index = matches[i][2];
        if(vel_index == -1){
            offset++;
            continue;
        }

        cv::Mat img = cv::imread(camera_files[cam_index], cv::IMREAD_COLOR);
        // cv::Mat img;
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
        imu_recorded[imu_index](6) = vel_data[vel_index];
        
        (*trajectory)[i-offset] = new trajectoryFrame(i-offset, camera_times[cam_index], loaded_keypt, loaded_desc, loaded_score,\
        									          img, imu_recorded[imu_index], frame2world_transforms[imu_index]);
    }
    
    return match;
}

void load_recorded_lidar_keyframes(std::string main_path){
    std::vector<std::string> dir_paths;
    for(const auto &entry : std::experimental::filesystem::directory_iterator(main_path)){
        if(std::experimental::filesystem::is_directory(entry.path())){
            dir_paths.push_back(entry.path());
        }
    }

    std::sort(dir_paths.begin(), dir_paths.end());
    std::fstream file;
    file.open(main_path + "/num_keypoints.txt", std::ios::in);

    ukf_tracker::Frame* previous_frame = 0;
    for(int i=0; i < dir_paths.size(); i++){
        cnpy::NpyArray desc = cnpy::npy_load(dir_paths[i] + "/descriptors.npy");
        float* loaded_desc = desc.data<float>();

        cnpy::NpyArray keypt = cnpy::npy_load(dir_paths[i] + "/keypoints.npy");
        float* loaded_keypt = keypt.data<float>();
        int num_keypoints = (int)loaded_keypt[0];
        loaded_keypt++;

        cnpy::NpyArray transform = cnpy::npy_load(dir_paths[i] + "/transform.npy");
        float* loaded_transform = transform.data<float>();

        cv::Mat img = cv::imread(dir_paths[i] + "/image.png", cv::IMREAD_COLOR);

        ukf_tracker::Frame* lidar_frame = new ukf_tracker::Frame(i, loaded_keypt, loaded_desc, num_keypoints, loaded_transform, previous_frame, img);
        
        if(i)
            previous_frame->next_keyframe = lidar_frame;
        previous_frame = lidar_frame;
    }
}

