#include "RGBDNode.h"
#include <fstream>

#define PI 3.14159265

// Couldnt add this to RGBNode.h since there was a conflict and it required
// static member variables
void pointPickingEventOccurred(
    const pcl::visualization::PointPickingEvent &event, void *viewer_void);

float x_clicking, y_clicking, z_clicking, clicked_point_id;
bool new_clicked_point = false;

int main(int argc, char **argv) {

  ros::init(argc, argv, "RGBD");
  ros::start();

  if (argc > 1) {
    ROS_WARN("Arguments supplied via command line are neglected.");
  }

  ros::NodeHandle node_handle;

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  image_transport::ImageTransport image_transport(node_handle);

  RGBDNode node(ORB_SLAM2::System::RGBD, node_handle, image_transport);

  pcl::visualization::PCLVisualizer::Ptr newViewer(
      new pcl::visualization::PCLVisualizer("3D Viewer"));

  node.viewer = newViewer;
  // node.viewer->setBackgroundColor(0.0, 0.0, 0.0);
  node.viewer->setBackgroundColor(0.5, 0.5, 0.5);
  // node.viewer->setBackgroundColor(1.0, 1.0, 1.0);

  ros::spin();

  ros::shutdown();

  return 0;
}

RGBDNode::RGBDNode(const ORB_SLAM2::System::eSensor sensor,
                   ros::NodeHandle &node_handle,
                   image_transport::ImageTransport &image_transport)
    : Node(sensor, node_handle, image_transport) {
  myfile.open("/home/pc-campero2/DatosExperimentos/datos.txt",
              std::ios_base::app);

  start_general_time = std::chrono::system_clock::now(); // measuring fps
  frame_count = 0;

  initializeParameters();

  rgb_subscriber_ = new message_filters::Subscriber<sensor_msgs::Image>(
      node_handle, "/camera/rgb/image_raw", 1);
  depth_subscriber_ = new message_filters::Subscriber<sensor_msgs::Image>(
      node_handle, "/camera/depth_registered/image_raw", 1);

  sync_ = new message_filters::Synchronizer<sync_pol>(
      sync_pol(10), *rgb_subscriber_, *depth_subscriber_);
  sync_->registerCallback(boost::bind(&RGBDNode::ImageCallback, this, _1, _2));

  mObjectTrackingState = NO_OBJECT;
}

void RGBDNode::initializeParameters() {

  cv::FileStorage fSettings(getDOTsettingsFileNameParam(),
                            cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(cameraIntrinsics);

  CameraHeight = fSettings["Camera.height"];
  CameraWidth = fSettings["Camera.width"];

  // Values For Global Variables
  activate_SLAM = true;

  // visualization s and parameters

  use_new_method = true;
  visualize_rgba_pointcloud = true;

  // constants of VCCS
  visualize_selected_object_stuff = false;
  show_normals = false;
  visualize_main_inertia_axes = false;
  show_adjacency = false;
  show_supervoxels = false;
  show_help = true;
  disable_transform = false;
  voxel_resolution =
      fSettings["VoxelResolution"]; // 0.01 good for tracking large movements
                                    // //for a good computation/performance
                                    // optimization: 0.01f // 0.007
  seed_resolution =
      fSettings["SeedResolution"]; // 0.04 good for tracking large movements
                                   // //for a good computation/performance
                                   // optimization: 0.038f // 0.02

  cout << endl << "VCCS parameters:" << endl;

  cout << "- voxel resolution set to: " << voxel_resolution << endl;
  cout << "- seed resolution set to: " << seed_resolution << endl;

  color_importance = fSettings["ColorImportance"];
  spatial_importance = fSettings["SpatialImportance"];
  normal_importance = fSettings["NormalImportance"];

  cout << "- Color importance set to: " << color_importance << endl;
  cout << "- Spatial importance set to: " << spatial_importance << endl;
  cout << "- Normal importance set to: " << normal_importance << endl;

  use_single_cam_transform = false;
  normals_scale = 1.0; // seed_resolution / 2.0;

  resolution = voxel_resolution / 2;

  frame_count = 0;

  unsigned int k_factor = 0;
  use_supervoxel_refinement = false;

  // LCCPSegmentation Stuff
  cout << endl << "LCCP parameters:" << endl;

  concavity_tolerance_threshold =
      static_cast<float>(fSettings["ConcavityToleranceThreshold"]);
  smoothness_threshold = static_cast<float>(fSettings["SmoothnessThreshold"]);
  min_segment_size = static_cast<float>(fSettings["MinSegmentSize"]);
  use_extended_convexity = true;
  use_sanity_criterion = true;
  show_visualization = true;

  cout << "- Concavity tolerance threshold: " << concavity_tolerance_threshold
       << endl;
  cout << "- Smoothness threshold: " << smoothness_threshold << endl;
  cout << "- Min number of supervoxels in LCCP segment: " << min_segment_size
       << endl;

  // Temporal coherence stuff
  // Object Segmentation color thresholds
  toleranceFactor = 0.0f; // the % of the bounding box boundaries ...

  // OctreeObjectTrackingMethod
  cout << endl << "Deformable Object Tracking parameters:" << endl;
  neighbouringFactor = fSettings["NeighbouringFactor"];
  explorationFactor = fSettings["ExplorationFactor"];

  neighbouringRadius = seed_resolution * neighbouringFactor; // related to Rseed
  explorationRadius = seed_resolution * explorationFactor;   // related to Rseed

  segment_manager.max_segment_manager_history_size_ =
      fSettings["SegmentHistorySize"];
  segment_manager.colorThreshold = fSettings["ColorThreshold"];
  segment_manager.normalThreshold = fSettings["NormalThreshold"];

  cout << "- Neighbouring factor: " << neighbouringFactor << endl;
  cout << "- Exploration factor: " << explorationFactor << endl;
  cout << "- Color threshold: " << segment_manager.colorThreshold << endl;
  cout << "- Normal threshold: " << segment_manager.normalThreshold << endl;
  cout << endl;

  int VisualizeColorDenseCloudm = fSettings["VisualizeColorDenseCloud"];
  VisualizeColorDenseCloud_ = VisualizeColorDenseCloudm;
  VisualizeLCCPSegments_ = fSettings["VisualizeLCCPSegments"];
  VisualizeSupervoxels_ = fSettings["VisualizeSupervoxels"];
  VisualizeNormals_ = fSettings["VisualizeNormals"];
  VisualizeGraphs_ = fSettings["VisualizeGraphs"];
  VisualizeLCCPConnections_ = fSettings["VisualizeLCCPConnections"];

  myfile << "%% frame_count | Total nºSV | nºSV in object | Residual_0 | "
            "Residual_f | Delta_residual | Solver_time | Solver_threads | SLAM "
            "time | SV segmentation time | DOT time | Optimization time | fps "
            "| general time"
         << endl;
}

RGBDNode::~RGBDNode() {
  delete rgb_subscriber_;
  delete depth_subscriber_;
  delete sync_;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//  __  __          _____ _   _   _      ____   ____  _____
// |  \/  |   /\   |_   _| \ | | | |    / __ \ / __ \|  __ \ 
// | \  / |  /  \    | | |  \| | | |   | |  | | |  | | |__) |
// | |\/| | / /\ \   | | | . ` | | |   | |  | | |  | |  ___/
// | |  | |/ ____ \ _| |_| |\  | | |___| |__| | |__| | |
// |_|  |_/_/    \_\_____|_| \_| |______\____/ \____/|_|

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void RGBDNode::ImageCallback(const sensor_msgs::ImageConstPtr &msgRGB,
                             const sensor_msgs::ImageConstPtr &msgD) {

  std::ostringstream number_of_frames;
  number_of_frames << "Frame: " << frame_count;
  viewer->addText(number_of_frames.str(), 10, 300, 20, 0, 0, 0,
                  "number_of_frames text");
  myfile << frame_count << ", ";

  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptrRGB;
  try {
    cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptrD;
  try {
    cv_ptrD = cv_bridge::toCvShare(msgD);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  current_frame_time_ = msgRGB->header.stamp;

  PointCloudTSuperVoxel::Ptr temp_cloud_rgba;

  long int totalMilliseconds_slam_time, totalMilliseconds_sv_segmentation_time,
      totalMilliseconds_DOT_time, totalMilliseconds_Optimization_time,
      totalMilliseconds_general_time;
  float fps_measurement;

  if (activate_SLAM) {
    auto start_slam_time = std::chrono::system_clock::now(); // measuring fps

    orb_slam_->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image,
                         cv_ptrRGB->header.stamp.toSec());

    auto end_slam_time = std::chrono::system_clock::now();
    auto elapsed_slam_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_slam_time -
                                                              start_slam_time);
    totalMilliseconds_slam_time = elapsed_slam_time.count();
    std::ostringstream slam_time;
    slam_time << "SLAM time: " << totalMilliseconds_slam_time << " ms";
    // viewer->addText(slam_time.str(), 10, 100, 20, 0, 0, 0, "SLAM time text");

    // UPDATING ROS NODE

    Update();

    const map<ORB_SLAM2::MapPoint *, cv::Vec3f> &vpMPs =
        orb_slam_->GetAllDepthMapPoints();

    temp_cloud_rgba = getPointcloudFromORBSlamMapDataStruct(vpMPs);

    // visualize camera
    cv::Mat camera_position = orb_slam_->GetCurrentPosition();
    if (!camera_position.empty()) {
      cv::Mat invPosition = camera_position.inv();
      Eigen::Affine3f t;
      for (int iAffine = 0; iAffine < 3; iAffine++) {
        for (int jAffine = 0; jAffine < 4; jAffine++) {
          t(iAffine, jAffine) = invPosition.at<float>(iAffine, jAffine);
        }
      }

      viewer->addCoordinateSystem(0.05, t, "ref"); // camera visualization
    }
  } else {
    temp_cloud_rgba = PointCloudCreation(cv_ptrRGB->image, cv_ptrD->image);
  }

  // CREATING THE DENSE POINTCLOUD PCL STRUCTURE

  // Filtering
  // PointCloudTSuperVoxel::Ptr temp_cloud_rgba_fully_sampled =
  // getPointcloudFromORBSlamMapDataStruct(vpMPs); PointCloudTSuperVoxel::Ptr
  // temp_cloud_rgba(new PointCloudTSuperVoxel);
  // gridSampleApprox(temp_cloud_rgba_fully_sampled, *temp_cloud_rgba,
  // downsampling_grid_size_);

  if (mObjectTrackingState == NO_OBJECT) {
    pcl::visualization::PointCloudColorHandlerRGBField<PointTSuperVoxel> rgb(
        temp_cloud_rgba);
    viewer->addPointCloud<PointTSuperVoxel>(temp_cloud_rgba, rgb, "ColorCloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ColorCloud");
    viewer->addCoordinateSystem(0.01);
  }

  // VCCS SEGMENTATION
  auto start_sv_segmentation_time = std::chrono::system_clock::now();

  pcl::SupervoxelClustering<PointTSuperVoxel> supervoxel_cluster =
      MakeSuperVoxelCluster(temp_cloud_rgba);
  std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>
      supervoxel_cluster_map;
  supervoxel_cluster.extract(supervoxel_cluster_map);

  auto end_sv_segmentation_time = std::chrono::system_clock::now();
  auto elapsed_sv_segmentation_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          end_sv_segmentation_time - start_sv_segmentation_time);
  totalMilliseconds_sv_segmentation_time = elapsed_sv_segmentation_time.count();
  std::ostringstream sv_segmentation_time;
  sv_segmentation_time << "SV time: " << totalMilliseconds_sv_segmentation_time
                       << " ms";
  // viewer->addText(sv_segmentation_time.str(), 10, 80, 20, 0, 0, 0, "sv time
  // text");

  if (mObjectTrackingState == NO_OBJECT) {
    segment_manager.segment_manager_history_size_ = 0;
    segment_manager.segment_history_.clear();

    // LCCP SEGMENTATION
    pcl::LCCPSegmentation<PointT> lccp_segmentation =
        doLCCPSegmentation(supervoxel_cluster, supervoxel_cluster_map);

    viewer->registerPointPickingCallback(pointPickingEventOccurred,
                                         (void *)&viewer);
    // OBJECT SELECTION
    if (new_clicked_point == true) {
      std::cout << "[INFO] Point picking event occurred." << std::endl;
      std::cout << "[INFO] Point coordinate ( " << x_clicking << ", "
                << y_clicking << ", " << z_clicking << ")" << std::endl;
      std::cout << "[INFO] Point id ( " << clicked_point_id << ")" << std::endl;
      new_clicked_point = false;

      std::shared_ptr<DEF_OBJ_TRACK::Segment> closest_object = getClosestObject(
          lccp_segmentation, x_clicking, y_clicking, z_clicking,
          supervoxel_cluster, supervoxel_cluster_map);

      std::map<uint32_t, std::shared_ptr<DEF_OBJ_TRACK::Segment>>
          segment_list_now;
      segment_list_now.emplace(1, closest_object);
      segment_manager.SegmentManagerUpdate(segment_list_now);

      mLastProcessedState = mObjectTrackingState;
      mObjectTrackingState = NEW_OBJECT_SELECTED;
      cout << "NEW OBJECT SELECTED" << endl;
    }
  } else if (mObjectTrackingState == NEW_OBJECT_SELECTED) {
    auto start_DOT_time = std::chrono::system_clock::now();

    trackObject(segment_manager.segment_history_[1][1], supervoxel_cluster,
                supervoxel_cluster_map);

    auto end_DOT_time = std::chrono::system_clock::now();
    auto elapsed_DOT_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_DOT_time -
                                                              start_DOT_time);
    totalMilliseconds_DOT_time = elapsed_DOT_time.count();
    // std::ostringstream DOT_time;
    // DOT_time << "DOT time: " << totalMilliseconds_DOT_time << " ms";
    // viewer->addText(DOT_time.str(), 10, 60, 20, 0, 0, 0, "DOT time text");

    segment_manager.SegmentManagerUpdate(segment_manager.segment_list_now_);

    if ((mObjectTrackingState != NO_OBJECT)) {

      auto start_Optimization_time = std::chrono::system_clock::now();

      computeOptimalCameraLocation(segment_manager.segment_list_now_[1],
                                   segment_manager.segment_list_now_[3],
                                   segment_manager.segment_list_now_[4]);

      // computeOptimalCameraLocationNoOcclusions(segment_manager.segment_list_now_[1]);

      auto end_Optimization_time = std::chrono::system_clock::now();
      auto elapsed_Optimization_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              end_Optimization_time - start_Optimization_time);
      totalMilliseconds_Optimization_time = elapsed_Optimization_time.count();
      // std::ostringstream Optimization_time;
      // Optimization_time << "Opt. time: " <<
      // totalMilliseconds_Optimization_time << " ms";
      // viewer->addText(Optimization_time.str(), 10, 40, 20, 0, 0, 0, "Opt.
      // time text");

      mLastProcessedState = mObjectTrackingState;

      std::ostringstream FPS_total_process;
      fps_measurement = 1.0f / ((totalMilliseconds_slam_time +
                                 totalMilliseconds_sv_segmentation_time +
                                 totalMilliseconds_DOT_time +
                                 totalMilliseconds_Optimization_time) *
                                0.001f);
      FPS_total_process << "fps: " << fps_measurement;
      viewer->addText(FPS_total_process.str(), 10, 120, 20, 0, 0, 0,
                      "fps text");

      // Normal histogram visualization

      // std::vector<double> data_vec;

      // for (int i = 1; i <=
      // segment_manager.segment_list_now_[1]->number_of_sv_in_segment_; i++)
      // {
      //   pcl::PointNormal NormalVector =
      //   segment_manager.segment_list_now_[1]->segments_normals_[i];
      //   data_vec.push_back(NormalVector.normal_x);
      // }

      // plotter->addHistogramData(data_vec, 20);

      if (save_maps_to_file) {
        frame_count++;
        char frame_filename_cloud[1000];
        sprintf(frame_filename_cloud, "/home/pc-campero2/SavedClouds/%06d.txt",
                frame_count);
        pcl::io::savePCDFileASCII(frame_filename_cloud,
                                  *(segment_manager.segment_list_now_[1]
                                        ->selected_cloud_with_labeled_sv_));
      }
    }
  }

  auto end_general_time = std::chrono::system_clock::now();
  auto elapsed_general_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_general_time -
                                                            start_general_time);
  totalMilliseconds_general_time = elapsed_general_time.count();

  // std::ostringstream general_time;
  // general_time << "Time: " << totalMilliseconds_general_time * 0.001f << "
  // s"; viewer->addText(general_time.str(), 10, 20, 20, 0, 0, 0, "general_time
  // text");

  if ((mObjectTrackingState != NO_OBJECT)) {
    myfile << totalMilliseconds_slam_time << ", "
           << totalMilliseconds_sv_segmentation_time << ", "
           << totalMilliseconds_DOT_time << ", "
           << totalMilliseconds_Optimization_time << ", " << fps_measurement
           << ", " << totalMilliseconds_general_time << endl;
    myfile.flush();
  }

  // plotter->setYRange(0, 25);
  // plotter->setXRange(-1, 1);
  // plotter->spinOnce();
  viewer->spinOnce();
  // CLEANING ALL VISUALIZATION

  viewer->removeAllShapes();
  viewer->removeAllPointClouds();
  viewer->removeCoordinateSystem("ref");
  viewer->removeCoordinateSystem("ref_objetivo");
  viewer->removeCoordinateSystem("refObject");
  // plotter->clearPlots();

  frame_count++;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

PointCloudTSuperVoxel::Ptr RGBDNode::getPointcloudFromORBSlamMapDataStruct(
    const std::map<ORB_SLAM2::MapPoint *, cv::Vec3f> &depthmap_points) {
  PointCloudTSuperVoxel::Ptr temp_cloud(new PointCloudTSuperVoxel);

  cv::Vec3f color_dense;
  ORB_SLAM2::MapPoint *MP;

  for (std::map<ORB_SLAM2::MapPoint *, cv::Vec3f>::const_iterator it =
           depthmap_points.begin();
       it != depthmap_points.end(); ++it) {

    MP = (it->first);
    color_dense = (it->second);
    cv::Mat pos = MP->GetWorldPos();

    int rgb = ((int)color_dense[0]) << 16 | ((int)color_dense[1]) << 8 |
              ((int)color_dense[2]);

    PointTSuperVoxel point_in_temp_cloud;

    point_in_temp_cloud.x = pos.at<float>(0);
    point_in_temp_cloud.y = pos.at<float>(1);
    point_in_temp_cloud.z = pos.at<float>(2);

    point_in_temp_cloud.rgba = rgb;

    temp_cloud->push_back(point_in_temp_cloud);
  }

  if (temp_cloud->empty()) {
    int rgb = (1) << 16 | (1) << 8 | (1);

    PointTSuperVoxel point_in_temp_cloud;

    point_in_temp_cloud.x = 0.0f;
    point_in_temp_cloud.y = 0.0f;
    point_in_temp_cloud.z = 0.0f;
    point_in_temp_cloud.rgba = rgb;

    temp_cloud->push_back(point_in_temp_cloud);
  }

  return temp_cloud;
}

PointCloudTSuperVoxel::Ptr
RGBDNode::PointCloudCreation(const cv::Mat dmcImColor,
                             const cv::Mat dmcImDepth) {
  float mDepthMapFactor = 1000.0;

  if (fabs(mDepthMapFactor) < 1e-5)
    mDepthMapFactor = 1;
  else
    mDepthMapFactor = 1.0f / mDepthMapFactor;

  cv::Mat imDepth;

  if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || dmcImDepth.type() != CV_32F)
    dmcImDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

  PointCloudTSuperVoxel::Ptr temp_cloud(new PointCloudTSuperVoxel);

  for (int j = 0; j < ((imDepth.cols) - 0); j = j + 2) // 10)
  {                                                    // j< siempre "<" no "<="
    for (int i = 0; i < ((imDepth.rows) - 0); i = i + 2) // 10)
    {

      float d = imDepth.at<float>(i, j);

      if ((d >= 0.15) &&
          (d <= 0.7)) // Setting the close/far distance thresholds
      {
        PointTSuperVoxel point_in_temp_cloud;
        cv::Mat color_pixel_coord = cv::Mat::zeros(4, 1, CV_32F);

        point_in_temp_cloud.x =
            d * (j - cameraIntrinsics.at<float>(0, 2)) /
            cameraIntrinsics.at<float>(
                0, 0); // d * (j - cameraIntrinsics.at<float>(0, 2)) /
                       // cameraIntrinsics.at<float>(0, 0); //
        point_in_temp_cloud.y =
            d * (i - cameraIntrinsics.at<float>(1, 2)) /
            cameraIntrinsics.at<float>(
                1, 1); // d * (i - cameraIntrinsics.at<float>(1, 2)) /
                       // cameraIntrinsics.at<float>(1, 1);    //
        point_in_temp_cloud.z = d;

        cv::Vec3f color_dense = dmcImColor.at<cv::Vec3b>(cv::Point(j, i));
        int rgb = ((int)color_dense[0]) << 16 | ((int)color_dense[1]) << 8 |
                  ((int)color_dense[2]);
        point_in_temp_cloud.rgba = rgb;
        temp_cloud->push_back(point_in_temp_cloud);
      }
    }
  }

  if (temp_cloud->empty()) {
    int rgb = (1) << 16 | (1) << 8 | (1);

    PointTSuperVoxel point_in_temp_cloud;

    point_in_temp_cloud.x = 0.0f;
    point_in_temp_cloud.y = 0.0f;
    point_in_temp_cloud.z = 0.0f;
    point_in_temp_cloud.rgba = rgb;

    temp_cloud->push_back(point_in_temp_cloud);
  }
  return temp_cloud;
}

pcl::SupervoxelClustering<PointTSuperVoxel>
RGBDNode::MakeSuperVoxelCluster(PointCloudTSuperVoxel::Ptr cloud) {

  pcl::SupervoxelClustering<PointTSuperVoxel> super(voxel_resolution,
                                                    seed_resolution);
  // std::cout<< seed_resolution << endl;

  super.setUseSingleCameraTransform(true);

  if (cloud->size() == 0) {
    std::cout << "Map point vector is empty! 2" << std::endl;
  } else {
    super.setInputCloud(cloud);
  }

  // if (has_normals)
  //   super.setNormalCloud (input_normals_ptr);
  super.setColorImportance(color_importance);
  super.setSpatialImportance(spatial_importance);
  super.setNormalImportance(normal_importance);
  super.setUseSingleCameraTransform(true);

  return super;
}

pcl::LCCPSegmentation<PointT> RGBDNode::doLCCPSegmentation(
    pcl::SupervoxelClustering<PointTSuperVoxel> supervoxel_cluster,
    std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>
        supervoxel_cluster_map) {

  if (use_supervoxel_refinement) {
    supervoxel_cluster.refineSupervoxels(2, supervoxel_cluster_map);
  }

  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
  supervoxel_cluster.getSupervoxelAdjacency(supervoxel_adjacency);

  /// Get the cloud of supervoxel centroid with normals and the colored cloud
  /// with supervoxel coloring (this is used for visulization)
  pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud =
      pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud(
          supervoxel_cluster_map);

  /// The Main Step: Perform LCCPSegmentation

  // PCL_INFO ("Starting Segmentation\n");
  pcl::LCCPSegmentation<PointT> lccp;
  lccp.setConcavityToleranceThreshold(10.0f);
  lccp.setSanityCheck(use_sanity_criterion);
  lccp.setSmoothnessCheck(true, supervoxel_cluster.getVoxelResolution(),
                          supervoxel_cluster.getSeedResolution(), 0.1f);
  lccp.setKFactor(0);
  lccp.setInputSupervoxels(supervoxel_cluster_map, supervoxel_adjacency);
  lccp.setMinSegmentSize(5);
  lccp.segment();

  // PCL_INFO ("Interpolation voxel cloud -> input cloud and relabeling\n");
  pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud =
      supervoxel_cluster.getLabeledCloud();
  pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud =
      sv_labeled_cloud->makeShared();
  lccp.relabelCloud(*lccp_labeled_cloud);
  SuperVoxelAdjacencyList sv_adjacency_list;
  lccp.getSVAdjacencyList(sv_adjacency_list); // Needed for visualization

  /// -----------------------------------|  Visualization
  /// |-----------------------------------

  if (VisualizeLCCPConnections_) {
    /// Calculate visualization of adjacency graph
    std::set<EdgeID> edge_drawn;
    // The vertices in the supervoxel adjacency list are the supervoxel
    // centroids This iterates through them, finding the edges
    std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
    vertex_iterator_range = boost::vertices(sv_adjacency_list);

    /// Create a cloud of the voxelcenters and map: VertexID in adjacency graph
    /// -> Point index in cloud

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkUnsignedCharArray> colors =
        vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("Colors");
    // Create a polydata to store everything in

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    for (VertexIterator itr = vertex_iterator_range.first;
         itr != vertex_iterator_range.second; ++itr) {
      const uint32_t sv_label = sv_adjacency_list[*itr];
      std::pair<AdjacencyIterator, AdjacencyIterator> neighbors =
          boost::adjacent_vertices(*itr, sv_adjacency_list);

      for (AdjacencyIterator itr_neighbor = neighbors.first;
           itr_neighbor != neighbors.second; ++itr_neighbor) {
        EdgeID connecting_edge =
            boost::edge(*itr, *itr_neighbor, sv_adjacency_list)
                .first; // Get the edge connecting these supervoxels
        if (sv_adjacency_list[connecting_edge].is_convex)
          color = convex_color;
        else
          color = concave_color;

        // two times since we add also two points per edge
        colors->InsertNextTupleValue(color);
        colors->InsertNextTupleValue(color);

        pcl::Supervoxel<PointT>::Ptr supervoxel =
            supervoxel_cluster_map.at(sv_label);
        pcl::PointXYZRGBA vert_curr = supervoxel->centroid_;

        const uint32_t sv_neighbor_label = sv_adjacency_list[*itr_neighbor];
        pcl::Supervoxel<PointT>::Ptr supervoxel_neigh =
            supervoxel_cluster_map.at(sv_neighbor_label);
        pcl::PointXYZRGBA vert_neigh = supervoxel_neigh->centroid_;

        points->InsertNextPoint(vert_curr.data);
        points->InsertNextPoint(vert_neigh.data);

        // Add the points to the dataset
        vtkSmartPointer<vtkPolyLine> polyLine =
            vtkSmartPointer<vtkPolyLine>::New();
        polyLine->GetPointIds()->SetNumberOfIds(2);
        polyLine->GetPointIds()->SetId(0, points->GetNumberOfPoints() - 2);
        polyLine->GetPointIds()->SetId(1, points->GetNumberOfPoints() - 1);
        cells->InsertNextCell(polyLine);
      }
    }

    polyData->SetPoints(points);
    // Add the lines to the dataset
    polyData->SetLines(cells);
    polyData->GetPointData()->SetScalars(colors);

    viewer->removeShape("adjacency_graph");
    viewer->addModelFromPolyData(polyData, "adjacency_graph");
  } else {
    viewer->removeShape("adjacency_graph");
  }

  if (VisualizeLCCPSegments_) {
    viewer->addPointCloud(lccp_labeled_cloud, "maincloud");

    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "maincloud");
    viewer->updatePointCloud((VisualizeSupervoxels_) ? sv_labeled_cloud
                                                     : lccp_labeled_cloud,
                             "maincloud");
  }

  // /// Show Normals
  // if (show_normals)
  // {
  //  viewer->addPointCloudNormals<pcl::PointNormal>(sv_centroid_normal_cloud,
  //  1, 1.25, "normals");
  // }
  // viewer->spinOnce(100);
  return lccp;
}

std::shared_ptr<DEF_OBJ_TRACK::Segment> RGBDNode::getClosestObject(
    const pcl::LCCPSegmentation<PointT> &lccp, float xClicked, float yClicked,
    float zClicked,
    const pcl::SupervoxelClustering<PointTSuperVoxel> &supervoxel_cluster,
    const std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>
        &supervoxel_cluster_map) {

  float minclickeddistance = 100.0f;
  uint32_t closest_supervoxel_label;
  pcl::Supervoxel<PointTSuperVoxel>::Ptr supervoxel;

  pcl::PointXYZRGBA clickedCentroid;
  clickedCentroid.x = x_clicking;
  clickedCentroid.y = y_clicking;
  clickedCentroid.z = z_clicking;

  cout << clickedCentroid.x << " " << clickedCentroid.y << " "
       << clickedCentroid.z << endl;

  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
  supervoxel_cluster.getSupervoxelAdjacency(supervoxel_adjacency);

  for (std::multimap<uint32_t, uint32_t>::iterator label_itr =
           supervoxel_adjacency.begin();
       label_itr != supervoxel_adjacency.end();) {
    // First get the label
    uint32_t supervoxel_label = label_itr->first;
    // Now get the supervoxel corresponding to the label
    supervoxel = supervoxel_cluster_map.at(supervoxel_label);

    pcl::PointXYZRGBA supervoxelCentroid = supervoxel->centroid_;
    float clickeddistnace =
        pcl::geometry::distance(clickedCentroid, supervoxelCentroid);

    if (clickeddistnace < minclickeddistance) {

      minclickeddistance = clickeddistnace;
      closest_supervoxel_label = label_itr->first;
    }

    label_itr = supervoxel_adjacency.upper_bound(supervoxel_label);
  }

  cout << "minclickeddistance" << endl;
  cout << minclickeddistance << endl;

  std::map<uint32_t, std::set<uint32_t>> segment_supervoxel_map_arg;
  std::map<uint32_t, uint32_t> supervoxel_segment_map_arg;
  uint32_t closest_segment_label;

  // Geting the segment in which the closest sv to the clicked point is

  lccp.getSegmentToSupervoxelMap(segment_supervoxel_map_arg);
  lccp.getSupervoxelToSegmentMap(supervoxel_segment_map_arg);

  for (std::map<uint32_t, std::set<uint32_t>>::iterator label_LCCP_iterator =
           segment_supervoxel_map_arg.begin();
       label_LCCP_iterator != segment_supervoxel_map_arg.end();) {
    std::set<uint32_t> SVs_in_segment = label_LCCP_iterator->second;

    std::set<uint32_t>::iterator it;
    for (it = SVs_in_segment.begin(); it != SVs_in_segment.end(); ++it) {

      uint32_t supervoxelIterando = *it; // Note the "*" here
      if (supervoxelIterando == closest_supervoxel_label) {
        closest_segment_label = label_LCCP_iterator->first;
      }
    }

    label_LCCP_iterator =
        segment_supervoxel_map_arg.upper_bound(label_LCCP_iterator->first);
  }

  cout << "closestmObjectTrackingState supervoxel label: "
       << closest_supervoxel_label << endl;
  cout << "closest segment label: " << closest_segment_label << endl;

  std::set<uint32_t> SVs_in_closest_segment =
      segment_supervoxel_map_arg[closest_segment_label];
  std::set<uint32_t>::iterator itr_closest_segment;

  pcl::PointCloud<PointTSuperVoxel>::Ptr selected_cloud_with_labeled_sv_(
      new pcl::PointCloud<PointTSuperVoxel>);
  std::shared_ptr<DEF_OBJ_TRACK::Segment> initial_object =
      std::make_shared<DEF_OBJ_TRACK::Segment>(
          selected_cloud_with_labeled_sv_, neighbouringRadius,
          explorationRadius,
          DEF_OBJ_TRACK::Segment::SegmentType::TARGET_OBJECT);

  for (itr_closest_segment = SVs_in_closest_segment.begin();
       itr_closest_segment != SVs_in_closest_segment.end();
       ++itr_closest_segment) // iterando en los supervoxels de pertenecen al
                              // segmento en le que se ha clickado
  {
    pcl::PointXYZRGBA SVcentroid;
    pcl::PointNormal newNormal;

    uint32_t supervoxel_label = *itr_closest_segment;
    pcl::Supervoxel<PointTSuperVoxel>::Ptr supervoxel =
        supervoxel_cluster_map.at(supervoxel_label);

    supervoxel->getCentroidPoint(SVcentroid);
    pcl::PointXYZRGBA SVcolorRGB = computeColorRGB(SVcentroid);
    supervoxel->getCentroidPointNormal(newNormal);

    pcl::PointCloud<PointTSuperVoxel>::Ptr voxelsPointcloud =
        supervoxel->voxels_;

    initial_object->label_of_sv_++;
    initial_object->segments_sv_map_.insert(
        std::pair<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>(
            initial_object->label_of_sv_, supervoxel));
    initial_object->segments_colors_RGB_.insert(
        std::pair<uint32_t, pcl::PointXYZRGBA>(initial_object->label_of_sv_,
                                               SVcolorRGB));
    initial_object->segments_normals_.insert(
        std::pair<uint32_t, pcl::PointNormal>(initial_object->label_of_sv_,
                                              newNormal));

    *initial_object->selected_cloud_with_labeled_sv_ +=
        *supervoxel->voxels_; // adding the pointcloud of the accepted voxel to
                              // the objects pointcloud
  }

  initial_object->computeFeatureExtraction();
  initial_object->computeOctreeAdjacencyAndDegree(viewer);

  // TODO... but much later in the future: automatise all of this for several
  // objects, several interactors etc
  pcl::PointCloud<pcl::PointNormal>::Ptr average_normal_cloud(
      new pcl::PointCloud<pcl::PointNormal>);
  initial_object->visualization_of_vectors_cloud_ = average_normal_cloud;
  // BestNextView
  cv::Mat camera_position = orb_slam_->GetCurrentPosition();
  cv::Mat camera_intrinsics = orb_slam_->GetCameraIntrinsics();

  if (!camera_position.empty() && (initial_object->label_of_sv_ > 0)) {
    initial_object->camera_position_ = camera_position;
    initial_object->camera_intrinsics_ = camera_intrinsics;
    initial_object->computeInverseOfCameraPositionAndExtendedIntrinsics();
    initial_object->computeCameraZaxisOnWorldReference();

    initial_object->computeLargestDistanceToCamera(
        initial_object->visualization_of_vectors_cloud_);
    initial_object->computeInterestfrustum(
        initial_object->visualization_of_vectors_cloud_);
  }

  initial_object->camera_position_ = camera_position;
  initial_object->camera_intrinsics_ = camera_intrinsics;

  initial_object->segmentType_ =
      DEF_OBJ_TRACK::Segment::SegmentType::TARGET_OBJECT;

  return initial_object;
}

void RGBDNode::trackObject(
    std::shared_ptr<DEF_OBJ_TRACK::Segment> previousObject,
    pcl::SupervoxelClustering<PointTSuperVoxel> &supervoxel_cluster,
    const std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>
        &supervoxel_cluster_map) {

  segment_manager.segment_list_now_.clear();
  // DEFINITION OF OBJECT, INTERACTORS, OCCLUSIONS AND HARD OCCLUSIONS

  pcl::PointCloud<PointTSuperVoxel>::Ptr selected_cloud_with_labeled_sv(
      new pcl::PointCloud<PointTSuperVoxel>);
  std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject =
      std::make_shared<DEF_OBJ_TRACK::Segment>(
          selected_cloud_with_labeled_sv, neighbouringRadius, explorationRadius,
          DEF_OBJ_TRACK::Segment::SegmentType::TARGET_OBJECT);
  NewObject->segments_voxel_labeled_cloud_ =
      supervoxel_cluster.getLabeledCloud();

  pcl::PointCloud<PointTSuperVoxel>::Ptr interactor_cloud_with_labeled_sv(
      new pcl::PointCloud<PointTSuperVoxel>);
  std::shared_ptr<DEF_OBJ_TRACK::Segment> Interactor =
      std::make_shared<DEF_OBJ_TRACK::Segment>(
          interactor_cloud_with_labeled_sv, neighbouringRadius,
          explorationRadius,
          DEF_OBJ_TRACK::Segment::SegmentType::INTERACTING_ELEMENT);

  pcl::PointCloud<PointTSuperVoxel>::Ptr occlusions_cloud_with_labeled_sv(
      new pcl::PointCloud<PointTSuperVoxel>);
  std::shared_ptr<DEF_OBJ_TRACK::Segment> Occlusions =
      std::make_shared<DEF_OBJ_TRACK::Segment>(
          occlusions_cloud_with_labeled_sv, neighbouringRadius,
          explorationRadius,
          DEF_OBJ_TRACK::Segment::SegmentType::OCCLUDING_ELEMENT);

  pcl::PointCloud<PointTSuperVoxel>::Ptr hard_occlusions_cloud_with_labeled_sv(
      new pcl::PointCloud<PointTSuperVoxel>);
  std::shared_ptr<DEF_OBJ_TRACK::Segment> HardOcclusions =
      std::make_shared<DEF_OBJ_TRACK::Segment>(
          hard_occlusions_cloud_with_labeled_sv, neighbouringRadius,
          explorationRadius,
          DEF_OBJ_TRACK::Segment::SegmentType::HARD_OCCLUDING_ELEMENT);

  // ANALYSIS OF NEW INCOMING SUPERVOXELS
  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
  supervoxel_cluster.getSupervoxelAdjacency(supervoxel_adjacency);

  pcl::PointXYZRGBA SVcentroid;
  pcl::PointNormal newNormal;
  pcl::PointXYZRGBA SVcolorRGB;

  // for visualization....
  pcl::PointCloud<pcl::PointNormal>::Ptr average_normal_cloud(
      new pcl::PointCloud<pcl::PointNormal>);
  NewObject->visualization_of_vectors_cloud_ = average_normal_cloud;

  int total_number_of_sv = 0;

  for (std::multimap<uint32_t, uint32_t>::iterator label_itr =
           supervoxel_adjacency.begin();
       label_itr != supervoxel_adjacency.end();) {
    total_number_of_sv++;

    // First get the label
    uint32_t supervoxel_label = label_itr->first;

    // Now get the supervoxel corresponding to the label
    pcl::Supervoxel<PointTSuperVoxel>::Ptr supervoxel =
        supervoxel_cluster_map.at(supervoxel_label);

    // getting SV's centroid (also on objects reference)
    supervoxel->getCentroidPoint(SVcentroid);
    pcl::PointXYZRGBA SVcentroid_on_objects_ref =
        previousObject->computeCentroidCoordInObjectsReference(SVcentroid);

    // getting SV's rgb value
    SVcolorRGB = computeColorRGB(SVcentroid);
    // getting SV's normal
    supervoxel->getCentroidPointNormal(newNormal);

    if (previousObject->isItInsideAugmentedOBB(SVcentroid_on_objects_ref)) {

      bool not_belonging_yet = true;
      for (int i = 1; ((i <= (segment_manager.segment_manager_history_size_)) &&
                       not_belonging_yet);
           i++) {
        if (segment_manager.doesItBelongToSegment(
                SVcolorRGB, newNormal, segment_manager.segment_history_[i][1],
                viewer)) {

          NewObject->label_of_sv_++;
          NewObject->segments_sv_map_.insert(
              std::pair<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>(
                  NewObject->label_of_sv_, supervoxel));
          NewObject->segments_colors_RGB_.insert(
              std::pair<uint32_t, pcl::PointXYZRGBA>(NewObject->label_of_sv_,
                                                     SVcolorRGB));
          NewObject->segments_normals_.insert(
              std::pair<uint32_t, pcl::PointNormal>(NewObject->label_of_sv_,
                                                    newNormal));
          // pcl::PointCloud<PointTSuperVoxel>::Ptr voxelsPointcloud =
          // supervoxel->voxels_;
          *NewObject->selected_cloud_with_labeled_sv_ += *supervoxel->voxels_;

          // pcl::PointNormal normalWorldRef;
          // normalWorldRef.x = 0.0;
          // normalWorldRef.y = 0.0;
          // normalWorldRef.z = 0.0;
          // normalWorldRef.normal_x = newNormal.normal_x - newNormal.x;
          // normalWorldRef.normal_y = newNormal.normal_y - newNormal.y;
          // normalWorldRef.normal_z = newNormal.normal_z - newNormal.z;

          // Eigen::Matrix<float, 3, 1> normal_world_reference;
          // normal_world_reference << normalWorldRef.normal_x,
          //     normalWorldRef.normal_y,
          //     normalWorldRef.normal_z;

          // Eigen::Matrix<float, 3, 1> normal_world_reference_normalized;
          // normal_world_reference_normalized =
          // normal_world_reference.normalized(); float normal_norm =
          // normal_world_reference_normalized.norm(); cout << "normal_norm: "
          // << normal_norm << endl;

          // NewObject->segment_normals_on_world_reference_.insert(std::pair<uint32_t,
          // pcl::PointNormal>(NewObject->label_of_sv_, normalWorldRef));
          // NewObject->visualization_of_vectors_cloud_->push_back(normalWorldRef);

          not_belonging_yet = false;
        } else if (i == segment_manager.segment_manager_history_size_) {

          Interactor->label_of_sv_++;
          Interactor->segments_sv_map_.insert(
              std::pair<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>(
                  Interactor->label_of_sv_, supervoxel));
          Interactor->segments_colors_RGB_.insert(
              std::pair<uint32_t, pcl::PointXYZRGBA>(Interactor->label_of_sv_,
                                                     SVcolorRGB));
          Interactor->segments_normals_.insert(
              std::pair<uint32_t, pcl::PointNormal>(Interactor->label_of_sv_,
                                                    newNormal));
          // pcl::PointCloud<PointTSuperVoxel>::Ptr voxelsPointcloud =
          // supervoxel->voxels_;
          *Interactor->selected_cloud_with_labeled_sv_ += *supervoxel->voxels_;

          // TODO: visualize weights aswell
          // std::stringstream ssText, slam_time;
          // slam_time << "Ws: " << std::setprecision(3) <<
          // centroidDistanceTotalWeighted; ssText << "text_" << SVid << "_" <<
          // ComparisonObjectID; viewer->addText3D(slam_time.str(),
          // sv_centroidRadiusSearch, 0.004, 1, 1, 1, ssText.str());
        }
      }
    } else if (!(previousObject->camera_position_.empty())) {
      if (previousObject->isItInOcclusionsDepthRange(SVcentroid)) {
        if (previousObject->isItInPerceptionSphere(
                SVcentroid,
                sphere_radius)) //(previousObject->isItInCameraToObjectFrustum(SVcentroid))
        {
          // cout << "it is!! it iss :D" << endl;
          HardOcclusions->label_of_sv_++;
          HardOcclusions->segments_sv_map_.insert(
              std::pair<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>(
                  HardOcclusions->label_of_sv_, supervoxel));
          HardOcclusions->segments_colors_RGB_.insert(
              std::pair<uint32_t, pcl::PointXYZRGBA>(
                  HardOcclusions->label_of_sv_, SVcolorRGB));
          HardOcclusions->segments_normals_.insert(
              std::pair<uint32_t, pcl::PointNormal>(
                  HardOcclusions->label_of_sv_, newNormal));
          // pcl::PointCloud<PointTSuperVoxel>::Ptr voxelsPointcloud =
          // supervoxel->voxels_;
          *HardOcclusions->selected_cloud_with_labeled_sv_ +=
              *supervoxel->voxels_;
        } else {

          Occlusions->label_of_sv_++;
          Occlusions->segments_sv_map_.insert(
              std::pair<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr>(
                  Occlusions->label_of_sv_, supervoxel));
          Occlusions->segments_colors_RGB_.insert(
              std::pair<uint32_t, pcl::PointXYZRGBA>(Occlusions->label_of_sv_,
                                                     SVcolorRGB));
          Occlusions->segments_normals_.insert(
              std::pair<uint32_t, pcl::PointNormal>(Occlusions->label_of_sv_,
                                                    newNormal));
          // pcl::PointCloud<PointTSuperVoxel>::Ptr voxelsPointcloud =
          // supervoxel->voxels_;
          *Occlusions->selected_cloud_with_labeled_sv_ += *supervoxel->voxels_;
        }
      }
    }

    label_itr = supervoxel_adjacency.upper_bound(supervoxel_label);
  }
  // std::ostringstream total_number_sv;
  // total_number_sv << "Total nº SV: " << total_number_of_sv;
  // viewer->addText(total_number_sv.str(), 10, 220, 20, 0, 0, 0, "total n sv");
  myfile << total_number_of_sv << ", ";

  // std::ostringstream n_sv;
  int totalNumberOfSVinObject = NewObject->label_of_sv_;
  // n_sv << "nº SV Object:  " << totalNumberOfSVinObject;
  // viewer->addText(n_sv.str(), 10, 200, 20, 0, 0, 0, "v3text");
  myfile << totalNumberOfSVinObject << ", ";

  if (NewObject->label_of_sv_ == 0) {
    mLastProcessedState = mObjectTrackingState;
    mObjectTrackingState = NO_OBJECT;
    new_clicked_point = false;
    cout << "State: OBJECT_LOST" << endl;
    return;
  }
  // TODO... but much later in the future: automatise all of this for several
  // objects, several interactors etc
  segment_manager.segment_list_now_.emplace(1, NewObject);
  segment_manager.segment_list_now_.emplace(2, Interactor);
  segment_manager.segment_list_now_.emplace(3, Occlusions);
  segment_manager.segment_list_now_.emplace(4, HardOcclusions);

  NewObject->computeFeatureExtraction();
  NewObject->computeOctreeAdjacencyAndDegree(viewer);

  Interactor->computeFeatureExtraction();
  Interactor->computeOctreeAdjacencyAndDegree(viewer);

  Occlusions->computeFeatureExtraction();
  Occlusions->computeOctreeAdjacencyAndDegree(viewer);

  HardOcclusions->computeFeatureExtraction();
  HardOcclusions->computeOctreeAdjacencyAndDegree(viewer);

  NewObject->VisualizeCloudsAndBoundingBoxes(viewer);

  // viewer->addPointCloudNormals<PointNTSuperVoxel>(NewObject->visualization_of_vectors_cloud_,
  // 1, 1.0f, "vectors_cloud");
  // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
  // 0.0, 0.0, 1.0, "vectors_cloud");
  // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
  // 1, "vectors_cloud");

  return;
}

void RGBDNode::computeOptimalCameraLocation(
    std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject,
    std::shared_ptr<DEF_OBJ_TRACK::Segment> Occlusions,
    std::shared_ptr<DEF_OBJ_TRACK::Segment> HardOcclusions) {
  // BestNextView
  cv::Mat camera_position = orb_slam_->GetCurrentPosition();
  cv::Mat camera_intrinsics = orb_slam_->GetCameraIntrinsics();

  if (!camera_position.empty() && (NewObject->number_of_sv_in_segment_ > 0)) {

    std::map<uint32_t, pcl::PointNormal> object_normals =
        NewObject->segments_normals_;

    for (int i = 0; i < NewObject->number_of_sv_in_segment_; ++i) {
      pcl::PointNormal sv_normal =
          object_normals[i + 1]; // index in maps start at 1 for me (0 preserved
                                 // for special cases)
    }

    NewObject->camera_position_ = camera_position;
    NewObject->camera_intrinsics_ = camera_intrinsics;
    NewObject->computeInverseOfCameraPositionAndExtendedIntrinsics();
    NewObject->computeCameraZaxisOnWorldReference();

    NewObject->computeLargestDistanceToCamera(
        NewObject->visualization_of_vectors_cloud_);
    NewObject->computeInterestfrustum(
        NewObject->visualization_of_vectors_cloud_);

    NewObject->visualizeObjectsReprojectionOnCamera(Occlusions, HardOcclusions);

    // NewObject->computeAverageNormal();
    // NewObject->computeDesiredPositionVector();
    // NewObject->computeAngularError();
    // NewObject->computePositionError();

    std::map<uint32_t, Eigen::Matrix<float, 3, 1>> object_sphere_intersections =
        NewObject->normalsToSphereIntersectionPoints(viewer, sphere_radius);
    std::map<uint32_t, Eigen::Matrix<float, 3, 1>>
        occlusion_sphere_intersections =
            HardOcclusions->centroidsToOcclussorRays(viewer, sphere_radius,
                                                     NewObject);

    Eigen::Matrix<float, 3, 1> initial_camera_position_vector =
        NewObject->computeIdealOptimalCameraPosition(
            sphere_radius, object_sphere_intersections);

    Eigen::Matrix<float, 3, 1> sphere_center;
    sphere_center << NewObject->mass_center_(0), NewObject->mass_center_(1),
        NewObject->mass_center_(2);

    Eigen::Matrix<float, 3, 1> W_vector =
        sphere_center - initial_camera_position_vector;

    W_vector = W_vector.normalized();

    Eigen::Matrix<float, 3, 1> U_vector =
        NewObject->computePerpendicularVector(W_vector);

    U_vector = U_vector.normalized();
    Eigen::Matrix<float, 3, 1> V_vector = W_vector.cross(U_vector);
    V_vector = V_vector.normalized();

    Eigen::Matrix<float, 3, 3> R_matrix;
    R_matrix << U_vector, V_vector, W_vector;

    Eigen::Matrix<float, 3, 4> Rt_matrix; // camera representation
    Rt_matrix << R_matrix, initial_camera_position_vector;

    Eigen::Affine3f t_objetivo_xy_fijos;
    for (int iAffine = 0; iAffine < 3; iAffine++) {
      for (int jAffine = 0; jAffine < 4; jAffine++) {
        t_objetivo_xy_fijos(iAffine, jAffine) =
            static_cast<float>(Rt_matrix(iAffine, jAffine));
      }
    }

    cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
    cout << "U_vector: " << U_vector << endl;
    cout << "V_vector: " << V_vector << endl;
    cout << "W_vector: " << W_vector << endl;

    cout << "Rt_matrix: " << Rt_matrix << endl;

    viewer->addCoordinateSystem(0.1, t_objetivo_xy_fijos,
                                "ref_objetivo"); // camera visualization
    // NewObject->optimal_position_ = t_objetivo;

    // visualization of camera and errors on viewer
    viewer->addPointCloudNormals<PointNTSuperVoxel>(
        NewObject->visualization_of_vectors_cloud_, 1, 1.0f,
        "visualization_of_vectors_cloud_");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0,
        "visualization_of_vectors_cloud_");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1,
        "visualization_of_vectors_cloud_");

    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF OLD STUF
    // OLD STUF OLD STUF

    // Eigen::Matrix<float, 3, 1> sphere_center;
    // sphere_center << NewObject->mass_center_(0), NewObject->mass_center_(1),
    //     NewObject->mass_center_(2);

    // Eigen::Matrix<float, 3, 1> normal_vector_unitary;
    // normal_vector_unitary << sphere_center(0) -
    //                              initial_camera_position_vector(0),
    //     sphere_center(1) - initial_camera_position_vector(1),
    //     sphere_center(2) - initial_camera_position_vector(2);

    // normal_vector_unitary = normal_vector_unitary.normalized();

    // Eigen::Matrix<float, 3, 1> U_vector; // 2nd basis vector
    // U_vector << float(1.0),
    //     float(0.0), //(1.0)
    //     float((-normal_vector_unitary(0)) /
    //           normal_vector_unitary(
    //               2)); //(-normal_vector_unitary(0) -
    //               normal_vector_unitary(1))
    //                    /// normal_vector_unitary(2)
    // U_vector = U_vector.normalized();

    // Eigen::Matrix<float, 3, 1> V_vector; // 3rd basis vector
    // V_vector = (normal_vector_unitary.cross(U_vector)).normalized();

    // Eigen::Matrix<float, 3, 1>
    //     W_vector; // Comprobation vector, it must be = normal_vector_unitary
    // W_vector = (U_vector.cross(V_vector)).normalized();

    // Eigen::Matrix<float, 3, 3> R_matrix; // Rwc (camera to world)
    // R_matrix << U_vector, V_vector, W_vector;

    // Eigen::Matrix<float, 3, 4> Rt_matrix; //
    // Rt_matrix << R_matrix, initial_camera_position_vector;

    // Eigen::Affine3f t_objetivo_xy_fijos;
    // for (int iAffine = 0; iAffine < 3; iAffine++) {
    //   for (int jAffine = 0; jAffine < 4; jAffine++) {
    //     t_objetivo_xy_fijos(iAffine, jAffine) =
    //         static_cast<float>(Rt_matrix(iAffine, jAffine));
    //   }
    // }

    // viewer->addCoordinateSystem(0.1, t_objetivo_xy_fijos,
    //                             "ref_objetivo"); // camera visualization
    // // NewObject->optimal_position_ = t_objetivo;

    // // visualization of camera and errors on viewer
    // viewer->addPointCloudNormals<PointNTSuperVoxel>(
    //     NewObject->visualization_of_vectors_cloud_, 1, 1.0f,
    //     "visualization_of_vectors_cloud_");
    // viewer->setPointCloudRenderingProperties(
    //     pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0,
    //     "visualization_of_vectors_cloud_");
    // viewer->setPointCloudRenderingProperties(
    //     pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1,
    //     "visualization_of_vectors_cloud_");
  }
}

void RGBDNode::computeOptimalCameraLocationNoOcclusions(
    std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject) {
  // BestNextView
  cv::Mat camera_position = orb_slam_->GetCurrentPosition();
  cv::Mat camera_intrinsics = orb_slam_->GetCameraIntrinsics();

  if (!camera_position.empty() && (NewObject->label_of_sv_ > 0)) {
    NewObject->camera_position_ = camera_position;
    NewObject->camera_intrinsics_ = camera_intrinsics;
    NewObject->computeInverseOfCameraPositionAndExtendedIntrinsics();
    // NewObject->computeCameraZaxisOnWorldReference();

    // NewObject->computeLargestDistanceToCamera(NewObject->visualization_of_vectors_cloud_);
    // NewObject->computeInterestfrustum(NewObject->visualization_of_vectors_cloud_);

    // NewObject->visualizeObjectsReprojectionOnCamera(Occlusions,
    // HardOcclusions);

    // NewObject->computeAverageNormal();
    // NewObject->computeDesiredPositionVector();
    // NewObject->computeAngularError();
    // NewObject->computePositionError();

    DEF_OBJ_TRACK::BestNextView *OptimizationProblem =
        new (DEF_OBJ_TRACK::BestNextView);
    double *parameters = OptimizationProblem->computeBestNextViewNoOcclusions(
        NewObject->segments_normals_, NewObject->number_of_sv_in_segment_,
        NewObject->Twc_depth_, NewObject->camera_intrinsics_extended_,
        NewObject->visualization_of_vectors_cloud_, viewer, myfile);
    // double *parameters =
    // OptimizationProblem->computeBestNextViewSimple(NewObject->segments_normals_,
    // NewObject->number_of_sv_in_segment_,
    //                                                                          NewObject->Twc_depth_, NewObject->camera_intrinsics_extended_,
    //                                                                          NewObject->visualization_of_vectors_cloud_);
    Eigen::Affine3f t_objetivo;
    for (int iAffine = 0; iAffine < 3; iAffine++) {
      for (int jAffine = 0; jAffine < 4; jAffine++) {
        t_objetivo(iAffine, jAffine) =
            static_cast<float>(parameters[4 * iAffine + jAffine]);
      }
    }
    Eigen::Matrix<float, 3, 1> position_vector;
    position_vector << t_objetivo(0, 3), t_objetivo(1, 3), t_objetivo(2, 3);

    Eigen::Matrix<float, 3, 1> normal_vector_unitary;
    normal_vector_unitary << static_cast<float>(parameters[2]),
        t_objetivo(1, 2), t_objetivo(2, 2);

    normal_vector_unitary = normal_vector_unitary.normalized();

    Eigen::Matrix<float, 3, 1> U_vector; // 2nd basis vector
    U_vector << float(1.0),
        float(0.0), //(1.0)
        float((-normal_vector_unitary(0)) /
              normal_vector_unitary(
                  2)); //(-normal_vector_unitary(0) - normal_vector_unitary(1))
                       /// normal_vector_unitary(2)
    U_vector = U_vector.normalized();

    Eigen::Matrix<float, 3, 1> V_vector; // 3rd basis vector
    V_vector = (normal_vector_unitary.cross(U_vector)).normalized();

    Eigen::Matrix<float, 3, 1>
        W_vector; // Comprobation vector, it must be = normal_vector_unitary
    W_vector = (U_vector.cross(V_vector)).normalized();

    Eigen::Matrix<float, 3, 3> R_matrix; // Rwc (camera to world)
    R_matrix << U_vector, V_vector, W_vector;

    Eigen::Matrix<float, 3, 4> Rt_matrix; //
    Rt_matrix << R_matrix, position_vector;

    Eigen::Affine3f t_objetivo_xy_fijos;
    for (int iAffine = 0; iAffine < 3; iAffine++) {
      for (int jAffine = 0; jAffine < 4; jAffine++) {
        t_objetivo_xy_fijos(iAffine, jAffine) =
            static_cast<float>(Rt_matrix(iAffine, jAffine));
      }
    }

    viewer->addCoordinateSystem(0.1, t_objetivo_xy_fijos,
                                "ref_objetivo"); // camera visualization
    NewObject->optimal_position_ = t_objetivo;

    delete OptimizationProblem;
    delete parameters;

    pcl::PointXYZ optimalCameraPostion;

    optimalCameraPostion.x = position_vector(0);
    optimalCameraPostion.y = position_vector(1);
    optimalCameraPostion.z = position_vector(2);

    pcl::PointXYZ actualCameraPosition;
    actualCameraPosition.x = NewObject->Twc_depth_.at<float>(0, 3);
    actualCameraPosition.y = NewObject->Twc_depth_.at<float>(1, 3);
    actualCameraPosition.z = NewObject->Twc_depth_.at<float>(2, 3);

    float cubeSize = 0.004;
    viewer->addCube(
        actualCameraPosition.x - cubeSize, actualCameraPosition.x + cubeSize,
        actualCameraPosition.y - cubeSize, actualCameraPosition.y + cubeSize,
        actualCameraPosition.z - cubeSize, actualCameraPosition.z + cubeSize,
        1.0, 0.0, 0.0, "camera bad");
    viewer->addCube(
        optimalCameraPostion.x - cubeSize, optimalCameraPostion.x + cubeSize,
        optimalCameraPostion.y - cubeSize, optimalCameraPostion.y + cubeSize,
        optimalCameraPostion.z - cubeSize, optimalCameraPostion.z + cubeSize,
        0.0, 1.0, 0.0, "camera good");

    // std::ostringstream optimal_text;
    // optimal_text << "Optimal" << endl
    //              << "camera";

    // std::ostringstream actual_camera_text;
    // actual_camera_text << "Real" << endl
    //                    << "camera";

    // viewer->addText3D(optimal_text.str(), optimalCameraPostion, 0.01, 0.0,
    // 0.0, 0.0, "optimal camera position 3d txt");
    // viewer->addText3D(actual_camera_text.str(), actualCameraPosition, 0.01,
    // 0.0, 0.0, 0.0, "actual camera position 3d txt");

    // Occlusions->OcclusionVisualChecking(NewObject, t_objetivo,viewer);

    // visualization of camera and errors on viewer
    viewer->addPointCloudNormals<PointNTSuperVoxel>(
        NewObject->visualization_of_vectors_cloud_, 1, 1.0f, "average_normal");
  }
}

void RGBDNode::addSupervoxelConnectionsToViewer(
    PointTSuperVoxel &supervoxel_center,
    PointCloudTSuperVoxel &adjacent_supervoxel_centers,
    std::string supervoxel_name) {
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

  // Iterate through all adjacent points, and add a center point to adjacent
  // point pair
  for (pcl::PointCloud<PointTSuperVoxel>::iterator adjacent_itr =
           adjacent_supervoxel_centers.begin();
       adjacent_itr != adjacent_supervoxel_centers.end(); ++adjacent_itr) {
    points->InsertNextPoint(supervoxel_center.data);
    points->InsertNextPoint(adjacent_itr->data);
  }
  // Create a polydata to store everything in
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
  // Add the points to the dataset
  polyData->SetPoints(points);
  polyLine->GetPointIds()->SetNumberOfIds(points->GetNumberOfPoints());
  for (unsigned int i = 0; i < points->GetNumberOfPoints(); i++)
    polyLine->GetPointIds()->SetId(i, i);
  cells->InsertNextCell(polyLine);
  // Add the lines to the dataset
  polyData->SetLines(cells);

  viewer->addModelFromPolyData(polyData, supervoxel_name);
  // std::cout<<"jejeeeeeeeeeeeeeeee"<<std::endl;
}

void pointPickingEventOccurred(
    const pcl::visualization::PointPickingEvent &event, void *viewer_void) {

  if (event.getPointIndex() == -1) {
    return;
  }
  event.getPoint(x_clicking, y_clicking, z_clicking);

  clicked_point_id = event.getPointIndex();
  new_clicked_point = true;
}

pcl::PointXYZRGBA
RGBDNode::computeColorRGB(const pcl::PointXYZRGBA &SVcentroid) {
  pcl::PointXYZRGBA SVcolorRGB;
  uint32_t SVrgb = (int)(SVcentroid.rgba);
  SVcolorRGB.x = (SVrgb >> 16) & 0x0000ff;
  SVcolorRGB.y = (SVrgb >> 8) & 0x0000ff;
  SVcolorRGB.z = (SVrgb)&0x0000ff;

  return SVcolorRGB;
}

void RGBDNode::gridSampleApprox(const PointCloudTSuperVoxel::Ptr &cloud,
                                PointCloudTSuperVoxel &result,
                                double leaf_size) {
  pcl::ApproximateVoxelGrid<pcl::PointXYZRGBA> grid;

  grid.setLeafSize(static_cast<float>(leaf_size), static_cast<float>(leaf_size),
                   static_cast<float>(leaf_size));
  grid.setInputCloud(cloud);
  grid.filter(result);
}
