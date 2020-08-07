/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBSLAM2_ROS_RGBDODE_H_
#define ORBSLAM2_ROS_RGBDODE_H_

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <tf/transform_broadcaster.h>

#include "System.h"
#include "Node.h"

#include <iostream>
#include <thread>

//Defined Object
#include "SegmentManager.h"

//BestNextView
#include "BestNextView.h"

class RGBDNode : public Node
{
public:
  RGBDNode(const ORB_SLAM2::System::eSensor sensor, ros::NodeHandle &node_handle, image_transport::ImageTransport &image_transport);
  ~RGBDNode();

  void ImageCallback(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD);

  PointCloudTSuperVoxel::Ptr getPointcloudFromORBSlamMapDataStruct(const std::map<ORB_SLAM2::MapPoint *, cv::Vec3f> &depthmap_points);

  pcl::SupervoxelClustering<PointTSuperVoxel> MakeSuperVoxelCluster(PointCloudTSuperVoxel::Ptr cloud);

  std::shared_ptr<DEF_OBJ_TRACK::Segment> getClosestObject(const pcl::LCCPSegmentation<PointT> &lccp, float xClicked, float yClicked, float zClicked,
                                                           const pcl::SupervoxelClustering<PointTSuperVoxel> &supervoxel_cluster,
                                                           const std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr> &supervoxel_cluster_map);

  pcl::LCCPSegmentation<PointT> doLCCPSegmentation(pcl::SupervoxelClustering<PointTSuperVoxel> supervoxel_cluster,
                                                   std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr> supervoxel_cluster_map);

  void addSupervoxelConnectionsToViewer(PointTSuperVoxel &supervoxel_center,
                                        PointCloudTSuperVoxel &adjacent_supervoxel_centers,
                                        std::string supervoxel_name);

  void trackObject(std::shared_ptr<DEF_OBJ_TRACK::Segment> previousObject, pcl::SupervoxelClustering<PointTSuperVoxel> &supervoxel_cluster,
                   const std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr> &supervoxel_cluster_map);

  void computeOptimalCameraLocation(int number_of_frame,std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject, std::shared_ptr<DEF_OBJ_TRACK::Segment> Occlusions, std::shared_ptr<DEF_OBJ_TRACK::Segment> HardOcclusions);
  void computeOptimalCameraLocationNoOcclusions(std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject);

  pcl::PointXYZRGBA
  computeColorRGB(const pcl::PointXYZRGBA &SVcentroid);

  void gridSampleApprox(const PointCloudTSuperVoxel::Ptr &cloud, PointCloudTSuperVoxel &result, double leaf_size);

  PointCloudTSuperVoxel::Ptr PointCloudCreation(const cv::Mat dmcImColor, const cv::Mat dmcImDepth);

  void initializeParameters();

public:
  cv::Mat cameraIntrinsics;
  int CameraWidth, CameraHeight;

  double downsampling_grid_size_ = 0.002;

  pcl::visualization::PCLVisualizer::Ptr viewer;
  pcl::visualization::PCLPlotter *plotter = new pcl::visualization::PCLPlotter();

  enum eObjectTrackingState
  {
    NO_OBJECT = 0,
    NEW_OBJECT_SELECTED = 1,
    OBJECT_TRACKING = 2,
  };

  DEF_OBJ_TRACK::SegmentManager segment_manager;

  eObjectTrackingState mObjectTrackingState;
  eObjectTrackingState mLastProcessedState;

  float fijoMaxX, fijoMaxY, fijoMaxZ;
  float fijoMinX, fijoMinY, fijoMinZ;

  std::ofstream outfile;

  std::map<uint32_t, pcl::PointNormal> object_normals_ant;
  std::map<uint32_t, pcl::PointXYZRGBA> object_colors_RGB_ant;

public:
  //Parameters
  double sphere_radius = 0.5;

  bool activate_SLAM;

  float neighbouringRadius; //related to Rseed
  float explorationRadius;  //fixed to 0.05

  //visualization bools and parameters

  bool use_new_method;
  bool visualize_rgba_pointcloud;

  //constants of VCCS
  bool visualize_selected_object_stuff;
  bool show_normals;
  bool visualize_main_inertia_axes;
  bool show_adjacency;
  bool show_supervoxels;
  bool show_help;
  bool disable_transform;
  float voxel_resolution; //for a good computation/performance optimization: 0.01f // 0.007
  double resolution;      //related to Rvoxel
  float seed_resolution;  //for a good computation/performance optimization: 0.038f // 0.02
  float color_importance;
  float spatial_importance;
  float normal_importance;
  bool use_single_cam_transform;
  float normals_scale;

  const unsigned char convex_color[3] = {255, 255, 255};
  const unsigned char concave_color[3] = {255, 0, 0};
  const unsigned char *color;
  unsigned int k_factor;
  bool use_supervoxel_refinement;

  // LCCPSegmentation Stuff
  float concavity_tolerance_threshold;
  float smoothness_threshold;
  int min_segment_size;
  bool use_extended_convexity;
  bool use_sanity_criterion;
  bool show_visualization;

  //Temporal coherence stuff
  //Object Segmentation color thresholds
  float toleranceFactor; //the % of the bounding box boundaries ...

  int numberOfSVinAntObject;
  int numberOfSVinObject;

  //Time Measuring
  long int totalMilliseconds;

  //Searching and graphing

  float neighbouringFactor, explorationFactor;

  //Visualization

  int VisualizeColorDenseCloud_;
  int VisualizeLCCPSegments_, VisualizeLCCPConnections_;
  int VisualizeSupervoxels_;
  int VisualizeNormals_;
  int VisualizeGraphs_;

  //time measuring

  std::chrono::time_point<std::chrono::high_resolution_clock> start_general_time;
  int frame_count;

  //saving parameters

  bool save_maps_to_file = false;
  ofstream myfile;

private:
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  message_filters::Subscriber<sensor_msgs::Image> *rgb_subscriber_;
  message_filters::Subscriber<sensor_msgs::Image> *depth_subscriber_;
  message_filters::Synchronizer<sync_pol> *sync_;
};

#endif //ORBSLAM2_ROS_RGBDODE_H_
