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

#ifndef ORBSLAM2_ROS_NODE_H_
#define ORBSLAM2_ROS_NODE_H_

#include <vector>
#include <ros/ros.h>
#include <ros/time.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include <dynamic_reconfigure/server.h>
#include <orb_slam2_ros/dynamic_reconfigureConfig.h>

#include "orb_slam2_ros/SaveMap.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

#include "System.h"

class Node
{
public:
  Node(ORB_SLAM2::System::eSensor sensor, ros::NodeHandle &node_handle, image_transport::ImageTransport &image_transport);
  ~Node();

  bool publish_densemap_on_ros = false;

  std::string getDOTsettingsFileNameParam();

protected:
  void Update();

  ORB_SLAM2::System *orb_slam_;

  ros::Time current_frame_time_;

private:
  void PublishMapPoints(std::vector<ORB_SLAM2::MapPoint *> map_points);
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------
  sensor_msgs::PointCloud2 PublishDepthMapPoints(const std::map<ORB_SLAM2::MapPoint *, cv::Vec3f> &depthmap_points);
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------
  void PublishPositionAsTransform(cv::Mat position);
  void PublishPositionAsPoseStamped(cv::Mat position);
  void PublishRenderedImage(cv::Mat image);
  void ParamsChangedCallback(orb_slam2_ros::dynamic_reconfigureConfig &config, uint32_t level);
  bool SaveMapSrv(orb_slam2_ros::SaveMap::Request &req, orb_slam2_ros::SaveMap::Response &res);

  tf::Transform TransformFromMat(cv::Mat position_mat);
  sensor_msgs::PointCloud2 MapPointsToPointCloud(std::vector<ORB_SLAM2::MapPoint *> map_points);
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------
  sensor_msgs::PointCloud2 DepthMapPointsToPointCloud(const std::map<ORB_SLAM2::MapPoint *, cv::Vec3f> &depthmap_points);
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------

  dynamic_reconfigure::Server<orb_slam2_ros::dynamic_reconfigureConfig> dynamic_param_server_;

  image_transport::Publisher rendered_image_publisher_;
  ros::Publisher map_points_publisher_;
  ros::Publisher pose_publisher_;
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------
  ros::Publisher depthmap_points_publisher_;
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------

  ros::ServiceServer service_server_;

  std::string name_of_node_;
  ros::NodeHandle node_handle_;

  std::string map_frame_id_param_;
  std::string camera_frame_id_param_;
  std::string map_file_name_param_;
  std::string voc_file_name_param_;
  std::string settings_file_name_param_;
  std::string DOT_settings_file_name_param_;
  bool load_map_param_;
  bool publish_pointcloud_param_;
  bool publish_pose_param_;
  bool publish_depthpointcloud_param_;
  int min_observations_per_point_;
};

#endif //ORBSLAM2_ROS_NODE_H_
