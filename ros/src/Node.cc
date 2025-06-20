#include "Node.h"

#include <iostream>

Node::Node(ORB_SLAM2::System::eSensor sensor, ros::NodeHandle &node_handle, image_transport::ImageTransport &image_transport)
{
  name_of_node_ = ros::this_node::getName();
  node_handle_ = node_handle;
  min_observations_per_point_ = 2;

  //static parameters
  node_handle_.param(name_of_node_ + "/publish_pointcloud", publish_pointcloud_param_, true);
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------
  node_handle_.param(name_of_node_ + "/publish_depthpointcloud", publish_depthpointcloud_param_, true);
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------
  node_handle_.param(name_of_node_ + "/publish_pose", publish_pose_param_, true);
  node_handle_.param<std::string>(name_of_node_ + "/pointcloud_frame_id", map_frame_id_param_, "map");
  node_handle_.param<std::string>(name_of_node_ + "/camera_frame_id", camera_frame_id_param_, "camera_link");
  node_handle_.param<std::string>(name_of_node_ + "/map_file", map_file_name_param_, "map.bin");
  node_handle_.param<std::string>(name_of_node_ + "/voc_file", voc_file_name_param_, "file_not_set");
  node_handle_.param<std::string>(name_of_node_ + "/settings_file", settings_file_name_param_, "file_not_set");
  node_handle_.param<std::string>(name_of_node_ + "/DOT_settings_file", DOT_settings_file_name_param_, "file_not_set");

  node_handle_.param(name_of_node_ + "/load_map", load_map_param_, false);

  orb_slam_ = new ORB_SLAM2::System(voc_file_name_param_, settings_file_name_param_, sensor, map_file_name_param_, load_map_param_);

  service_server_ = node_handle_.advertiseService(name_of_node_ + "/save_map", &Node::SaveMapSrv, this);

  //Setup dynamic reconfigure
  dynamic_reconfigure::Server<orb_slam2_ros::dynamic_reconfigureConfig>::CallbackType dynamic_param_callback;
  dynamic_param_callback = boost::bind(&Node::ParamsChangedCallback, this, _1, _2);
  dynamic_param_server_.setCallback(dynamic_param_callback);

  rendered_image_publisher_ = image_transport.advertise(name_of_node_ + "/debug_image", 1);
  if (publish_pointcloud_param_)
  {
    map_points_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(name_of_node_ + "/map_points", 1);
  }

  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------
  if (publish_depthpointcloud_param_)
  {
    depthmap_points_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(name_of_node_ + "/depthmap_points", 1);
  }
  //------------------------------------------------ROS DensePointCloud Topic Publishing-----------------------------------

  // Enable publishing camera's pose as PoseStamped message
  if (publish_pose_param_)
  {
    pose_publisher_ = node_handle_.advertise<geometry_msgs::PoseStamped>(name_of_node_ + "/pose", 1);
  }
}

Node::~Node()
{
  // Stop all threads
  orb_slam_->Shutdown();

  // Save camera trajectory
  orb_slam_->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  delete orb_slam_;
}

void Node::Update()
{
  cv::Mat position = orb_slam_->GetCurrentPosition();

  if (!position.empty())
  {
    PublishPositionAsTransform(position);

    if (publish_pose_param_)
    {
      PublishPositionAsPoseStamped(position);
    }
  }

  PublishRenderedImage(orb_slam_->DrawCurrentFrame());

  if (publish_pointcloud_param_)
  {
    PublishMapPoints(orb_slam_->GetAllMapPoints());
  }

  //------------------------Dense map publication------------------------

  if (publish_densemap_on_ros)
  {
    sensor_msgs::PointCloud2 depthcloud = PublishDepthMapPoints(orb_slam_->GetAllDepthMapPoints());
  }
  //------------------------Dense map publication------------------------
}

void Node::PublishMapPoints(std::vector<ORB_SLAM2::MapPoint *> map_points)
{
  sensor_msgs::PointCloud2 cloud = MapPointsToPointCloud(map_points);
  map_points_publisher_.publish(cloud);
}

//------------------------Dense map publication------------------------
sensor_msgs::PointCloud2 Node::PublishDepthMapPoints(const std::map<ORB_SLAM2::MapPoint *, cv::Vec3f> &depthmap_points)
{
  sensor_msgs::PointCloud2 depthcloud = DepthMapPointsToPointCloud(depthmap_points);
  depthmap_points_publisher_.publish(depthcloud);
  return depthcloud;
}
//------------------------Dense map publication------------------------

void Node::PublishPositionAsTransform(cv::Mat position)
{
  tf::Transform transform = TransformFromMat(position);
  static tf::TransformBroadcaster tf_broadcaster;
  tf_broadcaster.sendTransform(tf::StampedTransform(transform, current_frame_time_, map_frame_id_param_, camera_frame_id_param_));
}

void Node::PublishPositionAsPoseStamped(cv::Mat position)
{
  tf::Transform grasp_tf = TransformFromMat(position);
  tf::Stamped<tf::Pose> grasp_tf_pose(grasp_tf, current_frame_time_, map_frame_id_param_);
  geometry_msgs::PoseStamped pose_msg;
  tf::poseStampedTFToMsg(grasp_tf_pose, pose_msg);
  pose_publisher_.publish(pose_msg);
}

void Node::PublishRenderedImage(cv::Mat image)
{
  std_msgs::Header header;
  header.stamp = current_frame_time_;
  header.frame_id = map_frame_id_param_;
  const sensor_msgs::ImagePtr rendered_image_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  rendered_image_publisher_.publish(rendered_image_msg);
}

tf::Transform Node::TransformFromMat(cv::Mat position_mat)
{
  cv::Mat rotation(3, 3, CV_32F);
  cv::Mat translation(3, 1, CV_32F);

  rotation = position_mat.rowRange(0, 3).colRange(0, 3);
  translation = position_mat.rowRange(0, 3).col(3);

  tf::Matrix3x3 tf_camera_rotation(rotation.at<float>(0, 0), rotation.at<float>(0, 1), rotation.at<float>(0, 2),
                                   rotation.at<float>(1, 0), rotation.at<float>(1, 1), rotation.at<float>(1, 2),
                                   rotation.at<float>(2, 0), rotation.at<float>(2, 1), rotation.at<float>(2, 2));

  tf::Vector3 tf_camera_translation(translation.at<float>(0), translation.at<float>(1), translation.at<float>(2));

  //Coordinate transformation matrix from orb coordinate system to ros coordinate system
  const tf::Matrix3x3 tf_orb_to_ros(0, 0, 1,
                                    -1, 0, 0,
                                    0, -1, 0);

  //Transform from orb coordinate system to ros coordinate system on camera coordinates
  tf_camera_rotation = tf_orb_to_ros * tf_camera_rotation;
  tf_camera_translation = tf_orb_to_ros * tf_camera_translation;

  //Inverse matrix
  tf_camera_rotation = tf_camera_rotation.transpose();
  tf_camera_translation = -(tf_camera_rotation * tf_camera_translation);

  //Transform from orb coordinate system to ros coordinate system on map coordinates
  tf_camera_rotation = tf_orb_to_ros * tf_camera_rotation;
  tf_camera_translation = tf_orb_to_ros * tf_camera_translation;

  return tf::Transform(tf_camera_rotation, tf_camera_translation);
}

sensor_msgs::PointCloud2 Node::MapPointsToPointCloud(std::vector<ORB_SLAM2::MapPoint *> map_points)
{
  if (map_points.size() == 0)
  {
    std::cout << "Map point vector is empty!" << std::endl;
  }

  sensor_msgs::PointCloud2 cloud;

  const int num_channels = 3; // x y z

  cloud.header.stamp = current_frame_time_;
  cloud.header.frame_id = map_frame_id_param_;
  cloud.height = 1;
  cloud.width = map_points.size();
  cloud.is_bigendian = false;
  cloud.is_dense = true;
  cloud.point_step = num_channels * sizeof(float);
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.fields.resize(num_channels);

  std::string channel_id[] = {"x", "y", "z"};
  for (int i = 0; i < num_channels; i++)
  {
    cloud.fields[i].name = channel_id[i];
    cloud.fields[i].offset = i * sizeof(float);
    cloud.fields[i].count = 1;
    cloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
  }

  cloud.data.resize(cloud.row_step * cloud.height);

  unsigned char *cloud_data_ptr = &(cloud.data[0]);

  float data_array[num_channels];
  for (unsigned int i = 0; i < cloud.width; i++)
  {
    if (map_points.at(i)->nObs >= min_observations_per_point_)
    {
      data_array[0] = map_points.at(i)->GetWorldPos().at<float>(2);        //x. Do the transformation by just reading at the position of z instead of x
      data_array[1] = -1.0 * map_points.at(i)->GetWorldPos().at<float>(0); //y. Do the transformation by just reading at the position of x instead of y
      data_array[2] = -1.0 * map_points.at(i)->GetWorldPos().at<float>(1); //z. Do the transformation by just reading at the position of y instead of z
      //TODO dont hack the transformation but have a central conversion function for MapPointsToPointCloud and TransformFromMat

      memcpy(cloud_data_ptr + (i * cloud.point_step), data_array, num_channels * sizeof(float));
    }
  }

  return cloud;
}

//--------------------------------------------------------Comienza el Salseo del bueno-------------------------------------------------------
sensor_msgs::PointCloud2 Node::DepthMapPointsToPointCloud(const std::map<ORB_SLAM2::MapPoint *, cv::Vec3f> &depthmap_points)
{
  if (depthmap_points.size() == 0)
  {
    std::cout << "DepthMap point vector is empty!" << std::endl;
  }

  //std::cout << "PARADA 3" << std::endl;

  sensor_msgs::PointCloud2 depthcloud;

  const int num_channels = 6; // x y z --------- r g b

  depthcloud.header.stamp = current_frame_time_;
  depthcloud.header.frame_id = map_frame_id_param_;
  depthcloud.height = 1;
  depthcloud.width = depthmap_points.size();

  //std::cout<< "depthcloudwith (depthmapPointsSize): "<<depthcloud.width<<std::endl;
  depthcloud.is_bigendian = false;
  depthcloud.is_dense = true;
  depthcloud.point_step = num_channels * sizeof(float);
  depthcloud.row_step = depthcloud.point_step * depthcloud.width;
  depthcloud.fields.resize(num_channels);

  std::string channel_id[] = {"x", "y", "z", "r", "g", "b"};
  for (int i = 0; i < (num_channels - 3); i++)
  {
    depthcloud.fields[i].name = channel_id[i];
    depthcloud.fields[i].offset = i * sizeof(float);
    depthcloud.fields[i].count = 1;
    depthcloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
  }
  //TODO hacer esto un poco más limpio...
  for (int i = 3; i < num_channels; i++)
  {
    depthcloud.fields[i].name = channel_id[i];
    depthcloud.fields[i].offset = i * sizeof(float);
    depthcloud.fields[i].count = 1;
    depthcloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
  }

  depthcloud.data.resize(depthcloud.row_step * depthcloud.height);

  unsigned char *depthcloud_data_ptr = &(depthcloud.data[0]);

  float data_array_2[num_channels];

  cv::Vec3f color_dense;
  ORB_SLAM2::MapPoint *MP;

  int i = 0;

  for (std::map<ORB_SLAM2::MapPoint *, cv::Vec3f>::const_iterator it = depthmap_points.begin(); it != depthmap_points.end(); ++it)
  {

    MP = (it->first);
    color_dense = (it->second);
    cv::Mat pos = MP->GetWorldPos();

    //std::cout << "Pos: " <<  pos << std::endl;

    data_array_2[0] = pos.at<float>(2);
    data_array_2[1] = -1.0 * pos.at<float>(0);
    data_array_2[2] = -1.0 * pos.at<float>(1);

    data_array_2[3] = color_dense[0] / 255;
    data_array_2[4] = color_dense[1] / 255;
    data_array_2[5] = color_dense[2] / 255;

    memcpy(depthcloud_data_ptr + (i * depthcloud.point_step), data_array_2, num_channels * sizeof(float));

    i++;
  }

  //std::cout << "Parada 3.3:Fuera  bucle" << std::endl;
  return depthcloud;
}
//--------------------------------------------------------Fin del Salseo del bueno-------------------------------------------------------

void Node::ParamsChangedCallback(orb_slam2_ros::dynamic_reconfigureConfig &config, uint32_t level)
{
  orb_slam_->EnableLocalizationOnly(config.localize_only);
  min_observations_per_point_ = config.min_observations_for_ros_map;

  if (config.reset_map)
  {
    orb_slam_->Reset();
    config.reset_map = false;
  }

  orb_slam_->SetMinimumKeyFrames(config.min_num_kf_in_map);
}

bool Node::SaveMapSrv(orb_slam2_ros::SaveMap::Request &req, orb_slam2_ros::SaveMap::Response &res)
{
  res.success = orb_slam_->SaveMap(req.name);

  if (res.success)
  {
    ROS_INFO_STREAM("Map was saved as " << req.name);
  }
  else
  {
    ROS_ERROR("Map could not be saved.");
  }

  return res.success;
}

std::string Node::getDOTsettingsFileNameParam()
{
  return DOT_settings_file_name_param_;
}