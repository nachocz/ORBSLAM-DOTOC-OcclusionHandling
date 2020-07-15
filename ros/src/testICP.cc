
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
//PCL

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/visualization/mouse_event.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <algorithm>
#include <fstream>
#include <chrono>

bool next_iteration = false;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                           void *nothing)
{
  if (event.getKeySym() == "space" && event.keyDown())
    next_iteration = true;
}

int frame_count = 1;
int frame_step = 1;
float waiting_time = 150;
float point_visualization_size = 3;

int main(int argc, char **argv)
{
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
  viewer->setBackgroundColor(0.5, 0.5, 0.5);

  // Register keyboard callback :
  viewer->registerKeyboardCallback(&keyboardEventOccurred, (void *)NULL);

  while (1)
  {
    viewer->spinOnce();

    // The user pressed "space" :
    if (next_iteration)
    {
      viewer->removeAllPointClouds();

      std::stringstream ss, ss2, ssfinal;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);

      char frame_filename_cloud_initial_state[1000];
      char frame_filename_cloud_1[1000];
      char frame_filename_cloud_2[1000];

      sprintf(frame_filename_cloud_initial_state, "/home/pc-campero2/SavedClouds/%06d.txt", 2);            //Cloud1 is the previous state cloud
      sprintf(frame_filename_cloud_1, "/home/pc-campero2/SavedClouds/%06d.txt", 1); //Cloud1 is the previous state cloud
      sprintf(frame_filename_cloud_2, "/home/pc-campero2/SavedClouds/%06d.txt", frame_count);              //Cloud2 is the actual state cloud
      pcl::io::loadPCDFile<pcl::PointXYZRGBA>(frame_filename_cloud_1, *cloud);

      if ((pcl::io::loadPCDFile<pcl::PointXYZRGBA>(frame_filename_cloud_2, *cloud2) == -1)) //* load the file
      {
        PCL_ERROR("Couldn't read file  \n");
        return (-1);
      }

      // std::cout << "Loaded "
      //           << cloud2->width * cloud2->height
      //           << " data points from " << frame_filename_cloud_2 << std::endl;

      std::cout << "FRAME: " << frame_count << std::endl;

      auto start = std::chrono::system_clock::now(); //measuring fps

      pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
      icp.setInputSource(cloud);
      icp.setInputTarget(cloud2);
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Final(new pcl::PointCloud<pcl::PointXYZRGBA>);
      icp.align(*Final);
      // std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      // std::cout << icp.getFinalTransformation() << std::endl;

      auto end = std::chrono::system_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      float totalMilliseconds = elapsed.count();

      std::ostringstream txt_fps;
      txt_fps << "ICP time [ms]: " << totalMilliseconds;
      std::cout << txt_fps.str() << std::endl;

      ss << "cloud_" << frame_count - 1;
      ss2 << "cloud2_" << frame_count;
      ssfinal << "final_" << frame_count - 1;

      // non aligned previous state is red
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> cloud_icp_color_h(cloud, 180, 20, 20);
      viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, cloud_icp_color_h, ss.str());
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_visualization_size, ss.str());

      //actual state is RGB
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb2(cloud2);
      viewer->addPointCloud<pcl::PointXYZRGBA>(cloud2, rgb2, ss2.str());
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_visualization_size, ss2.str());

      viewer->addCoordinateSystem(0.01);
      // viewer->addCoordinateSystem(0.01, t, "ref_objetivo");

      //viewer->spinOnce(waiting_time);

      //Corrected previous state position is green

      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> cloud_tr_color_h(Final, 20, 180, 20);
      viewer->addPointCloud(Final, cloud_tr_color_h, ssfinal.str());
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_visualization_size, ssfinal.str());

      //viewer->spinOnce(waiting_time);

      //viewer->removePointCloud(ss.str());
      //viewer->spinOnce(waiting_time);

      frame_count = frame_count + frame_step;
    }
    next_iteration = false;
  }

  return (0);
}