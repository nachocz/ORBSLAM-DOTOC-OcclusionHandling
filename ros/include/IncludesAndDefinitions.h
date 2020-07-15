

#ifndef PARAMETERS_H
#define PARAMETERS_H

//Operations

#define PI 3.14159265
#define SQR(x) ((x) * (x))
#define POW2(x) SQR(x)
#define POW3(x) ((x) * (x) * (x))
#define POW4(x) (POW2(x) * POW2(x))
#define POW7(x) (POW3(x) * POW3(x) * (x))
#define DegToRad(x) ((x)*M_PI / 180)
#define RadToDeg(x) ((x) / M_PI * 180)

//General
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>

//Shared pointers

#include <memory>

//Opencv

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp> //draw circles
#include <opencv2/highgui.hpp> //imshow... etc

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

#include <pcl/visualization/pcl_plotter.h>

//octree
#include <pcl/octree/octree_search.h>

//fpfh
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <vector>

//NARF
#include <pcl/io/pcd_io.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/visualization/range_image_visualizer.h>

// VTK
#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkImageData.h>
#include <vtkImageFlip.h>
#include <vtkPolyLine.h>

typedef pcl::PointXYZ PointType;

typedef pcl::PointXYZRGBA PointTSuperVoxel;
typedef pcl::PointCloud<PointTSuperVoxel> PointCloudTSuperVoxel;
typedef pcl::PointNormal PointNTSuperVoxel;
typedef pcl::PointCloud<PointNTSuperVoxel> PointNCloudTSuperVoxel;
typedef pcl::PointXYZL PointLTSuperVoxel;
typedef pcl::PointCloud<PointLTSuperVoxel> PointLCloudTSuperVoxel;

typedef pcl::PointXYZRGBA PointT; // The point type used for input
typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;

typedef pcl::PointXYZRGBA PointT; // The point type used for input
typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;
typedef pcl::LCCPSegmentation<PointT>::VertexIterator VertexIterator;
typedef pcl::LCCPSegmentation<PointT>::AdjacencyIterator AdjacencyIterator;
typedef pcl::LCCPSegmentation<PointT>::EdgeID EdgeID;

#endif //PARAMETERS_H