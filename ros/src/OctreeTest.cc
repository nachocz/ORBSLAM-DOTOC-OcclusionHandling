#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>

#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>

#include <iostream>
#include <vector>
#include <ctime>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

void addSupervoxelConnectionsToViewer(PointT &supervoxel_center,
                                      PointCloudT &adjacent_supervoxel_centers,
                                      std::string supervoxel_name,
                                      pcl::visualization::PCLVisualizer::Ptr &viewer);

int main(int argc, char **argv)
{

    srand((unsigned int)time(NULL));

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate pointcloud data
    cloud->width = 10;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    for (std::size_t i = 0; i < cloud->points.size(); ++i)
    {
        cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);

        cout << cloud->points[i].x << endl;
    }
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    auto t1 = Clock::now();

    viewer->addPointCloud(cloud, "labeled voxels");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 3.0, "labeled voxels");

    float resolution = 128.0f;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);

    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    pcl::PointXYZ searchPoint;

    searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    float radius = 250;
    for (int j = 0; j < cloud->width; j++)
    {
        if (octree.radiusSearch(cloud->points[j], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            PointCloudT adjacent_supervoxel_centers;

            std::stringstream ss;
            ss << "supervoxel_" << j;
            cout << "-----------------------------------" << endl;
            cout << ss.str() << endl;

            for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
            {
                // std::cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
                //           << " " << cloud->points[pointIdxRadiusSearch[i]].y
                //           << " " << cloud->points[pointIdxRadiusSearch[i]].z
                //           << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;

                adjacent_supervoxel_centers.push_back(cloud->points[pointIdxRadiusSearch[i]]);

                cout << pointIdxRadiusSearch[i] << endl;
            }
            addSupervoxelConnectionsToViewer(cloud->points[j], adjacent_supervoxel_centers, ss.str(), viewer);
        }
    }

    auto t2 = Clock::now();
    auto tmicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout
        << "Delta t2-t1: " << tmicroseconds << " microseconds" << std::endl;

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }
    return (0);
}

void addSupervoxelConnectionsToViewer(PointT &supervoxel_center,
                                      PointCloudT &adjacent_supervoxel_centers,
                                      std::string supervoxel_name,
                                      pcl::visualization::PCLVisualizer::Ptr &viewer)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

    //Iterate through all adjacent points, and add a center point to adjacent point pair
    for (auto adjacent_itr = adjacent_supervoxel_centers.begin(); adjacent_itr != adjacent_supervoxel_centers.end(); ++adjacent_itr)
    {
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
}

// // Neighbors within voxel search

// std::vector<int> pointIdxVec;

// if (octree.voxelSearch(searchPoint, pointIdxVec))
// {
//     std::cout << "Neighbors within voxel search at (" << searchPoint.x
//               << " " << searchPoint.y
//               << " " << searchPoint.z << ")"
//               << std::endl;

//     for (std::size_t i = 0; i < pointIdxVec.size(); ++i)
//         std::cout << "    " << cloud->points[pointIdxVec[i]].x
//                   << " " << cloud->points[pointIdxVec[i]].y
//                   << " " << cloud->points[pointIdxVec[i]].z << std::endl;
// }

// // K nearest neighbor search

// int K = 10;

// std::vector<int> pointIdxNKNSearch;
// std::vector<float> pointNKNSquaredDistance;

// std::cout << "K nearest neighbor search at (" << searchPoint.x
//           << " " << searchPoint.y
//           << " " << searchPoint.z
//           << ") with K=" << K << std::endl;

// if (octree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
// {
//     for (std::size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
//         std::cout << "    " << cloud->points[pointIdxNKNSearch[i]].x
//                   << " " << cloud->points[pointIdxNKNSearch[i]].y
//                   << " " << cloud->points[pointIdxNKNSearch[i]].z
//                   << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
// }

// Neighbors within radius search
