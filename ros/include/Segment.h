#include "IncludesAndDefinitions.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#ifndef SEGMENT_
#define SEGMENT_

namespace DEF_OBJ_TRACK
{

    class Segment
    {
    public:
        enum SegmentType
        {
            TARGET_OBJECT = 0,          //The object I want to track
            INTERACTING_ELEMENT = 1,    //When it is inside the objects interacting zone (i.e. OBB)
            OCCLUDING_ELEMENT = 2,      //When it's depth (camera reference) is smaller than the furthest point of the target object
            HARD_OCCLUDING_ELEMENT = 3, //When is in the frustum defined by the OBB and the camera centre
        };
        SegmentType segmentType_;

    public:
        Segment();
        ~Segment();
        Segment(pcl::PointCloud<PointTSuperVoxel>::Ptr selected_cloud_with_labeled_sv, float neighbouring_radius, float exploration_radius, Segment::SegmentType typeOfSegment);

        //Computing global charactersitics
        void computeFeatureExtraction();
        //Utilities
        pcl::PointXYZRGBA computeCentroidCoordInObjectsReference(pcl::PointXYZRGBA SVcentroid);
        void computeInterestfrustum(pcl::PointCloud<pcl::PointNormal>::Ptr visualization_normal_cloud);

        bool isItInsideAugmentedOBB(const pcl::PointXYZRGBA &SVcentroid_on_objects_ref);

        //bool isItInCameraToObjectFrustum(pcl::PointXYZRGBA sv_centroid);

        //Best next view calculations
        void computeAverageNormal();
        void computeDesiredPositionVector();

        //Error and optimization parameter calculations
        void computeInverseOfCameraPositionAndExtendedIntrinsics();
        void computeCameraZaxisOnWorldReference();
        void computeAngularError();
        void computePositionError();

        //computes octree, neighbourhoods and node degrees

        void computeOctreeAdjacencyAndDegree(pcl::visualization::PCLVisualizer::Ptr viewer);

        //adjacency visualization

        void addSupervoxelConnectionsToViewer2(pcl::PointXYZ &supervoxel_center,
                                               pcl::PointCloud<pcl::PointXYZ> &adjacent_supervoxel_centers,
                                               std::string supervoxel_name, pcl::visualization::PCLVisualizer::Ptr viewer);

        //TO-DO: ADD DOES IT BELONG TO frustum FUNCTION

        void computeLargestDistanceToCamera(pcl::PointCloud<pcl::PointNormal>::Ptr visualization_normal_cloud);
        bool isItInOcclusionsDepthRange(pcl::PointXYZRGBA sv_centroid);
        bool isItInCameraToObjectFrustum(pcl::PointXYZRGBA sv_centroid);
        bool isItInPerceptionSphere(pcl::PointXYZRGBA sv_centroid, const double &sphere_radius);
        void visualizeObjectsReprojectionOnCamera(std::shared_ptr<DEF_OBJ_TRACK::Segment> occlusions, std::shared_ptr<DEF_OBJ_TRACK::Segment> hardOcclusions);

        //General visualization
        void OcclusionVisualChecking(std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject, Eigen::Affine3f t_objetivo, pcl::visualization::PCLVisualizer::Ptr viewer);
        void VisualizeCloudsAndBoundingBoxes(pcl::visualization::PCLVisualizer::Ptr viewer);

        //Next Best View with occlusions
        std::map<uint32_t, Eigen::Matrix<float, 3, 1>> normalsToSphereIntersectionPoints(pcl::visualization::PCLVisualizer::Ptr viewer, const double &sphere_radius);
        std::map<uint32_t, Eigen::Matrix<float, 3, 1>> centroidsToOcclussorRays(pcl::visualization::PCLVisualizer::Ptr viewer, const double &sphere_radius, std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject);

    public:
        //Main structure of a Segment

        //Max value of sv_label_ = nº of SVs in the segment. Use sv_label_ to access each SV
        uint32_t number_of_sv_in_segment_;
        uint32_t label_of_sv_;

        //Pointcloud of the Segment, don't use its labels as a reference since they wont correspond the ones of the calss Segment
        pcl::PointCloud<pcl::PointXYZL>::Ptr segments_voxel_labeled_cloud_; //previously selected_cloud_with_labeled_sv_
        pcl::PointCloud<PointTSuperVoxel>::Ptr selected_cloud_with_labeled_sv_;

        //Maps that storea the  Segment's Supervoxels. RGB and normal information is already in each sv, but having separated maps is more clear
        std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr> segments_sv_map_;
        std::map<uint32_t, pcl::PointXYZRGBA> segments_colors_RGB_;
        std::map<uint32_t, pcl::PointNormal> segments_normals_;                   //Each normalPoint also stores the SV's centroid information, starting and ending point of the normal vector
        std::map<uint32_t, pcl::PointNormal> segment_normals_on_world_reference_; //point= world referenc origin, normal= vector in world reference

        enum SegmentState
        {
            LOST = 0,     //Nothing can be done, need to go back to lccp segmentation of the scene
            VISIBLE = 1,  //When it is fully or partially visible
            OCCLUDED = 2, //When it is completely occluded but re-location tasks are running
        };

        SegmentState segmentState_;

    public:
        //Segment's global charactersitics information

        pcl::MomentOfInertiaEstimation<PointTSuperVoxel> feature_extractor_;
        std::vector<float> moment_of_inertia_;
        std::vector<float> eccentricity_;
        PointTSuperVoxel min_point_AABB_;
        PointTSuperVoxel max_point_AABB_;
        PointTSuperVoxel min_point_OBB_;
        PointTSuperVoxel max_point_OBB_;
        PointTSuperVoxel position_OBB_;
        Eigen::Matrix3f rotational_matrix_OBB_;
        float major_value_, middle_value_, minor_value_;
        Eigen::Vector3f major_vector_, middle_vector_, minor_vector_;
        Eigen::Vector3f mass_center_;
        float tolX_, tolY_, tolZ_;
        float tol1X_, tol1Y_, tol1Z_;

    public:
        float neighbouring_radius_; //related to Rseed*(1...1.5)
        float exploration_radius_;  //related to Rseed*(1...2) > neighbouring_radius_

        //Octree for adjacency and R-search
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr Segments_sv_AdjacencyOctree_;

        std::map<uint32_t, uint32_t> nodes_degrees_;
        std::map<uint32_t, std::vector<int>> nodes_neighbours_;
        pcl::PointCloud<pcl::PointXYZ>::Ptr adjacency_cloud;
        int max_graph_degree_;

    public:
        //Optimization of next best view
        pcl::PointNormal average_normal_, average_normal_global_reference_;
        //Average Normal Visualization cloud
        pcl::PointCloud<pcl::PointNormal>::Ptr visualization_of_vectors_cloud_;
        //Desired position of object's perception information
        float desired_distance_from_object_ = 0.3; //m
        float inv_distance_from_object_ = 1 / desired_distance_from_object_;
        float error_alfa_, error_beta_, position_error_;

        Eigen::Affine3f optimal_position_;

    public:
        //Camera intrinsics and extrinsics information
        cv::Mat camera_position_, camera_intrinsics_, camera_intrinsics_extended_; //T world->camera
        cv::Mat Twc_depth_;                                                        //T camera->world
        pcl::PointNormal camera_z_axis_on_world_reference_;

    public:
        //Occlusion information
        pcl::PointNormal position_vector_;
        pcl::PointCloud<pcl::PointNormal>::Ptr position_vector_normal_cloud_;

        float max_distance_camera_to_object_;

        std::map<uint32_t, pcl::PointXYZ> camera_to_object_frustum_; //stores the 4 corners (world referenced) of the middle plane of the augmented OBB
        std::map<uint32_t, cv::Mat_<float>> camera_to_object_frustum_image_coord_;

    private:
        //visualization parameters
        float cubeSize = 0.004;
        bool visualize_main_inertia_axes = false;
        bool visualize_rgba_pointcloud = true;
        bool show_normals = true;
        bool show_adjacency = false;

    public:
        //Occlusion handling
    };

} // namespace DEF_OBJ_TRACK

#endif //SEGMENT_