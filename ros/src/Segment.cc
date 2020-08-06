#include "Segment.h"
namespace DEF_OBJ_TRACK {

Segment::Segment(/* args */) {}

Segment::~Segment() {}

Segment::Segment(
    pcl::PointCloud<PointTSuperVoxel>::Ptr selected_cloud_with_labeled_sv,
    float neighbouring_radius, float exploration_radius,
    Segment::SegmentType typeOfSegment)
    : selected_cloud_with_labeled_sv_(selected_cloud_with_labeled_sv),
      neighbouring_radius_(neighbouring_radius),
      exploration_radius_(exploration_radius), label_of_sv_(0),
      segmentType_(typeOfSegment) {
  // Computing global charactersitics
  computeFeatureExtraction();
}
// FEATURE EXTRACTION
void Segment::computeFeatureExtraction() {
  if (segmentType_ == TARGET_OBJECT) {
    feature_extractor_.setInputCloud(selected_cloud_with_labeled_sv_);
    feature_extractor_.compute();

    feature_extractor_.getMomentOfInertia(moment_of_inertia_);
    feature_extractor_.getEccentricity(eccentricity_);
    feature_extractor_.getAABB(min_point_AABB_, max_point_AABB_);
    feature_extractor_.getOBB(min_point_OBB_, max_point_OBB_, position_OBB_,
                              rotational_matrix_OBB_);
    feature_extractor_.getEigenValues(major_value_, middle_value_,
                                      minor_value_);
    feature_extractor_.getEigenVectors(major_vector_, middle_vector_,
                                       minor_vector_);
    feature_extractor_.getMassCenter(mass_center_);
    tolX_ = -0.25 * (max_point_OBB_.x - min_point_OBB_.x);
    tolY_ = -0.25 * (max_point_OBB_.y - min_point_OBB_.y);
    tolZ_ = -0.25 * (max_point_OBB_.z - min_point_OBB_.z);
    tol1X_ = exploration_radius_ / 2;
    tol1Y_ = exploration_radius_ / 2;
    tol1Z_ = exploration_radius_ / 2;
  }

  number_of_sv_in_segment_ = label_of_sv_;
}
// AVERAGE NORMAL CALCULATIONS
pcl::PointXYZRGBA
Segment::computeCentroidCoordInObjectsReference(pcl::PointXYZRGBA SVcentroid) {
  pcl::PointXYZRGBA SVcentroid_on_objects_ref;

  SVcentroid_on_objects_ref.x =
      (SVcentroid.x - mass_center_(0)) * major_vector_(0) +
      (SVcentroid.y - mass_center_(1)) * major_vector_(1) +
      (SVcentroid.z - mass_center_(2)) * major_vector_(2);
  SVcentroid_on_objects_ref.y =
      (SVcentroid.x - mass_center_(0)) * middle_vector_(0) +
      (SVcentroid.y - mass_center_(1)) * middle_vector_(1) +
      (SVcentroid.z - mass_center_(2)) * middle_vector_(2);
  SVcentroid_on_objects_ref.z =
      (SVcentroid.x - mass_center_(0)) * minor_vector_(0) +
      (SVcentroid.y - mass_center_(1)) * minor_vector_(1) +
      (SVcentroid.z - mass_center_(2)) * minor_vector_(2);

  return SVcentroid_on_objects_ref;
}

void Segment::computeAverageNormal() {
  pcl::PointNormal object_normal;
  std::map<uint32_t, pcl::PointNormal>::iterator itr;
  for (itr = segments_normals_.begin(); itr != segments_normals_.end(); ++itr) {
    object_normal = segments_normals_.at(itr->first);

    average_normal_.x += object_normal.x;
    average_normal_.y += object_normal.y;
    average_normal_.z += object_normal.z;

    average_normal_.normal_x += object_normal.normal_x;
    average_normal_.normal_y += object_normal.normal_y;
    average_normal_.normal_z += object_normal.normal_z;
  }
  average_normal_.x /= label_of_sv_;
  average_normal_.y /= label_of_sv_;
  average_normal_.z /= label_of_sv_;

  average_normal_.normal_x /= label_of_sv_;
  average_normal_.normal_y /= label_of_sv_;
  average_normal_.normal_z /= label_of_sv_;
}

void Segment::computeDesiredPositionVector() {
  float avgModule =
      sqrt(((average_normal_.normal_x) * (average_normal_.normal_x)) +
           ((average_normal_.normal_y) * (average_normal_.normal_y)) +
           ((average_normal_.normal_z) * (average_normal_.normal_z)));

  average_normal_.normal_x /= (inv_distance_from_object_ * avgModule);
  average_normal_.normal_y /= (inv_distance_from_object_ * avgModule);
  average_normal_.normal_z /= (inv_distance_from_object_ * avgModule);

  average_normal_.normal_x = mass_center_(0) + average_normal_.normal_x;
  average_normal_.normal_y = mass_center_(1) + average_normal_.normal_y;
  average_normal_.normal_z = mass_center_(2) + average_normal_.normal_z;
}

void Segment::computeAngularError() {

  cv::Mat_<float> centroid_position(4, 1);
  cv::Mat_<float> image_plane_coord(3, 1);

  centroid_position(0, 0) = mass_center_(0); //
  centroid_position(1) = mass_center_(1);
  centroid_position(2) = mass_center_(2);
  centroid_position(3) = 1;

  float fx = camera_intrinsics_extended_.at<float>(0, 0);
  float fy = camera_intrinsics_extended_.at<float>(1, 1);
  float cx = camera_intrinsics_extended_.at<float>(0, 2);
  float cy = camera_intrinsics_extended_.at<float>(1, 2);

  image_plane_coord =
      camera_intrinsics_extended_ * camera_position_ *
      centroid_position; // reprojectiong 3D points from RGB camera center ref.
                         // to the RGB camera projection plane
  image_plane_coord /= image_plane_coord.at<float>(2, 0);

  error_alfa_ = atan((image_plane_coord(0, 0) - cx) / fx) * 180 / PI;
  error_beta_ = atan((image_plane_coord(1, 0) - cy) / fy) * 180 / PI;
}

void Segment::computePositionError() {
  cv::Mat invPosition = camera_position_.inv();
  Eigen::Affine3f t_objetivo;
  // for (int iAffine = 0; iAffine < 3; iAffine++)
  // {
  //     for (int jAffine = 0; jAffine < 3; jAffine++)
  //     {
  //         t_objetivo(iAffine, jAffine) = invPosition.at<float>(iAffine,
  //         jAffine);
  //     }
  // }

  t_objetivo(0, 3) = average_normal_.normal_x;
  t_objetivo(1, 3) = average_normal_.normal_y;
  t_objetivo(2, 3) = average_normal_.normal_z;
  t_objetivo(3, 3) = 1.0f;

  average_normal_global_reference_.normal_x =
      average_normal_.normal_x - invPosition.at<float>(0, 3);
  average_normal_global_reference_.normal_y =
      average_normal_.normal_y - invPosition.at<float>(1, 3);
  average_normal_global_reference_.normal_z =
      average_normal_.normal_z - invPosition.at<float>(2, 3);

  average_normal_global_reference_.x = invPosition.at<float>(0, 3);
  average_normal_global_reference_.y = invPosition.at<float>(1, 3);
  average_normal_global_reference_.z = invPosition.at<float>(2, 3);

  visualization_of_vectors_cloud_->push_back(average_normal_global_reference_);

  position_error_ = sqrt(((average_normal_global_reference_.normal_x -
                           average_normal_global_reference_.x) *
                          (average_normal_global_reference_.normal_x -
                           average_normal_global_reference_.x)) +
                         ((average_normal_global_reference_.normal_y -
                           average_normal_global_reference_.y) *
                          (average_normal_global_reference_.normal_y -
                           average_normal_global_reference_.y)) +
                         ((average_normal_global_reference_.normal_z -
                           average_normal_global_reference_.z) *
                          (average_normal_global_reference_.normal_z -
                           average_normal_global_reference_.z)));
}
// CLASS OF SEGMENT CRITERIA
// OBB
bool Segment::isItInsideAugmentedOBB(
    const pcl::PointXYZRGBA &SVcentroid_on_objects_ref) {
  if ((SVcentroid_on_objects_ref.x >= (min_point_OBB_.x - tol1X_)) &&
      (SVcentroid_on_objects_ref.x <= (max_point_OBB_.x + tol1X_)) &&
      (SVcentroid_on_objects_ref.y >= (min_point_OBB_.y - tol1Y_)) &&
      (SVcentroid_on_objects_ref.y <= (max_point_OBB_.y + tol1Y_)) &&
      (SVcentroid_on_objects_ref.z >= (min_point_OBB_.z - tol1Z_)) &&
      (SVcentroid_on_objects_ref.z <= (max_point_OBB_.z + tol1Z_))) {
    return true;
  } else {
    return false;
  }
}
// OCCLUSIONS

void Segment::computeLargestDistanceToCamera(
    pcl::PointCloud<pcl::PointNormal>::Ptr visualization_normal_cloud) {
  float maxdistance = 0.0;

  for (uint32_t i = 1; i <= number_of_sv_in_segment_; i++) {
    pcl::PointNormal sv_centroid = segments_normals_[i];

    cv::Mat_<float> sv_centroid_position(4, 1);
    cv::Mat_<float> sv_centroid_position_camera_ref(4, 1);

    // cv::Mat_<float> image_plane_coord(3, 1);

    sv_centroid_position(0, 0) = sv_centroid.x; //
    sv_centroid_position(1) = sv_centroid.y;
    sv_centroid_position(2) = sv_centroid.z;
    sv_centroid_position(3) = 1;

    sv_centroid_position_camera_ref = camera_position_ * sv_centroid_position;

    if (sv_centroid_position_camera_ref(2) > maxdistance) {
      maxdistance = sv_centroid_position_camera_ref(2);
    }
  }
  max_distance_camera_to_object_ = maxdistance;
}

bool Segment::isItInOcclusionsDepthRange(pcl::PointXYZRGBA sv_centroid) {
  bool itsInObservationfrustum = false;

  cv::Mat_<float> sv_centroid_position(4, 1);
  cv::Mat_<float> sv_centroid_position_camera_ref(4, 1);

  // cv::Mat_<float> image_plane_coord(3, 1);

  sv_centroid_position(0, 0) = sv_centroid.x; //
  sv_centroid_position(1) = sv_centroid.y;
  sv_centroid_position(2) = sv_centroid.z;
  sv_centroid_position(3) = 1;

  sv_centroid_position_camera_ref = camera_position_ * sv_centroid_position;

  if (sv_centroid_position_camera_ref(2) <= max_distance_camera_to_object_) {
    itsInObservationfrustum = true;
  }

  return itsInObservationfrustum;
}
// HARD OCCLUSIONS

void Segment::computeInterestfrustum(
    pcl::PointCloud<pcl::PointNormal>::Ptr
        visualization_normal_cloud) // Not working properly, objects frame of
                                    // reference variates too much
{

  // NOT WORTH IT.........
  // //computing max and min obb points on camera refrence to check which face
  // of the OBB is closer to the camera
  // //////////////////////MAX////////////////////////

  // Eigen::Vector3f max_OBB_object_ref;
  // max_OBB_object_ref << max_point_OBB_.x,
  //     max_point_OBB_.y,
  //     max_point_OBB_.z;
  // Eigen::Vector3f max_OBB_world_ref = mass_center_ + rotational_matrix_OBB_ *
  // max_OBB_object_ref;

  // cv::Mat_<float> max_point_OBB_world_ref_mat(4, 1);
  // cv::Mat_<float> max_point_OBB_camera_ref(3, 1);

  // max_point_OBB_world_ref_mat(0, 0) = max_OBB_world_ref(0); //
  // max_point_OBB_world_ref_mat(1) = max_OBB_world_ref(1);
  // max_point_OBB_world_ref_mat(2) = max_OBB_world_ref(2);
  // max_point_OBB_world_ref_mat(3) = 1;

  // max_point_OBB_camera_ref = camera_position_ * max_point_OBB_world_ref_mat;
  // //reprojectiong 3D points from RGB camera center ref. to the RGB camera
  // projection plane

  // //////////////////////MIN////////////////////////

  // Eigen::Vector3f min_OBB_object_ref;
  // min_OBB_object_ref << min_point_OBB_.x,
  //     min_point_OBB_.y,
  //     min_point_OBB_.z;
  // Eigen::Vector3f min_OBB_world_ref = mass_center_ + rotational_matrix_OBB_ *
  // min_OBB_object_ref;

  // cv::Mat_<float> min_point_OBB_world_ref_mat(4, 1);
  // cv::Mat_<float> min_point_OBB_camera_ref(3, 1);

  // min_point_OBB_world_ref_mat(0, 0) = min_OBB_world_ref(0); //
  // min_point_OBB_world_ref_mat(1) = min_OBB_world_ref(1);
  // min_point_OBB_world_ref_mat(2) = min_OBB_world_ref(2);
  // min_point_OBB_world_ref_mat(3) = 1;

  // min_point_OBB_camera_ref = camera_position_ * min_point_OBB_world_ref_mat;
  // //reprojectiong 3D points from RGB camera center ref. to the RGB camera
  // projection plane

  // //computing frustum corners on objects reference

  // Eigen::Matrix<float, 3, 4> corners;
  // cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
  // cout << "max_point_OBB_camera_ref(2): " << max_point_OBB_camera_ref(2) <<
  // endl; cout << "min_point_OBB_camera_ref(2): " <<
  // min_point_OBB_camera_ref(2) << endl;

  // if (max_point_OBB_camera_ref(2) <= min_point_OBB_camera_ref(2))
  // {
  //     cout << "max point OBB  <= min point OBB" << endl;
  //     corners << max_point_OBB_.x + tol1X_, max_point_OBB_.x + tol1X_,
  //     min_point_OBB_.x - tol1X_, min_point_OBB_.x - tol1X_,
  //         max_point_OBB_.y + tol1Y_, min_point_OBB_.y - tol1Y_,
  //         min_point_OBB_.y - tol1Y_, max_point_OBB_.y + tol1Y_,
  //         max_point_OBB_.z + tol1Z_, max_point_OBB_.z + tol1Z_,
  //         max_point_OBB_.z + tol1Z_, max_point_OBB_.z + tol1Z_;
  // }
  // else
  // {
  //     cout << "min point OBB < max point OBB" << endl;
  //     corners << max_point_OBB_.x + tol1X_, max_point_OBB_.x + tol1X_,
  //     min_point_OBB_.x - tol1X_, min_point_OBB_.x - tol1X_,
  //         max_point_OBB_.y + tol1Y_, min_point_OBB_.y - tol1Y_,
  //         min_point_OBB_.y - tol1Y_, max_point_OBB_.y + tol1Y_,
  //         min_point_OBB_.z - tol1Z_, min_point_OBB_.z - tol1Z_,
  //         min_point_OBB_.z - tol1Z_, min_point_OBB_.z - tol1Z_;
  // }

  Eigen::Matrix<float, 3, 4> corners;
  corners << max_point_OBB_.x + tol1X_, max_point_OBB_.x + tol1X_,
      min_point_OBB_.x - tol1X_, min_point_OBB_.x - tol1X_,
      max_point_OBB_.y + tol1Y_, min_point_OBB_.y - tol1Y_,
      min_point_OBB_.y - tol1Y_, max_point_OBB_.y + tol1Y_, 0.0, 0.0, 0.0, 0.0;

  // Eigen::Matrix<float, 3, 4> corners;
  // corners << max_point_OBB_.x, max_point_OBB_.x, min_point_OBB_.x,
  // min_point_OBB_.x,
  //     max_point_OBB_.y, min_point_OBB_.y, min_point_OBB_.y, max_point_OBB_.y,
  //     0.0, 0.0, 0.0, 0.0;

  for (uint32_t i = 0; i < 4; i++) // watchout, eigen begins in 0 index
  {
    Eigen::Vector3f shift;
    shift = corners.col(i);

    Eigen::Vector3f point_position =
        mass_center_ + rotational_matrix_OBB_ * shift;

    pcl::PointNormal visualizationVector;
    visualizationVector.x = Twc_depth_.at<float>(0, 3);
    visualizationVector.y = Twc_depth_.at<float>(1, 3);
    visualizationVector.z = Twc_depth_.at<float>(2, 3);
    visualizationVector.normal_x =
        point_position(0) - Twc_depth_.at<float>(0, 3);
    visualizationVector.normal_y =
        point_position(1) - Twc_depth_.at<float>(1, 3);
    visualizationVector.normal_z =
        point_position(2) - Twc_depth_.at<float>(2, 3);

    visualization_normal_cloud->push_back(visualizationVector);

    pcl::PointXYZ informationVector;

    informationVector.x = point_position(0);
    informationVector.y = point_position(1);
    informationVector.z = point_position(2);

    camera_to_object_frustum_.emplace(i + 1, informationVector);

    cv::Mat_<float> centroid_position(4, 1);
    cv::Mat_<float> image_plane_coord(3, 1);

    centroid_position(0, 0) = point_position(0); //
    centroid_position(1) = point_position(1);
    centroid_position(2) = point_position(2);
    centroid_position(3) = 1;

    image_plane_coord =
        camera_intrinsics_extended_ * camera_position_ *
        centroid_position; // reprojectiong 3D points from RGB camera center
                           // ref. to the RGB camera projection plane
    image_plane_coord /= image_plane_coord.at<float>(2, 0);
    camera_to_object_frustum_image_coord_.emplace(i + 1, image_plane_coord);
  }

  pcl::PointXYZ OBBcenter;
  OBBcenter.x = mass_center_(0);
  OBBcenter.y = mass_center_(1);
  OBBcenter.z = mass_center_(2);

  camera_to_object_frustum_.emplace(5, OBBcenter);
}
bool Segment::isItInCameraToObjectFrustum(pcl::PointXYZRGBA sv_centroid) {
  bool it_is_in_frustum = false;

  cv::Mat_<float> centroid_position(4, 1);
  cv::Mat_<float> image_plane_coord(3, 1);

  centroid_position(0, 0) = sv_centroid.x; //
  centroid_position(1) = sv_centroid.y;
  centroid_position(2) = sv_centroid.z;
  centroid_position(3) = 1;

  image_plane_coord =
      camera_intrinsics_extended_ * camera_position_ *
      centroid_position; // reprojectiong 3D points from RGB camera center ref.
                         // to the RGB camera projection plane
  image_plane_coord /= image_plane_coord.at<float>(2, 0);
  float testx = image_plane_coord.at<float>(0, 0);
  float testy = image_plane_coord.at<float>(1, 0);

  int nvert = 4;
  int i, j;
  j = nvert;

  for (i = 1; i <= nvert;
       i++) // Jordan curve theorem --from
            // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
  {
    float vertx_i = camera_to_object_frustum_image_coord_[i].at<float>(0, 0);
    float verty_i = camera_to_object_frustum_image_coord_[i].at<float>(1, 0);

    float vertx_j = camera_to_object_frustum_image_coord_[j].at<float>(0, 0);
    float verty_j = camera_to_object_frustum_image_coord_[j].at<float>(1, 0);

    if (((verty_i > testy) != (verty_j > testy)) &&
        (testx < (vertx_j - vertx_i) * (testy - verty_i) / (verty_j - verty_i) +
                     vertx_i))
      it_is_in_frustum = !it_is_in_frustum;

    j = i;
  }

  return it_is_in_frustum;
}

bool Segment::isItInPerceptionSphere(pcl::PointXYZRGBA sv_centroid,
                                     const double &sphere_radius) {
  bool it_is_in_sphere = false;

  Eigen::Matrix<float, 3, 1> supervoxel_centroid;
  supervoxel_centroid << sv_centroid.x, sv_centroid.y, sv_centroid.z;

  Eigen::Matrix<float, 3, 1> mass_center_to_sv_vector;
  mass_center_to_sv_vector = supervoxel_centroid - mass_center_;

  if (mass_center_to_sv_vector.norm() < sphere_radius) {
    it_is_in_sphere = true;
  }

  return it_is_in_sphere;
}

// OTHER UTILITIES
void Segment::computeInverseOfCameraPositionAndExtendedIntrinsics() {
  cv::Mat camera_intrinsics_extended =
      cv::Mat::eye(4, 4, camera_position_.type());
  camera_intrinsics_.copyTo(
      camera_intrinsics_extended.rowRange(0, 3).colRange(0, 3));
  camera_intrinsics_extended_ = camera_intrinsics_extended;

  cv::Mat Twc_depth = cv::Mat::eye(4, 4, camera_position_.type());
  cv::Mat mRcw = camera_position_.rowRange(0, 3).colRange(0, 3);
  cv::Mat mRwc = mRcw.t();
  cv::Mat mtcw = camera_position_.rowRange(0, 3).col(3);
  cv::Mat mOw = -mRcw.t() * mtcw;

  mRwc.copyTo(Twc_depth.rowRange(0, 3).colRange(0, 3));
  mOw.copyTo(Twc_depth.rowRange(0, 3).col(3));
  Twc_depth.copyTo(Twc_depth_);
}

void Segment::computeCameraZaxisOnWorldReference() {

  cv::Mat x3D;
  cv::Mat_<float> vec_tmp(4, 1);
  vec_tmp(0, 0) = 0; // d * (j - mK.at<float>(0, 2)) / mK.at<float>(0, 0); //
  vec_tmp(1) = 0;    // d * (i - mK.at<float>(1, 2)) / mK.at<float>(1, 1);    //
  vec_tmp(2) = 1;
  vec_tmp(3) = 1;

  x3D = Twc_depth_ * vec_tmp;
  x3D /= x3D.at<float>(3);

  pcl::PointNormal camera_z_axis_on_world_reference_;
  camera_z_axis_on_world_reference_.x = 0;
  camera_z_axis_on_world_reference_.y = 0;
  camera_z_axis_on_world_reference_.z = 0;

  camera_z_axis_on_world_reference_.normal_x = x3D.at<float>(0);
  camera_z_axis_on_world_reference_.normal_y = x3D.at<float>(1);
  camera_z_axis_on_world_reference_.normal_z = x3D.at<float>(2);

  // visualization_of_vectors_cloud_->push_back(camera_z_axis_on_world_reference_);
}

void Segment::addSupervoxelConnectionsToViewer2(
    pcl::PointXYZ &supervoxel_center,
    pcl::PointCloud<pcl::PointXYZ> &adjacent_supervoxel_centers,
    std::string supervoxel_name,
    pcl::visualization::PCLVisualizer::Ptr viewer) {
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

  // Iterate through all adjacent points, and add a center point to adjacent
  // point pair
  for (pcl::PointCloud<pcl::PointXYZ>::iterator adjacent_itr =
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

void Segment::visualizeObjectsReprojectionOnCamera(
    std::shared_ptr<DEF_OBJ_TRACK::Segment> occlusions,
    std::shared_ptr<DEF_OBJ_TRACK::Segment> hardOcclusions) {

  /// Windows names
  std::string atom_window = "Drawing 1: Atom";
  int radius = 10;
  int radius_15 = 15;
  cv::Scalar blue(240, 0, 0);
  cv::Scalar red(0, 0, 240);
  cv::Scalar yellow(0, 240, 240);

  /// Create black empty images
  cv::Mat atom_image = cv::Mat::zeros(480, 640, CV_8UC3);
  cv::circle(
      atom_image,
      cv::Point(camera_to_object_frustum_image_coord_[1].at<float>(0, 0),
                camera_to_object_frustum_image_coord_[1].at<float>(1, 0)),
      radius, blue);
  cv::circle(
      atom_image,
      cv::Point(camera_to_object_frustum_image_coord_[2].at<float>(0, 0),
                camera_to_object_frustum_image_coord_[2].at<float>(1, 0)),
      radius, blue);
  cv::circle(
      atom_image,
      cv::Point(camera_to_object_frustum_image_coord_[3].at<float>(0, 0),
                camera_to_object_frustum_image_coord_[3].at<float>(1, 0)),
      radius, blue);
  cv::circle(
      atom_image,
      cv::Point(camera_to_object_frustum_image_coord_[4].at<float>(0, 0),
                camera_to_object_frustum_image_coord_[4].at<float>(1, 0)),
      radius, blue);

  for (int i = 1; i <= occlusions->label_of_sv_; i++) {
    pcl::PointNormal sv_centroid;
    sv_centroid = occlusions->segments_normals_[i];

    cv::Mat_<float> centroid_position(4, 1);
    cv::Mat_<float> image_plane_coord(3, 1);

    centroid_position(0, 0) = sv_centroid.x; //
    centroid_position(1) = sv_centroid.y;
    centroid_position(2) = sv_centroid.z;
    centroid_position(3) = 1;

    image_plane_coord =
        camera_intrinsics_extended_ * camera_position_ *
        centroid_position; // reprojectiong 3D points from RGB camera center
                           // ref. to the RGB camera projection plane
    image_plane_coord /= image_plane_coord.at<float>(2, 0);

    cv::circle(atom_image,
               cv::Point(image_plane_coord.at<float>(0, 0),
                         image_plane_coord.at<float>(1, 0)),
               radius, yellow);
  }

  for (int i = 1; i <= hardOcclusions->label_of_sv_; i++) {
    pcl::PointNormal sv_centroid;
    sv_centroid = hardOcclusions->segments_normals_[i];

    cv::Mat_<float> centroid_position(4, 1);
    cv::Mat_<float> image_plane_coord(3, 1);

    centroid_position(0, 0) = sv_centroid.x; //
    centroid_position(1) = sv_centroid.y;
    centroid_position(2) = sv_centroid.z;
    centroid_position(3) = 1;

    image_plane_coord =
        camera_intrinsics_extended_ * camera_position_ *
        centroid_position; // reprojectiong 3D points from RGB camera center
                           // ref. to the RGB camera projection plane
    image_plane_coord /= image_plane_coord.at<float>(2, 0);

    cv::circle(atom_image,
               cv::Point(image_plane_coord.at<float>(0, 0),
                         image_plane_coord.at<float>(1, 0)),
               radius_15, red);
  }

  cv::imshow(atom_window, atom_image);
  cv::waitKey(1);
}

// OBJECTS STRUCTURE COMPUTATION (OCTREE)
void Segment::computeOctreeAdjacencyAndDegree(
    pcl::visualization::PCLVisualizer::Ptr viewer) {
  float resolution = 128.0f;
  int maxGraphDegree = 0;

  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree(
      new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(resolution));

  Segments_sv_AdjacencyOctree_ = octree;

  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);

  adjacency_cloud = temp_cloud;

  temp_cloud->width = label_of_sv_;
  temp_cloud->height = 1;
  temp_cloud->points.resize(temp_cloud->width * temp_cloud->height);

  for (std::size_t i = 0; i < label_of_sv_;
       ++i) // index for the cloud is one less than for the maps
  {
    pcl::PointNormal object_normal_idx = segments_normals_[i];

    temp_cloud->points[i].x = segments_normals_[i + 1].x;
    temp_cloud->points[i].y = segments_normals_[i + 1].y;
    temp_cloud->points[i].z = segments_normals_[i + 1].z;
  }

  Segments_sv_AdjacencyOctree_->setInputCloud(temp_cloud);
  Segments_sv_AdjacencyOctree_->addPointsFromInputCloud();

  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;

  for (int j = 0; j < temp_cloud->width; j++) {
    if (Segments_sv_AdjacencyOctree_->radiusSearch(
            temp_cloud->points[j], neighbouring_radius_, pointIdxRadiusSearch,
            pointRadiusSquaredDistance) > 0) {
      pcl::PointCloud<pcl::PointXYZ> adjacent_supervoxel_centers;

      pcl::PointNormal sv_centroid = segments_normals_[j + 1];
      pcl::PointXYZRGBA sv_color = segments_colors_RGB_[j + 1];
      pcl::PointXYZ sv_normal;

      sv_normal.x = sv_centroid.normal_x;
      sv_normal.y = sv_centroid.normal_y;
      sv_normal.z = sv_centroid.normal_z;

      std::stringstream ss;
      ss << segmentType_ << "node_" << j + 1;

      for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
        adjacent_supervoxel_centers.push_back(
            temp_cloud->points[pointIdxRadiusSearch[i]]);
      }

      nodes_degrees_[j + 1] = pointIdxRadiusSearch.size() - 1;

      if ((pointIdxRadiusSearch.size() - 1) >= maxGraphDegree) {
        maxGraphDegree = (pointIdxRadiusSearch.size() - 1);
      }

      nodes_neighbours_[j + 1] = pointIdxRadiusSearch;

      if (!adjacent_supervoxel_centers.empty()) {
        addSupervoxelConnectionsToViewer2(temp_cloud->points[j],
                                          adjacent_supervoxel_centers, ss.str(),
                                          viewer);
      }
      if (segmentType_ == INTERACTING_ELEMENT) {
        std::stringstream ssCube;
        ssCube << "interaction_" << j << "_";
        viewer->addCube(sv_centroid.x - cubeSize, sv_centroid.x + cubeSize,
                        sv_centroid.y - cubeSize, sv_centroid.y + cubeSize,
                        sv_centroid.z - cubeSize, sv_centroid.z + cubeSize, 0.0,
                        1.0, 0.0, ssCube.str());
      } else if (segmentType_ == OCCLUDING_ELEMENT) {
        std::stringstream ssCube;
        ssCube << "occlusion_" << j << "_";
        viewer->addCube(sv_centroid.x - cubeSize, sv_centroid.x + cubeSize,
                        sv_centroid.y - cubeSize, sv_centroid.y + cubeSize,
                        sv_centroid.z - cubeSize, sv_centroid.z + cubeSize, 1.0,
                        1.0, 0.0, ssCube.str());
      } else if (segmentType_ == HARD_OCCLUDING_ELEMENT) {
        std::stringstream ssCube;
        ssCube << "hard_occlusion_" << j << "_";
        viewer->addCube(sv_centroid.x - cubeSize, sv_centroid.x + cubeSize,
                        sv_centroid.y - cubeSize, sv_centroid.y + cubeSize,
                        sv_centroid.z - cubeSize, sv_centroid.z + cubeSize, 1.0,
                        0.0, 0.0, ssCube.str());
      }
    }
  }
  max_graph_degree_ = maxGraphDegree;
}

// VISUALIZATION STUFF

void Segment::VisualizeCloudsAndBoundingBoxes(
    pcl::visualization::PCLVisualizer::Ptr viewer) {
  // Visualization of clouds and bounding boxes

  // std::ostringstream n_sv;
  // n_sv << "nº SV Object:  " << label_of_sv_;
  // viewer->addText(n_sv.str(), 10, 200, 20, 0, 0, 0, "v2 text");

  // viewer->addCube( min_point_AABB_.x,  max_point_AABB_.x,  min_point_AABB_.y,
  // max_point_AABB_.y,  min_point_AABB_.z,  max_point_AABB_.z, 1.0, 1.0, 0.0,
  // "AABB");
  // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  // pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB");

  Eigen::Vector3f position(position_OBB_.x, position_OBB_.y, position_OBB_.z);
  Eigen::Quaternionf quat(rotational_matrix_OBB_);
  // viewer->addCube(position, quat,  max_point_OBB_.x -  min_point_OBB_.x,
  // max_point_OBB_.y -  min_point_OBB_.y,  max_point_OBB_.z - min_point_OBB_.z,
  // "OBB");
  // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  // pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB");

  viewer->addCube(
      position, quat, max_point_OBB_.x - min_point_OBB_.x + 2 * tol1X_,
      max_point_OBB_.y - min_point_OBB_.y + 2 * tol1Y_,
      max_point_OBB_.z - min_point_OBB_.z + 2 * tol1Z_, "augmentedOBB");
  viewer->setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
      "augmentedOBB");

  // viewer->addCube(position, quat,  max_point_OBB_.x -  min_point_OBB_.x + 2 *
  // tolX_,  max_point_OBB_.y -  min_point_OBB_.y + 2 *  tolY_, max_point_OBB_.z
  // -  min_point_OBB_.z + 2 *  tolZ_, "reducedOBB");
  // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  // pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "reducedOBB");
  if (visualize_main_inertia_axes) {
    pcl::PointXYZ center(mass_center_(0), mass_center_(1), mass_center_(2));
    pcl::PointXYZ x_axis(major_vector_(0) + mass_center_(0),
                         major_vector_(1) + mass_center_(1),
                         major_vector_(2) + mass_center_(2));
    pcl::PointXYZ y_axis(middle_vector_(0) + mass_center_(0),
                         middle_vector_(1) + mass_center_(1),
                         middle_vector_(2) + mass_center_(2));
    pcl::PointXYZ z_axis(minor_vector_(0) + mass_center_(0),
                         minor_vector_(1) + mass_center_(1),
                         minor_vector_(2) + mass_center_(2));
    viewer->addLine(center, x_axis, 0.2f, 0.0f, 0.0f, "major eigen vector");
    viewer->addLine(center, y_axis, 0.0f, 0.2f, 0.0f, "middle eigen vector");
    viewer->addLine(center, z_axis, 0.0f, 0.0f, 0.2f, "minor eigen vector");
  }

  if (visualize_rgba_pointcloud) {

    viewer->addPointCloud<PointTSuperVoxel>(selected_cloud_with_labeled_sv_,
                                            "sample cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud");

    // PointCloudTSuperVoxel::Ptr voxel_centroid_cloud =
    // supervoxel_cluster.getVoxelCentroidCloud();
    // viewer->addPointCloud(voxel_centroid_cloud, "voxel centroids");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
    // 50.0, "voxel centroids");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
    // 1.0, "voxel centroids");

    // PointLCloudTSuperVoxel::Ptr labeled_voxel_cloud =
    // supervoxel_cluster.getLabeledVoxelCloud();
    // viewer->addPointCloud(labeled_voxel_cloud, "labeled voxels");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
    // 3.0, "labeled voxels");
  }

  if (show_normals) {
    pcl::PointCloud<pcl::PointNormal>::Ptr sv_normal_cloud(
        new pcl::PointCloud<pcl::PointNormal>);

    for (int i = 1; i <= number_of_sv_in_segment_; i++) {
      sv_normal_cloud->push_back(segments_normals_[i]);
    }

    // We have this disabled so graph is easy to see, uncomment to see
    // supervoxel normals
    viewer->addPointCloudNormals<PointNTSuperVoxel>(sv_normal_cloud, 1, 0.05f,
                                                    "supervoxel_normals");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0,
        "supervoxel_normals");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "supervoxel_normals");
  }
}

void Segment::OcclusionVisualChecking(
    std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject,
    Eigen::Affine3f t_objetivo, pcl::visualization::PCLVisualizer::Ptr viewer) {
  // visual checking of optimization process
  /// Windows names
  std::string atom_window = "What the optimal camera position is seeing";
  int radius = 5;
  int radius_15 = 8;
  cv::Scalar blue(240, 0, 0);
  cv::Scalar red(0, 0, 240);
  cv::Scalar yellow(0, 240, 240);

  /// Create black empty images
  cv::Mat atom_image = cv::Mat::zeros(480, 640, CV_8UC3);

  int num_occlusions_ =
      static_cast<int>(number_of_sv_in_segment_); // remember, I was using index
                                                  // from 1 to inf, here from 0
  double *occlusions_ = new double[6 * number_of_sv_in_segment_];

  for (int i_checking = 0; i_checking < num_occlusions_; ++i_checking) {
    pcl::PointNormal sv_normal =
        segments_normals_[i_checking +
                          1]; // i_checkingndex i_checkingn maps start at 1 for
                              // me (0 preserved for special cases)

    occlusions_[i_checking * 6 + 0] = static_cast<double>(sv_normal.x);
    occlusions_[i_checking * 6 + 1] = static_cast<double>(sv_normal.y);
    occlusions_[i_checking * 6 + 2] = static_cast<double>(sv_normal.z);
    occlusions_[i_checking * 6 + 3] =
        static_cast<double>(sv_normal.x + sv_normal.normal_x);
    occlusions_[i_checking * 6 + 4] =
        static_cast<double>(sv_normal.y + sv_normal.normal_y);
    occlusions_[i_checking * 6 + 5] =
        static_cast<double>(sv_normal.z + sv_normal.normal_z);

    double occlusion_x = occlusions_[i_checking * 6 + 0];
    double occlusion_y = occlusions_[i_checking * 6 + 1];
    double occlusion_z = occlusions_[i_checking * 6 + 2];

    double OBBCorner_1_x = NewObject->camera_to_object_frustum_[1].x;
    double OBBCorner_1_y = NewObject->camera_to_object_frustum_[1].y;
    double OBBCorner_1_z = NewObject->camera_to_object_frustum_[1].z;

    double OBBCorner_2_x = NewObject->camera_to_object_frustum_[2].x;
    double OBBCorner_2_y = NewObject->camera_to_object_frustum_[2].y;
    double OBBCorner_2_z = NewObject->camera_to_object_frustum_[2].z;

    double OBBCorner_3_x = NewObject->camera_to_object_frustum_[3].x;
    double OBBCorner_3_y = NewObject->camera_to_object_frustum_[3].y;
    double OBBCorner_3_z = NewObject->camera_to_object_frustum_[3].z;

    double OBBCorner_4_x = NewObject->camera_to_object_frustum_[4].x;
    double OBBCorner_4_y = NewObject->camera_to_object_frustum_[4].y;
    double OBBCorner_4_z = NewObject->camera_to_object_frustum_[4].z;

    // normal_visualization_cloud->push_back(sv_normal);

    Eigen::Matrix<double, 3, 1> OBBCenter; // OBB center position
    OBBCenter << double(NewObject->camera_to_object_frustum_[5].x),
        double(NewObject->camera_to_object_frustum_[5].y),
        double(NewObject->camera_to_object_frustum_[5].z);

    Eigen::Matrix<double, 3, 1> camera_center; // camera position
    camera_center << t_objetivo(0, 3), t_objetivo(1, 3), t_objetivo(2, 3);

    Eigen::Matrix<double, 3, 1> normal_vector;
    normal_vector = OBBCenter - camera_center;

    normal_vector = normal_vector.normalized();

    Eigen::Matrix<double, 3, 1> U_vector; // 2nd basis vector
    U_vector << double(1.0), double(1.0),
        double((-normal_vector(0) - normal_vector(1)) / normal_vector(2));
    U_vector = U_vector.normalized();

    Eigen::Matrix<double, 3, 1> V_vector; // 3rd basis vector
    V_vector = (normal_vector.cross(U_vector)).normalized();

    Eigen::Matrix<double, 3, 1>
        W_vector; // Comprobation vector, it must be = normal_vector
    W_vector = (U_vector.cross(V_vector)).normalized();

    pcl::PointNormal uvector, vvector, wvector;
    uvector.x = t_objetivo(0, 3);
    uvector.y = t_objetivo(1, 3);
    uvector.z = t_objetivo(2, 3);

    vvector.x = t_objetivo(0, 3);
    vvector.y = t_objetivo(1, 3);
    vvector.z = t_objetivo(2, 3);

    wvector.x = t_objetivo(0, 3);
    wvector.y = t_objetivo(1, 3);
    wvector.z = t_objetivo(2, 3);

    uvector.normal_x = U_vector(0);
    uvector.normal_y = U_vector(1);
    uvector.normal_z = U_vector(2);

    vvector.normal_x = V_vector(0);
    vvector.normal_y = V_vector(1);
    vvector.normal_z = V_vector(2);

    wvector.normal_x = W_vector(0);
    wvector.normal_y = W_vector(1);
    wvector.normal_z = W_vector(2);

    NewObject->visualization_of_vectors_cloud_->push_back(uvector);
    NewObject->visualization_of_vectors_cloud_->push_back(vvector);
    NewObject->visualization_of_vectors_cloud_->push_back(wvector);

    Eigen::Matrix<double, 3, 3> R_matrix;
    R_matrix << U_vector, V_vector, W_vector;
    Eigen::Matrix<double, 3, 3> R_matrix_inverse;
    R_matrix_inverse = R_matrix.transpose();

    Eigen::Matrix<double, 3, 4> Rt_matrix; //
    Rt_matrix << R_matrix_inverse, -R_matrix_inverse * camera_center;

    // Camera.cx : 328.0010681152344 Camera.cy : 241.31031799316406;

    double fx = camera_intrinsics_extended_.at<double>(0, 0);
    double fy = camera_intrinsics_extended_.at<double>(1, 1);
    double cx = camera_intrinsics_extended_.at<double>(0, 2);
    double cy = camera_intrinsics_extended_.at<double>(1, 2);

    Eigen::Matrix<double, 3, 3>
        k_matrix; // this could be any intrinsics... but for possible future
                  // expansions of the optimization function -> TODO: add
                  // intrinsics as variables
    k_matrix << fx, double(0.0), cx, double(0.0), fy, cy, double(0.0),
        double(0.0), double(1.0);

    Eigen::Matrix<double, 3, 5>
        OBBCorners_and_sv_vector; // OBB center position [SV, corner1, corner2,
                                  // corner3, corner4]
    OBBCorners_and_sv_vector << double(occlusion_x), double(OBBCorner_1_x),
        double(OBBCorner_2_x), double(OBBCorner_3_x), double(OBBCorner_4_x),
        double(occlusion_y), double(OBBCorner_1_y), double(OBBCorner_2_y),
        double(OBBCorner_3_y), double(OBBCorner_4_y), double(occlusion_z),
        double(OBBCorner_1_z), double(OBBCorner_2_z), double(OBBCorner_3_z),
        double(OBBCorner_4_z);

    Eigen::Matrix<double, 3, 5> OBBcorn_and_sv_plane;

    // check if occlusion is actually infront of the object or behind

    bool occlusion_between_cam_obj = false;

    Eigen::Matrix<double, 4, 1> OBB_center_extended;
    Eigen::Matrix<double, 4, 1> occlusion_vector;
    Eigen::Matrix<double, 3, 1> OBBcenter_camera_ref;
    Eigen::Matrix<double, 3, 1> occlusion_vector_camera_ref;

    occlusion_vector << double(occlusion_x), double(occlusion_y),
        double(occlusion_z), double(1.0);
    occlusion_vector_camera_ref = Rt_matrix * occlusion_vector;

    OBB_center_extended << double(NewObject->camera_to_object_frustum_[5].x),
        double(NewObject->camera_to_object_frustum_[5].y),
        double(NewObject->camera_to_object_frustum_[5].z), double(1.0);
    OBBcenter_camera_ref = Rt_matrix * OBB_center_extended;

    if (occlusion_vector_camera_ref(2) < OBBcenter_camera_ref(2)) {
      occlusion_between_cam_obj = true;
    }

    for (uint32_t i = 0; i < 5; i++) // watchout, eigen begins in 0 index
    {
      Eigen::Matrix<double, 4, 1> temp_vector;
      Eigen::Matrix<double, 3, 1> temp_vector_camera_coord;

      temp_vector << OBBCorners_and_sv_vector.col(i), double(1.0);

      temp_vector_camera_coord = k_matrix * Rt_matrix * temp_vector;

      OBBcorn_and_sv_plane.col(i) =
          temp_vector_camera_coord / temp_vector_camera_coord(2);
    }
    // checking if it is inside frustum

    int nvert = 4;
    int i, j;
    j = nvert;

    double testx = OBBcorn_and_sv_plane(0, 0);
    double testy = OBBcorn_and_sv_plane(1, 0);

    bool it_is_in_frustum = false;

    for (i = 1; i <= nvert;
         i++) // Jordan curve theorem --from
              // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    {
      double vertx_i = OBBcorn_and_sv_plane(0, i); // plane coord x of point i
      double verty_i = OBBcorn_and_sv_plane(1, i); // plane coord y of point i

      double vertx_j = OBBcorn_and_sv_plane(0, j); // plane coord x of point j
      double verty_j = OBBcorn_and_sv_plane(1, j); // plane coord y of point j

      if (((verty_i > testy) != (verty_j > testy)) &&
          (testx <
           (vertx_j - vertx_i) * (testy - verty_i) / (verty_j - verty_i) +
               vertx_i))
        it_is_in_frustum = !it_is_in_frustum;

      j = i;
    }

    if (it_is_in_frustum && occlusion_between_cam_obj) {
      cv::circle(atom_image,
                 cv::Point(float(OBBcorn_and_sv_plane(0, 0)),
                           float(OBBcorn_and_sv_plane(1, 0))),
                 radius_15, red);
    } else {
      cv::circle(atom_image,
                 cv::Point(float(OBBcorn_and_sv_plane(0, 0)),
                           float(OBBcorn_and_sv_plane(1, 0))),
                 radius_15, yellow);
    }

    Eigen::Matrix<double, 4, 1> reprojected_OBBcorner_distances;
    reprojected_OBBcorner_distances
        << sqrt(OBBcorn_and_sv_plane(0, 1) * OBBcorn_and_sv_plane(0, 1) +
                OBBcorn_and_sv_plane(1, 1) *
                    OBBcorn_and_sv_plane(1, 1)), // plane coord y of point i
        sqrt(OBBcorn_and_sv_plane(0, 2) * OBBcorn_and_sv_plane(0, 2) +
             OBBcorn_and_sv_plane(1, 2) * OBBcorn_and_sv_plane(1, 2)),
        sqrt(OBBcorn_and_sv_plane(0, 3) * OBBcorn_and_sv_plane(0, 3) +
             OBBcorn_and_sv_plane(1, 3) * OBBcorn_and_sv_plane(1, 3)),
        sqrt(OBBcorn_and_sv_plane(0, 4) * OBBcorn_and_sv_plane(0, 4) +
             OBBcorn_and_sv_plane(1, 4) * OBBcorn_and_sv_plane(1, 4));

    cv::circle(atom_image,
               cv::Point(float(OBBcorn_and_sv_plane(0, 1)),
                         float(OBBcorn_and_sv_plane(1, 1))),
               radius, blue);
    cv::circle(atom_image,
               cv::Point(float(OBBcorn_and_sv_plane(0, 2)),
                         float(OBBcorn_and_sv_plane(1, 2))),
               radius, blue);
    cv::circle(atom_image,
               cv::Point(float(OBBcorn_and_sv_plane(0, 3)),
                         float(OBBcorn_and_sv_plane(1, 3))),
               radius, blue);
    cv::circle(atom_image,
               cv::Point(float(OBBcorn_and_sv_plane(0, 4)),
                         float(OBBcorn_and_sv_plane(1, 4))),
               radius, blue);

    double cone_radius = reprojected_OBBcorner_distances.maxCoeff();

    double weight;
    double weight_scale = double(PI / 2.0);

    if (it_is_in_frustum && occlusion_between_cam_obj) {
      weight = ((-weight_scale / cone_radius) *
                (sqrt(testx * testx + testy * testy))) +
               (weight_scale); // / normalizing_distance;
    } else {
      weight = double(0.0);
    }
  }

  for (uint32_t i = 1; i < 5; i++) // watchout, eigen begins in 0 index
  {

    pcl::PointNormal visualizationVector;
    visualizationVector.x = t_objetivo(0, 3);
    visualizationVector.y = t_objetivo(1, 3);
    visualizationVector.z = t_objetivo(2, 3);
    visualizationVector.normal_x =
        NewObject->camera_to_object_frustum_[i].x - t_objetivo(0, 3);
    visualizationVector.normal_y =
        NewObject->camera_to_object_frustum_[i].y - t_objetivo(1, 3);
    visualizationVector.normal_z =
        NewObject->camera_to_object_frustum_[i].z - t_objetivo(2, 3);

    NewObject->visualization_of_vectors_cloud_->push_back(visualizationVector);
  }

  cv::imshow(atom_window, atom_image);
  cv::waitKey(1);

  // End of visual checking of optimization process
}

std::map<uint32_t, Eigen::Matrix<float, 3, 1>>
Segment::normalsToSphereIntersectionPoints(
    pcl::visualization::PCLVisualizer::Ptr viewer,
    const double &sphere_radius) {
  std::map<uint32_t, Eigen::Matrix<float, 3, 1>> object_sphere_intersections;

  Eigen::Matrix<float, 3, 1> sphere_center;
  sphere_center << mass_center_(0), mass_center_(1), mass_center_(2);

  pcl::PointCloud<pcl::PointNormal>::Ptr visualization_of_sphere_rays(
      new pcl::PointCloud<pcl::PointNormal>);

  for (int i = 0; i < number_of_sv_in_segment_; ++i) {
    pcl::PointNormal sv_normal =
        segments_normals_[i + 1]; // index in maps start at 1 for me (0
                                  // preserved for special cases)

    pcl::PointNormal sv_normal_world_reference;
    sv_normal_world_reference.normal_x = sv_normal.normal_x - sv_normal.x;
    sv_normal_world_reference.normal_y = sv_normal.normal_y - sv_normal.y;
    sv_normal_world_reference.normal_z = sv_normal.normal_z - sv_normal.z;

    // cout << "sv_normal_world_reference: " << sv_normal_world_reference <<
    // endl;

    Eigen::Matrix<float, 3, 1> o_vector;
    o_vector << sv_normal.x, sv_normal.y, sv_normal.z;

    // cout << "o_vector: " << o_vector << endl;

    Eigen::Matrix<float, 3, 1> normal_world_reference;
    normal_world_reference << sv_normal_world_reference.normal_x,
        sv_normal_world_reference.normal_y, sv_normal_world_reference.normal_z;

    // cout << "normal_world_reference: " << normal_world_reference << endl;

    Eigen::Matrix<float, 3, 1> l_vector;
    l_vector << normal_world_reference(0), normal_world_reference(1),
        normal_world_reference(2);

    l_vector = l_vector.normalized();
    float normal_norm = l_vector.norm();

    // cout << "l_vector: " << l_vector << endl;
    // cout << "l_vector_norm: " << normal_norm << endl;

    float a, b, c;

    a = 1.0;
    b = 2 * (l_vector.dot(o_vector - sphere_center));
    c = ((o_vector - sphere_center).norm()) *
            ((o_vector - sphere_center).norm()) -
        (sphere_radius * sphere_radius);

    // cout << " a, b, c: " << a << " " << b << " " << c << endl;

    float d_line_parameter = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);

    // cout << "d_line_parameter: " << d_line_parameter << endl;

    Eigen::Matrix<float, 3, 1> intersection_point;

    intersection_point = o_vector + l_vector * d_line_parameter;

    // cout << "intersection_point: " << intersection_point << endl;

    pcl::PointNormal visualization_rays;

    visualization_rays.x = sphere_center(0);
    visualization_rays.y = sphere_center(1);
    visualization_rays.z = sphere_center(2);
    visualization_rays.normal_x = intersection_point(0) - sphere_center(0);
    visualization_rays.normal_y = intersection_point(1) - sphere_center(1);
    visualization_rays.normal_z = intersection_point(2) - sphere_center(2);

    Eigen::Matrix<float, 3, 1> intersection_point_sphere_ref;

    intersection_point_sphere_ref << intersection_point(0) - sphere_center(0),
        intersection_point(1) - sphere_center(0),
        intersection_point(2) - sphere_center(0);

    visualization_of_sphere_rays->push_back(visualization_rays);

    // cout << "vector_module_checking: " << vector_module_checking.norm() <<
    // endl; cout << "sphere_radius: " << sphere_radius << endl;
    object_sphere_intersections[i + 1] = intersection_point_sphere_ref;

    // cout << "object_sphere_intersections[i + 1] = " <<
    // object_sphere_intersections[i + 1] << endl;
  }
  // cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;

  // UNCOMENT TO VISUALIZE SPHERE RAYS
  viewer->addPointCloudNormals<PointNTSuperVoxel>(
      visualization_of_sphere_rays, 1, 1.0f, "visualization_rays");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0,
      "visualization_rays");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "visualization_rays");

  return object_sphere_intersections;
}

std::map<uint32_t, Eigen::Matrix<float, 3, 1>>
Segment::centroidsToOcclussorRays(
    pcl::visualization::PCLVisualizer::Ptr viewer, const double &sphere_radius,
    std::shared_ptr<DEF_OBJ_TRACK::Segment> NewObject) {
  std::map<uint32_t, Eigen::Matrix<float, 3, 1>> occlusion_sphere_intersections;

  Eigen::Matrix<float, 3, 1> sphere_center;
  sphere_center << NewObject->mass_center_(0), NewObject->mass_center_(1),
      NewObject->mass_center_(2);

  pcl::PointCloud<pcl::PointNormal>::Ptr visualization_of_sphere_rays(
      new pcl::PointCloud<pcl::PointNormal>);

  for (int i = 0; i < NewObject->number_of_sv_in_segment_; ++i) {
    for (int j = 0; j < number_of_sv_in_segment_; ++j) {
      pcl::PointNormal sv_normal =
          NewObject
              ->segments_normals_[i + 1]; // index in maps start at 1 for me (0
                                          // preserved for special cases)
      pcl::PointNormal occlusor_normal = segments_normals_[j + 1];

      // cout << "sv_normal_world_reference: " << sv_normal_world_reference <<
      // endl;

      Eigen::Matrix<float, 3, 1> o_vector;
      o_vector << sv_normal.x, sv_normal.y, sv_normal.z;

      // cout << "o_vector: " << o_vector << endl;

      Eigen::Matrix<float, 3, 1> l_vector;
      l_vector << occlusor_normal.x - sv_normal.x,
          occlusor_normal.y - sv_normal.y, occlusor_normal.z - sv_normal.z;

      l_vector = l_vector.normalized();
      float normal_norm = l_vector.norm();

      // cout << "l_vector: " << l_vector << endl;
      // cout << "l_vector_norm: " << normal_norm << endl;

      float a, b, c;

      a = 1.0;
      b = 2 * (l_vector.dot(o_vector - sphere_center));
      c = ((o_vector - sphere_center).norm()) *
              ((o_vector - sphere_center).norm()) -
          (sphere_radius * sphere_radius);

      // cout << " a, b, c: " << a << " " << b << " " << c << endl;

      float d_line_parameter = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);

      // cout << "d_line_parameter: " << d_line_parameter << endl;

      Eigen::Matrix<float, 3, 1> intersection_point;

      intersection_point = o_vector + l_vector * d_line_parameter;

      // cout << "intersection_point: " << intersection_point << endl;

      Eigen::Matrix<float, 3, 1> l_antipodes;
      l_antipodes << intersection_point(0) - sphere_center(0),
          intersection_point(1) - sphere_center(1),
          intersection_point(2) - sphere_center(2);

      l_antipodes = l_antipodes.normalized();

      Eigen::Matrix<float, 3, 1> antipodes_intersection_point;

      antipodes_intersection_point =
          intersection_point -
          l_antipodes * (sphere_radius * 2.0); // =-intersection_point

      pcl::PointNormal visualization_rays;

      visualization_rays.x = sphere_center(0);
      visualization_rays.y = sphere_center(1);
      visualization_rays.z = sphere_center(2);
      visualization_rays.normal_x =
          antipodes_intersection_point(0) - sphere_center(0);
      visualization_rays.normal_y =
          antipodes_intersection_point(1) - sphere_center(1);
      visualization_rays.normal_z =
          antipodes_intersection_point(2) - sphere_center(2);

      visualization_of_sphere_rays->push_back(visualization_rays);

      occlusion_sphere_intersections[i * number_of_sv_in_segment_ + j + 1] =
          antipodes_intersection_point;
      // cout << "i * number_of_sv_in_segment_ + j+1: " << i *
      // number_of_sv_in_segment_ + j + 1; cout <<
      // "occlusion_sphere_intersections[i * number_of_sv_in_segment_ + j+1] = "
      // << occlusion_sphere_intersections[i * number_of_sv_in_segment_ + j + 1]
      // << endl;
    }
  }
  // UNCOMENT TO VISUALIZE SPHERE RAYS
  // viewer->addPointCloudNormals<PointNTSuperVoxel>(visualization_of_sphere_rays,
  // 1, 1.0f, "visualization_rays_2");
  // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
  // 1.0, 0.0, 0.0, "visualization_rays_2");
  // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
  // 1, "visualization_rays_2");

  return occlusion_sphere_intersections;
}

Eigen::Matrix<float, 1, 3> Segment::computeIdealOptimalCameraPosition(
    const double &sphere_radius, std::map<uint32_t, Eigen::Matrix<float, 3, 1>>
                                     object_sphere_intersections_sphere_reference) {
  int number_of_sv_in_object = number_of_sv_in_segment_;
  // double number_of_sv_occluding = HardOcclusions->number_of_sv_in_segment_;

  float average_theta,average_phi;

  float theta_sin_sum = 0.0;
  float theta_cos_sum = 0.0;
  float phi_sin_sum = 0.0;
  float phi_cos_sum = 0.0;

  for (int i = 0; i < number_of_sv_in_segment_; ++i) {

    float x = object_sphere_intersections_sphere_reference[i + 1](0);
    float y = object_sphere_intersections_sphere_reference[i + 1](1);
    float z = object_sphere_intersections_sphere_reference[i + 1](2);

    float theta, phi;

    if (z > 0) {
      theta = atan(sqrt(x * x + y * y) / z);
    } else if (z == 0) {
      theta = PI / 2;
    } else {
      theta = PI + atan(sqrt(x * x + y * y) / z);
    }

    if ((x > 0) && (y > 0)) // 1st Q
    {
      phi = atan(y / x);
    } else if ((x > 0) && (y < 0)) // 4º Q
    {
      phi = 2 * PI + atan(y / x);
    } else if (x == 0) {
      phi = (PI / 2) * ((y > 0) ? 1 : ((y < 0) ? -1 : 0)); //...sign(y)
    } else if (x < 0) {
      phi = PI + atan(y / x); // 2nd and 3rd Q
    }

    theta_sin_sum += sin(theta);
    theta_cos_sum += cos(theta);

    phi_sin_sum += sin(phi);
    phi_cos_sum += cos(phi);
  }

  // mean of angles

  average_theta = atan2((theta_sin_sum / number_of_sv_in_object),
                        (theta_cos_sum / number_of_sv_in_object));
  average_phi = atan2((phi_sin_sum / number_of_sv_in_object),
                      (phi_cos_sum / number_of_sv_in_object));

  // Camera coordenates generator

  Eigen::Matrix<float, 3, 1> initial_camera_position_vector;
  initial_camera_position_vector
      << sphere_radius * sin(average_theta) * cos(average_phi),
      sphere_radius * sin(average_theta) * sin(average_phi),
      sphere_radius * cos(average_theta);

  return initial_camera_position_vector;
}

Eigen::Matrix<float, 1, 3>
Segment::computePerpendicularVector(Eigen::Matrix<float, 1, 3> v_input) {

  Eigen::Matrix<float, 3, 1> v_perp;

  int number_of_non_zero_elements = 0;
  for (int i = 0; i < 3; i++) {
    if (v_input(i) != 0)
      cout << "v_input(i): " << v_input(i) << endl;
    number_of_non_zero_elements++;
  }
  cout << "number_of_non_zero_elements: " << number_of_non_zero_elements
       << endl;

  if (number_of_non_zero_elements == 3) {
    v_perp(0) = 1.0;
    v_perp(1) = 0.0;
    v_perp(2) = -1 * (v_input(0)) / (v_input(2));
  } else if (number_of_non_zero_elements == 2) {
    if (v_input(1) == 0) {
      v_perp(0) = 1.0;
      v_perp(1) = 1.0;
      v_perp(2) = -v_input(1) / (v_input(2));
    } else if (v_input(2) == 0) {
      v_perp(0) = 1.0f;
      v_perp(1) = 1.0f;
      v_perp(2) = -v_input(0) / (v_input(2));
    } else {
      v_perp(0) = 1.0f;
      v_perp(1) = 1.0f;
      v_perp(2) = -v_input(1) / (v_input(0));
    }
  } else {
    if (number_of_non_zero_elements < 1) {
      v_perp(0) = 0.0f;
      v_perp(1) = 1.0f;
      v_perp(2) = 0.0f;
    } else {

      v_perp(0) = 1.0f;
      v_perp(1) = 0.0f;
      v_perp(2) = 0.0f;
    }
  }

  cout << "v_perp: " << v_perp << endl;
  return v_perp;
} // namespace DEF_OBJ_TRACK
} // namespace DEF_OBJ_TRACK
