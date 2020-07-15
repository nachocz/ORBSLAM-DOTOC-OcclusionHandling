#include "BestNextView.h"

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
namespace DEF_OBJ_TRACK
{
  BestNextView::BestNextView(/* args */)
  {
  }

  BestNextView::~BestNextView()
  {
  }

  void BestNextView::testFunction()
  {
    std::map<uint32_t, pcl::Supervoxel<PointTSuperVoxel>::Ptr> segments_sv_map_;
    //cout << "testFunction" << endl;
  }

  double *BestNextView::computeBestNextViewNoOcclusions(std::map<uint32_t, pcl::PointNormal> segments_normals, uint32_t number_of_sv_in_segment,
                                                        cv::Mat Twc_depth, cv::Mat camera_intrinsics_extended,
                                                        pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud,
                                                        pcl::visualization::PCLVisualizer::Ptr viewer,
                                                        std::ofstream &myfile)
  {
    //OBSERVATIONS

    //intrinsics
    double *focalDistance_ = new double[1];
    focalDistance_[0] = static_cast<double>(camera_intrinsics_extended.at<float>(0, 0)) * 100 / 640;
    const double *focalDistance = focalDistance_;

    //OBJECT SUPERVOXELS
    //cout << "number_of_sv_in_segment: " << number_of_sv_in_segment << endl;

    int num_observations_ = static_cast<int>(number_of_sv_in_segment); //remember, I was using index from 1 to inf, here from 0
    double *observations_ = new double[6 * num_observations_];

    for (int i = 0; i < num_observations_; ++i)
    {
      pcl::PointNormal sv_normal = segments_normals[i + 1]; //index in maps start at 1 for me (0 preserved for special cases)

      observations_[i * 6 + 0] = static_cast<double>(sv_normal.x);
      observations_[i * 6 + 1] = static_cast<double>(sv_normal.y);
      observations_[i * 6 + 2] = static_cast<double>(sv_normal.z);
      observations_[i * 6 + 3] = static_cast<double>(sv_normal.x + sv_normal.normal_x);
      observations_[i * 6 + 4] = static_cast<double>(sv_normal.y + sv_normal.normal_y);
      observations_[i * 6 + 5] = static_cast<double>(sv_normal.z + sv_normal.normal_z);

      //normal_visualization_cloud->push_back(sv_normal);
    }
    const double *observations = observations_;

    //IMAGE PLANE CENTER

    int num_image_plane_center_ = 3;
    double *image_plane_center_ = new double[num_image_plane_center_];

    image_plane_center_[0] = 0.0f;
    image_plane_center_[1] = 0.0f;
    image_plane_center_[2] = 0.5f;

    const double *image_plane_center = image_plane_center_;

    //PARAMETERES

    int num_rotation_parameters_ = 12;
    double *rotation_parameters_ = new double[num_rotation_parameters_];

    //cout << "Twc_: " << endl
    //<< Twc_depth << endl;

    for (int j = 0; j < 3; j++) //majoring colums
    {
      for (int i = 0; i < 3; i++)
      {
        rotation_parameters_[3 * j + i] = static_cast<double>(Twc_depth.at<float>(i, j));
        //cout << 3 * j + i << " " << rotation_parameters_[3 * j + i] << ", ";
      }
      //cout << endl;
    }

    double *rodrigues_rotation_parameteres_ = new double[3];
    ceres::RotationMatrixToAngleAxis(rotation_parameters_, rodrigues_rotation_parameteres_);

    int num_translation_parameters_ = 3;
    double *translation_parameters_ = new double[num_translation_parameters_];

    for (int i = 0; i < 3; i++)
    {
      translation_parameters_[i] = static_cast<double>(Twc_depth.at<float>(i, 3));
    }

    //cout << "%%%%%%%%%%%% POSITION OPTIMIZATION BEGINS %%%%%%%%%%%%%%%%%%%%" << endl;
    //testing values
    //cout << "OBBcenter" << endl;
    //cout << augmentedOBBcornersAndCentre[3 * 4 + 0] << endl
    //     << augmentedOBBcornersAndCentre[3 * 4 + 1] << endl
    //     << augmentedOBBcornersAndCentre[3 * 4 + 2] << endl;
    // Position Solving:

    ceres::Problem problemPosition;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).

    //Position Optimization functions regarding the object
    for (int i = 0; i < num_observations_; i++)
    {

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<BestNextViewNormalAlignmentPosition, 2, 3>(new BestNextViewNormalAlignmentPosition(observations[6 * i + 0],
                                                                                                                             observations[6 * i + 1],
                                                                                                                             observations[6 * i + 2],
                                                                                                                             observations[6 * i + 3],
                                                                                                                             observations[6 * i + 4],
                                                                                                                             observations[6 * i + 5]));
      problemPosition.AddResidualBlock(cost_function, NULL /*new ceres::HuberLoss(0.01)*/, translation_parameters_);
    }

    //Poisition Optimization functions regarding occlusions
    //cout << "num_occlusions_: " << num_occlusions_ << endl;
    // for (int i = 0; i < num_occlusions_; i++)
    // {

    //   ceres::CostFunction *cost_function =
    //       new ceres::AutoDiffCostFunction<BestNextViewAvoidOcclusions, 1, 3>(new BestNextViewAvoidOcclusions(occlusions[6 * i + 0],
    //                                                                                                          occlusions[6 * i + 1],
    //                                                                                                          occlusions[6 * i + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 0 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 0 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 0 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 1 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 1 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 1 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 2 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 2 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 2 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 3 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 3 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 3 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 4 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 4 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 4 + 2]));
    //   problemPosition.AddResidualBlock(cost_function, NULL /*new ceres::HuberLoss(0.01)*/, translation_parameters_);
    // }

    // Run the solver!
    ceres::Solver::Options optionsPosition;
    optionsPosition.num_threads = 4;
    optionsPosition.max_num_iterations = 1000;
    optionsPosition.linear_solver_type = ceres::DENSE_SCHUR;
    optionsPosition.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summaryPosition;
    ceres::Solve(optionsPosition, &problemPosition, &summaryPosition);

    std::ostringstream opt_1_text;
    opt_1_text //<< "Optimization" << endl
        //<< "Init. cost: " << summaryPosition.initial_cost << endl
        //<< "Final. cost: " << summaryPosition.final_cost << endl
        << "Cost Incr.: " << summaryPosition.initial_cost - summaryPosition.final_cost << endl;
    //<< "Optimisation time: " << summaryPosition.total_time_in_seconds << " s " << endl
    //<< "nº Threads used: " << summaryPosition.num_threads_used << endl;

    //cout << opt_1_text.str() << endl;

    viewer->addText(opt_1_text.str(), 10, 800, 20, 0, 0, 0, "opt_1_text text");
    //std::cout << summaryPosition.FullReport() << "\n";

    //cout << "translation_parameters_: " << translation_parameters_[0] << ", " << translation_parameters_[1] << ", " << translation_parameters_[2] << ";" << endl;

    //cout << "%%%%%%%%%%%% POSITION OPTIMIZATION ENDS %%%%%%%%%%%%%%%%%%%%" << endl;

    // Orientation solving:

    //cout << "%%%%%%%%%%%% ANGLE OPTIMIZATION BEGINS %%%%%%%%%%%%%%%%%%%%" << endl;

    const double *translation_parameters = translation_parameters_;

    ceres::Problem problemOrientation;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).

    //Optimization funtions regarding orientation
    for (int i = 0; i < num_observations_; i++)
    {

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<BestNextViewNormalAlignmentOrientation, 1, 3>(new BestNextViewNormalAlignmentOrientation(focalDistance[0],
                                                                                                                                   image_plane_center[0],
                                                                                                                                   image_plane_center[1],
                                                                                                                                   image_plane_center[2],
                                                                                                                                   translation_parameters[0],
                                                                                                                                   translation_parameters[1],
                                                                                                                                   translation_parameters[2],
                                                                                                                                   observations[6 * i + 0],
                                                                                                                                   observations[6 * i + 1],
                                                                                                                                   observations[6 * i + 2],
                                                                                                                                   observations[6 * i + 3],
                                                                                                                                   observations[6 * i + 4],
                                                                                                                                   observations[6 * i + 5]));
      problemOrientation.AddResidualBlock(cost_function, NULL, rodrigues_rotation_parameteres_);
    }

    // Run the solver!
    ceres::Solver::Options optionsOrientation;
    optionsOrientation.max_num_iterations = 10000;
    optionsOrientation.linear_solver_type = ceres::DENSE_SCHUR;
    optionsOrientation.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summaryOrientation;
    ceres::Solve(optionsOrientation, &problemOrientation, &summaryOrientation);
    //std::cout << summaryOrientation.FullReport() << "\n";
    // std::ostringstream opt_2_text;
    // opt_2_text << "Optimization 2 (Rotation)" << endl
    //            << "Init. cost: " << summaryOrientation.initial_cost << endl
    //            << "Final. cost: " << summaryOrientation.final_cost << endl
    //            << "Cost Incr.: " << summaryOrientation.initial_cost - summaryOrientation.final_cost << endl
    //            << "Optimisation time: " << summaryOrientation.total_time_in_seconds << " s " << endl
    //            << "nº Threads used: " << summaryOrientation.num_threads_used << endl;

    // cout << opt_2_text.str() << endl;
    // cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;

    myfile << summaryPosition.initial_cost + summaryOrientation.initial_cost << ", "
           << summaryPosition.final_cost + summaryOrientation.final_cost << ", "
           << (summaryPosition.initial_cost - summaryPosition.final_cost) + (summaryOrientation.initial_cost - summaryOrientation.final_cost) << ", "
           << summaryPosition.total_time_in_seconds + summaryOrientation.total_time_in_seconds << ", "
           << summaryPosition.num_threads_used << ", ";

    //cout << "%%%%%%%%%%%% ANGLE OPTIMIZATION ENDS %%%%%%%%%%%%%%%%%%%%" << endl;

    //End of camera problems

    double *imagePlaneCenterWorldCoord = new double[num_image_plane_center_];
    ceres::AngleAxisRotatePoint(rodrigues_rotation_parameteres_, image_plane_center, imagePlaneCenterWorldCoord);

    imagePlaneCenterWorldCoord[0] += translation_parameters_[0];
    imagePlaneCenterWorldCoord[1] += translation_parameters_[1];
    imagePlaneCenterWorldCoord[2] += translation_parameters_[2];

    pcl::PointNormal z_axis_world_coord; //index in maps start at 1 for me (0 preserved for special cases)

    z_axis_world_coord.x = static_cast<float>(translation_parameters_[0]);
    z_axis_world_coord.y = static_cast<float>(translation_parameters_[1]);
    z_axis_world_coord.z = static_cast<float>(translation_parameters_[2]);
    z_axis_world_coord.normal_x = static_cast<float>(imagePlaneCenterWorldCoord[0] - translation_parameters_[0]);
    z_axis_world_coord.normal_y = static_cast<float>(imagePlaneCenterWorldCoord[1] - translation_parameters_[1]);
    z_axis_world_coord.normal_z = static_cast<float>(imagePlaneCenterWorldCoord[2] - translation_parameters_[2]);

    normal_visualization_cloud->push_back(z_axis_world_coord);

    //cout << "End of summary, rodrigues rotation:" << endl;
    //cout << rodrigues_rotation_parameteres_[0] << ", " << rodrigues_rotation_parameteres_[1] << ", " << rodrigues_rotation_parameteres_[2] << ", " << endl;

    double *rotation_parameters_out_ = new double[3 * 3];

    ceres::AngleAxisToRotationMatrix(rodrigues_rotation_parameteres_, rotation_parameters_out_);

    int num_parameters_ = 12;
    double *parameters_ = new double[num_parameters_];

    parameters_[0] = rotation_parameters_out_[0];
    parameters_[1] = rotation_parameters_out_[3];
    parameters_[2] = rotation_parameters_out_[6];

    parameters_[3] = translation_parameters_[0];

    parameters_[4] = rotation_parameters_out_[1];
    parameters_[5] = rotation_parameters_out_[4];
    parameters_[6] = rotation_parameters_out_[7];

    parameters_[7] = translation_parameters_[1];

    parameters_[8] = rotation_parameters_out_[2];
    parameters_[9] = rotation_parameters_out_[5];
    parameters_[10] = rotation_parameters_out_[8];

    parameters_[11] = translation_parameters_[2];

    return parameters_;

    //cout << "End of optimization" << endl;
  }

  double *BestNextView::computeBestNextView(std::map<uint32_t, pcl::PointNormal> segments_normals, uint32_t number_of_sv_in_segment,
                                            cv::Mat Twc_depth, cv::Mat camera_intrinsics_extended,
                                            std::map<uint32_t, pcl::PointXYZ> OBB_mid_plane_corners_world_ref,
                                            std::map<uint32_t, pcl::PointNormal> occlusion_normals, uint32_t number_of_occlusions,
                                            pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud,
                                            pcl::visualization::PCLVisualizer::Ptr viewer,
                                            std::ofstream &myfile)
  {
    //OBSERVATIONS

    //intrinsics
    double *focalDistance_ = new double[1];
    focalDistance_[0] = static_cast<double>(camera_intrinsics_extended.at<float>(0, 0)) * 100 / 640;
    const double *focalDistance = focalDistance_;

    //OBJECT SUPERVOXELS
    //cout << "number_of_sv_in_segment: " << number_of_sv_in_segment << endl;

    int num_observations_ = static_cast<int>(number_of_sv_in_segment); //remember, I was using index from 1 to inf, here from 0
    double *observations_ = new double[6 * num_observations_];

    for (int i = 0; i < num_observations_; ++i)
    {
      pcl::PointNormal sv_normal = segments_normals[i + 1]; //index in maps start at 1 for me (0 preserved for special cases)

      observations_[i * 6 + 0] = static_cast<double>(sv_normal.x);
      observations_[i * 6 + 1] = static_cast<double>(sv_normal.y);
      observations_[i * 6 + 2] = static_cast<double>(sv_normal.z);
      observations_[i * 6 + 3] = static_cast<double>(sv_normal.x + sv_normal.normal_x);
      observations_[i * 6 + 4] = static_cast<double>(sv_normal.y + sv_normal.normal_y);
      observations_[i * 6 + 5] = static_cast<double>(sv_normal.z + sv_normal.normal_z);

      //normal_visualization_cloud->push_back(sv_normal);
    }
    const double *observations = observations_;

    //OBB CORNERS WORLD REFERENCE

    int num_augmentedOBBcornersAndCentre_ = static_cast<int>(5); //remember, I was using index from 1 to inf, here from 0
    double *augmentedOBBcornersAndCentre_ = new double[3 * num_augmentedOBBcornersAndCentre_];
    for (int i = 0; i < num_augmentedOBBcornersAndCentre_; ++i)
    {
      pcl::PointXYZ OBB_corner = OBB_mid_plane_corners_world_ref[i + 1]; //index in maps start at 1 for me (0 preserved for special cases)

      augmentedOBBcornersAndCentre_[i * 3 + 0] = static_cast<double>(OBB_corner.x);
      augmentedOBBcornersAndCentre_[i * 3 + 1] = static_cast<double>(OBB_corner.y);
      augmentedOBBcornersAndCentre_[i * 3 + 2] = static_cast<double>(OBB_corner.z);

      //normal_visualization_cloud->push_back(sv_normal);
    }
    const double *augmentedOBBcornersAndCentre = augmentedOBBcornersAndCentre_;

    //OCCLUSIONS SUPERVOXEL CENTERS
    //cout << "number_of_occlusions: " << number_of_occlusions << endl;

    int num_occlusions_ = static_cast<int>(number_of_occlusions); //remember, I was using index from 1 to inf, here from 0
    double *occlusions_ = new double[6 * number_of_occlusions];

    for (int i = 0; i < num_occlusions_; ++i)
    {
      pcl::PointNormal sv_normal = segments_normals[i + 1]; //index in maps start at 1 for me (0 preserved for special cases)

      occlusions_[i * 6 + 0] = static_cast<double>(sv_normal.x);
      occlusions_[i * 6 + 1] = static_cast<double>(sv_normal.y);
      occlusions_[i * 6 + 2] = static_cast<double>(sv_normal.z);
      occlusions_[i * 6 + 3] = static_cast<double>(sv_normal.x + sv_normal.normal_x);
      occlusions_[i * 6 + 4] = static_cast<double>(sv_normal.y + sv_normal.normal_y);
      occlusions_[i * 6 + 5] = static_cast<double>(sv_normal.z + sv_normal.normal_z);

      //normal_visualization_cloud->push_back(sv_normal);
    }
    const double *occlusions = occlusions_;

    //IMAGE PLANE CENTER

    int num_image_plane_center_ = 3;
    double *image_plane_center_ = new double[num_image_plane_center_];

    image_plane_center_[0] = 0.0f;
    image_plane_center_[1] = 0.0f;
    image_plane_center_[2] = 0.5f;

    const double *image_plane_center = image_plane_center_;

    //PARAMETERES

    int num_rotation_parameters_ = 12;
    double *rotation_parameters_ = new double[num_rotation_parameters_];

    //cout << "Twc_: " << endl
    //<< Twc_depth << endl;

    for (int j = 0; j < 3; j++) //majoring colums
    {
      for (int i = 0; i < 3; i++)
      {
        rotation_parameters_[3 * j + i] = static_cast<double>(Twc_depth.at<float>(i, j));
        //cout << 3 * j + i << " " << rotation_parameters_[3 * j + i] << ", ";
      }
      //cout << endl;
    }

    double *rodrigues_rotation_parameteres_ = new double[3];
    ceres::RotationMatrixToAngleAxis(rotation_parameters_, rodrigues_rotation_parameteres_);

    int num_translation_parameters_ = 3;
    double *translation_parameters_ = new double[num_translation_parameters_];

    for (int i = 0; i < 3; i++)
    {
      translation_parameters_[i] = static_cast<double>(Twc_depth.at<float>(i, 3));
    }

    //cout << "%%%%%%%%%%%% POSITION OPTIMIZATION BEGINS %%%%%%%%%%%%%%%%%%%%" << endl;
    //testing values
    //cout << "OBBcenter" << endl;
    //cout << augmentedOBBcornersAndCentre[3 * 4 + 0] << endl
    //     << augmentedOBBcornersAndCentre[3 * 4 + 1] << endl
    //     << augmentedOBBcornersAndCentre[3 * 4 + 2] << endl;
    // Position Solving:

    ceres::Problem problemPosition;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).

    //Position Optimization functions regarding the object
    for (int i = 0; i < num_observations_; i++)
    {

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<BestNextViewNormalAlignmentPosition, 2, 3>(new BestNextViewNormalAlignmentPosition(observations[6 * i + 0],
                                                                                                                             observations[6 * i + 1],
                                                                                                                             observations[6 * i + 2],
                                                                                                                             observations[6 * i + 3],
                                                                                                                             observations[6 * i + 4],
                                                                                                                             observations[6 * i + 5]));
      problemPosition.AddResidualBlock(cost_function, NULL /*new ceres::HuberLoss(0.01)*/, translation_parameters_);
    }

    //Poisition Optimization functions regarding occlusions
    //cout << "num_occlusions_: " << num_occlusions_ << endl;
    // for (int i = 0; i < num_occlusions_; i++)
    // {

    //   ceres::CostFunction *cost_function =
    //       new ceres::AutoDiffCostFunction<BestNextViewAvoidOcclusions, 1, 3>(new BestNextViewAvoidOcclusions(occlusions[6 * i + 0],
    //                                                                                                          occlusions[6 * i + 1],
    //                                                                                                          occlusions[6 * i + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 0 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 0 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 0 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 1 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 1 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 1 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 2 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 2 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 2 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 3 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 3 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 3 + 2],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 4 + 0],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 4 + 1],
    //                                                                                                          augmentedOBBcornersAndCentre[3 * 4 + 2]));
    //   problemPosition.AddResidualBlock(cost_function, NULL /*new ceres::HuberLoss(0.01)*/, translation_parameters_);
    // }

    // Run the solver!
    ceres::Solver::Options optionsPosition;
    optionsPosition.num_threads = 4;
    optionsPosition.max_num_iterations = 1000;
    optionsPosition.linear_solver_type = ceres::DENSE_SCHUR;
    optionsPosition.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summaryPosition;
    ceres::Solve(optionsPosition, &problemPosition, &summaryPosition);

    // std::ostringstream opt_1_text;
    // opt_1_text << "Optimization" << endl
    //            << "Init. cost: " << summaryPosition.initial_cost << endl
    //            << "Final. cost: " << summaryPosition.final_cost << endl
    //            << "Cost Incr.: " << summaryPosition.initial_cost - summaryPosition.final_cost << endl
    //            << "Optimisation time: " << summaryPosition.total_time_in_seconds << " s " << endl
    //            << "nº Threads used: " << summaryPosition.num_threads_used << endl;

    // cout << opt_1_text.str() << endl;

    //viewer->addText(opt_1_text.str(), 10, 800, 20, 0, 0, 0, "opt_1_text text");
    //std::cout << summaryPosition.FullReport() << "\n";

    //cout << "translation_parameters_: " << translation_parameters_[0] << ", " << translation_parameters_[1] << ", " << translation_parameters_[2] << ";" << endl;

    //cout << "%%%%%%%%%%%% POSITION OPTIMIZATION ENDS %%%%%%%%%%%%%%%%%%%%" << endl;

    // Orientation solving:

    //cout << "%%%%%%%%%%%% ANGLE OPTIMIZATION BEGINS %%%%%%%%%%%%%%%%%%%%" << endl;

    const double *translation_parameters = translation_parameters_;

    ceres::Problem problemOrientation;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).

    //Optimization funtions regarding orientation
    for (int i = 0; i < num_observations_; i++)
    {

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<BestNextViewNormalAlignmentOrientation, 1, 3>(new BestNextViewNormalAlignmentOrientation(focalDistance[0],
                                                                                                                                   image_plane_center[0],
                                                                                                                                   image_plane_center[1],
                                                                                                                                   image_plane_center[2],
                                                                                                                                   translation_parameters[0],
                                                                                                                                   translation_parameters[1],
                                                                                                                                   translation_parameters[2],
                                                                                                                                   observations[6 * i + 0],
                                                                                                                                   observations[6 * i + 1],
                                                                                                                                   observations[6 * i + 2],
                                                                                                                                   observations[6 * i + 3],
                                                                                                                                   observations[6 * i + 4],
                                                                                                                                   observations[6 * i + 5]));
      problemOrientation.AddResidualBlock(cost_function, NULL, rodrigues_rotation_parameteres_);
    }

    // Run the solver!
    ceres::Solver::Options optionsOrientation;
    optionsOrientation.max_num_iterations = 10000;
    optionsOrientation.linear_solver_type = ceres::DENSE_SCHUR;
    optionsOrientation.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summaryOrientation;
    ceres::Solve(optionsOrientation, &problemOrientation, &summaryOrientation);
    //std::cout << summaryOrientation.FullReport() << "\n";
    // std::ostringstream opt_2_text;
    // opt_2_text << "Optimization 2 (Rotation)" << endl
    //            << "Init. cost: " << summaryOrientation.initial_cost << endl
    //            << "Final. cost: " << summaryOrientation.final_cost << endl
    //            << "Cost Incr.: " << summaryOrientation.initial_cost - summaryOrientation.final_cost << endl
    //            << "Optimisation time: " << summaryOrientation.total_time_in_seconds << " s " << endl
    //            << "nº Threads used: " << summaryOrientation.num_threads_used << endl;

    // cout << opt_2_text.str() << endl;
    // cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;

    myfile << summaryPosition.initial_cost + summaryOrientation.initial_cost << ", "
           << summaryPosition.final_cost + summaryOrientation.final_cost << ", "
           << (summaryPosition.initial_cost - summaryPosition.final_cost) + (summaryOrientation.initial_cost - summaryOrientation.final_cost) << ", "
           << summaryPosition.total_time_in_seconds + summaryOrientation.total_time_in_seconds << ", "
           << summaryPosition.num_threads_used << ", ";

    //cout << "%%%%%%%%%%%% ANGLE OPTIMIZATION ENDS %%%%%%%%%%%%%%%%%%%%" << endl;

    //End of camera problems

    double *imagePlaneCenterWorldCoord = new double[num_image_plane_center_];
    ceres::AngleAxisRotatePoint(rodrigues_rotation_parameteres_, image_plane_center, imagePlaneCenterWorldCoord);

    imagePlaneCenterWorldCoord[0] += translation_parameters_[0];
    imagePlaneCenterWorldCoord[1] += translation_parameters_[1];
    imagePlaneCenterWorldCoord[2] += translation_parameters_[2];

    pcl::PointNormal z_axis_world_coord; //index in maps start at 1 for me (0 preserved for special cases)

    z_axis_world_coord.x = static_cast<float>(translation_parameters_[0]);
    z_axis_world_coord.y = static_cast<float>(translation_parameters_[1]);
    z_axis_world_coord.z = static_cast<float>(translation_parameters_[2]);
    z_axis_world_coord.normal_x = static_cast<float>(imagePlaneCenterWorldCoord[0] - translation_parameters_[0]);
    z_axis_world_coord.normal_y = static_cast<float>(imagePlaneCenterWorldCoord[1] - translation_parameters_[1]);
    z_axis_world_coord.normal_z = static_cast<float>(imagePlaneCenterWorldCoord[2] - translation_parameters_[2]);

    normal_visualization_cloud->push_back(z_axis_world_coord);

    //cout << "End of summary, rodrigues rotation:" << endl;
    //cout << rodrigues_rotation_parameteres_[0] << ", " << rodrigues_rotation_parameteres_[1] << ", " << rodrigues_rotation_parameteres_[2] << ", " << endl;

    double *rotation_parameters_out_ = new double[3 * 3];

    ceres::AngleAxisToRotationMatrix(rodrigues_rotation_parameteres_, rotation_parameters_out_);

    int num_parameters_ = 12;
    double *parameters_ = new double[num_parameters_];

    parameters_[0] = rotation_parameters_out_[0];
    parameters_[1] = rotation_parameters_out_[3];
    parameters_[2] = rotation_parameters_out_[6];

    parameters_[3] = translation_parameters_[0];

    parameters_[4] = rotation_parameters_out_[1];
    parameters_[5] = rotation_parameters_out_[4];
    parameters_[6] = rotation_parameters_out_[7];

    parameters_[7] = translation_parameters_[1];

    parameters_[8] = rotation_parameters_out_[2];
    parameters_[9] = rotation_parameters_out_[5];
    parameters_[10] = rotation_parameters_out_[8];

    parameters_[11] = translation_parameters_[2];

    return parameters_;

    //cout << "End of optimization" << endl;
  }

  double *BestNextView::computeBestNextViewSimple(std::map<uint32_t, pcl::PointNormal> segments_normals,
                                                  uint32_t number_of_sv_in_segment, cv::Mat Twc_depth,
                                                  cv::Mat camera_intrinsics_extended, pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud)
  {
    //OBSERVATION

    double *focalDistance_ = new double[1];
    focalDistance_[0] = static_cast<double>(camera_intrinsics_extended.at<float>(0, 0)) * 100 / 640;
    const double *focalDistance = focalDistance_;

    //cout << "number_of_sv_in_segment: " << number_of_sv_in_segment << endl;

    int num_observations_ = static_cast<int>(number_of_sv_in_segment); //remember, I was using index from 1 to inf, here from 0
    double *observations_ = new double[6 * num_observations_];

    for (int i = 0; i < num_observations_; ++i)
    {
      pcl::PointNormal sv_normal = segments_normals[i + 1]; //index in maps start at 1 for me (0 preserved for special cases)

      observations_[i * 6 + 0] = static_cast<double>(sv_normal.x);
      observations_[i * 6 + 1] = static_cast<double>(sv_normal.y);
      observations_[i * 6 + 2] = static_cast<double>(sv_normal.z);
      observations_[i * 6 + 3] = static_cast<double>(sv_normal.x + sv_normal.normal_x);
      observations_[i * 6 + 4] = static_cast<double>(sv_normal.y + sv_normal.normal_y);
      observations_[i * 6 + 5] = static_cast<double>(sv_normal.z + sv_normal.normal_z);

      //normal_visualization_cloud->push_back(sv_normal);
    }
    const double *observations = observations_;

    int num_image_plane_center_ = 3;
    double *image_plane_center_ = new double[num_image_plane_center_];

    image_plane_center_[0] = 0.0f;
    image_plane_center_[1] = 0.0f;
    image_plane_center_[2] = 0.5f;

    const double *image_plane_center = image_plane_center_;

    //PARAMETERES

    int num_rotation_parameters_ = 12;
    double *rotation_parameters_ = new double[num_rotation_parameters_];

    //cout << "Twc_: " << endl
    //<< Twc_depth << endl;

    for (int j = 0; j < 3; j++) //majoring colums
    {
      for (int i = 0; i < 3; i++)
      {
        rotation_parameters_[3 * j + i] = static_cast<double>(Twc_depth.at<float>(i, j));
        //cout << 3 * j + i << " " << rotation_parameters_[3 * j + i] << ", ";
      }
      //cout << endl;
    }

    double *rodrigues_rotation_parameteres_ = new double[3];
    ceres::RotationMatrixToAngleAxis(rotation_parameters_, rodrigues_rotation_parameteres_);

    int num_translation_parameters_ = 3;
    double *translation_parameters_ = new double[num_translation_parameters_];

    for (int i = 0; i < 3; i++)
    {
      translation_parameters_[i] = static_cast<double>(Twc_depth.at<float>(i, 3));
    }

    //cout << "%%%%%%%%%%%% POSITION OPTIMIZATION BEGINS %%%%%%%%%%%%%%%%%%%%" << endl;

    // Position Solving:

    ceres::Problem problemPosition;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).

    for (int i = 0; i < num_observations_; i++)
    {

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<BestNextViewNormalAlignmentPosition, 2, 3>(new BestNextViewNormalAlignmentPosition(observations[6 * i + 0],
                                                                                                                             observations[6 * i + 1],
                                                                                                                             observations[6 * i + 2],
                                                                                                                             observations[6 * i + 3],
                                                                                                                             observations[6 * i + 4],
                                                                                                                             observations[6 * i + 5]));
      problemPosition.AddResidualBlock(cost_function, NULL /*new ceres::HuberLoss(0.01)*/, translation_parameters_);
    }

    // Run the solver!
    ceres::Solver::Options optionsPosition;
    optionsPosition.max_num_iterations = 1000;
    optionsPosition.linear_solver_type = ceres::DENSE_SCHUR;
    optionsPosition.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summaryPosition;
    ceres::Solve(optionsPosition, &problemPosition, &summaryPosition);
    ////cout << summaryPosition.FullReport() << "\n";

    //cout << "translation_parameters_: " << translation_parameters_[0] << ", " << translation_parameters_[1] << ", " << translation_parameters_[2] << ";" << endl;

    //cout << "%%%%%%%%%%%% POSITION OPTIMIZATION ENDS %%%%%%%%%%%%%%%%%%%%" << endl;

    // Orientation solving:

    //cout << "%%%%%%%%%%%% ANGLE OPTIMIZATION BEGINS %%%%%%%%%%%%%%%%%%%%" << endl;

    const double *translation_parameters = translation_parameters_;

    ceres::Problem problemOrientation;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).

    for (int i = 0; i < num_observations_; i++)
    {

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<BestNextViewNormalAlignmentOrientation, 1, 3>(new BestNextViewNormalAlignmentOrientation(focalDistance[0],
                                                                                                                                   image_plane_center[0],
                                                                                                                                   image_plane_center[1],
                                                                                                                                   image_plane_center[2],
                                                                                                                                   translation_parameters[0],
                                                                                                                                   translation_parameters[1],
                                                                                                                                   translation_parameters[2],
                                                                                                                                   observations[6 * i + 0],
                                                                                                                                   observations[6 * i + 1],
                                                                                                                                   observations[6 * i + 2],
                                                                                                                                   observations[6 * i + 3],
                                                                                                                                   observations[6 * i + 4],
                                                                                                                                   observations[6 * i + 5]));
      problemOrientation.AddResidualBlock(cost_function, NULL, rodrigues_rotation_parameteres_);
    }

    // Run the solver!
    ceres::Solver::Options optionsOrientation;
    optionsOrientation.max_num_iterations = 10000;
    optionsOrientation.linear_solver_type = ceres::DENSE_SCHUR;
    optionsOrientation.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summaryOrientation;
    ceres::Solve(optionsOrientation, &problemOrientation, &summaryOrientation);
    //std::cout << summaryOrientation.FullReport() << "\n";

    //cout << "%%%%%%%%%%%% ANGLE OPTIMIZATION ENDS %%%%%%%%%%%%%%%%%%%%" << endl;

    //End of camera problems

    double *imagePlaneCenterWorldCoord = new double[num_image_plane_center_];
    ceres::AngleAxisRotatePoint(rodrigues_rotation_parameteres_, image_plane_center, imagePlaneCenterWorldCoord);

    imagePlaneCenterWorldCoord[0] += translation_parameters_[0];
    imagePlaneCenterWorldCoord[1] += translation_parameters_[1];
    imagePlaneCenterWorldCoord[2] += translation_parameters_[2];

    pcl::PointNormal z_axis_world_coord; //index in maps start at 1 for me (0 preserved for special cases)

    z_axis_world_coord.x = static_cast<float>(translation_parameters_[0]);
    z_axis_world_coord.y = static_cast<float>(translation_parameters_[1]);
    z_axis_world_coord.z = static_cast<float>(translation_parameters_[2]);
    z_axis_world_coord.normal_x = static_cast<float>(imagePlaneCenterWorldCoord[0] - translation_parameters_[0]);
    z_axis_world_coord.normal_y = static_cast<float>(imagePlaneCenterWorldCoord[1] - translation_parameters_[1]);
    z_axis_world_coord.normal_z = static_cast<float>(imagePlaneCenterWorldCoord[2] - translation_parameters_[2]);

    normal_visualization_cloud->push_back(z_axis_world_coord);

    //cout << "End of summary, rodrigues rotation:" << endl;
    //cout << rodrigues_rotation_parameteres_[0] << ", " << rodrigues_rotation_parameteres_[1] << ", " << rodrigues_rotation_parameteres_[2] << ", " << endl;

    double *rotation_parameters_out_ = new double[3 * 3];

    ceres::AngleAxisToRotationMatrix(rodrigues_rotation_parameteres_, rotation_parameters_out_);

    int num_parameters_ = 12;
    double *parameters_ = new double[num_parameters_];

    parameters_[0] = rotation_parameters_out_[0];
    parameters_[1] = rotation_parameters_out_[3];
    parameters_[2] = rotation_parameters_out_[6];

    parameters_[3] = translation_parameters_[0];

    parameters_[4] = rotation_parameters_out_[1];
    parameters_[5] = rotation_parameters_out_[4];
    parameters_[6] = rotation_parameters_out_[7];

    parameters_[7] = translation_parameters_[1];

    parameters_[8] = rotation_parameters_out_[2];
    parameters_[9] = rotation_parameters_out_[5];
    parameters_[10] = rotation_parameters_out_[8];

    parameters_[11] = translation_parameters_[2];

    return parameters_;

    //cout << "End of optimization" << endl;
  }

  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  //ALL TOGETHER
  double *BestNextView::computeBestNextViewAllTogether(std::map<uint32_t, pcl::PointNormal> segments_normals,
                                                       uint32_t number_of_sv_in_segment, cv::Mat Twc_depth,
                                                       cv::Mat camera_intrinsics_extended, pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud)
  {
    //OBSERVATION

    double *focalDistance_ = new double[1];
    focalDistance_[0] = static_cast<double>(camera_intrinsics_extended.at<float>(0, 0)) * 100 / 640;
    const double *focalDistance = focalDistance_;

    //cout << "number_of_sv_in_segment: " << number_of_sv_in_segment << endl;

    int num_observations_ = static_cast<int>(number_of_sv_in_segment); //remember, I was using index from 1 to inf, here from 0
    double *observations_ = new double[6 * num_observations_];

    for (int i = 0; i < num_observations_; ++i)
    {
      pcl::PointNormal sv_normal = segments_normals[i + 1]; //index in maps start at 1 for me (0 preserved for special cases)

      observations_[i * 6 + 0] = static_cast<double>(sv_normal.x);
      observations_[i * 6 + 1] = static_cast<double>(sv_normal.y);
      observations_[i * 6 + 2] = static_cast<double>(sv_normal.z);
      observations_[i * 6 + 3] = static_cast<double>(sv_normal.x + sv_normal.normal_x);
      observations_[i * 6 + 4] = static_cast<double>(sv_normal.y + sv_normal.normal_y);
      observations_[i * 6 + 5] = static_cast<double>(sv_normal.z + sv_normal.normal_z);

      //normal_visualization_cloud->push_back(sv_normal);
    }
    const double *observations = observations_;

    int num_image_plane_center_ = 3;
    double *image_plane_center_ = new double[num_image_plane_center_];

    image_plane_center_[0] = 0.0f;
    image_plane_center_[1] = 0.0f;
    image_plane_center_[2] = 0.5f;

    const double *image_plane_center = image_plane_center_;

    //PARAMETERES

    int num_rotation_parameters_ = 12;
    double *rotation_parameters_ = new double[num_rotation_parameters_];

    //cout << "Twc_: " << endl
    //<< Twc_depth << endl;

    for (int j = 0; j < 3; j++) //majoring colums
    {
      for (int i = 0; i < 3; i++)
      {
        rotation_parameters_[3 * j + i] = static_cast<double>(Twc_depth.at<float>(i, j));
        //cout << 3 * j + i << " " << rotation_parameters_[3 * j + i] << ", ";
      }
      //cout << endl;
    }

    double *rodrigues_rotation_parameteres_ = new double[3];
    ceres::RotationMatrixToAngleAxis(rotation_parameters_, rodrigues_rotation_parameteres_);

    int num_translation_parameters_ = 3;
    double *translation_parameters_ = new double[num_translation_parameters_];

    for (int i = 0; i < 3; i++)
    {
      translation_parameters_[i] = static_cast<double>(Twc_depth.at<float>(i, 3));
    }

    //cout << "%%%%%%%%%%%% POSITION OPTIMIZATION BEGINS %%%%%%%%%%%%%%%%%%%%" << endl;

    // Position Solving:

    ceres::Problem problem;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).

    for (int i = 0; i < num_observations_; i++)
    {

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<BestNextViewAligment, 3, 3, 3>(new BestNextViewAligment(focalDistance[0],
                                                                                                  image_plane_center[0],
                                                                                                  image_plane_center[1],
                                                                                                  image_plane_center[2],
                                                                                                  observations[6 * i + 0],
                                                                                                  observations[6 * i + 1],
                                                                                                  observations[6 * i + 2],
                                                                                                  observations[6 * i + 3],
                                                                                                  observations[6 * i + 4],
                                                                                                  observations[6 * i + 5]));
      problem.AddResidualBlock(cost_function, NULL, rodrigues_rotation_parameteres_, translation_parameters_);
    }

    // Run the solver!
    ceres::Solver::Options options;
    options.max_num_iterations = 10000;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.FullReport() << "\n";

    //cout << "%%%%%%%%%%%% ANGLE OPTIMIZATION ENDS %%%%%%%%%%%%%%%%%%%%" << endl;

    //End of camera problems

    double *imagePlaneCenterWorldCoord = new double[num_image_plane_center_];
    ceres::AngleAxisRotatePoint(rodrigues_rotation_parameteres_, image_plane_center, imagePlaneCenterWorldCoord);

    imagePlaneCenterWorldCoord[0] += translation_parameters_[0];
    imagePlaneCenterWorldCoord[1] += translation_parameters_[1];
    imagePlaneCenterWorldCoord[2] += translation_parameters_[2];

    pcl::PointNormal z_axis_world_coord; //index in maps start at 1 for me (0 preserved for special cases)

    z_axis_world_coord.x = static_cast<float>(translation_parameters_[0]);
    z_axis_world_coord.y = static_cast<float>(translation_parameters_[1]);
    z_axis_world_coord.z = static_cast<float>(translation_parameters_[2]);
    z_axis_world_coord.normal_x = static_cast<float>(imagePlaneCenterWorldCoord[0] - translation_parameters_[0]);
    z_axis_world_coord.normal_y = static_cast<float>(imagePlaneCenterWorldCoord[1] - translation_parameters_[1]);
    z_axis_world_coord.normal_z = static_cast<float>(imagePlaneCenterWorldCoord[2] - translation_parameters_[2]);

    normal_visualization_cloud->push_back(z_axis_world_coord);

    //cout << "End of summary, rodrigues rotation:" << endl;
    //cout << rodrigues_rotation_parameteres_[0] << ", " << rodrigues_rotation_parameteres_[1] << ", " << rodrigues_rotation_parameteres_[2] << ", " << endl;

    double *rotation_parameters_out_ = new double[3 * 3];

    ceres::AngleAxisToRotationMatrix(rodrigues_rotation_parameteres_, rotation_parameters_out_);

    int num_parameters_ = 12;
    double *parameters_ = new double[num_parameters_];

    parameters_[0] = rotation_parameters_out_[0];
    parameters_[1] = rotation_parameters_out_[3];
    parameters_[2] = rotation_parameters_out_[6];

    parameters_[3] = translation_parameters_[0];

    parameters_[4] = rotation_parameters_out_[1];
    parameters_[5] = rotation_parameters_out_[4];
    parameters_[6] = rotation_parameters_out_[7];

    parameters_[7] = translation_parameters_[1];

    parameters_[8] = rotation_parameters_out_[2];
    parameters_[9] = rotation_parameters_out_[5];
    parameters_[10] = rotation_parameters_out_[8];

    parameters_[11] = translation_parameters_[2];

    return parameters_;

    //cout << "End of optimization" << endl;
  }

} // namespace DEF_OBJ_TRACK

//CREATION OF SYNTHETIC OBSERVATIONS: 33 points and normals belonging to an ELLIPSOID

//  int num_observations_ = static_cast<int>(33); //remember, I was using index from 1 to inf, here from 0
//   double *observations_ = new double[6 * num_observations_];

//   float elips_x[num_observations_] = {3.000000e-01, 2.476007e-01, 1.087073e-01, 2.476007e-01, 2.043537e-01, 8.972003e-02, 1.087073e-01, 8.972003e-02, 3.939094e-02, -6.816063e-02, -5.625539e-02, -2.469853e-02, -2.212181e-01, -1.825792e-01, -8.016010e-02, -2.969977e-01, -2.451228e-01, -1.076194e-01, -2.690275e-01, -2.220380e-01, -9.748421e-02, -1.470782e-01, -1.213889e-01, -5.329494e-02, 2.624970e-02, 2.166481e-02, 9.511781e-03, 1.904079e-01, 1.571504e-01, 6.899577e-02, 2.880511e-01, 2.377388e-01, 1.043775e-01};
//   float elips_y[num_observations_] = {0, 0, 0, 2.258570e-01, 1.864078e-01, 8.184103e-02, 3.728156e-01, 3.076980e-01, 1.350926e-01, 3.895391e-01, 3.215005e-01, 1.411525e-01, 2.701853e-01, 2.229935e-01, 9.790373e-02, 5.644800e-02, 4.658855e-02, 2.045437e-02, -1.770082e-01, -1.460912e-01, -6.414029e-02, -3.486303e-01, -2.877370e-01, -1.263289e-01, -3.984658e-01, -3.288681e-01, -1.443872e-01, -3.091058e-01, -2.551160e-01, -1.120069e-01, -1.117662e-01, -9.224462e-02, -4.049935e-02};
//   float elips_z[num_observations_] = {5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01, 5.000000e-01, 6.693927e-01, 7.796117e-01};

//   float elips_xn[num_observations_] = {-6.666667e-01, -5.502237e-01, -2.415718e-01, -5.502237e-01, -4.541193e-01, -1.993778e-01, -2.415718e-01, -1.993778e-01, -8.753543e-02, 1.514681e-01, 1.250120e-01, 5.488563e-02, 4.915958e-01, 4.057315e-01, 1.781336e-01, 6.599950e-01, 5.447174e-01, 2.391543e-01, 5.978389e-01, 4.934178e-01, 2.166316e-01, 3.268405e-01, 2.697531e-01, 1.184332e-01, -5.833266e-02, -4.814402e-02, -2.113729e-02, -4.231286e-01, -3.492231e-01, -1.533239e-01, -6.401135e-01, -5.283085e-01, -2.319501e-01};
//   float elips_yn[num_observations_] = {0, 0, 0, -2.823212e-01, -2.330098e-01, -1.023013e-01, -4.660195e-01, -3.846225e-01, -1.688658e-01, -4.869238e-01, -4.018756e-01, -1.764406e-01, -3.377316e-01, -2.787419e-01, -1.223797e-01, -7.056000e-02, -5.823568e-02, -2.556796e-02, 2.212602e-01, 1.826139e-01, 8.017536e-02, 4.357879e-01, 3.596713e-01, 1.579111e-01, 4.980823e-01, 4.110851e-01, 1.804840e-01, 3.863822e-01, 3.188950e-01, 1.400086e-01, 1.397077e-01, 1.153058e-01, 5.062419e-02};
//   float elips_zn[num_observations_] = {0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01, 0, -3.764283e-01, -6.213594e-01};

//   for (int i = 0; i < num_observations_; ++i)
//   {
//     //pcl::PointNormal sv_normal = segments_normals[i + 1]; //index in maps start at 1 for me (0 preserved for special cases)

//     // observations_[i * 6 + 0] = static_cast<double>(sv_normal.x);
//     // observations_[i * 6 + 1] = static_cast<double>(sv_normal.y);
//     // observations_[i * 6 + 2] = static_cast<double>(sv_normal.z);
//     // observations_[i * 6 + 3] = static_cast<double>(sv_normal.x + sv_normal.normal_x);
//     // observations_[i * 6 + 4] = static_cast<double>(sv_normal.y + sv_normal.normal_y);
//     // observations_[i * 6 + 5] = static_cast<double>(sv_normal.z + sv_normal.normal_z);

//     pcl::PointNormal sv_normal;

//     sv_normal.x = elips_x[i];
//     sv_normal.y = elips_y[i];
//     sv_normal.z = elips_z[i];
//     sv_normal.normal_x = elips_xn[i]/10;
//     sv_normal.normal_y = elips_yn[i]/10;
//     sv_normal.normal_z = elips_zn[i]/10;

//     observations_[i * 6 + 0] = static_cast<double>(sv_normal.x);
//     observations_[i * 6 + 1] = static_cast<double>(sv_normal.y);
//     observations_[i * 6 + 2] = static_cast<double>(sv_normal.z);
//     observations_[i * 6 + 3] = static_cast<double>(sv_normal.x + sv_normal.normal_x);
//     observations_[i * 6 + 4] = static_cast<double>(sv_normal.y + sv_normal.normal_y);
//     observations_[i * 6 + 5] = static_cast<double>(sv_normal.z + sv_normal.normal_z);

//     normal_visualization_cloud->push_back(sv_normal);
//   }
//   const double *observations = observations_;