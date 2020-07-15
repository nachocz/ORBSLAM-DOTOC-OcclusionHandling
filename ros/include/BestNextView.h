// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/loss_function.h"

#include "IncludesAndDefinitions.h"

// Read a Bundle Adjustment in the Large dataset.
#ifndef BEST_NEXT_VIEW_
#define BEST_NEXT_VIEW_

namespace DEF_OBJ_TRACK
{

  class BestNextView
  {
  public:
    BestNextView();
    ~BestNextView();
    double *computeBestNextView(std::map<uint32_t, pcl::PointNormal> segments_normals, uint32_t number_of_sv_in_segment,
                                cv::Mat Twc_depth, cv::Mat camera_intrinsics_extended,
                                std::map<uint32_t, pcl::PointXYZ> OBB_mid_plane_corners_world_ref,
                                std::map<uint32_t, pcl::PointNormal> occlusion_normals, uint32_t number_of_occlusions,
                                pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud,
                                pcl::visualization::PCLVisualizer::Ptr viewer,
                                std::ofstream &myfile);

    double *computeBestNextViewNoOcclusions(std::map<uint32_t, pcl::PointNormal> segments_normals, uint32_t number_of_sv_in_segment,
                                cv::Mat Twc_depth, cv::Mat camera_intrinsics_extended,
                                pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud,
                                pcl::visualization::PCLVisualizer::Ptr viewer,
                                std::ofstream &myfile);

    double *computeBestNextViewSimple(std::map<uint32_t, pcl::PointNormal> segments_normals,
                                      uint32_t number_of_sv_in_segment, cv::Mat Twc_depth,
                                      cv::Mat camera_intrinsics_extended, pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud);

    double *computeBestNextViewAllTogether(std::map<uint32_t, pcl::PointNormal> segments_normals,
                                           uint32_t number_of_sv_in_segment, cv::Mat Twc_depth,
                                           cv::Mat camera_intrinsics_extended, pcl::PointCloud<pcl::PointNormal>::Ptr normal_visualization_cloud);

    void testFunction();

  public:
    struct BestNextViewNormalAlignmentPosition
    {
      BestNextViewNormalAlignmentPosition(double observed_x, double observed_y, double observed_z,
                                          double observed_normal_x, double observed_normal_y, double observed_normal_z)
          : observed_x(observed_x), observed_y(observed_y), observed_z(observed_z),
            observed_normal_x(observed_normal_x), observed_normal_y(observed_normal_y), observed_normal_z(observed_normal_z) {}
      template <typename T>
      bool operator()(const T *const cameraTranslation,
                      T *residuals) const
      {

        //Position optimization

        T C[3], D[3];
        C[0] = T(observed_normal_x) - T(observed_x);
        C[1] = T(observed_normal_y) - T(observed_y);
        C[2] = T(observed_normal_z) - T(observed_z);

        D[0] = cameraTranslation[0] - T(observed_x);
        D[1] = cameraTranslation[1] - T(observed_y);
        D[2] = cameraTranslation[2] - T(observed_z);

        T crossCxD[3];
        ceres::CrossProduct(C, D, crossCxD);

        T vectorN2[3];
        vectorN2[0] = crossCxD[0] / (sqrt(ceres::DotProduct(crossCxD, crossCxD)));
        vectorN2[1] = crossCxD[1] / (sqrt(ceres::DotProduct(crossCxD, crossCxD)));
        vectorN2[2] = crossCxD[2] / (sqrt(ceres::DotProduct(crossCxD, crossCxD)));

        T crossDxC[3];
        ceres::CrossProduct(D, C, crossDxC);

        //Beta (position error)
        //cout << "resiudals[0]: " << atan2(ceres::DotProduct(crossCxD, vectorN2), ceres::DotProduct(C, D)) << endl;
        //cout << "resiudals[1]: " << T(1.0) * (T(0.5) - sqrt(ceres::DotProduct(D, D))) << endl;

        residuals[0] = atan2(ceres::DotProduct(crossCxD, vectorN2), ceres::DotProduct(C, D));
        residuals[1] = T(1.0) * (exp(sqrt((T(0.5) - sqrt(ceres::DotProduct(D, D))) * (T(0.5) - sqrt(ceres::DotProduct(D, D))))) - T(1.0));

        return true;
      }
      double observed_x;
      double observed_y;
      double observed_z;
      double observed_normal_x;
      double observed_normal_y;
      double observed_normal_z;
    };

    struct BestNextViewAvoidOcclusions
    {
      BestNextViewAvoidOcclusions(double occlusion_x, double occlusion_y, double occlusion_z,
                                  double OBBCorner_1_x, double OBBCorner_1_y, double OBBCorner_1_z,
                                  double OBBCorner_2_x, double OBBCorner_2_y, double OBBCorner_2_z,
                                  double OBBCorner_3_x, double OBBCorner_3_y, double OBBCorner_3_z,
                                  double OBBCorner_4_x, double OBBCorner_4_y, double OBBCorner_4_z,
                                  double OBBCenter_x, double OBBCenter_y, double OBBCenter_z)
          : occlusion_x(occlusion_x), occlusion_y(occlusion_y), occlusion_z(occlusion_z),
            OBBCorner_1_x(OBBCorner_1_x), OBBCorner_1_y(OBBCorner_1_y), OBBCorner_1_z(OBBCorner_1_z),
            OBBCorner_2_x(OBBCorner_2_x), OBBCorner_2_y(OBBCorner_2_y), OBBCorner_2_z(OBBCorner_2_z),
            OBBCorner_3_x(OBBCorner_3_x), OBBCorner_3_y(OBBCorner_3_y), OBBCorner_3_z(OBBCorner_3_z),
            OBBCorner_4_x(OBBCorner_4_x), OBBCorner_4_y(OBBCorner_4_y), OBBCorner_4_z(OBBCorner_4_z),
            OBBCenter_x(OBBCenter_x), OBBCenter_y(OBBCenter_y), OBBCenter_z(OBBCenter_z)
      {
      }
      template <typename T>
      bool operator()(const T *const cameraTranslation,
                      T *residuals_occlusions) const
      {
        // cout << "Functor OBBCenter: " << endl;
        // cout << OBBCenter_x << endl
        //      << OBBCenter_y << endl
        //      << OBBCenter_z << endl;

        Eigen::Matrix<T, 3, 1> OBBCenter; //OBB center position
        OBBCenter << T(OBBCenter_x),
            T(OBBCenter_y),
            T(OBBCenter_z);

        Eigen::Matrix<T, 3, 1> camera_center; //camera position
        camera_center << cameraTranslation[0],
            cameraTranslation[1],
            cameraTranslation[2];

        Eigen::Matrix<T, 3, 1> normal_vector;
        normal_vector = OBBCenter - camera_center;

        Eigen::Matrix<T, 3, 1> normal_vector_unitary;

        normal_vector_unitary = normal_vector.normalized();

        Eigen::Matrix<T, 3, 1> U_vector; //2nd basis vector
        U_vector << T(1.0),
            T(1.0),
            T((-normal_vector_unitary(0) - normal_vector_unitary(1)) / normal_vector_unitary(2));
        U_vector = U_vector.normalized();

        Eigen::Matrix<T, 3, 1> V_vector; //3rd basis vector
        V_vector = (normal_vector_unitary.cross(U_vector)).normalized();

        Eigen::Matrix<T, 3, 1> W_vector; //Comprobation vector, it must be = normal_vector_unitary
        W_vector = (U_vector.cross(V_vector)).normalized();

        Eigen::Matrix<T, 3, 3> R_matrix; //Rwc (camera to world)
        R_matrix << U_vector, V_vector, W_vector;
        Eigen::Matrix<T, 3, 3> R_matrix_inverse; //Rcw (world to camera)
        R_matrix_inverse = R_matrix.transpose();

        Eigen::Matrix<T, 3, 4> Rt_matrix; //
        Rt_matrix << R_matrix_inverse, -R_matrix_inverse * camera_center;

        Eigen::Matrix<T, 3, 3> k_matrix; //this could be any intrinsics... but for possible future expansions of the optimization function -> TODO: add intrinsics as variables
        k_matrix << T(617.82), T(0.0), T(0.0),
            T(0.0), T(617.858), T(0.0),
            T(0.0), T(0.0), T(1.0);

        Eigen::Matrix<T, 3, 5> OBBCorners_and_sv_vector; //OBB center position [SV, corner1, corner2, corner3, corner4]
        OBBCorners_and_sv_vector << T(occlusion_x), T(OBBCorner_1_x), T(OBBCorner_2_x), T(OBBCorner_3_x), T(OBBCorner_4_x),
            T(occlusion_y), T(OBBCorner_1_y), T(OBBCorner_2_y), T(OBBCorner_3_y), T(OBBCorner_4_y),
            T(occlusion_z), T(OBBCorner_1_z), T(OBBCorner_2_z), T(OBBCorner_3_z), T(OBBCorner_4_z);

        Eigen::Matrix<T, 3, 5> OBBcorn_and_sv_plane;

        //check if occlusion is actually infront of the object or behind

        bool occlusion_between_cam_obj = false;

        Eigen::Matrix<T, 4, 1> OBB_center_extended;
        Eigen::Matrix<T, 4, 1> occlusion_vector;
        Eigen::Matrix<T, 3, 1> OBBcenter_camera_ref;
        Eigen::Matrix<T, 3, 1> occlusion_vector_camera_ref;

        occlusion_vector << T(occlusion_x),
            T(occlusion_y),
            T(occlusion_z),
            T(1.0);
        occlusion_vector_camera_ref = Rt_matrix * occlusion_vector;

        OBB_center_extended << T(OBBCenter_x),
            T(OBBCenter_y),
            T(OBBCenter_z),
            T(1.0);
        OBBcenter_camera_ref = Rt_matrix * OBB_center_extended;

        if (occlusion_vector_camera_ref(2) < OBBcenter_camera_ref(2))
        {
          occlusion_between_cam_obj = true;
        }

        for (uint32_t i = 0; i < 5; i++) //watchout, eigen begins in 0 index
        {
          Eigen::Matrix<T, 4, 1> temp_vector;
          Eigen::Matrix<T, 3, 1> temp_vector_camera_coord;

          temp_vector << OBBCorners_and_sv_vector.col(i),
              T(1.0);

          temp_vector_camera_coord = k_matrix * Rt_matrix * temp_vector;

          OBBcorn_and_sv_plane.col(i) = temp_vector_camera_coord / temp_vector_camera_coord(2);
        }
        //checking if it is inside frustum

        int nvert = 4;
        int i, j;
        j = nvert;

        T testx = OBBcorn_and_sv_plane(0, 0);
        T testy = OBBcorn_and_sv_plane(1, 0);

        bool it_is_in_frustrum = false;

        for (i = 1; i <= nvert; i++) //Jordan curve theorem --from https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
        {
          T vertx_i = OBBcorn_and_sv_plane(0, i); //plane coord x of point i
          T verty_i = OBBcorn_and_sv_plane(1, i); //plane coord y of point i

          T vertx_j = OBBcorn_and_sv_plane(0, j); //plane coord x of point j
          T verty_j = OBBcorn_and_sv_plane(1, j); //plane coord y of point j

          if (((verty_i > testy) != (verty_j > testy)) &&
              (testx < (vertx_j - vertx_i) * (testy - verty_i) / (verty_j - verty_i) + vertx_i))
            it_is_in_frustrum = !it_is_in_frustrum;

          j = i;
        }

        Eigen::Matrix<T, 4, 1> reprojected_OBBcorner_distances;
        reprojected_OBBcorner_distances << sqrt(OBBcorn_and_sv_plane(0, 1) * OBBcorn_and_sv_plane(0, 1) + OBBcorn_and_sv_plane(1, 1) * OBBcorn_and_sv_plane(1, 1)), //plane coord y of point i
            sqrt(OBBcorn_and_sv_plane(0, 2) * OBBcorn_and_sv_plane(0, 2) + OBBcorn_and_sv_plane(1, 2) * OBBcorn_and_sv_plane(1, 2)),
            sqrt(OBBcorn_and_sv_plane(0, 3) * OBBcorn_and_sv_plane(0, 3) + OBBcorn_and_sv_plane(1, 3) * OBBcorn_and_sv_plane(1, 3)),
            sqrt(OBBcorn_and_sv_plane(0, 4) * OBBcorn_and_sv_plane(0, 4) + OBBcorn_and_sv_plane(1, 4) * OBBcorn_and_sv_plane(1, 4));

        T cone_radius = reprojected_OBBcorner_distances.maxCoeff();

        T weight;

        //Angle between z_cam and occlusion sv projection ray

        Eigen::Matrix<T, 3, 1>
            occlusion;
        occlusion << T(occlusion_x),
            T(occlusion_y),
            T(occlusion_z);
        Eigen::Matrix<T, 3, 1> occlusion_ray;
        occlusion_ray = occlusion - camera_center;
        //xi angle : camera axis to furthest corner of OBB (image coord)

        T A_xi[3], B_xi[3];
        A_xi[0] = T(0.0);
        A_xi[1] = T(0.0);
        A_xi[2] = T(1.0);

        B_xi[0] = cone_radius;
        B_xi[1] = T(0.0);
        B_xi[2] = T(617.82);

        T crossAxB_xi[3];
        ceres::CrossProduct(A_xi, B_xi, crossAxB_xi);

        T vectorN_xi[3];
        vectorN_xi[0] = crossAxB_xi[0] / (sqrt(ceres::DotProduct(crossAxB_xi, crossAxB_xi)));
        vectorN_xi[1] = crossAxB_xi[1] / (sqrt(ceres::DotProduct(crossAxB_xi, crossAxB_xi)));
        vectorN_xi[2] = crossAxB_xi[2] / (sqrt(ceres::DotProduct(crossAxB_xi, crossAxB_xi)));

        T xiAngle = atan2(ceres::DotProduct(crossAxB_xi, vectorN_xi), ceres::DotProduct(A_xi, B_xi));

        //Gamma angle: camera axis to occlusion projection ray
        T A[3], B[3];
        A[0] = W_vector(0);
        A[1] = W_vector(1);
        A[2] = W_vector(2);

        B[0] = occlusion_ray(0);
        B[1] = occlusion_ray(1);
        B[2] = occlusion_ray(2);

        T crossAxB[3];
        ceres::CrossProduct(A, B, crossAxB);

        T vectorN[3];
        vectorN[0] = crossAxB[0] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));
        vectorN[1] = crossAxB[1] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));
        vectorN[2] = crossAxB[2] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));

        T crossBxA[3];
        ceres::CrossProduct(B, A, crossBxA);

        T gammaAngle = atan2(ceres::DotProduct(crossAxB, vectorN), ceres::DotProduct(A, B));

        T weight_scale = T(PI / 2.0);

        if (it_is_in_frustrum && occlusion_between_cam_obj)
        {

          // T a_value = sqrt(xiAngle * xiAngle) * sqrt(xiAngle * xiAngle) * sqrt(xiAngle * xiAngle);
          // T x_value = sqrt(gammaAngle * gammaAngle) * sqrt(gammaAngle * gammaAngle) * sqrt(gammaAngle * gammaAngle);

          // weight = ((-weight_scale / a_value) * (x_value)) + (weight_scale); // / normalizing_distance;

          weight = T(1.0);

          //cout << "it_is_in_frustrum && occlusion_between_cam_obj" << endl;
        }
        else
        {
          //cout << "OUTSIDE OUTSIDE OUTSIDE" << endl;
          weight = T(0.0);
        }
        //cout << "weight: " << weight << endl;
        //weight = T(0.0);
        residuals_occlusions[0] = weight;
        //residuals_occlusions[1] = exp(sqrt((T(0.5) - sqrt(ceres::DotProduct(D, D))) * (T(0.5) - sqrt(ceres::DotProduct(D, D))))) - T(1.0);
        //residuals_occlusions[1] = exp(sqrt((T(0.5) - sqrt(normal_vector.dot(normal_vector))) * (T(0.5) - sqrt(normal_vector.dot(normal_vector))))) - T(1.0);

        return true;
      }

      double occlusion_x;
      double occlusion_y;
      double occlusion_z;

      double OBBCenter_x;
      double OBBCenter_y;
      double OBBCenter_z;

      double OBBCorner_1_x;
      double OBBCorner_1_y;
      double OBBCorner_1_z;

      double OBBCorner_2_x;
      double OBBCorner_2_y;
      double OBBCorner_2_z;

      double OBBCorner_3_x;
      double OBBCorner_3_y;
      double OBBCorner_3_z;

      double OBBCorner_4_x;
      double OBBCorner_4_y;
      double OBBCorner_4_z;
    };

    struct BestNextViewNormalAlignmentOrientation
    {
      BestNextViewNormalAlignmentOrientation(double focalDistance,
                                             double image_plane_center_x, double image_plane_center_y, double image_plane_center_z,
                                             double cameraTranslation_x, double cameraTranslation_y, double cameraTranslation_z,
                                             double observed_x, double observed_y, double observed_z,
                                             double observed_normal_x, double observed_normal_y, double observed_normal_z)
          : focalDistance(focalDistance),
            image_plane_center_x(image_plane_center_x), image_plane_center_y(image_plane_center_y), image_plane_center_z(image_plane_center_z),
            cameraTranslation_x(cameraTranslation_x), cameraTranslation_y(cameraTranslation_y), cameraTranslation_z(cameraTranslation_z),
            observed_x(observed_x), observed_y(observed_y), observed_z(observed_z),
            observed_normal_x(observed_normal_x), observed_normal_y(observed_normal_y), observed_normal_z(observed_normal_z) {}
      template <typename T>
      bool operator()(const T *const cameraRotation,
                      T *residuals2) const
      {
        //Orientation optimization: alfa angle
        T image_plane_center[3];
        image_plane_center[0] = T(image_plane_center_x);
        image_plane_center[1] = T(image_plane_center_y);
        image_plane_center[2] = T(image_plane_center_z);

        T imagePlaneCenterWorldCoord[3];
        ceres::AngleAxisRotatePoint(cameraRotation, image_plane_center, imagePlaneCenterWorldCoord);

        imagePlaneCenterWorldCoord[0] += T(cameraTranslation_x);
        imagePlaneCenterWorldCoord[1] += T(cameraTranslation_y);
        imagePlaneCenterWorldCoord[2] += T(cameraTranslation_z);

        // cout << "imagePlaneCenterWorldCoord: " << endl
        //      << imagePlaneCenterWorldCoord[0] << " " << imagePlaneCenterWorldCoord[1] << " " << imagePlaneCenterWorldCoord[2] << endl;

        T A[3], B[3];
        A[0] = imagePlaneCenterWorldCoord[0] - T(cameraTranslation_x);
        A[1] = imagePlaneCenterWorldCoord[1] - T(cameraTranslation_y);
        A[2] = imagePlaneCenterWorldCoord[2] - T(cameraTranslation_z);

        // B[0] = T(observed_x) - T(observed_normal_x);
        // B[1] = T(observed_y) - T(observed_normal_y);
        // B[2] = T(observed_z) - T(observed_normal_z);

        B[0] = T(observed_x) - T(cameraTranslation_x);
        B[1] = T(observed_y) - T(cameraTranslation_y);
        B[2] = T(observed_z) - T(cameraTranslation_z);

        T crossAxB[3];
        ceres::CrossProduct(A, B, crossAxB);

        T vectorN[3];
        vectorN[0] = crossAxB[0] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));
        vectorN[1] = crossAxB[1] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));
        vectorN[2] = crossAxB[2] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));

        T crossBxA[3];
        ceres::CrossProduct(B, A, crossBxA);

        //Alfa (orientation error)
        residuals2[0] = atan2(ceres::DotProduct(crossAxB, vectorN), ceres::DotProduct(A, B));

        return true;
      }
      double focalDistance;
      double cameraTranslation_x;
      double cameraTranslation_y;
      double cameraTranslation_z;
      double image_plane_center_x;
      double image_plane_center_y;
      double image_plane_center_z;
      double observed_x;
      double observed_y;
      double observed_z;
      double observed_normal_x;
      double observed_normal_y;
      double observed_normal_z;
    }; // namespace DEF_OBJ_TRACK

    struct BestNextViewAligment
    {
      BestNextViewAligment(double focalDistance,
                           double image_plane_center_x, double image_plane_center_y, double image_plane_center_z,

                           double observed_x, double observed_y, double observed_z,
                           double observed_normal_x, double observed_normal_y, double observed_normal_z)
          : focalDistance(focalDistance),
            image_plane_center_x(image_plane_center_x), image_plane_center_y(image_plane_center_y), image_plane_center_z(image_plane_center_z),
            observed_x(observed_x), observed_y(observed_y), observed_z(observed_z),
            observed_normal_x(observed_normal_x), observed_normal_y(observed_normal_y), observed_normal_z(observed_normal_z)
      {
      }
      template <typename T>
      bool operator()(const T *const cameraTranslation,
                      const T *const cameraRotation,
                      T *residuals) const
      {
        //Position optimization

        T C[3], D[3];
        C[0] = T(observed_normal_x) - T(observed_x);
        C[1] = T(observed_normal_y) - T(observed_y);
        C[2] = T(observed_normal_z) - T(observed_z);

        D[0] = cameraTranslation[0] - T(observed_x);
        D[1] = cameraTranslation[1] - T(observed_y);
        D[2] = cameraTranslation[2] - T(observed_z);

        T crossCxD[3];
        ceres::CrossProduct(C, D, crossCxD);

        T vectorN2[3];
        vectorN2[0] = crossCxD[0] / (sqrt(ceres::DotProduct(crossCxD, crossCxD)));
        vectorN2[1] = crossCxD[1] / (sqrt(ceres::DotProduct(crossCxD, crossCxD)));
        vectorN2[2] = crossCxD[2] / (sqrt(ceres::DotProduct(crossCxD, crossCxD)));

        T crossDxC[3];
        ceres::CrossProduct(D, C, crossDxC);

        //Orientation optimization: alfa angle
        T image_plane_center[3];
        image_plane_center[0] = T(image_plane_center_x);
        image_plane_center[1] = T(image_plane_center_y);
        image_plane_center[2] = T(image_plane_center_z);

        T imagePlaneCenterWorldCoord[3];
        ceres::AngleAxisRotatePoint(cameraRotation, image_plane_center, imagePlaneCenterWorldCoord);

        imagePlaneCenterWorldCoord[0] += cameraTranslation[0];
        imagePlaneCenterWorldCoord[1] += cameraTranslation[1];
        imagePlaneCenterWorldCoord[2] += cameraTranslation[2];

        // cout << "imagePlaneCenterWorldCoord: " << endl
        //      << imagePlaneCenterWorldCoord[0] << " " << imagePlaneCenterWorldCoord[1] << " " << imagePlaneCenterWorldCoord[2] << endl;

        T A[3], B[3];
        A[0] = imagePlaneCenterWorldCoord[0] - cameraTranslation[0];
        A[1] = imagePlaneCenterWorldCoord[1] - cameraTranslation[1];
        A[2] = imagePlaneCenterWorldCoord[2] - cameraTranslation[2];

        // B[0] = T(observed_x) - T(observed_normal_x);
        // B[1] = T(observed_y) - T(observed_normal_y);
        // B[2] = T(observed_z) - T(observed_normal_z);

        B[0] = -D[0];
        B[1] = -D[1];
        B[2] = -D[2];

        T crossAxB[3];
        ceres::CrossProduct(A, B, crossAxB);

        T vectorN[3];
        vectorN[0] = crossAxB[0] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));
        vectorN[1] = crossAxB[1] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));
        vectorN[2] = crossAxB[2] / (sqrt(ceres::DotProduct(crossAxB, crossAxB)));

        T crossBxA[3];
        ceres::CrossProduct(B, A, crossBxA);

        //Alfa (orientation error)
        residuals[0] = atan2(ceres::DotProduct(crossCxD, vectorN2), ceres::DotProduct(C, D));
        residuals[1] = T(10) * (T(0.5) - sqrt(ceres::DotProduct(D, D)));
        residuals[2] = atan2(ceres::DotProduct(crossAxB, vectorN), ceres::DotProduct(A, B));

        return true;
      }
      double focalDistance;
      double image_plane_center_x;
      double image_plane_center_y;
      double image_plane_center_z;
      double observed_x;
      double observed_y;
      double observed_z;
      double observed_normal_x;
      double observed_normal_y;
      double observed_normal_z;
    };
  };
} // namespace DEF_OBJ_TRACK
#endif //BEST_NEXT_VIEW_