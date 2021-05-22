#pragma once
#include <Eigen/Dense>
#include "cassert"
#include <iostream>
#include <sophus/se3.hpp>
#include <sophus/rotation_matrix.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
namespace rgbd_utils{
    using interp2D=ceres::BiCubicInterpolator<ceres::Grid2D<double>>;
    constexpr float kDepthScale = 5000.f;
    double getSubPix(const cv::Mat& img, double dx, double dy);

    cv::Mat getErrorImage(const std::shared_ptr<const cv::Mat> &I1, const std::shared_ptr<const cv::Mat> &I2,const std::shared_ptr<const cv::Mat> &D1,
                          const Eigen::Matrix3d& K, Eigen::Matrix<double, 6,1> params);

    cv::Mat getErrorImageGrid(const std::shared_ptr<const cv::Mat> &I1,
                          const std::shared_ptr<const cv::Mat> &I2,
                          const std::shared_ptr<const cv::Mat> &D1,
                          const Eigen::Matrix3d& K, Eigen::Matrix<double, 6,1> params);

    cv::Mat getProjectedImage(const std::shared_ptr<const cv::Mat> &I1, const std::shared_ptr<const cv::Mat> &I2,const std::shared_ptr<const cv::Mat> &D1,
                          const Eigen::Matrix3d& K, Eigen::Matrix<double, 6,1> params);


    void downscale (const cv::Mat &inI, const cv::Mat &inD, const Eigen::Matrix3d &inK,
                    cv::Mat &outI, cv::Mat &outD, Eigen::Matrix3d &outK);

    struct pyramidLevel{
        cv::Mat I;
        cv::Mat D;
        Eigen::Matrix3d K;
    };



    struct PhotometricErrorGrid{
        const Eigen::Matrix3d K;
        const std::shared_ptr<const cv::Mat> I1;
        const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> I2;
        const std::shared_ptr<const cv::Mat> D1;

        int u;
        int v;
        PhotometricErrorGrid( const std::shared_ptr<const cv::Mat> &I1,
                              const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> &I2,
                              const std::shared_ptr<const cv::Mat> &D1,
                          const Eigen::Matrix3d& K, int u, int v) :
                I1(I1), I2(I2),D1(D1),K(K), u(u),v(v){}

        template <typename T>
        bool operator()(const T* const params1tangent, T* residuals) const {
            Eigen::Map<const Eigen::Matrix<T,6,1>> params(params1tangent);
            T error = T(0);
            Sophus::SE3<T> rotation = Sophus::SE3<T>::exp(params);
            const T cx (K(0,2));
            const T cy (K(1,2));
            const T fx (K(0,0));
            const T fy (K(1,1));
            const T depth = T(1.0*D1->at<uint16_t>(u, v)/kDepthScale);

            Eigen::Matrix<T, 4, 1> p3d_frame1;
            p3d_frame1 << (T(u) - cx) / fx * depth, (T(v) - cy) / fy * depth, depth, T(1.0);
            const Eigen::Matrix<T, 4, 1> p3d_frame2 = rotation.matrix() * p3d_frame1;

            const T x = T(fx * p3d_frame2[0] / p3d_frame2[2] + cx);
            const T y = T(fy * p3d_frame2[1] / p3d_frame2[2] + cy);

            //T i2 = getSubPix(*I2, y, x);
            if (x > T(0) && y> T(0) && y < T(I1->cols) && x< T(I1->rows)) {
                T i2;
                I2->Evaluate(x, y, &i2);
                T i1 = (T) I1->at<uint8_t>(int(u), int(v));
                error = (i1 - i2) * (i1 - i2);
            }
            else
            {
                error = T(255.0*255.0);
            }

            residuals[0]=error;
            return true;
        }

        static ceres::CostFunction* Create(
                const std::shared_ptr<const cv::Mat> &I1,
                const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> &I2,
                const std::shared_ptr<const cv::Mat> &D1,
                const Eigen::Matrix3d& K, int u, int v) {
//            return (new ceres::NumericDiffCostFunction<PhotometricErrorGrid,ceres::CENTRAL, 1, 6>(
//                    new PhotometricErrorGrid(I1,I2,D1,K,u,v)));
            return (new ceres::AutoDiffCostFunction<PhotometricErrorGrid, 1, 6>(
                    new PhotometricErrorGrid(I1,I2,D1,K,u,v)));
        }
    };
    struct PhotometricError{
        const Eigen::Matrix3d K;
        const std::shared_ptr<const cv::Mat> I1;
        const std::shared_ptr<const cv::Mat> I2;
        const std::shared_ptr<const cv::Mat> D1;

        int u;
        int v;
        PhotometricError( const std::shared_ptr<const cv::Mat> &I1, const std::shared_ptr<const cv::Mat> &I2,const std::shared_ptr<const cv::Mat> &D1,
                          const Eigen::Matrix3d& K, int u, int v) :
                I1(I1), I2(I2),D1(D1),K(K), u(u),v(v){}

        template <typename T>
        bool operator()(const T* const params1tangent, T* residuals) const {
            Eigen::Map<const Eigen::Matrix<T,6,1>> params(params1tangent);
            T error = T(0);
            Sophus::SE3<T> rotation = Sophus::SE3<T>::exp(params);
            const double cx = K(0,2);
            const double cy = K(1,2);
            const double fx = K(0,0);
            const double fy = K(1,1);
            const double depth = T(1.0*D1->at<uint16_t>(u, v)/kDepthScale);

            Eigen::Matrix<T, 4, 1> p3d_frame1;
            p3d_frame1 << (T(u) - cx) / fx * depth, (T(v) - cy) / fy * depth, depth, T(1.0);
            const Eigen::Matrix<T, 4, 1> p3d_frame2 = rotation.matrix() * p3d_frame1;

            const double x = (fx * p3d_frame2[0] / p3d_frame2[2] + cx);
            const double y = (fy * p3d_frame2[1] / p3d_frame2[2] + cy);
            if (x > 0 && y> 0 && y < I2->cols && x< I2->rows) {
                T i2 = getSubPix(*I2, y, x);
                T i1 = (T) I1->at<uint8_t>(int(u), int(v));
                error = (i1 - i2) * (i1 - i2);
            }
            else
            {
                error =0;
            }
            residuals[0]=error;
            return true;
        }

        static ceres::CostFunction* Create(
                const std::shared_ptr<const cv::Mat> &I1,
                const std::shared_ptr<const cv::Mat> &I2,
                const std::shared_ptr<const cv::Mat> &D1,
                const Eigen::Matrix3d& K, int u, int v) {
            return (new ceres::NumericDiffCostFunction<PhotometricError,ceres::CENTRAL, 1, 6>(
                    new PhotometricError(I1,I2,D1,K,u,v)));
//            return (new ceres::AutoDiffCostFunction<PhotometricError, 1, 6>(
//                    new PhotometricError(I1,I2,D1,K,u,v)));
        }
    };

}

