#include "rgbd_utils.h"
double rgbd_utils::getSubPix(const cv::Mat& img, double dx, double dy)
{
    assert(!img.empty());

    int x = (int)dx;
    int y = (int)dy;

    int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

    float a = dx - (double)x;
    float c = dy - (double)y;

    double b = (1.*img.at<uint8_t>(y0, x0) * (1. - a) + 1.*img.at<uint8_t>(y0, x1) * a) * (1. - c)
              + (1.*img.at<uint8_t>(y1, x0) * (1. - a) + 1.*img.at<uint8_t>(y1, x1) * a) * c;

    return b;
}

cv::Mat rgbd_utils::getProjectedImage(const std::shared_ptr<const cv::Mat> &I1,
                                      const std::shared_ptr<const cv::Mat> &I2,
                                      const std::shared_ptr<const cv::Mat> &D1,
                          const Eigen::Matrix3d& K, Eigen::Matrix<double, 6,1> params)
{
    cv::Mat error_map(I1->rows, I1->cols, CV_8U);
    Sophus::SE3d rotation = Sophus::SE3d::exp(params);
    const double cx = K(0, 2);
    const double cy = K(1, 2);
    const double fx = K(0, 0);
    const double fy = K(1, 1);
    for (int u = 0; u < D1->rows; u++) {
        for (int v = 0; v < D1->cols; v++) {
            error_map.at<uint8_t>(u, v) = 0;
            double i1 = (I1->at<uint8_t>(int(u), int(v)));

            const double depth = 1.0f*D1->at<uint16_t>(u, v)/kDepthScale;
            Eigen::Matrix<double, 4, 1> p3d_frame1;
            p3d_frame1 << (u - cx) / fx * depth, (v - cy) / fy * depth, depth, 1.0;

            const Eigen::Matrix<double, 4, 1> p3d_frame2 = rotation.matrix() * p3d_frame1;
            const double x = (fx * p3d_frame2[0] / p3d_frame2[2] + cx);
            const double y = (fy * p3d_frame2[1] / p3d_frame2[2] + cy);
            if (x > 0 && y> 0 && y < I2->cols && x< I2->rows) {
                double i2 = (getSubPix(*I2, y, x));
                error_map.at<uint8_t>(u, v) = i2;
            }
        }
    }
    return error_map;
}

cv::Mat rgbd_utils::getErrorImageGrid(const std::shared_ptr<const cv::Mat> &I1,
                                  const std::shared_ptr<const cv::Mat> &I2,
                      const std::shared_ptr<const cv::Mat> &D1,
                      const Eigen::Matrix3d& K, Eigen::Matrix<double, 6,1> params)
{
    double datai2f[I2->cols*I2->rows];
    for(int i=0; i<I2->rows; i++) {
        for (int j = 0; j < I2->cols; j++) {
            datai2f[i*I2->cols + j ] = I2->at<uint8_t>(i,j);
        }
    }
    ceres::Grid2D<double> gridI2f(datai2f, 0,I2->rows,0, I2->cols);
    ceres::BiCubicInterpolator<ceres::Grid2D<double>> grid_interpolatorI2f(gridI2f);

    cv::Mat error_map(I1->rows, I1->cols, CV_8U);
    Sophus::SE3d rotation = Sophus::SE3d::exp(params);
    const double cx = K(0, 2);
    const double cy = K(1, 2);
    const double fx = K(0, 0);
    const double fy = K(1, 1);

    for (int u = 0; u < D1->rows; u++) {
        for (int v = 0; v < D1->cols; v++) {
            error_map.at<uint8_t>(u, v) = 0;
            double i1 = (I1->at<uint8_t>(int(u), int(v)));

            const double depth = 1.0f*D1->at<uint16_t>(u, v)/kDepthScale;
            Eigen::Matrix<double, 4, 1> p3d_frame1;
            p3d_frame1 << (u - cx) / fx * depth, (v - cy) / fy * depth, depth, 1.0;

            const Eigen::Matrix<double, 4, 1> p3d_frame2 = rotation.matrix() * p3d_frame1;
            const double x = (fx * p3d_frame2[0] / p3d_frame2[2] + cx);
            const double y = (fy * p3d_frame2[1] / p3d_frame2[2] + cy);
            //if (x > 0 && y> 0 && y < I2->cols && x< I2->rows) {
                double i2;
                gridI2f.GetValue(x,y,&i2);
                //double i1 = (I1->at<uint8_t>(int(u), int(v)));
                grid_interpolatorI2f.Evaluate(x,y, &i2, nullptr, nullptr);
                //double i2 = (getSubPix(*I2, y, x));
                error_map.at<uint8_t>(u, v) += abs(i1-i2);
            //}
        }
    }
    return error_map;
}


cv::Mat rgbd_utils::getErrorImage(const std::shared_ptr<const cv::Mat> &I1, const std::shared_ptr<const cv::Mat> &I2,const std::shared_ptr<const cv::Mat> &D1,
                      const Eigen::Matrix3d& K, Eigen::Matrix<double, 6,1> params) {
    cv::Mat error_map(I1->rows, I1->cols, CV_8U);

    Sophus::SE3d rotation = Sophus::SE3d::exp(params);
    const double cx = K(0, 2);
    const double cy = K(1, 2);
    const double fx = K(0, 0);
    const double fy = K(1, 1);
    for (int u = 0; u < D1->rows; u++) {
        for (int v = 0; v < D1->cols; v++) {
            error_map.at<uint8_t>(u, v) = 0;

            const double depth = 1.0f*D1->at<uint16_t>(u, v)/kDepthScale;
            Eigen::Matrix<double, 4, 1> p3d_frame1;
            p3d_frame1 << (u - cx) / fx * depth, (v - cy) / fy * depth, depth, 1.0;

            const Eigen::Matrix<double, 4, 1> p3d_frame2 = rotation.matrix() * p3d_frame1;
            const double x = (fx * p3d_frame2[0] / p3d_frame2[2] + cx);
            const double y = (fy * p3d_frame2[1] / p3d_frame2[2] + cy);
            if (x > 0 && y> 0 && y < I2->cols && x< I2->rows) {
                double i1 = (I1->at<uint8_t>(int(u), int(v)));
                double i2 = (getSubPix(*I2, y, x));
                error_map.at<uint8_t>(u, v) += abs(i1-i2);
            }
        }
    }
    return error_map;
}


void rgbd_utils::downscale (const cv::Mat &inI, const cv::Mat &inD, const Eigen::Matrix3d &inK,
                cv::Mat &outI, cv::Mat &outD, Eigen::Matrix3d &outK)
{
    const int h = inI.rows/2;
    const int w = inI.cols/2;
    assert(inD.rows==inI.rows);
    assert(inD.cols==inI.cols);
    outI = cv::Mat(h,w,CV_8U);
    outD = cv::Mat(h,w,CV_16U);

    outK.setZero();
    outK(0,0) = 0.5 * inK (0,0);
    outK(1,1) = 0.5 * inK (1,1);
    outK(0,2) = (0.5 + inK (0,2)) /2.0 -0.5 ;
    outK(2,2) = (0.5 + inK (2,2)) /2.0 -0.5 ;

    for (int r = 0; r < h; r++)
    {
        for (int c = 0; c < w; c++)
        {
            outI.at<uint8_t>(r,c) = 0.25 * inI.at<uint8_t>(2*r+1,+2*c+1)+
                                    0.25 * inI.at<uint8_t>(2*r+1,+2*c)+
                                    0.25 * inI.at<uint8_t>(2*r,+2*c+1)+
                                    0.25 * inI.at<uint8_t>(2*r,+2*c);

            float pixel = 0.f;
            float p = 0.f;
            if (inD.at<uint16_t>(2*r,2*c)!=0)
            {
                pixel += inD.at<uint16_t>(2*r,2*c);
                p+=1.f;
            }
            if (inD.at<uint16_t>(2*r+1,2*c)!=0)
            {
                pixel += inD.at<uint16_t>(2*r+1,2*c);
                p+=1.f;
            }
            if (inD.at<uint16_t>(2*r,2*c+1)!=0)
            {
                pixel += inD.at<uint16_t>(2*r,2*c+1);
                p+=1.f;
            }
            if (inD.at<uint16_t>(2*r+1,2*c+1)!=0)
            {
                pixel += inD.at<uint16_t>(2*r+1,2*c+1);
                p+=1.f;
            }
            outD.at<uint16_t>(r,c) = pixel/p;
        }
    }
}
