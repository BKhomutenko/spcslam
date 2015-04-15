#ifndef _SPCMAP_MEI_H_
#define _SPCMAP_MEI_H_

#include <Eigen/Eigen>

#include "vision.h"

class MeiCamera : public Camera
{
public:
    double alpha;
    double beta;
    double fu;
    double fv;
    double u0;
    double v0;

    MeiCamera(unsigned int imageWidth, unsigned int imageHeight, double alpha, double fu, double fv, double u0, double v0)
    : alpha(alpha), fu(fu), fv(fv), u0(u0), v0(v0)
    {
        beta = 1.2;
        this->imageHeight = imageHeight;
        this->imageWidth = imageWidth;
    }

     /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Eigen::Vector2d & src, Eigen::Vector3d & dst) const;

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Eigen::Vector3d & src, Eigen::Vector2d & dst) const;

    virtual bool projectionJacobian(const Eigen::Vector3d & src, Eigen::Matrix<double, 2, 3> & Jac) const;

    virtual ~MeiCamera() {}
};

#endif
