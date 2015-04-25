#ifndef _SPCMAP_MEI_H_
#define _SPCMAP_MEI_H_

#include <Eigen/Eigen>

#include "vision.h"


double logistic(double x);

class MeiCamera : public Camera
{
public:
    MeiCamera(int W, int H, double * parameters);
    MeiCamera(double * parameters);
    
     /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Eigen::Vector2d & src, Eigen::Vector3d & dst) const;

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Eigen::Vector3d & src, Eigen::Vector2d & dst) const;

    virtual bool projectionJacobian(const Eigen::Vector3d & src, Eigen::Matrix<double, 2, 3> & Jac) const;

    virtual void setParameters(const double * const params);

    virtual ~MeiCamera() {}
};

#endif
