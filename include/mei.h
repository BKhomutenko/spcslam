#ifndef _SPCMAP_MEI_H_
#define _SPCMAP_MEI_H_

#include "vision.h"
#include <Eigen/Eigen>

using namespace Eigen;

class MeiCamera : public Camera
{
public:
    unsigned int imageWidth;
    unsigned int imageHeight;
    double xi;
    double fu;
    double fv; 
    double u0;
    double v0;
    
    MeiCamera(unsigned int imageWidth, unsigned int imageHeight, double xi, double fu, double fv, double u0, double v0)
    :imageHeight(imageHeight), imageWidth(imageWidth), xi(xi), fu(fu), fv(fv), u0(u0), v0(v0) {}

     /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Vector2d & src, Vector3d & dst) const;
    
    /// projects 3D points onto the original image
    virtual bool projectPoint(const Vector3d & src, Vector2d & dst) const;
    
    virtual bool projectionJacobian(const Vector3d & src, Eigen::Matrix<double, 2, 3> & Jac) const;
    
    virtual ~MeiCamera() {}
};

#endif
