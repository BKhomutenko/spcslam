/*
Abstract camera class
*/
#ifndef _SPCMAP_CAMERA_H_
#define _SPCMAP_CAMERA_H_

#include <Eigen/Eigen>
#include "geometry.h"

using Eigen::Matrix;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;

class ICamera
{
public:
    vector<double> params;
    int width, height;

    /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Vector2d & src, Vector3d & dst) const = 0;

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Vector3d & src, Vector2d & dst) const = 0;

    //TODO implement the projection and distortion Jacobian
    virtual bool projectionJacobian(const Vector3d & src,
            Eigen::Matrix<double, 2, 3> & Jac) const = 0;

    virtual void setParameters(const double * const newParams)
    {
        copy(newParams, newParams + params.size(), params.begin());
    }
    
    ICamera(int W, int H, int numParams) : width(W), height(H), params(numParams) {}

    virtual ~ICamera() {}
    
    virtual ICamera * clone() const = 0; 
    
    bool reconstructPointCloud(const vector<Vector2d> & src, vector<Vector3d> & dst) const
    {
        dst.resize(src.size());
        bool res = true;
        for (int i = 0; i < src.size(); i++)
        {
            res &= reconstructPoint(src[i], dst[i]);
        }  
        return res;
    }
    
    bool projectPointCloud(const vector<Vector3d> & src, vector<Vector2d> & dst) const
    {
        dst.resize(src.size());
        bool res = true;
        for (int i = 0; i < src.size(); i++)
        {
            res &= projectPoint(src[i], dst[i]);
        }  
        return res;
    }
};

#endif

