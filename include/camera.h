/*
Abstract camera class
*/
#ifndef _SPCMAP_CAMERA_H_
#define _SPCMAP_CAMERA_H_

#include "geometry.h"

template<typename T>
class Camera
{
public:
    vector<T> params;
    int width, height;

    /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Vector2<T> & src, Vector3<T> & dst) const = 0;

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Vector3<T> & src, Vector2<T> & dst) const = 0;

    //TODO implement the projection and distortion Jacobian
    virtual bool projectionJacobian(const Vector3<T> & src,
            Eigen::Matrix<T, 2, 3> & Jac) const = 0;

    virtual void setParameters(const T * const newParams)
    {
        copy(newParams, newParams + params.size(), params.begin());
    }
    
    Camera(int W, int H, int numParams) : width(W), height(H), params(numParams) {}

    virtual ~Camera() {}
    
    virtual Camera * clone() const = 0; 
    
    bool reconstructPointCloud(const vector<Vector2<T>> & src, vector<Vector3<T>> & dst) const
    {
        dst.resize(src.size());
        bool res = true;
        for (int i = 0; i < src.size(); i++)
        {
            res &= reconstructPoint(src[i], dst[i]);
        }  
        return res;
    }
    
    bool projectPointCloud(const vector<Vector3<T>> & src, vector<Vector2<T>> & dst) const
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

