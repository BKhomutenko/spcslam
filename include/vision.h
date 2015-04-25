/*
Abstract camera definition.
Stereo vision definition.
*/

#ifndef _SPCMAP_VISION_H_
#define _SPCMAP_VISION_H_

//STL
#include <vector>
#include <algorithm>

//Eigen
#include <Eigen/Eigen>

#include "geometry.h"

#define Vector2T Eigen::Matrix<T, 2, 1> 
#define Vector3T Eigen::Matrix<T, 3, 1> 

using namespace std;

enum CameraID {LEFT, RIGHT};

template<class T>
class Camera
{
public:
    vector<T> params;
    int width, height;

    /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Vector2T & src, Vector3T & dst) const = 0;

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Vector3T & src, Vector2T & dst) const = 0;

    //TODO implement the projection and distortion Jacobian
    virtual bool projectionJacobian(const Vector3T & src,
            Eigen::Matrix<T, 2, 3> & Jac) const = 0;

    virtual void setParameters(const T * const newParams)
    {
        copy(newParams, newParams + params.size(), params.begin());
    }
    
    Camera(int W, int H, int numParams) : width(W), height(H), params(numParams) {}

    virtual ~Camera() {}
    
    bool reconstructPointCloud(const vector<Vector2T> & src, vector<Vector3T> & dst) const
    {
        dst.resize(src.size());
        bool res = true;
        for (int i = 0; i < src.size(); i++)
        {
            res &= reconstructPoint(src[i], dst[i]);
        }  
        return res;
    }
    
    bool projectPointCloud(const vector<Vector3T> & src, vector<Vector2T> & dst) const
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

class StereoSystem
{
public:
    void projectPointCloud(const vector<Eigen::Vector3d> & src,
            vector<Vector2d> & dst1, vector<Eigen::Vector2d> & dst2) const;

    void reconstructPointCloud(const vector<Eigen::Vector2d> & src1, const vector<Eigen::Vector2d> & src2,
            vector<Eigen::Vector3d> & dst) const;

    //TODO make smart constructor with calibration data passed
    StereoSystem(Transformation & p1, Transformation & p2, Camera<double> & c1, Camera<double> & c2)
            : pose1(p1), pose2(p2), cam1(c1), cam2(c2) {}

    static bool triangulate(const Eigen::Vector3d & v1, const Eigen::Vector3d & v2,
            const Eigen::Vector3d & t,  Eigen::Vector3d & X);
    void reconstruct2(const Eigen::Vector2d & p1,
            const Eigen::Vector2d & p2,
            Eigen::Vector3d & X) const;
    Transformation pose1;  // pose of the left camera in the base frame
    Transformation pose2;  // pose of the right camera in the base frame
    Camera<double> & cam1, & cam2;
};

void computeEssentialMatrix(const vector<Vector3d> & xVec1,
        const vector<Vector3d> & xVec2,
        Matrix3d & E);

#endif
