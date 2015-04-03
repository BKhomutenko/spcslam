/*
Abstract camera definition.
Stereo vision definition.
*/

#ifndef _SPCMAP_VISION_H_
#define _SPCMAP_VISION_H_

//STL
#include <vector>

//Eigen
#include <Eigen/Eigen>

#include "geometry.h"

using namespace std;

enum CameraID {LEFT, RIGHT};

class Camera
{
public:

    unsigned int imageWidth;
    unsigned int imageHeight;

    /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Eigen::Vector2d & src, Eigen::Vector3d & dst) const = 0;

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Eigen::Vector3d & src, Eigen::Vector2d & dst) const = 0;

    //TODO implement the projection and distortion Jacobian
    virtual bool projectionJacobian(const Eigen::Vector3d & src,
            Eigen::Matrix<double, 2, 3> & Jac) const = 0;

    virtual ~Camera() {}
};

class StereoSystem
{
public:
    void projectPointCloud(const vector<Eigen::Vector3d> & src,
            vector<Eigen::Vector2d> & dst1, vector<Eigen::Vector2d> & dst2) const;

    void reconstructPointCloud(const vector<Eigen::Vector2d> & src1, const vector<Eigen::Vector2d> & src2,
            vector<Eigen::Vector3d> & dst) const;

    ~StereoSystem();
    //TODO make smart constructor with calibration data passed
    StereoSystem(Transformation & p1, Transformation & p2, Camera * c1, Camera * c2)
            : pose1(p1), pose2(p2), cam1(c1), cam2(c2) {}

    static bool triangulate(const Eigen::Vector3d & v1, const Eigen::Vector3d & v2,
            const Eigen::Vector3d & t,  Eigen::Vector3d & X);
    void reconstruct2(const Eigen::Vector2d & p1, const Eigen::Vector2d & p2, Eigen::Vector3d & X) const;
    Transformation pose1;  // pose of the left camera in the base frame
    Transformation pose2;  // pose of the right camera in the base frame
    Camera * cam1, * cam2;
};

#endif
