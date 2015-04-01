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

//Ceres solver
#include <ceres/ceres.h>

#include "geometry.h"

using namespace std;
using namespace Eigen;
using namespace ceres;

enum CameraID {LEFT, RIGHT};


class Camera
{
public:
    /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Vector2d & src, Vector3d & dst) const = 0;
    
    /// projects 3D points onto the original image
    virtual bool projectPoint(const Vector3d & src, Vector2d & dst) const = 0; 
    
    //TODO implement the projection and distortion Jacobian
    virtual bool projectionJacobian(const Vector3d & src,
            Eigen::Matrix<double, 2, 3> & Jac) const = 0; 
    
    virtual ~Camera() {}
};

struct ReprojectionError : public ceres::SizedCostFunction<2, 3>
{
    ReprojectionError(double u, double v, const Transformation & xi, const Camera * const camera);
    
    bool Evaluate(double const* const* landmark,
                    double* residuals,
                    double** jac) const;
    
    //observed coordinates
    const double u, v;
    //Transformation information
    Vector3d trans;
    Matrix3d rot;
    
    //provides projection model
    const Camera * const camera;
    
};

//Performs the reconstruction of a point using its multiple 
//observations by calibrated cameras
class Reconstructor
{
public:
    Reconstructor (Vector3d & point) : point(point) {}
    
    void addObservation(double u, double v, const Transformation & pose, const Camera * const cam);
    
    void compute();
    
private:
    Problem problem;
    Vector3d & point;    
};


class StereoSystem
{
public:    
    void projectPointCloud(const vector<Vector3d> & src,
            vector<Vector2d> & dst1, vector<Vector2d> & dst2) const;
            
    void reconstructPointCloud(const vector<Vector2d> & src1, const vector<Vector2d> & src2,
            vector<Vector3d> & dst) const;
      
    ~StereoSystem(); 
    //TODO make smart constructor with calibration data passed
    StereoSystem(Transformation & p1, Transformation & p2, Camera * c1, Camera * c2) 
            : pose1(p1), pose2(p2), cam1(c1), cam2(c2) { initializeGeometrciParameter(); }
        
    void initializeGeometrciParameter();
    
    bool triangulate(const Vector2d & p1, const Vector2d & p2, Vector3d & X) const;
    void reconstruct2(const Vector2d & p1, const Vector2d & p2, Vector3d & X) const;
    Transformation pose1;  // pose of the left camera in the base frame
    Transformation pose2;  // pose of the right camera in the base frame
    Camera * cam1, * cam2; 
    
    // These parameters must be reinitialized after the pose is changed.
    Matrix3d R1, R2; // Rotations bring vectors to the base frame
    Vector3d t1, t2; // Positions of cameras in the base frame
    
    // position of the right camera wrt the left one (in the base frame)
    // t = t2 - t1;
    Vector3d t;         
};

#endif
