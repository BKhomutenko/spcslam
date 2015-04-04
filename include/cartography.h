/*
The mapping system itself
*/

#ifndef _SPCMAP_CARTOGRAPHY_H_
#define _SPCMAP_CARTOGRAPHY_H_

//STL
#include <vector>

//Eigen
#include <Eigen/Eigen>

//Ceres solver
#include <ceres/ceres.h>

#include "extractor.h"
#include "geometry.h"
#include "vision.h"

//Structure is used to perform map improvement

using namespace std;
using Eigen::Matrix;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;


struct Observation
{
    Observation(double u, double v, unsigned int poseIdx, CameraID camId)
        : u(u), v(v), poseIdx(poseIdx), cameraId(camId) {}
    //observed coordinates
    double u, v;
    
    //index of corresponding positions in StereoCartograpy::trajectory
    unsigned int poseIdx; 
    
    //Either left or right camera
    CameraID cameraId;
};

struct LandMark
{
    //3D position in the globa frame
    Vector3d X;
    
    //Feature descriptor
    Matrix<float, 64, 1> d;
    
    //All Vec6drealted measurements
    vector<Observation> observations;
};

struct ReprojectionErrorStereo : public ceres::SizedCostFunction<2, 3, 3, 3>
{
    ReprojectionErrorStereo(double u, double v, const Transformation & xi, const Camera * const camera);
    
    // args : double lm[3], double pose[6]
    bool Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const;
    
    //observed coordinates
    const double u, v;
    //Transformation information
    Vector3d Pcb;
    Matrix3d Rcb;
    
    //provides projection model
    const Camera * const camera;
    
};

struct ReprojectionErrorFixed : public ceres::SizedCostFunction<2, 3>
{
    ReprojectionErrorFixed(double u, double v, const Transformation & xi,
            const Transformation & camTransformation, const Camera * const camera);
    
    // args : double lm[3]
    bool Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const;
    
    //observed coordinates
    const double u, v;
    //Transformation information
    Vector3d Pcb, Pbo;
    Matrix3d Rcb, Rbo;
    //provides projection model
    const Camera * const camera;
    
};


//TODO implement camera calibration in the future
class MapInitializer
{
public:
   
    void addObservation(Vector3d & X, double u, double v, Transformation & pose,
            const Camera * const cam, const Transformation & camTransformation);
    
    void addFixedObservation(Vector3d & X, double u, double v, Transformation & pose,
            const Camera * const cam, const Transformation & camTransformation);       
//    void addObservationRight(Vector3d & X, double u, double v, Transformation & pose,
//            const Camera * const cam, Transformation & rightCamTransformation);
            
    void compute();
    
private:
    ceres::Problem problem;

};




class StereoCartography
{
public:
    StereoCartography (Transformation & p1, Transformation & p2, Camera * c1, Camera * c2) 
            : stereo(p1, p2, c1, c2) {}
//    virtual ~StereoCartography () { LM.clear(); trajectory.clear(); }
    
    StereoSystem stereo;
    
    void projectPointCloud(const vector<Vector3d> & src,
            vector<Vector2d> & dst1, vector<Vector2d> & dst2, int poseIdx) const;
            
    //performs optimization of all landmark positions wrt the actual path
    void improveTheMap();    
    
    // Observation are supposed to be made by the left camera (cam1, pose1)
    Transformation computeTransformation(
        const vector<Vector2d> & observationVec,
        const vector<Vector3d> & cloud, 
        const vector<bool> & inlierMask, 
        Transformation & xi);
            
    void odometryRansac(
        const vector<Vector2d> & observationVec,
        const vector<Vector3d> & cloud,
        vector<bool> & inlierMask,
        Transformation & xi);
            
    Transformation estimateOdometry(const vector<Feature> & featureVec);
    // take 300 last landmarks 
    // BF matching
    // RANSAC
    // transformation refinement
    
    void refineCloud(int maxStep);
    void refineTrajectory();
    //the library of all landmarks
    //to be replaced in the future with somth smarter than a vector
    vector<LandMark> LM;
    
    //a chain of camera positions
    //first initialized with the odometry measurements
    vector<Transformation> trajectory;
    //list<LandMark &> activeLM;

};

#endif
