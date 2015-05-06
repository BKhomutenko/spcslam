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

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::CauchyLoss;

using namespace std;
using Eigen::Matrix;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;


struct Observation
{
    Observation(const Vector2d pt, unsigned int poseIdx, CameraID camId)
        : pt(pt), poseIdx(poseIdx), cameraId(camId) {}
    // The observed coordinates
    Vector2d pt;
    
    // Index of corresponding positions in StereoCartograpy::trajectory
    unsigned int poseIdx; 
    
    // Either left or right camera
    CameraID cameraId;
};

struct LandMark
{
    // 3D position in the globa frame
    Vector3d X;
    
    // Feature descriptor
    Matrix<float, 64, 1> d;
    
    // All Vec6drealted measurements
    vector<Observation> observations;
};

template<template<typename> class Projector>
struct ReprojectionErrorStereo 
{
    ReprojectionErrorStereo(const Vector2d pt,
        const Transformation<double> & camPose,
        const vector<double> & params) 
        : u(pt[0]), v(pt[1]), params(params) 
    { camPose.toRotTransInv(Rcb, Pcb); }

    // The args are double[3] X, double[3] trans, double[3] rot
    template <typename T>
    bool operator()(const T * Xparams,
                    const T * transParams,
                    const T * rotParams,
                    T* residual) const 
    {
        Vector3<T> rot = Vector3<T>(rotParams);
        Matrix3<T> Rbo = rotationMatrix<T>(-rot);
        Vector3<T> Pob = Vector3<T>(transParams);    
        Vector3<T> X(Xparams);
        
        X = Rcb.template cast<T>() * (Rbo * (X - Pob)) + Pcb.template cast<T>();
        Vector2<T> point;
        vector<T> paramsT(params.begin(), params.end());
        Projector<T>::compute(paramsT.data(), X.data(), point.data());
        residual[0] = point[0] - T(u);
        residual[1] = point[1] - T(v);
        return true;    
    }
    
    // The observed landmark position
    const double u, v;
    
    // Transformation information
    Vector3d Pcb;
    Matrix3d Rcb;
    
    // Parameters needed for the Projector
    const vector<double> & params;
};

template<template<typename> class Projector>
struct ReprojectionErrorFixed 
{
    ReprojectionErrorFixed(const Vector2d pt,
        const Transformation<double> & TorigBase,
        const Transformation<double> & TbaseCam,
        const vector<double> & params) 
        : u(pt[0]), v(pt[1]), params(params) 
    { 
        Transformation<double> TorigCam = TorigBase.compose(TbaseCam);
        TorigCam.toRotTransInv(Rco, Pco); 
    }

    template <typename T>
    bool operator()(const T * Xparams,
                    T* residual) const 
    {
        Vector3<T> X(Xparams);
        X = Rco.template cast<T>() * X  + Pco.template cast<T>();
        Vector2<T> point;
        vector<T> paramsT(params.begin(), params.end());
        Projector<T>::compute(paramsT.data(), X.data(), point.data());
        residual[0] = point[0] - T(u);
        residual[1] = point[1] - T(v);
        return true;    
    }
    
    const double u, v;
    
    // Transformation information
    Vector3d Pco;
    Matrix3d Rco;
    
    // Parameters needed for the Projector
    const vector<double> & params;
};

/*
struct ReprojectionErrorStereo : public ceres::SizedCostFunction<2, 3, 3, 3>
{
    ReprojectionErrorStereo(double u, double v, const Transformation<double> & xi,
            const ICamera * camera);
    
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
    const ICamera * camera;
    
};*/
/*
struct ReprojectionErrorFixed : public ceres::SizedCostFunction<2, 3>
{
    ReprojectionErrorFixed(double u, double v, const Transformation<double> & xi,
            const Transformation<double> & camTransformation, const ICamera * camera);
    
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
    const ICamera * camera;
    
};*/


//TODO implement camera calibration in the future
template<template<typename> class Projector>
class MapInitializer
{
public:
   
    void addFixedObservation(Vector3d & X, const Vector2d pt,
            const Transformation<double> & TorigBase,
            const Transformation<double> & TbaseCam,
            const ICamera & camera)
    {
        using projectionCF = AutoDiffCostFunction<ReprojectionErrorFixed<Projector>, 2, 3>; 
        
        //create a functional object
        ReprojectionErrorFixed<Projector> * errorFunct;
        errorFunct = new ReprojectionErrorFixed<Projector>(pt, TorigBase,
                TbaseCam, camera.params);
                
        // initialize a costfunction and add it to the problem
        CostFunction * costFunc = new projectionCF(errorFunct);
        problem.AddResidualBlock(costFunc, NULL, X.data());
    }

    void addObservation(Vector3d & X, const Vector2d pt,
            Transformation<double> & TorigBase,
            const Transformation<double> & TbaseCam,
            const ICamera & camera)
    {
        using projectionCF = AutoDiffCostFunction<ReprojectionErrorStereo<Projector>, 2, 3, 3, 3>; 
        
        //create a functional object
        ReprojectionErrorStereo<Projector> * errorFunct;
        errorFunct = new ReprojectionErrorStereo<Projector>(pt, TbaseCam, camera.params);
                
        // initialize a costfunction and add it to the problem
        CostFunction * costFunc = new projectionCF(errorFunct);
        problem.AddResidualBlock(costFunc, NULL, X.data(),
                TorigBase.transData(), TorigBase.rotData());
    }

    void compute()
    {
        Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
    //    options.function_tolerance = 1e-3;
    //    options.gradient_tolerance = 1e-4;
    //    options.parameter_tolerance = 1e-4;
    //    options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;
        Solve(options, &problem, &summary);
    }
    
private:
    ceres::Problem problem;

};


class StereoCartography
{
public:
    StereoCartography (Transformation<double> & p1, Transformation<double> & p2,
            ICamera & c1, ICamera & c2) 
            : stereo(p1, p2, c1, c2) {}
//    virtual ~StereoCartography () { LM.clear(); trajectory.clear(); }
    
    StereoSystem stereo;
    
    void projectPointCloud(const vector<Vector3d> & src,
            vector<Vector2d> & dst1, vector<Vector2d> & dst2, int poseIdx) const;
            
    //performs optimization of all landmark positions wrt the actual path
    template<template<typename> class Projector>
    void improveTheMap()
    {   
        //BUNDLE ADJUSTMENT
        //TODO Make cartography a template as well
        MapInitializer<Projector> initializer;
        for (auto & landmark : LM)
        {
            for (auto & observation : landmark.observations)
            {
                int xiIdx = observation.poseIdx;
                if (xiIdx == 0)
                {
                    if (observation.cameraId == LEFT)
                    {
                        initializer.addFixedObservation(landmark.X, observation.pt,
                                trajectory[xiIdx], stereo.pose1, *(stereo.cam1));
                    }
                    else
                    {
                        initializer.addFixedObservation(landmark.X, observation.pt,
                                trajectory[xiIdx], stereo.pose2, *(stereo.cam2));
                    }
                }
                else if (observation.cameraId == LEFT)
                {
                    initializer.addObservation(landmark.X, observation.pt,
                            trajectory[xiIdx], stereo.pose1, *(stereo.cam1));
                }
                else
                {
                    initializer.addObservation(landmark.X, observation.pt,
                            trajectory[xiIdx], stereo.pose2, *(stereo.cam2));
                }
            }
        }
        initializer.compute();

    }   
    
    // Observation are supposed to be made by the left camera (cam1, pose1)
    Transformation<double> computeTransformation(
        const vector<Vector2d> & observationVec,
        const vector<Vector3d> & cloud, 
        const vector<bool> & inlierMask, 
        Transformation<double> & xi);
            
    void odometryRansac(
        const vector<Vector2d> & observationVec,
        const vector<Vector3d> & cloud,
        vector<bool> & inlierMask,
        Transformation<double> & xi);
            
    Transformation<double> estimateOdometry(const vector<Feature> & featureVec);
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
    vector<Transformation<double>> trajectory;
    //list<LandMark &> activeLM;

};

#endif
