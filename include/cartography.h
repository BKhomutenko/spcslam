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
#include "mei.h"
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

inline double sinc(const double x);

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
struct OdometryError 
{
    OdometryError(const Vector3d X, const Vector2d pt,
        const Transformation<double> & camPose,
        const vector<double> & params) 
        : X(X), u(pt[0]), v(pt[1]), params(params)
    { camPose.toRotTransInv(Rcb, Pcb); }

    // The args are double[3] trans, double[3] rot
    template <typename T>
    bool operator()(const T * transParams,
                    const T * rotParams,
                    T* residual) const 
    {
        Vector3<T> rot(rotParams);
        Quaternion<T> Qbo = Quaternion<T>(rot).inv();
        Vector3<T> Pob(transParams);    
        Vector3<T> XT = X.template cast<T>() - Pob;
        XT = Rcb.template cast<T>() * (Qbo.rotate(XT)) + Pcb.template cast<T>();
        Vector2<T> point;
        vector<T> paramsT(params.begin(), params.end());
        Projector<T>::compute(paramsT.data(), XT.data(), point.data());
        residual[0] = point[0] - T(u);
        residual[1] = point[1] - T(v);
        return true;    
    }
    
    // The observed landmark position
    const double u, v;
    
    // Transformation information
    Vector3d Pcb;
    Matrix3d Rcb;
    Vector3d X;
    // Parameters needed for the Projector
    const vector<double> & params;
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

//template<template<typename> class Projector>
class Odometry
{
public:
    vector<Vector2d> observationVec;
    vector<Vector3d> cloud;
    vector<bool> inlierMask;
    Transformation<double> TorigBase;
    const Transformation<double> TbaseCam;
    const ICamera & camera;
    
    Odometry(const Transformation<double> TorigBase,
            const Transformation<double> TbaseCam,
            const ICamera & camera) 
            : TorigBase(TorigBase), TbaseCam(TbaseCam), camera(camera) {}
    
    Odometry(const Transformation<double> TorigBase,
            const Transformation<double> TbaseCam,
            const ICamera * camera) 
            : TorigBase(TorigBase), TbaseCam(TbaseCam), camera(*camera) {}
            
    void computeTransformation()
    {
        assert(observationVec.size() == cloud.size());
        assert(observationVec.size() == inlierMask.size());
        Problem problem;
    
        using projectionCF = AutoDiffCostFunction<OdometryError<MeiProjector>, 2, 3, 3>; 
        for (unsigned int i = 0; i < cloud.size(); i++)
        {
            if (not inlierMask[i]) continue;
            //create a functional object
            OdometryError<MeiProjector> * errorFunct;
            errorFunct = new OdometryError<MeiProjector>(cloud[i], observationVec[i],
                    TbaseCam, camera.params);
                    
            // initialize a costfunction and add it to the problem
            CostFunction * costFunc = new projectionCF(errorFunct);
            problem.AddResidualBlock(costFunc, NULL,
                    TorigBase.transData(), TorigBase.rotData());
        }
        
        Solver::Options options;
        
//        options.linear_solver_type = ceres::DENSE_SCHUR;
        Solver::Summary summary;
        Solve(options, &problem, &summary);
        /*
        for (unsigned int optimIter = 0; optimIter < 10; optimIter++)
        {
            Matrix3d Rbo, LxiInv;
            Vector3d Pbo;
            TorigBase.toRotTransInv(Rbo, Pbo);
            Vector3d u = TorigBase.rot();
            double theta = u.norm();
            if ( theta != 0)
            {
                u /= theta;
                Matrix3d uhat = hat(u);
                LxiInv = Matrix3d::Identity() + 
                    theta/2*sinc(theta/2)*uhat + 
                    (1 - sinc(theta))*uhat*uhat;
            }
            else
            {
                LxiInv = Matrix3d::Identity();
            }
            Eigen::Matrix<double, 6, 6> JTJ = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> JTerr = Eigen::Matrix<double, 6, 1>::Zero();
            double ERR = 0;
            for (unsigned int i = 0; i < observationVec.size(); i++)
            {
                if (not inlierMask[i])
                {
                    continue;
                }

                Vector3d X = Rbo*cloud[i] + Pbo;
                X = Rcb*X + Pcb;
                Vector2d err;
                camera->projectPoint(X, err);
                err = observationVec[i] - err;
                ERR += err.norm();
                Matrix3d Rco = Rcb * Rbo;
                Eigen::Matrix<double, 2, 3> Jx;
                camera->projectionJacobian(X, Jx);
                
                Eigen::Matrix<double, 2, 6> Jcam;
                Jcam << -Jx * Rco, Jx * hat(X) * Rco * LxiInv;
                JTJ += Jcam.transpose()*Jcam;
                JTerr += Jcam.transpose()*err;           
            }
            auto dxi = JTJ.inverse()*JTerr;
            TorigBase.trans() += dxi.head<3>();
            TorigBase.rot() += dxi.tail<3>();
        }*/
    }
    
            
    void Ransac()
    {
        assert(observationVec.size() == cloud.size());
        int numPoints = observationVec.size();
        
        inlierMask.resize(numPoints);
        
        const int numIterMax = 25;
        const Transformation<double> initialPose = TorigBase;
        int bestInliers = 0;
        //TODO add a termination criterion
        for (unsigned int iteration = 0; iteration < numIterMax; iteration++)
        {
            Transformation<double> pose = initialPose;
            int maxIdx = observationVec.size();
            //choose three points at random
		    int idx1m = rand() % maxIdx;
		    int idx2m, idx3m;
		    do 
		    {
			    idx2m = rand() % maxIdx;
		    } while (idx2m == idx1m);
		
		    do 
		    {
			    idx3m = rand() % maxIdx;
		    } while (idx3m == idx1m or idx3m == idx2m);
            
    //        cout << cloud[idx1m].transpose() << " ###### " << observationVec[idx1m].transpose() << endl; 
    //        cout << cloud[idx2m].transpose() << " ###### " << observationVec[idx2m].transpose() << endl;
    //        cout << cloud[idx3m].transpose() << " ###### " << observationVec[idx3m].transpose() << endl;
    //        cout << endl;
                    
            
            //solve an optimization problem 
            
            Problem problem;
            using projectionCF = AutoDiffCostFunction<OdometryError<MeiProjector>, 2, 3, 3>; 
            for (auto i : {idx1m, idx2m, idx3m})
            {
                //create a functional object
                OdometryError<MeiProjector> * errorFunct;
                errorFunct = new OdometryError<MeiProjector>(cloud[i], observationVec[i],
                        TbaseCam, camera.params);
                        
                // initialize a costfunction and add it to the problem
                CostFunction * costFunc = new projectionCF(errorFunct);
                problem.AddResidualBlock(costFunc, NULL,
                        pose.transData(), pose.rotData());
            }
            
            Solver::Options options;
//            options.linear_solver_type = ceres::DENSE_SCHUR;
            Solver::Summary summary;
            options.max_num_iterations = 5;
            Solve(options, &problem, &summary);
            //compute xi using these points
            /*for (unsigned int i = 0; i < 10; i++)
            {
                double theta = pose.rot().norm();
                Matrix3d uhat = hat(pose.rot());
                Matrix3d LxiInv = Matrix3d::Identity() + 
                    theta/2*sinc(theta/2)*uhat + 
                    (1 - sinc(theta))*uhat*uhat;
                    
                Matrix3d Rbo, Rcb;
                Vector3d Pbo, Pcb;
                
                pose.toRotTransInv(Rbo, Pbo);
                stereo.pose1.toRotTransInv(Rcb, Pcb);
                
                Vector3d X1, X2, X3;            
                Vector2d Err1, Err2, Err3;
                Eigen::Matrix<double, 2, 3> J1, J2, J3;
                
                X1 = Rcb*(Rbo*cloud[idx1m] + Pbo) + Pcb;
                stereo.cam1->projectPoint(X1, Err1);
                stereo.cam1->projectionJacobian(X1, J1);
                
                X2 = Rcb*(Rbo*cloud[idx2m] + Pbo) + Pcb;
                stereo.cam1->projectPoint(X2, Err2);
                stereo.cam1->projectionJacobian(X2, J2);
                
                X3 = Rcb*(Rbo*cloud[idx3m] + Pbo) + Pcb;
                stereo.cam1->projectPoint(X3, Err3);            
                stereo.cam1->projectionJacobian(X3, J3);
                
                Eigen::Matrix<double, 6, 6> J;
                Matrix3d Rco = Rcb * Rbo;
                
                J << -J1 * Rco, J1 * hat(X1) * Rco * LxiInv,
                    -J2 * Rco, J2 * hat(X2) * Rco * LxiInv,
                    -J3 * Rco, J3 * hat(X3) * Rco * LxiInv;
                Eigen::Matrix<double, 6, 1> E, dxi;
                E << observationVec[idx1m] - Err1,
                     observationVec[idx2m] - Err2,
                     observationVec[idx3m] - Err3;
                dxi = J.inverse() * E;
                
                pose.trans() += dxi.head<3>();
                pose.rot() += dxi.tail<3>();
                
                if (iteration < 0)
                {
                    cout << (observationVec[idx1m] - Err1).transpose() << endl;
                    cout << (observationVec[idx2m] - Err2).transpose() << endl;
                    cout << (observationVec[idx3m] - Err3).transpose() << endl;
                    cout << " ##### " << endl;
                }
            }*/
            
            //count inliers
            vector<Vector3d> XcamVec(numPoints);
            Transformation<double> TorigCam = pose.compose(TbaseCam);
            TorigCam.inverseTransform(cloud, XcamVec);
            vector<Vector2d> projVec(numPoints);
            camera.projectPointCloud(XcamVec, projVec);
//            for (int i = 0; i < numPoints; i++)
//            {
//                cout << XcamVec[i].transpose() << " ";
////                MeiProjector<double>::compute(params.data(), XcamVec[i].data(), projVec[i].data());
//                cout << projVec[i].transpose() << endl;
//            }
            vector<bool> currentInlierMask(numPoints, false);
            
            int countInliers = 0;
            for (unsigned int i = 0; i < numPoints; i++)
            {   
//                cout << observationVec[i].transpose() << " " <<  projVec[i].transpose() << endl;
                Vector2d err = observationVec[i] - projVec[i];
                if (err.norm() < 2)
                {
                    currentInlierMask[i] = true;
                    countInliers++;
                }
            }
            //keep the best hypothesis
            if (countInliers > bestInliers)
            {
//                cout << "improvement "  << countInliers << " " <<  bestInliers << endl;
                //TODO copy in a bettegit lor way
                inlierMask = currentInlierMask;
                bestInliers = countInliers;
                TorigBase = pose;
            }        
        }
    }
};

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
//        options.linear_solver_type = ceres::DENSE_SCHUR;
    //    options.function_tolerance = 1e-3;
    //    options.gradient_tolerance = 1e-4;
    //    options.parameter_tolerance = 1e-4;
    //    options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;
        Solve(options, &problem, &summary);
        cout << summary.FullReport() << endl;
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
