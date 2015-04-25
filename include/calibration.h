/*
The calibration system for a generic camera
*/

#ifndef _SPCMAP_CALIBRATION_H_
#define _SPCMAP_CALIBRATION_H_

#include <Eigen/Eigen>
#include <vector>

#include "geometry.h"
#include "vision.h"

using namespace std;
using Eigen::Vector3d;
using Eigen::Vector2d;

//TODO create a stereo calibration object that keeps all the necessary information

template<template<typename> class Camera>
class IntrinsicCameraCalibration
{
    struct CostFunctor {
        template <typename T>
        bool operator()(const T * const params,
                        const T * const extrinsic,
                        T* residual) const 
        {
            Camera<T> camera(params);
            Transformation<T> transfo(extrinsic);
            
            vector<Vector3<T> > transformedPoints;
            transfo.transform(_orig, transformedPoints);
            
            vector<Vector2<T> > projectedPoints;
            camera.projectPointCloud(transformedPoints, projectedPoints);
            for (unsigned int i = 0; i < projectedPoints.size(); i++)
            {
                Vector2d diff = _proj[i] - projectedPoints[i];
                residual[2*i] = diff[0];
                residual[2*i + 1] = diff[1];
                if (std::isinf(projectedPoints[i][0]) or std::isnan(projectedPoints[i][0]))
                {
                    cout << _proj[i].transpose() << " " << projectedPoints[i].transpose() << endl;
                    cout << transformedPoints[i].transpose() << endl;
                    cout << transfo << endl;
                }
            }
        }
        
        vector<Vector2d> _proj;
        vector<Vector3d> _orig;
    };

};

struct BoardProjection 
{    
    BoardProjection(const vector<Eigen::Vector2d> & proj, const vector<Eigen::Vector3d> & orig,
            Camera<double> & camera) : _proj(proj), _orig(orig), _camera(camera) {}
    
    virtual bool operator()(const double * const * parameters, double * residual) const;
    
    Camera<double> & _camera;
    vector<Eigen::Vector2d> _proj;
    vector<Eigen::Vector3d> _orig;
};

struct StereoBoardProjection 
{
    StereoBoardProjection(const vector<Eigen::Vector2d> & proj,
             const vector<Eigen::Vector3d> & orig, Camera<double> & camera) 
             : _proj(proj), _orig(orig), _camera(camera) {}
    
    bool operator()(const double * const * parameters, double * residual) const;
    
    Camera<double> & _camera;
    vector<Eigen::Vector2d> _proj;
    vector<Eigen::Vector3d> _orig;
};

struct IntrinsicCalibrationData
{
    IntrinsicCalibrationData(const string & infoFileName, Camera<double> & camera);
    
    int Nx, Ny;
    double sqSize;
    double outlierThresh;
    vector<BoardProjection*> residualVec;
    vector<string> fileNameVec;
    vector< array<double, 6> > extrinsicVec;
    
    void residualAnalysis(const vector<double> & intrinsic);
};

//TODO add the residual analysis
struct ExtrinsicCalibrationData
{
    ExtrinsicCalibrationData(const string & infoFileName, Camera<double> & camera1, Camera<double> & camera2);
    
    int Nx, Ny;
    double sqSize;
    double outlierThresh;
    vector<BoardProjection*> residual1Vec;
    vector<StereoBoardProjection*> residual2Vec;
    vector<string> fileNameVec;
    vector< array<double, 6> > extrinsicVec;
};

void intrinsicCalibration(const string & infoFileName, Camera<double> & camera);

void extrinsicStereoCalibration(const string & infoFileName1, const string & infoFileName2,
         const string & infoFileNameStereo, Camera<double> & cam1, Camera<double> & cam2,
         Transformation<double> & T);

bool extractGridProjection(const string & fileName,
        int Nx, int Ny, vector<Eigen::Vector2d> & pointVec);
        
void generateOriginal(int Nx, int Ny, double sqSize, vector<Eigen::Vector3d> & pointVec);


#endif

