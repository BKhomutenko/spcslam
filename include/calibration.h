/*
The calibration system for a generic camera
*/

#ifndef _SPCMAP_CALIBRATION_H_
#define _SPCMAP_CALIBRATION_H_

#include <Eigen/Eigen>
#include <vector>

#include "vision.h"

using namespace std;

//TODO create a stereo calibration object that keeps all the necessary information

struct BoardProjection 
{    
    BoardProjection(const vector<Eigen::Vector2d> & proj, const vector<Eigen::Vector3d> & orig,
            Camera & camera) : _proj(proj), _orig(orig), _camera(camera) {}
    
    virtual bool operator()(const double * const * parameters, double * residual) const;
    
    Camera & _camera;
    vector<Eigen::Vector2d> _proj;
    vector<Eigen::Vector3d> _orig;
};

struct StereoBoardProjection 
{
    StereoBoardProjection(const vector<Eigen::Vector2d> & proj,
             const vector<Eigen::Vector3d> & orig, Camera & camera) 
             : _proj(proj), _orig(orig), _camera(camera) {}
    
    bool operator()(const double * const * parameters, double * residual) const;
    
    Camera & _camera;
    vector<Eigen::Vector2d> _proj;
    vector<Eigen::Vector3d> _orig;
};

struct IntrinsicCalibrationData
{
    IntrinsicCalibrationData(const string & infoFileName, Camera & camera);
    
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
    ExtrinsicCalibrationData(const string & infoFileName, Camera & camera1, Camera & camera2);
    
    int Nx, Ny;
    double sqSize;
    double outlierThresh;
    vector<BoardProjection*> residual1Vec;
    vector<StereoBoardProjection*> residual2Vec;
    vector<string> fileNameVec;
    vector< array<double, 6> > extrinsicVec;
};

void intrinsicCalibration(const string & infoFileName, Camera & camera);

void extrinsicStereoCalibration(const string & infoFileName1, const string & infoFileName2,
         const string & infoFileNameStereo, Camera & cam1, Camera & cam2, Transformation & T);

bool extractGridProjection(const string & fileName,
        int Nx, int Ny, vector<Eigen::Vector2d> & pointVec);
        
void generateOriginal(int Nx, int Ny, double sqSize, vector<Eigen::Vector3d> & pointVec);


#endif

