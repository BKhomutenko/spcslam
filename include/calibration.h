/*
The calibration system for a generic camera
*/

#ifndef _SPCMAP_CALIBRATION_H_
#define _SPCMAP_CALIBRATION_H_

#include <Eigen/Eigen>
#include <vector>

using namespace std;

//TODO create a stereo calibration object that keeps all the necessary information

struct BoardProjection 
{    
    BoardProjection(const vector<Eigen::Vector2d> & proj, const vector<Eigen::Vector3d> & orig)
    : _proj(proj), _orig(orig) {}
    
    virtual bool operator()(const double * const * parameters, double * residual) const;
    
    vector<Eigen::Vector2d> _proj;
    vector<Eigen::Vector3d> _orig;
};

struct StereoBoardProjection 
{
    StereoBoardProjection(const vector<Eigen::Vector2d> & proj, const vector<Eigen::Vector3d> & orig)
    : _proj(proj), _orig(orig) {}
    
    bool operator()(const double * const * parameters, double * residual) const;
    
    vector<Eigen::Vector2d> _proj;
    vector<Eigen::Vector3d> _orig;
};

struct IntrinsicCalibrationData
{
    IntrinsicCalibrationData(const string & infoFileName);
    
    int Nx, Ny;
    double sqSize;
    double outlierThresh;
    vector<BoardProjection*> residualVec;
    vector<string> fileNameVec;
    vector< array<double, 6> > extrinsicVec;
    
    void residualAnalysis(const array<double, 6> & intrinsic);
};

//TODO add the residual analysis
struct ExtrinsicCalibrationData
{
    ExtrinsicCalibrationData(const string & infoFileName);
    
    int Nx, Ny;
    double sqSize;
    double outlierThresh;
    vector<BoardProjection*> residual1Vec;
    vector<StereoBoardProjection*> residual2Vec;
    vector<string> fileNameVec;
    vector< array<double, 6> > extrinsicVec;
};

void intrinsicCalibration(const string & infoFileName, array<double, 6> & parameters);

void extrinsicStereoCalibration(const string & infoFileName1, const string & infoFileName2,
         const string & infoFileNameStereo, array<double, 6> & intrinsic1, 
         array<double, 6> & intrinsic2, array<double, 6> & extrinsic);

bool extractGridProjection(const string & fileName,
        int Nx, int Ny, vector<Eigen::Vector2d> & pointVec);
        
void generateOriginal(int Nx, int Ny, double sqSize, vector<Eigen::Vector3d> & pointVec);

double logistic(double x);

#endif

