#ifndef _SPCSLAM_INTRINSIC_CALIBRATION_H_
#define _SPCSLAM_INTRINSIC_CALIBRATION_H_

#include "generic_calibration.h"

template<template<typename> class Camera>
class IntrinsicCameraCalibration : GenericCameraCalibration<Camera>
{
private:
    vector<CalibrationData> monoCalibDataVec;
    
    using GenericCameraCalibration<Camera>::initializeIntrinsic;
    using GenericCameraCalibration<Camera>::extractGridProjection;
    using GenericCameraCalibration<Camera>::constructGrid;
    using GenericCameraCalibration<Camera>::estimateInitialGrid;
    using GenericCameraCalibration<Camera>::initIntrinsicProblem;
    using GenericCameraCalibration<Camera>::residualAnalysis;
        
public:


    //TODO chanche the file formatting
    bool initialize(const string & infoFileName)
    {
        cout << "### Initialize calibration data ###" << endl;
        if (not initializeIntrinsic(infoFileName, monoCalibDataVec)) return false;
        constructGrid();
        return true;
    }
    
    bool compute(ICamera & camera)
    {
        
        cout << "### Intrinsic parameters calibration ###" << endl;
        vector<double> intrinsic = camera.params;
        
        if (monoCalibDataVec.size() == 0)
        {
            cout << "ERROR : none of images were accepted" << endl;
            return false;
        } 
        
        //initial board positions estimation
        estimateInitialGrid(camera, monoCalibDataVec);
        
        // Problem initialization
        Problem problem;
        initIntrinsicProblem(problem, intrinsic, monoCalibDataVec);
               
        //run the solver
        Solver::Options options;
        options.max_num_iterations = 500;
        options.function_tolerance = 1e-10;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-10;
//        options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;
        Solve(options, &problem, &summary);
        cout << summary.BriefReport() << endl;
        
        camera.setParameters(intrinsic.data());
    } 
    
    void residualAnalysis(const ICamera & camera)
    {
        cout << "### Intrinsic caibration - residual analysis ###" << endl;
        residualAnalysis(camera, monoCalibDataVec);
    }
};

#endif
