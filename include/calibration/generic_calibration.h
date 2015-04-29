#ifndef _SPCMAP_GENERIC_CALIBRATION_H_
#define _SPCMAP_GENERIC_CALIBRATION_H_

#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>

#include "vision.h"
#include "cost_functors.h"

using namespace std;
using namespace cv;
using Eigen::Vector3d;
using Eigen::Vector2d;

using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::CauchyLoss;

int nObjects = 0;

typedef shared_ptr<array<double, 6>> ArraySharedPtr;

struct CalibrationData
{
    vector<Vector2d> projection;
    ArraySharedPtr extrinsic;
    string fileName;
};

template<template<typename> class Camera>
class GenericCameraCalibration
{
protected:
    int Nx, Ny;
    double sqSize;
    double outlierThresh;
    vector<Vector3d> grid;
    
public:

    //TODO chanche the file formatting
    bool initializeIntrinsic(const string & infoFileName, 
            vector<CalibrationData> & calibDataVec)
    {
        // open the file and read the data
        ifstream calibInfoFile(infoFileName);
        if (not calibInfoFile.is_open())
        {
            cout << infoFileName << " : ERROR, file is not found" << endl;
            return false;
        }
        bool checkExtraction;
        calibInfoFile >> Nx >> Ny >> sqSize >> outlierThresh >> checkExtraction;
        calibInfoFile.ignore();  // To get to the next line
        
        calibDataVec.clear();
        string imageFolder;
        string imageName;    
        getline(calibInfoFile, imageFolder);
        while (getline(calibInfoFile, imageName))
        {
            CalibrationData calibData;
            vector<Vector2d> projection;
            bool isExtracted;
            
            calibData.fileName = imageFolder + imageName;
            isExtracted = extractGridProjection(calibData, checkExtraction);
            
            if (not isExtracted)
            {
                continue;
            }      
                  
            calibData.extrinsic = ArraySharedPtr(new array<double, 6>{0, 0, 1, 0, 0, 0});
            calibDataVec.push_back(calibData);
            
            cout << "." << flush;
        }
        cout << "done" << endl;
        return true;
    }
    
    bool extractGridProjection(CalibrationData & calibData, bool checkExtraction)
    {
        Size patternSize(Nx, Ny);
        Mat frame = imread(calibData.fileName, 0);

        vector<Point2f> centers;
        bool patternIsFound = findChessboardCorners(frame, patternSize, centers);
        if (not patternIsFound)
        {
            cout << calibData.fileName << " : ERROR, pattern is not found" << endl;
            return false;
        }
        
        if (checkExtraction)
        {
            drawChessboardCorners(frame, patternSize, Mat(centers), patternIsFound);
            imshow("corners", frame);
            char key = waitKey();
            if (key == 'n' or key == 'N')
            {
                cout << calibData.fileName << " : ERROR, pattern is not accepted" << endl;
                return false; 
            }  
        } 
        
        calibData.projection.resize(Nx * Ny);
        for (unsigned int i = 0; i < Nx * Ny; i++)
        {
            calibData.projection[i] = Vector2d(centers[i].x, centers[i].y);
        }
        return true;
    }

    void constructGrid()
    {
        grid.resize(Nx * Ny);
        for (unsigned int i = 0; i < Nx * Ny; i++)
        {
            grid[i] = Vector3d(sqSize * (i % Nx), sqSize * (i / Nx), 0);
        }
    }

    void estimateInitialGrid(Camera<double> & camera,
            vector<CalibrationData> & calibDataVec)
    {
        for (int i = 0; i < calibDataVec.size(); i++)
        {
            Problem problem;
            typedef DynamicAutoDiffCostFunction<GridEstimate<Camera>> dynamicProjectionCF;

            GridEstimate<Camera> * boardEstimate;
            boardEstimate = new GridEstimate<Camera>(calibDataVec[i].projection,
                                        grid, camera.params);
            dynamicProjectionCF * costFunction = new dynamicProjectionCF(boardEstimate);
            costFunction->AddParameterBlock(6);
            costFunction->SetNumResiduals(2 * Nx * Ny);
            problem.AddResidualBlock(costFunction, new CauchyLoss(1),
                    calibDataVec[i].extrinsic->data());   
            
            //run the solver
            Solver::Options options;
            Solver::Summary summary;
            Solve(options, &problem, &summary);
        }
    }
    
    void initIntrinsicProblem(Problem & problem, vector<double> & intrinsic,
            vector<CalibrationData> & calibDataVec)
    {
        typedef DynamicAutoDiffCostFunction<GridProjection<Camera>> projectionCF;        
        for (unsigned int i = 0; i < calibDataVec.size(); i++)
        {
            GridProjection<Camera> * boardProjection;
            boardProjection = new GridProjection<Camera>(calibDataVec[i].projection, grid);
            projectionCF * costFunction = new projectionCF(boardProjection);
            costFunction->AddParameterBlock(intrinsic.size());
            costFunction->AddParameterBlock(6);
            costFunction->SetNumResiduals(2 * Nx * Ny);
            problem.AddResidualBlock(costFunction, NULL, intrinsic.data(),
                    calibDataVec[i].extrinsic->data());   
        }
    }
    
    void residualAnalysis(const Camera<double> & camera,
            const vector<CalibrationData> & calibDataVec)
    {
        residualAnalysis(camera, calibDataVec, Transformation<double>());
    }
            
    void residualAnalysis(const Camera<double> & camera,
            const vector<CalibrationData> & calibDataVec,
            const Transformation<double> & TrefCam)
    {
        
        double Ex = 0, Ey = 0;
        double Emax = 0;
        for (unsigned int ptIdx = 0; ptIdx < calibDataVec.size(); ptIdx++)
        {
                vector<Vector3d> transfModelVec;
                Transformation<double> TrefGrid(calibDataVec[ptIdx].extrinsic->data());
                Transformation<double> TcamGrid = TrefCam.inverseCompose(TrefGrid);
                TcamGrid.transform(grid, transfModelVec);
                
                vector<Vector2d> projModelVec;
                camera.projectPointCloud(transfModelVec, projModelVec);
                
                Mat frame = imread(calibDataVec[ptIdx].fileName, 0);
                bool outlierDetected = false;
                for (unsigned int i = 0; i < Nx * Ny; i++)
                {
                    Vector2d p = calibDataVec[ptIdx].projection[i];
                    Vector2d pModel = projModelVec[i];
                    
                    circle(frame, Point(p(0), p(1)), 7, 127, 1);
                    circle(frame, Point(pModel(0), pModel(1)), 4.5, 255, 1);
                    Vector2d delta = p - pModel;
                    double dx = delta[0] * delta[0];
                    double dy = delta[1] * delta[1];
                    if (outlierThresh != 0 and dx + dy > outlierThresh * outlierThresh)
                    {
                        outlierDetected = true;
                        cout << calibDataVec[ptIdx].fileName << " # " << i << endl;
                        cout << delta.transpose() << endl;
                    }
                    if (dx + dy > Emax)
                    {
                        Emax = dx + dy;
                    }
                    Ex += dx;
                    Ey += dy;
                }
                if (outlierDetected)
                {
                    imshow("reprojection", frame);
                    waitKey();
                }
        }
        Ex /= calibDataVec.size() * Nx * Ny;
        Ey /= calibDataVec.size() * Nx * Ny;
        Ex = sqrt(Ex);
        Ey = sqrt(Ey);
        Emax = sqrt(Emax);
        cout << "Ex = " << Ex << "; Ey = " << Ey << "; Emax = " << Emax << endl;  
    }
};

#endif
