#ifndef _SPCSLAM_EXTRINSIC_CALIBRATION_H_
#define _SPCSLAM_EXTRINSIC_CALIBRATION_H_

#include "generic_calibration.h"

template<template<typename> class Camera>
struct StereoGridProjection 
{
    StereoGridProjection(const vector<Eigen::Vector2d> & proj,
            const vector<Eigen::Vector3d> & orig) : _proj(proj), _orig(orig) {}
            
    template <typename T>
    bool operator()(const T * const* params,
                    T* residual) const 
    {
        Camera<T> camera(params[0]);
        Transformation<T> TrefGrid(params[1]);
        Transformation<T> TrefCam(params[2]);
        Transformation<T> TcamGrid = TrefCam.inverseCompose(TrefGrid);
        
        vector<Vector3<T>> transformedPoints(_orig.size());
        for (int i = 0; i < _orig.size(); i++)
        {
            transformedPoints[i] = _orig[i].template cast<T>();
        }
        TcamGrid.transform(transformedPoints, transformedPoints);
        
        vector<Vector2<T>> projectedPoints;
        camera.projectPointCloud(transformedPoints, projectedPoints);
        for (unsigned int i = 0; i < projectedPoints.size(); i++)
        {
            Vector2<T> diff = _proj[i].template cast<T>() - projectedPoints[i];
            residual[2*i] = T(diff[0]);
            residual[2*i + 1] = T(diff[1]);
        }
        return true;
    }
    
    const vector<Vector2d> & _proj;
    const vector<Vector3d> & _orig;
};

template<template<typename> class Camera>
struct StereoEstimate
{
    StereoEstimate(const vector<Vector2d> & proj, const vector<Vector3d> & orig,
    const vector<double> & camParams, const array<double, 6> & extrinsic) 
    : _proj(proj), _orig(orig), _camParams(camParams),
      _extrinsic(extrinsic) {}
            
    template <typename T>
    bool operator()(const T * const * params,
                    T* residual) const 
    {
        vector<T> camParamsT(_camParams.size());
        for (int i = 0; i < _camParams.size(); i++)
        {
            camParamsT[i] = T(_camParams[i]);
        }
        Camera<T> camera(camParamsT.data());
        
        array<T, 6> extrinsic;
        for (int i = 0; i < _extrinsic.size(); i++)
        {
            extrinsic[i] = T(_extrinsic[i]);
        }
        Transformation<T> TrefGrid(extrinsic.data());
        Transformation<T> TrefCam(params[0]);
        Transformation<T> TcamGrid = TrefCam.inverseCompose(TrefGrid);
        vector<Vector3<T>> transformedPoints(_orig.size());
        for (int i = 0; i < _orig.size(); i++)
        {
            transformedPoints[i] = _orig[i].template cast<T>();
        }
        TcamGrid.transform(transformedPoints, transformedPoints);
        
        vector<Vector2<T>> projectedPoints;
        camera.projectPointCloud(transformedPoints, projectedPoints);
        for (unsigned int i = 0; i < projectedPoints.size(); i++)
        {
            Vector2<T> diff = _proj[i].template cast<T>() - projectedPoints[i];
            residual[2*i] = T(diff[0]);
            residual[2*i + 1] = T(diff[1]);
        }
        return true;
    }
    
    const array<double, 6> & _extrinsic;
    const vector<double> & _camParams;
    const vector<Vector2d> & _proj;
    const vector<Vector3d> & _orig;
};

template<template<typename> class Camera>
class ExtrinsicCameraCalibration : GenericCameraCalibration<Camera>
{
private:
    vector<CalibrationData> monoCalibDataVec1;
    vector<CalibrationData> monoCalibDataVec2;
    vector<CalibrationData> stereoCalibDataVec1;
    vector<CalibrationData> stereoCalibDataVec2;
    
    // methods
    using GenericCameraCalibration<Camera>::initializeIntrinsic;
    using GenericCameraCalibration<Camera>::extractGridProjection;
    using GenericCameraCalibration<Camera>::constructGrid;
    using GenericCameraCalibration<Camera>::estimateInitialGrid;
    using GenericCameraCalibration<Camera>::initIntrinsicProblem;
    using GenericCameraCalibration<Camera>::residualAnalysis;
    
    // fields
    using GenericCameraCalibration<Camera>::grid;
    using GenericCameraCalibration<Camera>::Nx;
    using GenericCameraCalibration<Camera>::Ny;
    
public:

    bool initialize(const string & infoFileName1, const string & infoFileName2, 
            const string & infoFileNameStereo)
    {
        cout << "### Initialize calibration data ###" << endl;
        if (not initializeIntrinsic(infoFileName1, monoCalibDataVec1))
        {
            return false;
        }
        if (not initializeIntrinsic(infoFileName2, monoCalibDataVec2))
        {
            return false;
        }
        constructGrid();
        
        if (not initializeExtrinsic(infoFileNameStereo)) return false;
        return true;
    } 
    
    bool initializeExtrinsic(const string & infoFileName)
    {
            // open the file and read the data
        ifstream calibInfoFile(infoFileName);
        if (not calibInfoFile.is_open())
        {
            cout << infoFileName << " : ERROR, file is not found" << endl;
            return false;
        }
        bool checkExtraction;
        double foo1, foo2;
        calibInfoFile >> Nx >> Ny >> foo1 >> foo2 >> checkExtraction;
        calibInfoFile.ignore();  // To get to the next line
        
        stereoCalibDataVec1.clear();
        stereoCalibDataVec2.clear();
        
        string imageFolder;
        string imageName;    
        string leftPref, rightPref;
        
        getline(calibInfoFile, imageFolder);
        getline(calibInfoFile, leftPref);
        getline(calibInfoFile, rightPref);
        while (getline(calibInfoFile, imageName))
        {
            CalibrationData calibDataLeft, calibDataRight;
            vector<Vector2d> point1Vec, point2Vec;
            bool isExtracted1, isExtracted2;
            
            calibDataLeft.fileName = imageFolder + leftPref + imageName;
            isExtracted1 = extractGridProjection(calibDataLeft, checkExtraction);
            
            calibDataRight.fileName = imageFolder + rightPref + imageName;                                     
            isExtracted2 = extractGridProjection(calibDataRight, checkExtraction);
            
            if (not isExtracted1 or not isExtracted2) 
            {
                continue;
            }
            
            calibDataLeft.extrinsic = ArraySharedPtr(new array<double, 6>{0, 0, 1, 0, 0, 0});
            calibDataRight.extrinsic = calibDataLeft.extrinsic;
            
            stereoCalibDataVec1.push_back(calibDataLeft);
            stereoCalibDataVec2.push_back(calibDataRight);
            
            cout << "." << flush;
        }
        cout << "done" << endl;
        return true;
    }
    

    void initStereoProblem(Problem & problem, vector<double> & intrinsic, 
            array<double, 6> & extrinsic,
            vector<CalibrationData> & calibDataVec)
    {
        typedef DynamicAutoDiffCostFunction<StereoGridProjection<Camera>> stereoProjectionCF;
        for (unsigned int i = 0; i < calibDataVec.size(); i++)
        {
            StereoGridProjection<Camera> * stereoProjection;
            stereoProjection = new StereoGridProjection<Camera>(
                                        calibDataVec[i].projection,
                                        grid);
            
            stereoProjectionCF * costFunction = new stereoProjectionCF(stereoProjection);
            costFunction->AddParameterBlock(intrinsic.size());
            costFunction->AddParameterBlock(6);
            costFunction->AddParameterBlock(6);
            costFunction->SetNumResiduals(2 * Nx * Ny);
            problem.AddResidualBlock(costFunction, NULL, intrinsic.data(),
                    calibDataVec[i].extrinsic->data(), extrinsic.data());
            
        }
    }
    
    void estimateInitialExtrinsic(const Camera<double> & camera, array<double, 6> & extrinsic,
            vector<CalibrationData> & calibDataVec)
    {
        Problem problem;
        for (int i = 0; i < calibDataVec.size(); i++)
        {
            
            typedef DynamicAutoDiffCostFunction<StereoEstimate<Camera>> dynamicProjectionCF;

            StereoEstimate<Camera> * stereoEstimate;
            stereoEstimate = new StereoEstimate<Camera>(calibDataVec[i].projection, grid,
                                                camera.params, *(calibDataVec[i].extrinsic));
            dynamicProjectionCF * costFunction = new dynamicProjectionCF(stereoEstimate);
            costFunction->AddParameterBlock(6);
            costFunction->SetNumResiduals(2 * Nx * Ny);
            problem.AddResidualBlock(costFunction, new CauchyLoss(1), extrinsic.data());   
            
            //run the solver
            
        }
        Solver::Options options;
        Solver::Summary summary;
        Solve(options, &problem, &summary);
    }
    
    bool estimateExtrinsic(Camera<double> & cam1, Camera<double> & cam2,
            array<double, 6> & extrinsic)
    {
        estimateInitialGrid(cam1, monoCalibDataVec1);
        estimateInitialGrid(cam2, monoCalibDataVec2);
        estimateInitialGrid(cam1, stereoCalibDataVec1);
        estimateInitialExtrinsic(cam2, extrinsic, stereoCalibDataVec2);
    }
    
    bool compute(Camera<double> & cam1, Camera<double> & cam2, Transformation<double> & transfo)
    {
        
        vector<double> intrinsic1 = cam1.params; 
        vector<double> intrinsic2 = cam2.params; 
        array<double, 6> extrinsic = transfo.toArray();
        
        
        cout << "### Extrinsic parameters calibration ###" << endl;
        if (monoCalibDataVec1.size() == 0)
        {
            cout << "ERROR : none of images were accepted" << endl;
            return false;
        } 
           
        if (monoCalibDataVec2.size() == 0)
        {
            cout << "ERROR : none of images were accepted" << endl;
            return false;
        } 
        
        if (stereoCalibDataVec1.size() == 0)
        {
            cout << "ERROR : none of images were accepted" << endl;
            return false;
        } 
        
        cout << "Initial board poses estimation " << endl;
        estimateExtrinsic(cam1, cam2, extrinsic);
        
        cout << "Global optimization" << endl;
        Problem problem;
        // Intrinsic init
        initIntrinsicProblem(problem, intrinsic1, monoCalibDataVec1);
        initIntrinsicProblem(problem, intrinsic2, monoCalibDataVec2);
            
        // Extrinsic init
        initIntrinsicProblem(problem, intrinsic1, stereoCalibDataVec1);
        initStereoProblem(problem, intrinsic2, extrinsic, stereoCalibDataVec2);
            
        Solver::Options options;
        options.max_num_iterations = 500;
        
        options.function_tolerance = 1e-15;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-10;
//        options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;
        Solve(options, &problem, &summary);
//        cout << summary.BriefReport() << endl;
        
        cam1.setParameters(intrinsic1.data());
        cam2.setParameters(intrinsic2.data());
        
        transfo = Transformation<double>(extrinsic.data());
    } 
    
    bool residualAnalysis(const Camera<double> & cam1, const Camera<double> & cam2,
            const Transformation<double> & transfo)
    {
        residualAnalysis(cam1, monoCalibDataVec1);
        residualAnalysis(cam2, monoCalibDataVec2);
        residualAnalysis(cam1, stereoCalibDataVec1);
        residualAnalysis(cam2, stereoCalibDataVec2, transfo);
    }
};

#endif
