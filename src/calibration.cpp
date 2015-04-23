#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>

#include "mei.h"
#include "calibration.h"

using namespace std;
using namespace cv;

using ceres::DynamicNumericDiffCostFunction;
using ceres::CENTRAL;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

double logistic(double x)
{
    double ex = exp(x);
    return ex/(1 + ex);
}

bool BoardProjection::operator()(const double * const * parameters, double * residual) const 
{
    const double * const calib = parameters[0];
    MeiCamera cam(100, 100, logistic(calib[0]), exp(calib[1]), 
            calib[2], calib[3], calib[4], calib[5]);
    const double * const xi = parameters[1];
    Transformation T(xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]);
    vector<Vector3d> transformedPoints;
    T.transform(_orig, transformedPoints);
    vector<Vector2d> projectedPoints;
    cam.projectPointCloud(transformedPoints, projectedPoints);
    for (unsigned int i = 0; i < projectedPoints.size(); i++)
    {
        Vector2d diff = _proj[i] - projectedPoints[i];
        residual[2*i] = diff[0];
        residual[2*i + 1] = diff[1];
        if (std::isinf(projectedPoints[i][0]) or std::isnan(projectedPoints[i][0]))
        {
            cout << _proj[i].transpose() << " " << projectedPoints[i].transpose() << endl;
            cout << transformedPoints[i].transpose() << endl;
            cout << T << endl;
        }
    }
    return true;
}

bool StereoBoardProjection::operator()(const double * const * parameters, double * residual) const 
{
    const double * const calib = parameters[0];
    MeiCamera cam(100, 100, logistic(calib[0]), exp(calib[1]), 
            calib[2], calib[3], calib[4], calib[5]);
    const double * const xi = parameters[1];
    const double * const eta = parameters[2];
    Transformation T0b(xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]);
    Transformation T01(eta[0], eta[1], eta[2], eta[3], eta[4], eta[5]);
    Transformation T1b = T01.inverseCompose(T0b);
    vector<Vector3d> transformedPoints;
    T1b.transform(_orig, transformedPoints);
    
    vector<Vector2d> projectedPoints;
    cam.projectPointCloud(transformedPoints, projectedPoints);
    for (unsigned int i = 0; i < projectedPoints.size(); i++)
    {
        Vector2d diff = _proj[i] - projectedPoints[i];
        residual[2*i] = diff[0];
        residual[2*i + 1] = diff[1];
        if (std::isinf(projectedPoints[i][0]) or std::isnan(projectedPoints[i][0]))
        {
            cout << _proj[i].transpose() << " " << projectedPoints[i].transpose() << endl;
            cout << transformedPoints[i].transpose() << endl;
            cout << T01 << endl;
        }
    }
    return true;
}
  
IntrinsicCalibrationData::IntrinsicCalibrationData(const string & infoFileName)
{
    // open the file and read the data
    ifstream calibInfoFile(infoFileName);
    if (not calibInfoFile.is_open())
    {
        cout << infoFileName << " : ERROR, file is not found" << endl;
        return;
    }
    calibInfoFile >> Nx >> Ny >> sqSize >> outlierThresh;
    calibInfoFile.ignore();  // To get to the next line
    
    vector<Vector3d> original;
    generateOriginal(Nx, Ny, sqSize, original);
    
    string imageFolder;
    string imageName;    
    getline(calibInfoFile, imageFolder);
    while (getline(calibInfoFile, imageName))
    {
//        cout << imageName << endl;
        vector<Vector2d> pointVec;
        bool isExtracted;
        isExtracted = extractGridProjection(imageFolder + imageName, Nx, Ny, pointVec);
        if (isExtracted) 
        {
            BoardProjection * boardProj = new BoardProjection(pointVec, original);
            residualVec.push_back(boardProj);
            array<double, 6> xi = {-0.1, -0.1, 0.2, 0.1, 0.1, 0.1};
            extrinsicVec.push_back(xi);
            fileNameVec.push_back(imageFolder + imageName);
        }
    }
}

void IntrinsicCalibrationData::residualAnalysis(const array<double, 6> & intrinsic)
{
    vector<double> residual(2 * Nx * Ny);
    double Ex = 0, Ey = 0;
    double Emax = 0;
    double const * params[2];
    params[0] = intrinsic.data();
    for (unsigned int resIdx = 0; resIdx < residualVec.size(); resIdx++)
    {
            params[1] = extrinsicVec[resIdx].data();
            BoardProjection * boardProj = residualVec[resIdx];
            (*boardProj)(params, residual.data());
            Mat frame = imread(fileNameVec[resIdx], 0);
            bool outlierDetected = false;
            for (unsigned int i = 0; i < Nx * Ny; i++)
            {
                Vector2d p = boardProj->_proj[i];
                Vector2d dp(residual[2 * i], residual[2 * i + 1]);
                p -= dp;
                circle(frame, Point(p(0), p(1)), 4.5, 255, 2);
                double dx = residual[2 * i] * residual[2 * i];
                double dy = residual[2 * i + 1] * residual[2 * i + 1];
                if (outlierThresh != 0 and dx + dy > outlierThresh * outlierThresh)
                {
                    outlierDetected = true;
                    cout << fileNameVec[resIdx] << " # " << i << endl;
                    cout << residual[2 * i] << " " << residual[2 * i + 1] << endl;
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
    Ex /= residualVec.size() * Nx * Ny;
    Ey /= residualVec.size() * Nx * Ny;
    Ex = sqrt(Ex);
    Ey = sqrt(Ey);
    Emax = sqrt(Emax);
    cout << "Ex = " << Ex << "; Ey = " << Ey << "; Emax = " << Emax << endl;  
}

ExtrinsicCalibrationData::ExtrinsicCalibrationData(const string & infoFileName)
{
    // open the file and read the data
    ifstream calibInfoFile(infoFileName);
    if (not calibInfoFile.is_open())
    {
        cout << infoFileName << " : ERROR, file is not found" << endl;
        return;
    }
    calibInfoFile >> Nx >> Ny >> sqSize >> outlierThresh;
    calibInfoFile.ignore();  // To get to the next line
    
    vector<Vector3d> original;
    generateOriginal(Nx, Ny, sqSize, original);
    
    string imageFolder;
    string imageName;    
    string leftPref, rightPref;
    getline(calibInfoFile, imageFolder);
    getline(calibInfoFile, leftPref);
    getline(calibInfoFile, rightPref);
    while (getline(calibInfoFile, imageName))
    {
//        cout << imageName << endl;
        vector<Vector2d> point1Vec, point2Vec;
        bool isExtracted1, isExtracted2;
        isExtracted1 = extractGridProjection(imageFolder + leftPref + imageName,
                                             Nx, Ny, point1Vec);
        isExtracted2 = extractGridProjection(imageFolder + rightPref + imageName, 
                                             Nx, Ny, point2Vec);
                                             
        if (isExtracted1 and isExtracted2) 
        {
            BoardProjection * boardProj1 = new BoardProjection(point1Vec, original);
            StereoBoardProjection * boardProj2 = new StereoBoardProjection(point2Vec, original);
            residual1Vec.push_back(boardProj1);
            residual2Vec.push_back(boardProj2);
            array<double, 6> xi = {-0.1, -0.1, 0.5, 0.1, 0.1, 0.1};
            extrinsicVec.push_back(xi);
            fileNameVec.push_back(imageFolder + leftPref + imageName);
        }
    }
}

bool extractGridProjection(const string & fileName, int Nx, int Ny, vector<Vector2d> & pointVec)
{
    Size patternSize(Nx, Ny);
    Mat frame = imread(fileName, 0);

    vector<Point2f> centers;
    bool patternIsFound = findChessboardCorners(frame, patternSize, centers);
    if (not patternIsFound)
    {
        cout << fileName << " : ERROR, pattern is not found" << endl;
        return false;
    }
    
//    drawChessboardCorners(frame, patternSize, Mat(centers), patternIsFound);
//    imshow("corners", frame);
//    char key = waitKey();
//    if (key == 'n' or key == 'N')
//    {
//        cout << fileName << " : ERROR, pattern is not accepted" << endl;
//        return false; 
//    }   
    pointVec.resize(Nx * Ny);
    for (unsigned int i = 0; i < Nx * Ny; i++)
    {
        pointVec[i] = Vector2d(centers[i].x, centers[i].y);
    }
    return true;
}  // extractGridProjection

void generateOriginal(int Nx, int Ny, double sqSize, vector<Vector3d> & pointVec)
{
    pointVec.resize(Nx * Ny);
    for (unsigned int i = 0; i < Nx * Ny; i++)
    {
        pointVec[i] = Vector3d(sqSize * (i % Nx), sqSize * (i / Nx), 0);
    }
}

void extrinsicStereoCalibration(const string & infoFileName1, const string & infoFileName2,
         const string & infoFileNameStereo, array<double, 6> & intrinsic1, 
         array<double, 6> & intrinsic2, array<double, 6> & extrinsic)
{
    cout << "### Extrinsic parameters calibration ###" << endl;
    IntrinsicCalibrationData calibInfo1(infoFileName1); 
    if (calibInfo1.residualVec.size() == 0)
    {
        cout << "ERROR : none of images were accepted" << endl;
        return;
    } 
       
    IntrinsicCalibrationData calibInfo2(infoFileName2);
    if (calibInfo2.residualVec.size() == 0)
    {
        cout << "ERROR : none of images were accepted" << endl;
        return;
    } 
    ExtrinsicCalibrationData extCalibInfo(infoFileNameStereo);
    if (extCalibInfo.residual1Vec.size() == 0)
    {
        cout << "ERROR : none of images were accepted" << endl;
        return;
    } 
    Problem problem;
    typedef DynamicNumericDiffCostFunction<BoardProjection> projectionCF;
    typedef DynamicNumericDiffCostFunction<StereoBoardProjection> stereoProjectionCF;
    //TODO put to a function
    for (unsigned int i = 0; i < calibInfo1.residualVec.size(); i++)
    {
        projectionCF * costFunction = new projectionCF(calibInfo1.residualVec[i]);
        costFunction->AddParameterBlock(6);
        costFunction->AddParameterBlock(6);
        costFunction->SetNumResiduals(2 * calibInfo1.Nx * calibInfo1.Ny);
        auto & xi = calibInfo1.extrinsicVec[i];
        problem.AddResidualBlock(costFunction, NULL, intrinsic1.data(), xi.data());   
    }
    for (unsigned int i = 0; i < calibInfo2.residualVec.size(); i++)
    {
        projectionCF * costFunction = new projectionCF(calibInfo2.residualVec[i]);
        costFunction->AddParameterBlock(6);
        costFunction->AddParameterBlock(6);
        costFunction->SetNumResiduals(2 * calibInfo2.Nx * calibInfo2.Ny);
        auto & xi = calibInfo2.extrinsicVec[i];
        problem.AddResidualBlock(costFunction, NULL, intrinsic2.data(), xi.data());   
    }
    for (unsigned int i = 0; i < extCalibInfo.residual1Vec.size(); i++)
    {
        //the left camera
        projectionCF * costFunction1;
        costFunction1 = new projectionCF(extCalibInfo.residual1Vec[i]);
        costFunction1->AddParameterBlock(6);
        costFunction1->AddParameterBlock(6);
        costFunction1->SetNumResiduals(2 * extCalibInfo.Nx * extCalibInfo.Ny);
        auto & xi = extCalibInfo.extrinsicVec[i];
        problem.AddResidualBlock(costFunction1, NULL, intrinsic1.data(), xi.data());   
        
        //the right camera
        stereoProjectionCF * costFunction2;
        costFunction2 = new stereoProjectionCF(extCalibInfo.residual2Vec[i]);
        costFunction2->AddParameterBlock(6);
        costFunction2->AddParameterBlock(6);
        costFunction2->AddParameterBlock(6);
        costFunction2->SetNumResiduals(2 * extCalibInfo.Nx * extCalibInfo.Ny);
        problem.AddResidualBlock(costFunction2, NULL, intrinsic2.data(),
                xi.data(), extrinsic.data());
        
    }
    
    Solver::Options options;
    options.max_num_iterations = 500;
    options.numeric_derivative_relative_step_size = 1e-4;
//    options.linear_solver_type = ceres::DENSE_QR;
    
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;
//    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    
    calibInfo1.residualAnalysis(intrinsic1);
    calibInfo2.residualAnalysis(intrinsic2);
    
//    intrinsic1[0] = logistic(intrinsic1[0]);
//    intrinsic1[1] = exp(intrinsic1[1]);
//    
//    intrinsic2[0] = logistic(intrinsic2[0]);
//    intrinsic2[1] = exp(intrinsic2[1]);
}


void intrinsicCalibration(const string & infoFileName, array<double, 6> & intrinsic)
{
    
    cout << "### Intrinsic parameters calibration ###" << endl;
    IntrinsicCalibrationData calibInfo(infoFileName);
    
    if (calibInfo.residualVec.size() == 0)
    {
        cout << "ERROR : none of images were accepted" << endl;
        return;
    } 
    
    // Problem initialization
    Problem problem;
    typedef DynamicNumericDiffCostFunction<BoardProjection> dynamicProjectionCF;
    for (unsigned int i = 0; i < calibInfo.residualVec.size(); i++)
    {
        dynamicProjectionCF * costFunction = new dynamicProjectionCF(calibInfo.residualVec[i]);
        costFunction->AddParameterBlock(6);
        costFunction->AddParameterBlock(6);
        costFunction->SetNumResiduals(2 * calibInfo.Nx * calibInfo.Ny);
        auto & xi = calibInfo.extrinsicVec[i];
        problem.AddResidualBlock(costFunction, NULL, intrinsic.data(), xi.data());   
    }
    
    //run the solver
    Solver::Options options;
    options.max_num_iterations = 500;
    options.numeric_derivative_relative_step_size = 1e-4;
    //options.linear_solver_type = ceres::DENSE_QR;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;
//    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
//    cout << summary.BriefReport() << endl;
    
    calibInfo.residualAnalysis(intrinsic);
    
//    intrinsic[0] = logistic(intrinsic[0]);
//    intrinsic[1] = exp(intrinsic[1]);
    
//    cout << "Parameters : " <<  intrinsic[0] << " " << intrinsic[1] << " " << 
//                intrinsic[2] << " " << intrinsic[3] << " " <<
//                intrinsic[4] << " " << intrinsic[5] << endl;
}  // intrinsicCalibration


