#include "tests/cartography_tests.h"
#include <opencv2/opencv.hpp>


#include "ceres/ceres.h"
#include "glog/logging.h"

#include "mei.h"

using namespace cv;

using ceres::NumericDiffCostFunction;
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

struct BoardProjection {
    
    BoardProjection(const vector<Vector2d> & proj, const vector<Vector3d> & orig)
    : _proj(proj), _orig(orig) {}
    
    
    bool operator()(const double* const calib,
                  const double* const xi,
                  double* residual) const {
        MeiCamera cam(100, 100, logistic(calib[0]), exp(calib[1]), 
                calib[2], calib[3], calib[4], calib[5]);
        Transformation T(xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]);
        vector<Vector3d> transformedPoints;
        T.transform(_orig, transformedPoints);
        vector<Vector2d> projectedPoints;
        cam.projectPointCloud(transformedPoints, projectedPoints);
        for (unsigned int i = 0; i < projectedPoints.size(); i++)
        {
//            cout << _proj[i].transpose() << " # "  << projectedPoints[i].transpose() << endl;
            Vector2d diff = _proj[i] - projectedPoints[i];
            residual[2*i] = diff[0];
            residual[2*i + 1] = diff[1];
        }
        return true;
//        return false;
    }
    
    double _size;
    int _Nx, _Ny;
    vector<Vector2d> _proj;
    vector<Vector3d> _orig;
};

void addResidual(const string & fileName, Problem & problem, 
        double * params, double * xi)
{
    const int Nx = 8, Ny = 5;
    const double sqSize = 0.1;
    Size patternSize(Nx, Ny);
    Mat frame = imread(fileName, 0);

    vector<Point2f> centers;
    bool patternfound = findChessboardCorners(frame, patternSize, centers);
    drawChessboardCorners(frame, patternSize, Mat(centers), patternfound);
    imshow("corners", frame);
    waitKey();
    vector<Vector2d> proj;
    vector<Vector3d> orig;
    for (unsigned int i = 0; i < Nx*Ny; i++)
    {
        proj.push_back(Vector2d(centers[i].x, centers[i].y));
        orig.push_back(Vector3d(0.1 * (i % Nx), 0.1 * (i / Nx), 0));
    }
    

    // Set up the only cost function (also known as residual). This uses
    // numeric differentiation to obtain the derivative (jacobian).
    CostFunction* cost_function =
    new NumericDiffCostFunction<BoardProjection, CENTRAL, 2*Nx*Ny, 6, 6> (
        new BoardProjection(proj, orig)
    );
    
    problem.AddResidualBlock(cost_function, NULL, params, xi);
} 

int main(int argc, char** argv) {

    testCartography();
    
    //TRY CALIBRATION FUNCTIONS
    
    double params[6]{0, 0, 1000, 1000, 650, 450}, 
    xi1[6]{-0.5, -0.5, 1, 0, 0, 0},
    xi2[6]{-0.5, -0.5, 1, 0, 0, 0},
    xi3[6]{-0.5, -0.5, 1, 0, 0, 0},
    xi4[6]{-0.5, -0.5, 1, 0, 0, 0},
    xi5[6]{-0.5, -0.5, 1, 0, 0, 0},
    xi6[6]{-0.5, -0.5, 1, 0, 0, 0},
    xi7[6]{-0.5, -0.5, 1, 0, 0, 0};
        
    google::InitGoogleLogging(argv[0]);


    Problem problem;
    
    addResidual("/home/bogdan/projects/icars/calib_dataset/left_1426238811.pgm",
            problem, params, xi1);
    addResidual("/home/bogdan/projects/icars/calib_dataset/left_1426238815.pgm",
            problem, params, xi2);
    addResidual("/home/bogdan/projects/icars/calib_dataset/left_1426238818.pgm",
            problem, params, xi3);
    addResidual("/home/bogdan/projects/icars/calib_dataset/left_1426238820.pgm",
            problem, params, xi4);
    addResidual("/home/bogdan/projects/icars/calib_dataset/left_1426238823.pgm",
            problem, params, xi5);
    addResidual("/home/bogdan/projects/icars/calib_dataset/left_1426238826.pgm",
            problem, params, xi6);
    
    Solver::Options options;
    options.max_num_iterations = 50000;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;
    //options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;
    for (unsigned int i = 0; i < 6; i++)
    {
        cout << params[i] << " ";
    }
    cout << endl;
    for (unsigned int i = 0; i < 6; i++)
    {
        cout << xi1[i] << " ";
    }   
    cout << endl;
    for (unsigned int i = 0; i < 6; i++)
    {
        cout << xi2[i] << " ";
    }
    return 0;

//    
//    imshow("corners", frame);
//    waitKey();
}


