/*
Abstract camera definition.
Stereo vision definition.
*/

//Standard libraries
#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
//Eigen
#include <Eigen/Eigen>

//Ceres solver
#include <ceres/ceres.h>


#include "geometry.h"
#include "vision.h"

using namespace std;
using namespace Eigen;
using namespace ceres;
  
ReprojectionError::ReprojectionError(double u, double v, const Transformation & xi,
        const Camera * const camera) : u(u), v(v), camera(camera) 
{
    xi.toRotTransInv(rot, trans);
}

int callCounter = 0;

bool ReprojectionError::Evaluate(double const* const* landmark,
                                   double* residuals,
                                   double** jac) const
{
    Vector3d X(*landmark);
    X = rot*X + trans;
    
    Vector2d point;
    camera->projectPoint(X, point);

    residuals[0] = point[0] - u;
    residuals[1] = point[1] - v;
    
    if (jac)
    {
        Eigen::Matrix<double, 2, 3> J;
        camera->projectionJacobian(X, J);
        J = J * rot;
        double * jacPtr = jac[0];
        for (unsigned int i = 0; i < 2; i++)
        {
            for (unsigned int j = 0; j < 3; j++)
            {
                *jacPtr = J(i, j);
                jacPtr++;
            }
        }
    }
    
    callCounter++;
    return true;
}
  
  
void Reconstructor::addObservation(double u, double v, const Transformation & pose,
        const Camera * const cam)
{
//    CostFunction * costFunc = new NumericDiffCostFunction<ReprojectionError, CENTRAL, 2, 3>(
//        new ReprojectionError(u, v, pose, cam)
//    );

    CostFunction * costFunc = new ReprojectionError(u, v, pose, cam);
    problem.AddResidualBlock(costFunc, NULL, point.data());
}

void Reconstructor::compute()
{
    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.function_tolerance = 1e-1;
    options.gradient_tolerance = 1e-1;
    options.parameter_tolerance = 1e-1;
    options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    
}

void StereoSystem::projectPointCloud(const vector<Vector3d> & src,
        vector<Vector2d> & dst1, vector<Vector2d> & dst2) const
{
    assert(src.size() == dst1.size());
    assert(src.size() == dst2.size());
    Matrix3d R1inv = R1.transpose();
    Matrix3d R2inv = R2.transpose();
    for (unsigned int i = 0; i < src.size(); i++)
    {
        const Vector3d & X = src[i];
        cam1->projectPoint(R1inv*(X - t1), dst1[i]);
        cam2->projectPoint(R2inv*(X - t2), dst2[i]);
    }
}
 
void StereoSystem::initializeGeometrciParameter()
{
    pose1.toRotTrans(R1, t1);
    pose2.toRotTrans(R2, t2);
    t = t2 - t1;   
}


bool StereoSystem::triangulate(const Vector2d & p1, const Vector2d & p2, Vector3d & X) const
{
    Vector3d v1, v2;
    cam1->reconstructPoint(p1, v1);
    cam1->reconstructPoint(p2, v2);
    v1 = R1*v1;
    v2 = R2*v2;
    v1 /= v1.norm();
    v2 /= v2.norm();
    //cout << v1 << endl << v2 << endl;
    
    double v1v2 = v1.dot(v2);
    if (v1v2 > 1 - 1e-4) // TODO the constant to be revised
    {
        X << -1, -1, -1;
        return false;
    }
    double tv1 = t.dot(v1);
    double tv2 = t.dot(v2);
    double l1 = (tv1 - tv2 * v1v2)/(1 - v1v2*v1v2);
    double l2 = (-tv2 + tv1 * v1v2)/(1 - v1v2*v1v2);
    X = (v1*l1 + t + v2*l2)*0.5 + t1;
    return true;
}

void StereoSystem::reconstruct2(const Vector2d & p1, const Vector2d & p2, Vector3d & X) const
{
    X << 0, 0, 10;
    Matrix3d R1inv = R1.transpose();
    Matrix3d R2inv = R2.transpose();
    for (unsigned int i = 0; i < 5; i++)
    {
        Matrix3d JTJ = Matrix3d::Zero();
        Vector3d JTerr(0, 0, 0);
        Vector2d err(0, 0);
        
        Vector2d p1proj, p2proj;
        cam1->projectPoint(R1inv*(X-t1), p1proj);
        cam2->projectPoint(R2inv*(X-t2), p2proj);
        
        err = p1 - p1proj;
        Eigen::Matrix<double, 2, 3> J;
        cam1->projectionJacobian(R1inv*(X-t1), J);
        J = J*R1inv;
        JTJ += J.transpose()*J;
        JTerr += J.transpose()*err;   
        
        err = p2 - p2proj;
        cam2->projectionJacobian(R2inv*(X-t2), J);
        J = J*R2inv;
        JTJ += J.transpose()*J;
        JTerr += J.transpose()*err; 
        
        X += JTJ.inverse()*JTerr;
    }
}

     
void StereoSystem::reconstructPointCloud(const vector<Vector2d> & src1,
        const vector<Vector2d> & src2, vector<Vector3d> & dst) const
{
    assert(src1.size() == dst.size());
    assert(src2.size() == dst.size());
    for (unsigned int i = 0; i < src1.size(); i++)
    {
       
        triangulate(src1[i], src2[i], dst[i]);
//        dst[i] << 0, 0, 10;
//        Reconstructor reconstructor(dst[i]);
//        const Vector2d & p1 = src1[i];
//        const Vector2d & p2 = src2[i];
//        reconstructor.addObservation(p1(0), p1(1), pose1, cam1);
//        reconstructor.addObservation(p2(0), p2(1), pose2, cam2);
//        reconstructor.compute();
        
    }
}

StereoSystem::~StereoSystem()
{
    if (cam1)
        delete cam1;
    if (cam2)
        delete cam2;
}

////TODO make smart constructor with calibration data passed

//StereoSystem::StereoSystem();

