
#include <iostream>
#include <ctime>
#include <cmath>
#include <stdlib.h>
#include <random>

#include <ceres/rotation.h>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "cartography.h"
#include "geometry.h"
#include "vision.h"
#include "mei.h"
#include "tests/cartography_tests.h"

#define EPS 1e-6

using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

class Pinhole : public Camera<double>
{
public:

    Pinhole(double u0, double v0, double f)
    : Camera(2*u0, 2*v0, 3) 
    {
        params[0] = u0;
        params[1] = v0;
        params[2] = f;
    }
    virtual ~Pinhole() {}

    virtual bool reconstructPoint(const Vector2d & src, Vector3d & dst) const
    {
        const double & u0 = params[0];
        const double & v0 = params[1];
        const double & f = params[2];
        const double & u = src(0);
        const double & v = src(1);
        dst << (u - u0)/f, (v - v0)/f, 1;
        return true;
    }

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Vector3d & src, Vector2d & dst) const
    {
        const double & u0 = params[0];
        const double & v0 = params[1];
        const double & f = params[2];
        const double & x = src(0);
        const double & y = src(1);
        const double & z = src(2);
        if (z < 1e-2)
        {
            dst << -1, -1;
            return false;
        }
        dst << x * f / z + u0, y * f / z + v0;
        return true;
    }

    //TODO implement the projection and distortion Jacobian
    virtual bool projectionJacobian(const Vector3d & src, Eigen::Matrix<double, 2, 3> & Jac) const
    {
        const double & u0 = params[0];
        const double & v0 = params[1];
        const double & f = params[2];
        const double & x = src(0);
        const double & y = src(1);
        const double & z = src(2);
        double zz = z * z;
        Jac(0, 0) = f/z;
        Jac(0, 1) = 0;
        Jac(0, 2) = -x * f/ zz;
        Jac(1, 0)= 0;
        Jac(1, 1) = f/z;
        Jac(1, 2) = -y * f/ zz;
    }
};

void assertEqual(const Eigen::Vector2d & v1, const Eigen::Vector2d & v2)
{
    auto dv = v1 - v2;
    assert(dv.maxCoeff() < EPS and dv.minCoeff() > -EPS);
}


void assertEqual(const Eigen::Vector3d & v1, const Eigen::Vector3d & v2)
{
    auto dv = v1 - v2;
    assert(dv.maxCoeff() < EPS and dv.minCoeff() > -EPS);
}

void assertEqual(const Eigen::Matrix3d & m1, const Eigen::Matrix3d & m2)
{
    auto dm = m1 - m2;
    assert(dm.maxCoeff() < EPS and dm.minCoeff() > -EPS);
}

void assertEqual(const vector<Vector3d> cloud1, const vector<Vector3d> cloud2)
{
    assert(cloud1.size() == cloud2.size());
    int maxNum = cloud1.size();
    for (unsigned int i = 0; i < maxNum; i++)
    {
        auto delta = cloud1[i] -cloud2[i];
        assert(delta.norm() < EPS);
    }
}

void compare(const vector<Vector3d> cloud1, const vector<Vector3d> cloud2)
{
    assert(cloud1.size() == cloud2.size());
    int maxNum = cloud1.size();
    double errMax = 0;
    double errMean = 0;
    for (unsigned int i = 0; i < maxNum; i++)
    {
        Vector3d delta = cloud1[i] -cloud2[i];
        errMax = max(delta.norm(), errMax);
        errMean += delta.norm() * delta.norm();
    }
    cout << "standard deviation : " << std::sqrt(errMean/maxNum) << endl;
    cout << "max error : " << errMax << endl;

}

void testGeometry()
{
    Transformation<double> p1(1, 1, 1, 0.2, 0.3, 1);
    Transformation<double> p2(1, 0, -1, 0, -1, 0.7);
    
    auto p3 = p1.compose(p2);
    
    assertEqual(p3.rotMat(), p1.rotMat() * p2.rotMat());
    
    auto p4 = p1.inverseCompose(p2);
    assertEqual(p4.rotMat(), p1.rotMat().transpose() * p2.rotMat());
    
    Vector3d v(1, 2, 3.3);

    assertEqual(p1.rotMat()*v, p1.rotQuat().rotate(v));
    assertEqual(p2.rotMat()*v, p2.rotQuat().rotate(v));
    assertEqual(p3.rotMat()*v, p3.rotQuat().rotate(v));
}

void testVision()
{
    double params[6]{0.3, 0.2, 375, 375, 650, 470};
    MeiCamera cam1mei(params);   
    MeiCamera cam2mei(params);
    
    const Quaternion<double> qR(-0.0166921, 0.0961855, -0.0121137, 0.99515);
    const Vector3d tR(0.78, 0, 0);  // (x, y, z) OL-OR expressed in CR reference frame?
    Transformation<double> T1, T2(tR, qR);
    StereoSystem stereo(T1, T2, cam1mei, cam2mei);
    
    int maxNum = 15;
    
    vector<Vector2d> proj1, proj2;
    vector<Vector3d> cloud1, cloud2;
    
    for (unsigned int i = 0; i < maxNum; i++)
    {
        cloud1.push_back(Vector3d(10*sin(i),
                        10*std::cos(i*1.7),
                        15.2+5*std::sin(i/3.14)));
    }
    
    stereo.projectPointCloud(cloud1, proj1, proj2);
    stereo.reconstructPointCloud(proj1, proj2, cloud2);
    assertEqual(cloud1, cloud2);
}

void testMei()
{
    double params[6]{0.3, 0.2, 375, 375, 650, 470};
    MeiCamera cam1mei(params);
    
    for (int i = -3; i < 3; i++)
    {
        for (int j = -3; j < 3; j++)
        {
            Vector3d X1(i, j, 3);
            Matrix<double, 2, 3> J;
            Vector2d p1, p2;
            cam1mei.projectPoint(X1, p1);
            cam1mei.projectionJacobian(X1, J);
            for (int k = 0; k < 3; k++)
            {
               Vector3d dX = Vector3d::Zero();
               dX(k) = 100*EPS; 
               cam1mei.projectPoint(X1 + dX, p2);
               Vector2d dp = (p2 - p1);
               Vector2d Jdx = J*dX;
               assertEqual(dp, Jdx);
               dX(k) = 0; 
            }
        }
    }

}

void testBundleAdjustment()
{
    double params[6]{0.3, 0.2, 375, 375, 650, 470};
    MeiCamera cam1mei(params);
    MeiCamera cam2mei(params);
    const Quaternion<double> qR(-0.0166921, 0.0961855, -0.0121137, 0.99515);
    const Vector3d tR(0.78, 0, 0);  // (x, y, z) OL-OR expressed in CR reference frame?
    Transformation<double> T1, T2(tR, qR);
    StereoCartography cartograph(T1, T2, cam1mei, cam2mei);
    
    int maxNum = 250;
    
    vector<Vector2d> proj1, proj2;
    vector<Vector3d> cloud1, cloud2;
    
    cartograph.LM.resize(maxNum);
    for (unsigned int i = 0; i < maxNum; i++)
    {
        cloud1.push_back(Vector3d(10*sin(i),
                        10*std::cos(i*1.7),
                        15.2+5*std::sin(i/3.14)));
        cartograph.LM[i].X = cloud1[i];
    }
    
    cartograph.trajectory.push_back(Transformation<double>(0, 0, 0, 0, 0, 0));
    cartograph.trajectory.push_back(Transformation<double>(0, 0, 1, 0, 0.2, 0));
    cartograph.trajectory.push_back(Transformation<double>(0.1, 0, 2, 0, 0.3, 0));
    cartograph.trajectory.push_back(Transformation<double>(0.2, 0, 2.5, 0.1, 0.3, 0));
    cartograph.trajectory.push_back(Transformation<double>(0.3, 0, 2.7, 0.15, 0.3, 0));
    
    for (unsigned int j = 0; j < cartograph.trajectory.size(); j++)
    {
        cartograph.projectPointCloud(cloud1, proj1, proj2, j);
        for (unsigned int i = 0; i < maxNum; i++)
        {
            Observation obs1(proj1[i][0], proj1[i][1], j, LEFT);
            Observation obs2(proj2[i][0], proj2[i][1], j, RIGHT);
            cartograph.LM[i].observations.push_back(obs1);
            cartograph.LM[i].observations.push_back(obs2);
        }
    }
    
    for (unsigned int i = 0; i < maxNum; i++)
    {
        cartograph.LM[i].X += Vector3d::Random();
    }
    
    for (unsigned int j = 1; j < cartograph.trajectory.size(); j++)
    {
        cartograph.trajectory[j].rot() += Vector3d::Random()*0.1*j;
        cartograph.trajectory[j].trans() += Vector3d::Random()*0.2*j;
    }
    
    cartograph.improveTheMap();
    for (auto & lm : cartograph.LM)
    {
        cloud2.push_back(lm.X);
    }
    assertEqual(cloud1, cloud2);
}

void testOdometry()
{
    double params[6]{0.3, 0.2, 375, 375, 650, 470};
    MeiCamera cam1mei(params);   MeiCamera cam2mei(params);
    
    const Quaternion<double> qR(-0.0166921, 0.0961855, -0.0121137, 0.99515);
    const Vector3d tR(0.78, 0, 0);  // (x, y, z) OL-OR expressed in CR reference frame?
    Transformation<double> T1, T2(tR, qR);
    StereoCartography cartograph(T1, T2, cam1mei, cam2mei);
    
    int maxNum = 250;
    
    vector<Vector2d> proj1, proj2;
    vector<Vector3d> cloud;
    
    cartograph.LM.resize(maxNum);
    
    cartograph.trajectory.push_back(Transformation<double>(0, 0, 0, 0, 0, 0));
    cartograph.trajectory.push_back(Transformation<double>(0.1, 0.2, 0.5, 0.1, 0.1, 0.1));
       
     
    for (unsigned int i = 0; i < maxNum; i++)
    {
        cloud.push_back(Vector3d(10*sin(i),
                        10*std::cos(i*1.7),
                        15.2+5*std::sin(i/3.14)));
    }
    cartograph.projectPointCloud(cloud, proj1, proj2, 1); 
    
    for (unsigned int i = 0; i < maxNum; i += 3)
    {
        proj1[i] = Vector2d::Random()*100;
        proj1[i][0] += 100;
        proj1[i][1] += 100;
    }
    vector<bool> inlierMask(maxNum);
    
    Transformation<double> xi;
    cartograph.odometryRansac(proj1, cloud, inlierMask, xi);
    cartograph.computeTransformation(proj1, cloud, inlierMask, xi);
    assertEqual(xi.rot(), cartograph.trajectory[1].rot());
    assertEqual(xi.trans(), cartograph.trajectory[1].trans());
}

void testCartography()
{
    cout << "### Geometry tests ### " << flush;
    testGeometry();
    cout << "OK" << endl;
    
    cout << "### Mei tests ### " << flush;
    testMei();
    cout << "OK" << endl;
    
    cout << "### Stereo tests ### " << flush;
    testVision();
    cout << "OK" << endl;
    
    cout << "### Odometry tests ### " << flush;
    testOdometry();
    cout << "OK" << endl;
    
    cout << "### Bundle Adjustment tests ### " << flush;
    testBundleAdjustment();
    cout << "OK" << endl;
}

/* EPIPOLAR TESTS
TODO: TO BE DONE IN THE FUTURE

    /// RECOMPUTE THE CALIBRATION
    
    vector<Vector3d> xVec1, xVec2;
    cam1mei.reconstructPointCloud(pVec1, xVec1);
    cam2mei.reconstructPointCloud(pVec2, xVec2);
    
    Matrix3d E;
    computeEssentialMatrix(xVec1, xVec2, E);
    
    Matrix3d Eold = hat(T2.trans()) * T2.rotMat();
    Eold = Eold / Eold(2, 2);
    cout << "old E : " << endl;
    cout <<  Eold << endl;
    
    cout << xVec1[18].transpose() * E * xVec2[18] << endl;
    cout << xVec1[18].transpose() * Eold * xVec2[18] << endl;
    cout << "#######" << endl;
    
    JacobiSVD<Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    cout << svd.singularValues() << endl;
    
    Matrix3d R90;
    Matrix3d Rm90;
    R90 << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    Rm90 << 0, 1, 0, -1, 0, 0, 0, 0, 1;
    Matrix3d R = svd.matrixU() * Rm90 * svd.matrixV().transpose();
    cout << "new rotation : " << endl;
    cout << R << endl;
    R = -svd.matrixU() * R90 * svd.matrixV().transpose();
    Quaternion<double> newQ(R);
    cout << newQ << endl;
    cout << R << endl;
    cout << "original rotation : " << endl;
    cout << T2.rotMat() << endl;
    cout << "left sing vecs : " << endl;
    cout << svd.matrixU() << endl;    
    ///


*/




