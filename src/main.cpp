#include <iostream>
#include <ctime>
#include <cmath>
#include <stdlib.h>
#include <random>

#include <ceres/rotation.h>

#include "cartography.h"
#include "geometry.h"
#include "vision.h"

#define S 200
#define SIZE Size(S, S)

using namespace std;
extern int countCalls;

class Pinhole : public Camera
{
public:
    double u0, v0, f;
    
    Pinhole(double u0, double v0, double f)
    : u0(u0), v0(v0), f(f) {}
    virtual ~Pinhole() {}
    
    virtual bool reconstructPoint(const Vector2d & src, Vector3d & dst) const
    {
        const double & u = src(0);
        const double & v = src(1);        
        dst << (u - u0)/f, (v - v0)/f, 1;
        return true;
    }
    
    /// projects 3D points onto the original image
    virtual bool projectPoint(const Vector3d & src, Vector2d & dst) const 
    {
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

void GeometryTest()
{
    Transformation p1(1, 1, 1, 0.2, 0.3, 1);
    Transformation p2(1, 0, -1, 0, -1, 0.7);
    
    cout << p1.rotMat() << endl;
    cout << p2.rotMat() << endl;
    
    Transformation p3 = p1.compose(p2);
    cout << p3.rotMat()  - p1.rotMat() * p2.rotMat() << endl;
    p3 = p1.inverseCompose(p2);
    cout << endl << p3.rotMat()  - p1.rotMat().transpose() * p2.rotMat() << endl;
    
    Vector3d v(1, 2, 3.3);
    
    cout << p1.rotMat()*v - p1.rotQuat().rotate(v) << endl;
    cout << p2.rotMat()*v - p2.rotQuat().rotate(v) << endl;
    cout << p3.rotMat()*v - p3.rotQuat().rotate(v) << endl;
}

void compare(const vector<Vector3d> cloud1, const vector<Vector3d> cloud2)
{
    assert(cloud1.size() == cloud2.size());
    int maxNum = cloud1.size();
    double errMax = 0;
    double errMean = 0;
    for (unsigned int i = 0; i < maxNum; i++)
    {
//        cout << cloud[i].transpose() << " " <<  cartograph.LM[i].X.transpose() << endl;
        Vector3d delta = cloud1[i] -cloud2[i];
        errMax = max(delta.norm(), errMax);
        errMean += delta.norm() * delta.norm();
    }
    cout << "standard deviation : " << std::sqrt(errMean/maxNum) << endl;
    cout << "max error : " << errMax << endl;
    
} 

int main(int argc, char** argv) {
     google::InitGoogleLogging(argv[0]);
    //GeometryTest();
    Pinhole * cam1 = new Pinhole(100, 100, 100);
    Pinhole * cam2 = new Pinhole(100, 100, 100);
    Transformation p1(0, 0, 0, 0, 0, 0);
    Transformation p2(1, 0, 0, 0, 0, 0);; //p2(1, -0.2, -0.31, 0, 0.12, -0.12);

    StereoCartography cartograph(p1, p2, cam1, cam2);

    vector<Vector2d> proj1, proj2;
    vector<Vector3d> cloud, cloud2;
    int maxNum = 250000;

    cartograph.trajectory.push_back(Transformation(0, 0, 0, 0, 0, 0));
    cartograph.trajectory.push_back(Transformation(0, 0, 1, 0, 0.2, 0));
    cartograph.trajectory.push_back(Transformation(0, 0, 2, 0, 0.3, 0));
    cartograph.trajectory.push_back(Transformation(0, 0, 2.5, 0.1, 0.3, 0));
    cartograph.trajectory.push_back(Transformation(0, 0, 2.7, 0.15, 0.3, 0));
    
    vector<Transformation> refTraj = cartograph.trajectory;
    cartograph.LM.resize(maxNum);
//    cloud.push_back(Vector3d(1, 0, 10));
//    cloud.push_back(Vector3d(0, 1, 11));
//    cloud.push_back(Vector3d(0, 2, 12));
//    cloud.push_back(Vector3d(1, 0, 13));
//    cloud.push_back(Vector3d(2, 0, 14));
    for (unsigned int i = 0; i < maxNum; i++)
    {
        cloud.push_back(Vector3d(10*sin(i),
                        10*std::cos(i*1.7),
                        15.2+5*std::sin(i/3.14)));
        
        cartograph.LM[i].X = cloud[i];
    }
    

    
	proj1.resize(maxNum);
	proj2.resize(maxNum);
    double sigma = 0.1;
    std::normal_distribution<double> noise(0, sigma);
    std::default_random_engine re;
    noise(re);

    for (unsigned int j = 0; j < cartograph.trajectory.size(); j++)
    {
        cartograph.projectPointCloud(cloud, proj1, proj2, j);
        
        
        for (unsigned int i = 0; i < maxNum; i++)
        {
            Observation obs1(proj1[i][0] + noise(re), proj1[i][1] + noise(re), j, LEFT);
            Observation obs2(proj2[i][0] + noise(re), proj2[i][1] + noise(re), j, RIGHT);
            cartograph.LM[i].observations.push_back(obs1);
            cartograph.LM[i].observations.push_back(obs2);
        }  
    }	

    for (unsigned int i = 0; i < maxNum; i++)
    {
        cartograph.LM[i].X += Vector3d::Random(); 
    }

    cartograph.trajectory[1] = Transformation(0.1, -0.1, 1.1, 0.11, 0.22, -0.1);
    cartograph.trajectory[2] = Transformation(-0.1, 0.1, 2.08, -0.1, 0.27, 0.1);

//    Matrix3d R;
//    Vector3d t;
//    Transformation pose(1, 0, 0, 0, 3.141596/2, 0);
//    pose.toRotTrans(R, t);
//    cout << "##### test rot #####" << endl;
//    cout << R*Vector3d(2, 0, 0) + t << endl;
//    cout << R*Vector3d(0, 2, 0) + t << endl;
//    cout << R*Vector3d(0, 0, 2) + t << endl;
//    cout << R*Vector3d(2, 2, 0) + t << endl;
//    
//    ceres::AngleAxisRotatePoint(pose.data + 3, Vector3d(2, 0, 0).data(), t.data());
//    cout << t << endl;

    clock_t beginTime, entTime;

    beginTime = clock();
    
    cartograph.stereo.projectPointCloud(cloud, proj1, proj2);

    entTime = clock();
    cout << "Projection time : " << double(entTime - beginTime) / CLOCKS_PER_SEC << endl;


    beginTime = clock();
    
    cloud2.resize(proj1.size());
    cartograph.stereo.reconstructPointCloud(proj1, proj2, cloud2);

    entTime = clock();
    cout << "Reconstruction time : " << double(entTime - beginTime) / CLOCKS_PER_SEC << endl;

    compare(cloud, cloud2);
    
/*****************************************/


    beginTime = clock();
    
    cartograph.improveTheMap();
//    cartograph.projectPointCloud(cloud, proj1, proj2);

    entTime = clock();
    cout << "BA time : " << double(entTime - beginTime) / CLOCKS_PER_SEC << endl;
    
    vector<Vector3d> cloud3;
    for (auto & lm : cartograph.LM)
    {
        cloud3.push_back(lm.X);
    }
    
    compare(cloud, cloud3);
    
    for (unsigned int j = 0; j < cartograph.trajectory.size(); j++)
    {
        cout << cartograph.trajectory[j].inverseCompose(refTraj[j]) << endl;
//        cout << "### " << endl;
    }
    cout << countCalls << endl;
    
    

}


