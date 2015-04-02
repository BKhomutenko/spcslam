#include <random>

//STL
#include <vector>
//Eigen
#include <Eigen/Eigen>

//Ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "geometry.h"
#include "vision.h"
#include "cartography.h"

using namespace ceres;
using Eigen::Matrix;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

struct compareVectors
{
    bool operator()(const Vector3d & a, const Vector3d & b)
    {
        if (a[0] < b[0])
            return true;
        else if (a[0] == b[0] and a[1] < b[1])
            return true;
        else if (a[0] == b[0] and a[1] == b[1] and a[2] < b[2])
            return true;
        else
            return false;
    }
};

map<Vector3d, Matrix3d, compareVectors> InteractionCash;

int countCalls = 0;

inline double sinc(const double x)
{
    if (x==0)
        return 1;
    return std::sin(x)/x;
}

inline Matrix3d hat(const Vector3d & u)
{
    Matrix3d M;
    M << 0, -u(2), u(1),   u(2), 0, -u(0),   -u(1), u(0), 0;
    return M;
}

ReprojectionErrorStereo::ReprojectionErrorStereo(double u, double v,
        const Transformation & camPose,
        const Camera * const camera) 
        : u(u), v(v), camera(camera) 
{
    camPose.toRotTransInv(Rcb, Pcb);
}

ReprojectionErrorFixed::ReprojectionErrorFixed(double u, double v, const Transformation & xi,
        const Transformation & camPose, const Camera * const camera) 
        : u(u), v(v), camera(camera) 
{
    xi.toRotTransInv(Rbo, Pbo);
    camPose.toRotTransInv(Rcb, Pcb);
}

bool ReprojectionErrorFixed::Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const
{
    double newPoint[3];
    double rotationVec[3];
    const double * const landmark = args[0];
    Vector3d X(landmark);

    X = Rcb*(Rbo*X + Pbo) + Pcb;
    Vector2d point;
    camera->projectPoint(X, point);
    residuals[0] = point[0] - u;
    residuals[1] = point[1] - v;
    
    if (jac)
    {
        
        Eigen::Matrix<double, 2, 3> J;
        camera->projectionJacobian(X, J);
        
        // dp / dX
        Eigen::Matrix<double, 2, 3, RowMajor> dpdX = J * Rcb * Rbo;
        copy(dpdX.data(), dpdX.data() + 6, jac[0]);
   
    }
    
    return true;
}


//TODO unify the transformation system
bool ReprojectionErrorStereo::Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const
{
    Vector3d rot = Vector3d(args[2]);
    Matrix3d Rbo = rotationMatrix(-rot);
    Vector3d Pbo = -Rbo*Vector3d(args[1]);    
    Vector3d X(args[0]);
    
    X = Rcb * (Rbo * X + Pbo) + Pcb;
    Vector2d point;
    camera->projectPoint(X, point);
    residuals[0] = point[0] - u;
    residuals[1] = point[1] - v;

    if (jac)
    {
        
        Eigen::Matrix<double, 2, 3> J;
        camera->projectionJacobian(X, J);
        
        Matrix3d Rco = Rcb * Rbo;
        
        // dp / dX
        Eigen::Matrix<double, 2, 3, RowMajor> dpdX = J * Rco;
        copy(dpdX.data(), dpdX.data() + 6, jac[0]);
        
        // dp / dxi
        Eigen::Matrix<double, 3, 3> LxiInv;

        double theta = rot.norm();
        if ( theta != 0)
        {
            Matrix3d uhat = hat(rot / theta);
            LxiInv = Matrix3d::Identity() + 
                theta/2*sinc(theta/2)*uhat + 
                (1 - sinc(theta))*uhat*uhat;
        }
        else
        {
            LxiInv = Matrix3d::Identity();
        }

        Eigen::Matrix<double, 2, 3, RowMajor>  dpdxi2; // = (Eigen::Matrix<double, 2, 3, RowMajor> *) jac[2];
        dpdX *= -1;
        dpdxi2 = J*hat(X)*Rco*LxiInv;
        copy(dpdX.data(), dpdX.data() + 6, jac[1]);
        copy(dpdxi2.data(), dpdxi2.data() + 6, jac[2]);
    }

    return true;
}

void MapInitializer::addFixedObservation(Vector3d & X, double u, double v, Transformation & pose,
        const Camera * const cam, const Transformation & camPose)
{
    CostFunction * costFunc = new ReprojectionErrorFixed(u, v, pose, camPose, cam);
    problem.AddResidualBlock(costFunc, NULL, X.data());
}

void MapInitializer::addObservation(Vector3d & X, double u, double v, Transformation & pose,
        const Camera * const cam, const Transformation & camPose)
{
    CostFunction * costFunc = new ReprojectionErrorStereo(u, v, camPose, cam);
    problem.AddResidualBlock(costFunc, NULL, X.data(), pose.transData(), pose.rotData());
}

void MapInitializer::compute()
{
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.function_tolerance = 1e-3;
//    options.gradient_tolerance = 1e-4;
//    options.parameter_tolerance = 1e-4;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    cout << summary.FullReport();
}

void StereoCartography::projectPointCloud(const vector<Vector3d> & src,
        vector<Vector2d> & dst1, vector<Vector2d> & dst2, int poseIdx) const
{
    assert(src.size() == dst1.size());
    assert(src.size() == dst2.size());
    vector<Vector3d> Xb(src.size());
    trajectory[poseIdx].inverseTransform(src, Xb);
    stereo.projectPointCloud(Xb, dst1, dst2);
}

void StereoCartography::improveTheMap()
{   
    //BUNDLE ADJUSTMENT
    MapInitializer initializer;
    for (auto & landmark : LM)
    {
        for (auto & observation : landmark.observations)
        {
            int xiIdx = observation.poseIdx;
            if (xiIdx == 0)
            {
                if (observation.cameraId == LEFT)
                {
                    initializer.addFixedObservation(landmark.X, observation.u, observation.v,
                            trajectory[xiIdx], stereo.cam1, stereo.pose1);
                }
                else
                {
                    initializer.addFixedObservation(landmark.X, observation.u, observation.v,
                            trajectory[xiIdx], stereo.cam2, stereo.pose2);
                }
            }
            else if (observation.cameraId == LEFT)
            {
                initializer.addObservation(landmark.X, observation.u, observation.v,
                        trajectory[xiIdx], stereo.cam1, stereo.pose1);
            }
            else
            {
                initializer.addObservation(landmark.X, observation.u, observation.v,
                        trajectory[xiIdx], stereo.cam2, stereo.pose2);
            }
        }
    }
    initializer.compute();

}

Transformation StereoCartography::computeTransformation(
        const vector<Vector2d> & observationVec,
        const vector<Vector3d> & cloud, 
        const vector<bool> & inlierMask, 
        Transformation xi)
{
    assert(observationVec.size() == cloud.size());
    assert(observationVec.size() == inlierMask.size());
    Matrix3d Rcb;
    Vector3d Pcb;
    stereo.pose1.toRotTransInv(Rcb, Pcb);
    Camera * camera = stereo.cam1;
    // TODO put the iteration criterion
    Matrix3d Rbo, LxiInv;
    Vector3d Pbo;
    xi.toRotTransInv(Rbo, Pbo);
    Vector3d u = xi.rot();
    double theta = u.norm();
    if ( theta != 0)
    {
        u /= theta;
        Matrix3d uhat = hat(u);
        LxiInv = Matrix3d::Identity() + 
            theta/2*sinc(theta/2)*uhat + 
            (1 - sinc(theta))*uhat*uhat;
    }
    else
    {
        LxiInv = Matrix3d::Identity();
    }
    Eigen::Matrix<double, 6, 6> JTJ = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> JTerr = Eigen::Matrix<double, 6, 1>::Zero();
    double ERR = 0;
    for (unsigned int i = 0; i < observationVec.size(); i++)
    {
        if (not inlierMask[i])
        {
            continue;
        }

        Vector3d X = Rbo*cloud[i] + Pbo;
        X = Rcb*X + Pcb;
        Vector2d err;
        camera->projectPoint(X, err);
        err = observationVec[i] - err;
        ERR += err.norm();
        Matrix3d Rco = Rcb * Rbo;
        Eigen::Matrix<double, 2, 3> Jx;
        camera->projectionJacobian(X, Jx);
        
        Eigen::Matrix<double, 2, 6> Jcam;
        Jcam << -Jx * Rco, Jx * hat(X) * Rco * LxiInv;
        JTJ += Jcam.transpose()*Jcam;
        JTerr += Jcam.transpose()*err;           
    }
    auto dxi = JTJ.inverse()*JTerr;
    xi.trans() += dxi.head<3>();
    xi.rot() += dxi.tail<3>();
}
        
void StereoCartography::odometryRansac(
        const vector<Vector2d> & observationVec,
        const vector<Vector3d> & cloud,
        vector<bool> & inlierMask,
        Transformation & xi)
{
    assert(observationVec.size() == cloud.size());
    int numPoints = observationVec.size();
    
    inlierMask.resize(numPoints);
    
    int numIterMax = 25;
    const Transformation initialPose = xi;
    Transformation pose;
    int bestInliers = 0;
    for (unsigned int iteration = 0; iteration < numIterMax; iteration++)
    {
        int maxIdx = observationVec.size();
        //choose three points at random
		int idx1m = rand() % maxIdx;
		int idx2m = 0;
		int idx3m = 0;
		do 
		{
			idx2m = rand() % maxIdx;
		} while (idx2m == idx1m);
		
		do 
		{
			idx3m = rand() % maxIdx;
		} while (idx3m == idx1m or idx3m == idx2m);
        
        pose = initialPose;        
        //compute xi using these points
        for (unsigned int i = 0; i < 50; i++)
        {
            double theta = pose.rot().norm();
            Matrix3d uhat = hat(pose.rot());
            Matrix3d LxiInv = Matrix3d::Identity() + 
                theta/2*sinc(theta/2)*uhat + 
                (1 - sinc(theta))*uhat*uhat;
                
            Matrix3d Rbo, Rcb;
            Vector3d Pbo, Pcb;
            
            pose.toRotTransInv(Rbo, Pbo);
            stereo.pose1.toRotTransInv(Rcb, Pcb);
            
            Vector3d X1, X2, X3;            
            Vector2d Err1, Err2, Err3;
            Eigen::Matrix<double, 2, 3> J1, J2, J3;
            
            X1 = Rcb*(Rbo*cloud[idx1m] + Pbo) + Pcb;
            stereo.cam1->projectPoint(X1, Err1);
            stereo.cam1->projectionJacobian(X1, J1);
            
            X2 = Rcb*(Rbo*cloud[idx2m] + Pbo) + Pcb;
            stereo.cam1->projectPoint(X2, Err2);
            stereo.cam1->projectionJacobian(X2, J2);
            
            X3 = Rcb*(Rbo*cloud[idx3m] + Pbo) + Pcb;
            stereo.cam1->projectPoint(X3, Err3);            
            stereo.cam1->projectionJacobian(X3, J3);
            
            Eigen::Matrix<double, 6, 6> J;
            Matrix3d Rco = Rcb * Rbo;
            
            J << -J1 * Rco, J1 * hat(X1) * Rco * LxiInv,
                -J2 * Rco, J2 * hat(X2) * Rco * LxiInv,
                -J3 * Rco, J3 * hat(X3) * Rco * LxiInv;
            Eigen::Matrix<double, 6, 1> E, dxi;
            E << observationVec[idx1m] - Err1,
                 observationVec[idx2m] - Err2,
                 observationVec[idx3m] - Err3;
            dxi = J.inverse() * E;
            
            pose.trans() += dxi.head<3>();
            pose.rot() += dxi.tail<3>();
            
            if (iteration < 3)
            {
                cout << J << endl;
                cout << J.inverse() << endl << endl;
            }
        }
        
        //count inliers
        vector<Vector3d> XbaseVec(numPoints);
        pose.inverseTransform(cloud, XbaseVec);
        vector<Vector2d> p1Vec(numPoints), p2Vec(numPoints);
        stereo.projectPointCloud(XbaseVec, p1Vec, p2Vec);
        vector<bool> currentInlierMask(numPoints, false);
        
        int countInliers = 0;
        for (unsigned int i = 0; i < numPoints; i++)
        {   
//            cout << observationVec[i] << " " <<  p1Vec[i]<< endl;
            Vector2d err = observationVec[i] - p1Vec[i];
            if (err.norm() < 0.1)
            {
                currentInlierMask[i] = true;
                countInliers++;
            }
        }
        //keep the best hypothesis
        if (countInliers > bestInliers)
        {
            cout << "improvement "  << countInliers << " " <<  bestInliers << endl;
            //TODO copy in a better way
            inlierMask = currentInlierMask;
            bestInliers = countInliers;
            xi = pose;
        }        
    }
}

/*
Transformation StereoCartography::estimateOdometry(const vector<Feature> & observationVec)
{
    //Matching ###
    
        //create a vector of 3D points and corresponding descriptors (about 300)
    
        //perform mathing 
        
    //RANSAC
    
    //Final transformation computation
}
*/





