//STL
#include <vector>
#include <unordered_map>
//Eigen
#include <Eigen/Eigen>

//Ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "geometry.h"
#include "vision.h"
#include "cartography.h"

using namespace Eigen;
using namespace ceres;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

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
    Matrix3d Rbo;
    Vector3d Pbo;    
    Transformation Tbo(args[1][0], args[1][1], args[1][2],
            args[2][0], args[2][1], args[2][2]);
            
    Tbo.toRotTransInv(Rbo, Pbo);
    
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
        
        
        auto uCash = InteractionCash.find(Tbo.rot());
        if (uCash == InteractionCash.end())
        {
            double theta = Tbo.rot().norm();
            if ( theta != 0)
            {
                Vector3d u = Tbo.rot() / theta;
                Matrix3d uhat = hat(u);
                LxiInv = Matrix3d::Identity() + 
                    theta/2*sinc(theta/2)*uhat + 
                    (1 - sinc(theta))*uhat*uhat;
            }
            else
            {
                LxiInv = Matrix3d::Identity();
            }
            
            InteractionCash[Tbo.rot()] = LxiInv;
            countCalls++;
        }
        else
        {
           LxiInv = uCash->second; 
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
//    ReprojectionErrorFixed * f = new ReprojectionErrorFixed(u, v, pose, camPose, cam);
//    CostFunction * costFunc = 
//        new NumericDiffCostFunction<ReprojectionErrorFixed, CENTRAL, 2, 3>(f);
    CostFunction * costFunc = new ReprojectionErrorFixed(u, v, pose, camPose, cam);
    problem.AddResidualBlock(costFunc, NULL, X.data());
}

void MapInitializer::addObservation(Vector3d & X, double u, double v, Transformation & pose,
        const Camera * const cam, const Transformation & camPose)
{
//    ReprojectionErrorStereo * f = new ReprojectionErrorStereo(u, v, camPose, cam);
//    CostFunction * costFunc = 
//        new NumericDiffCostFunction<ReprojectionErrorStereo, CENTRAL, 2, 3, 6>(f);
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
    
    Matrix3d R;
    Vector3d t;
    trajectory[poseIdx].toRotTransInv(R, t);

    Matrix3d R1inv = stereo.pose1.rotMat().transpose();
    Matrix3d R2inv = stereo.pose2.rotMat().transpose();
    for (unsigned int i = 0; i < src.size(); i++)
    {
        const Vector3d & X = src[i];
        Vector3d Xt = R*X + t;
        stereo.cam1->projectPoint(R1inv*(Xt - stereo.pose1.trans()), dst1[i]);
        stereo.cam2->projectPoint(R2inv*(Xt - stereo.pose2.trans()), dst2[i]);
    }
}

void StereoCartography::refineTrajectory()
{
    //Precompute interaction matrices
    vector<Matrix3d> R, L;
    vector<Vector3d> P;
    for (auto & xi : trajectory)
    {
        Matrix3d Ri, LxiInv;
        Vector3d Pi;
        xi.toRotTransInv(Ri, Pi);
        R.push_back(Ri);
        P.push_back(Pi);
        
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
        L.push_back(Ri*LxiInv);
    }
    
    Matrix3d Rcb1, Rcb2;
    Vector3d Pcb1, Pcb2;    
    stereo.pose1.toRotTransInv(Rcb1, Pcb1);
    stereo.pose2.toRotTransInv(Rcb2, Pcb2);
    double ERR = 0;
    vector< Matrix6d > JTJ(trajectory.size(), Matrix6d::Zero());
    vector< Vector6d > JTerr(trajectory.size(), Vector6d::Zero());
    //Accumulate Jacobians
    for (auto & landmark : LM)
    {
        for (auto & observation : landmark.observations)
        {
            int idx = observation.poseIdx;
           /* if (idx == 0)
            {
                continue;
            }*/
            const Matrix3d & Rbo = R[idx];
            const Vector3d & Pbo = P[idx];
            Matrix3d Rcb;
            Vector3d Pcb;
            Camera * camera;
            if (observation.cameraId == LEFT)
            {
                Rcb = Rcb1;
                Pcb = Pcb1;
                camera = stereo.cam1;
            } 
            else
            {
                Rcb = Rcb2;
                Pcb = Pcb2;
                camera = stereo.cam2;
            }
            Vector3d X = Rbo*landmark.X + Pbo;
            X = Rcb*X + Pcb;
            Vector2d err;
            camera->projectPoint(X, err);
            err = Vector2d(observation.u, observation.v) - err;
            ERR += err.norm();
            Eigen::Matrix<double, 2, 3> J;
            camera->projectionJacobian(X, J);
            
            Eigen::Matrix<double, 2, 6> J2;
            J2 << -J * Rcb * Rbo, J * hat(X) * Rcb * Rbo * L[idx];
            JTJ[idx] += J2.transpose()*J2;
            JTerr[idx] += J2.transpose()*err;           
        }
    }   
    
//    cout << "TRAJ ERROR : " << ERR << endl;
    //Perform the improvement
    for (unsigned int i = 0; i < trajectory.size(); i++)
    {
        Vector6d res = JTJ[i].inverse() * JTerr[i];
        trajectory[i].trans() += res.head<3>();
        trajectory[i].rot() += res.tail<3>();
    }
} //  refineTrajectory


void StereoCartography::refineCloud(int maxStep)
{
    //To be improved
    vector<Matrix3d> R;
    vector<Vector3d> P;
    for (auto & xi : trajectory)
    {
        Matrix3d Ri;
        Vector3d Pi;
        xi.toRotTransInv(Ri, Pi);
        R.push_back(Ri);
        P.push_back(Pi);
    }
    
    Matrix3d Rcb1, Rcb2;
    Vector3d Pcb1, Pcb2;    
    stereo.pose1.toRotTransInv(Rcb1, Pcb1);
    stereo.pose2.toRotTransInv(Rcb2, Pcb2);
    double ERR = 0;
    for (auto & landmark : LM)
    {
        Matrix3d JTJ = Matrix3d::Zero();
        Vector3d JTerr = Vector3d::Zero();
        for (auto & observation : landmark.observations)
        {
            int idx = observation.poseIdx;
            if (idx > maxStep)
            {
                continue;
            }
            const Matrix3d & Rbo = R[idx];
            const Vector3d & Pbo = P[idx];
            Matrix3d Rcb;
            Vector3d Pcb;
            
            Camera * camera;
            if (observation.cameraId == LEFT)
            {
                Rcb = Rcb1;
                Pcb = Pcb1;
                camera = stereo.cam1;
            } 
            else
            {
                Rcb = Rcb2;
                Pcb = Pcb2;
                camera = stereo.cam2;
            }
            Vector3d X = Rbo*landmark.X + Pbo;
            X = Rcb*X + Pcb;
            Vector2d err;
            camera->projectPoint(X, err);
            err = Vector2d(observation.u, observation.v) - err;
            ERR += err.norm();
            Eigen::Matrix<double, 2, 3> J;
            camera->projectionJacobian(X, J);
            
            J = J * Rcb * Rbo;
            
            JTJ += J.transpose()*J;
            JTerr += J.transpose()*err;           
        }
        landmark.X += JTJ.inverse()*JTerr;
    }   
//    cout << "ERROR : " << ERR << endl;
} //  refineCloud


void StereoCartography::improveTheMap()
{   
    /*Vector3d a = Vector3d::Random(); 
    Vector3d b = Vector3d::Random(); 
    Vector3d c = Vector3d::Random(); 
    Vector3d d = Vector3d::Random(); 
    Matrix3d M;
    InteractionCash[a] = M;
    InteractionCash[b] = M;
    InteractionCash[c] = M;
    InteractionCash[d] = M;
    cout << "Set test" << endl;
    cout << (InteractionCash.find(a) != InteractionCash.end()) << endl;
    cout << (InteractionCash.find(b) != InteractionCash.end()) << endl;
    cout << (InteractionCash.find(c) != InteractionCash.end()) << endl;
    cout << (InteractionCash.find(d) != InteractionCash.end()) << endl;*/
    //BUNDLE ADJUSTMENT
   /* MapInitializer initializer;
    for (auto & landmark : LM)
    {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
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
        
        // Make Ceres automatically detect the bundle structure. Note that the
        // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
        // for standard bundle adjustment problems.
    }
    initializer.compute();
    */
     
    for (int i = 0; i < 50; i++)
    {       
        cout << i << endl;
        refineCloud(min(i, 6));                             
        refineTrajectory();
    }
    Transformation T0 = trajectory[0];
    for (auto & T : trajectory)
    {
        T = T0.inverseCompose(T);
    }
}








