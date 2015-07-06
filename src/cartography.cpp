#include <random>

//STL
#include <vector>
#include <algorithm>
//Eigen
#include <Eigen/Eigen>

//Ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "geometry.h"
#include "matcher.h"
#include "vision.h"
#include "cartography.h"

using namespace ceres;
using namespace std;
using Eigen::Matrix;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

int countCalls = 0; //FIXME

bool MotionModel::Evaluate(double const* const* args,
                double* residuals,
                double** jac) const
{
    Transformation<double> xi0(args[0]), xi1(args[1]);
    Transformation<double> delta = xi0.inverseCompose(xi1);
    Vector3d v = delta.trans();
    Vector3d w = delta.rot();
    Vector3d vz(0, 0, v[2]);
    v -= vz;
    Vector3d err = w.cross(vz) - 2*v;
    for (unsigned int i = 0; i < 3; i++)
    {
        residuals[i] = err[i];
    }
    residuals[3] = w[2];
    
    return true;
}

PriorPosition::PriorPosition(const Transformation<double> & xi, const Matrix6d & cov) 
{
    JacobiSVD<Matrix6d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
    Vector6d lambda = svd.singularValues();
    Matrix6d U = svd.matrixU(), V = svd.matrixV();
    
    for (unsigned int j = 0; j < 3; j++)
    {
        lambda[j] = 1./std::sqrt(lambda[j]);
    }
    
    M = U * lambda.asDiagonal() * V.transpose();
    xiPrior << xi.trans(), xi.rot();
}

bool PriorPosition::Evaluate(double const* const* args,
                double* residuals,
                double** jac) const
{
    Vector6d X(args[0]);
    Vector6d err = M*(X - xiPrior);
    if (jac)
    {
        copy(M.data(), M.data() + 36, jac[0]);
    }
    return true;
}


PriorLandmark::PriorLandmark(const Vector3d & Xprior, const Matrix3d & cov) 
: Xprior(Xprior)
{
    JacobiSVD<Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
    Vector3d lambda = svd.singularValues();
    Matrix3d U = svd.matrixU(), V = svd.matrixV();
    
    for (unsigned int j = 0; j < 3; j++)
    {
        lambda[j] = 1./std::sqrt(lambda[j]);
    }
    
    M = U * lambda.asDiagonal() * V.transpose();
}

bool PriorLandmark::Evaluate(double const* const* args,
                double* residuals,
                double** jac) const
{
    Vector3d X(args[0]);
    Vector3d err = M*(X - Xprior);
    if (jac)
    {
        copy(M.data(), M.data() + 9, jac[0]);
    }
    return true;
}


OdometryError::OdometryError(const Vector3d X, const Vector2d pt,
        const Transformation<double> & TbaseCam,
        const ICamera & camera)
        : X(X), u(pt[0]), v(pt[1]), camera(&camera)
{
    TbaseCam.toRotTransInv(RcamBase, PcamBase);
}

ReprojectionErrorStereo::ReprojectionErrorStereo(const Vector2d pt,
        const Transformation<double> & TbaseCam,
        const ICamera * camera)
        : u(pt[0]), v(pt[1]), camera(camera)
{
    TbaseCam.toRotTransInv(RcamBase, PcamBase);
}

ReprojectionErrorFixed::ReprojectionErrorFixed(const Vector2d pt,
        const Transformation<double> & TorigBase,
        const Transformation<double> & TbaseCam, const ICamera * camera)
        : u(pt[0]), v(pt[1]), camera(camera)
{
    TorigBase.toRotTransInv(RbaseOrig, PbaseOrig);
    TbaseCam.toRotTransInv(RcamBase, PcamBase);
}

bool ReprojectionErrorFixed::Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const
{
    double newPoint[3];
    double rotationVec[3];
    const double * const landmark = args[0];
    Vector3d X(landmark);

    X = RcamBase*(RbaseOrig*X + PbaseOrig) + PcamBase;
    Vector2d point;
    camera->projectPoint(X, point);
    residuals[0] = point[0] - u;
    residuals[1] = point[1] - v;

    if (jac)
    {

        Eigen::Matrix<double, 2, 3> J;
        camera->projectionJacobian(X, J);

        // dp / dX
        Eigen::Matrix<double, 2, 3, RowMajor> dpdX = J * RcamBase * RbaseOrig;
        copy(dpdX.data(), dpdX.data() + 6, jac[0]);

    }

    return true;
}


//TODO unify the transformation system
bool OdometryError::Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const
{
    Vector3d rotOrigBase(args[1]);
    Matrix3d RbaseOrig = rotationMatrix<double>(-rotOrigBase);
    Vector3d PorigBase(args[0]);

    Vector3d Xtr = RcamBase * (RbaseOrig * (X - PorigBase)) + PcamBase;
    Vector2d point;
    camera->projectPoint(Xtr, point);
    residuals[0] = point[0] - u;
    residuals[1] = point[1] - v;

    if (jac)
    {

        Eigen::Matrix<double, 2, 3> J;
        camera->projectionJacobian(Xtr, J);

        Matrix3d RcamOrig = RcamBase * RbaseOrig;

        // dp / dxi
        Matrix3d LxiInv = interOmegaRot(rotOrigBase);

        Eigen::Matrix<double, 2, 3, RowMajor> dpdxi1 = -J * RcamOrig;
        Eigen::Matrix<double, 2, 3, RowMajor>  dpdxi2; // = (Eigen::Matrix<double, 2, 3, RowMajor> *) jac[2];
        dpdxi2 = J*hat(Xtr)*RcamOrig*LxiInv;
        copy(dpdxi1.data(), dpdxi1.data() + 6, jac[0]);
        copy(dpdxi2.data(), dpdxi2.data() + 6, jac[1]);
    }

    return true;
}


bool ReprojectionErrorStereo::Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const
{
    Vector3d rotOrigBase(args[2]);
    Matrix3d RbaseOrig = rotationMatrix<double>(-rotOrigBase);
    Vector3d PorigBase(args[1]);
    Vector3d X(args[0]);

    X = RcamBase * (RbaseOrig * (X - PorigBase)) + PcamBase;
    Vector2d point;
    camera->projectPoint(X, point);
    residuals[0] = point[0] - u;
    residuals[1] = point[1] - v;

    if (jac)
    {

        Eigen::Matrix<double, 2, 3> J;
        camera->projectionJacobian(X, J);

        Matrix3d RcamOrig = RcamBase * RbaseOrig;

        // dp / dX
        Eigen::Matrix<double, 2, 3, RowMajor> dpdX = J * RcamOrig;
        copy(dpdX.data(), dpdX.data() + 6, jac[0]);

        // dp / dxi
        Matrix3d LxiInv = interOmegaRot(rotOrigBase);

        Eigen::Matrix<double, 2, 3, RowMajor>  dpdxi2; // = (Eigen::Matrix<double, 2, 3, RowMajor> *) jac[2];
        dpdX *= -1;
        dpdxi2 = J*hat(X)*RcamOrig*LxiInv;
        copy(dpdX.data(), dpdX.data() + 6, jac[1]);
        copy(dpdxi2.data(), dpdxi2.data() + 6, jac[2]);
    }

    return true;
}

void MapInitializer::addFixedObservation(Vector3d & X, Vector2d pt, Transformation<double> & pose,
        const ICamera * cam, const Transformation<double> & TbaseCam)
{
    CostFunction * costFunc = new ReprojectionErrorFixed(pt, pose, TbaseCam, cam);
    problem.AddResidualBlock(costFunc, NULL, X.data());
}

void MapInitializer::addObservation(Vector3d & X, Vector2d pt, Transformation<double> & pose,
        const ICamera * cam, const Transformation<double> & TbaseCam)
{
    CostFunction * costFunc = new ReprojectionErrorStereo(pt, TbaseCam, cam);
    problem.AddResidualBlock(costFunc, NULL, X.data(), pose.transData(), pose.rotData());
}

void MapInitializer::compute()
{
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.function_tolerance = 1e-3;
//    options.gradient_tolerance = 1e-4;
//    options.parameter_tolerance = 1e-4;
//    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
//    cout << summary.FullReport() << endl;
}

void StereoCartography::projectPointCloud(const vector<Vector3d> & src,
        vector<Vector2d> & dst1, vector<Vector2d> & dst2, int poseIdx) const
{
    dst1.resize(src.size());
    dst2.resize(src.size());
    vector<Vector3d> Xb(src.size());
    trajectory[poseIdx].inverseTransform(src, Xb);
    stereo.projectPointCloud(Xb, dst1, dst2);
}

void StereoCartography::improveTheMap(bool firstBA)
{
    //BUNDLE ADJUSTMENT
    int lastFixedPos;
    if (firstBA)
    {
        lastFixedPos = 0;
    }
    else
    {
        int step = trajectory.size()-1;
        lastFixedPos = max(1, step - 4);
    }
    if (WM.size() > 10)
    {
        /*int fixedPos = trajectory.size();
        for (auto & landmark : WM)
        {
            int firstOb = landmark.observations[0].poseIdx;
            if (firstOb < fixedPos)
            {
                fixedPos = firstOb;
            }
        }*/
        MapInitializer initializer;
        for (auto & landmark : WM)
        {
            for (auto & observation : landmark.observations)
            {
                int xiIdx = observation.poseIdx;
                if (xiIdx <= lastFixedPos)
                {
                    if (observation.cameraId == LEFT)
                    {
                        initializer.addFixedObservation(landmark.X, observation.pt,
                                trajectory[xiIdx], stereo.cam1, stereo.TbaseCam1);
                    }
                    else
                    {
                        initializer.addFixedObservation(landmark.X, observation.pt,
                                trajectory[xiIdx], stereo.cam2, stereo.TbaseCam2);
                    }
                }
                else if (observation.cameraId == LEFT)
                {
                    initializer.addObservation(landmark.X, observation.pt,
                            trajectory[xiIdx], stereo.cam1, stereo.TbaseCam1);
                }
                else
                {
                    initializer.addObservation(landmark.X, observation.pt,
                            trajectory[xiIdx], stereo.cam2, stereo.TbaseCam2);
                }
            }
        }
        initializer.compute();
    }

}

void StereoCartography::improveTheMap_2()
{
    //BUNDLE ADJUSTMENT
    MapInitializer initializer;
    for (auto & landmark : WM)
    {
        for (auto & observation : landmark.observations)
        {
            int xiIdx = observation.poseIdx;
            if (xiIdx == 0)
            {
                if (observation.cameraId == LEFT)
                {
                    initializer.addFixedObservation(landmark.X, observation.pt,
                            trajectory[xiIdx], stereo.cam1, stereo.TbaseCam1);
                }
                else
                {
                    initializer.addFixedObservation(landmark.X, observation.pt,
                            trajectory[xiIdx], stereo.cam2, stereo.TbaseCam2);
                }
            }
            else if (observation.cameraId == LEFT)
            {
                initializer.addObservation(landmark.X, observation.pt,
                        trajectory[xiIdx], stereo.cam1, stereo.TbaseCam1);
            }
            else
            {
                initializer.addObservation(landmark.X, observation.pt,
                        trajectory[xiIdx], stereo.cam2, stereo.TbaseCam2);
            }
        }
    }
    initializer.compute();

}

void Odometry::computeTransformation()
{
    assert(observationVec.size() == cloud.size());
    assert(observationVec.size() == inlierMask.size());
    Problem problem;
    for (unsigned int i = 0; i < cloud.size(); i++)
    {

        if (not inlierMask[i]) continue;
        CostFunction * costFunc = new OdometryError(cloud[i],
                                        observationVec[i], TbaseCam, camera);
        problem.AddResidualBlock(costFunc, NULL,
                    TorigBase.transData(), TorigBase.rotData());
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
}

void Odometry::computeTransformation_2()
{
    assert(observationVec_2.size() == cloud.size());
    assert(observationVec_2.size() == inlierMask_2.size());
    Problem problem;
    for (unsigned int i = 0; i < cloud.size(); i++)
    {
        for (int j = 0; j < inlierMask_2[i].size(); j++)
        {
            if (not inlierMask_2[i][j]) continue;
            CostFunction * costFunc = new OdometryError(cloud[i],
                                        observationVec_2[i][j], TbaseCam, camera);
            problem.AddResidualBlock(costFunc, NULL,
                                        TorigBase.transData(), TorigBase.rotData());
        }
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
}

void Odometry::Ransac()
{
    assert(observationVec.size() == cloud.size());
    int numPoints = observationVec.size();

    inlierMask.resize(numPoints);

    const int numIterMax = 300;
    const Transformation<double> initialPose = TorigBase;
    int bestInliers = 0;
    //TODO add a termination criterion
    for (unsigned int iteration = 0; iteration < numIterMax; iteration++)
    {
        Transformation<double> pose = initialPose;
        int maxIdx = observationVec.size();
        //choose three points at random
        int idx1m = rand() % maxIdx;
        int idx2m, idx3m;
        do
        {
                idx2m = rand() % maxIdx;
        } while (idx2m == idx1m);

        do
        {
                idx3m = rand() % maxIdx;
        } while (idx3m == idx1m or idx3m == idx2m);


        //solve an optimization problem

        Problem problem;
        for (auto i : {idx1m, idx2m, idx3m})
        {
            CostFunction * costFunc = new OdometryError(cloud[i],
                                        observationVec[i], TbaseCam, camera);
            problem.AddResidualBlock(costFunc, NULL,
                        pose.transData(), pose.rotData());
        }

        Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        Solver::Summary summary;
        options.max_num_iterations = 10;
        Solve(options, &problem, &summary);

        //count inliers
        vector<Vector3d> XcamVec(numPoints);
        Transformation<double> TorigCam = pose.compose(TbaseCam);
        TorigCam.inverseTransform(cloud, XcamVec);
        vector<Vector2d> projVec(numPoints);
        camera.projectPointCloud(XcamVec, projVec);
        vector<bool> currentInlierMask(numPoints, false);

        int countInliers = 0;
        for (unsigned int i = 0; i < numPoints; i++)
        {
            Vector2d err = observationVec[i] - projVec[i];
            if (err.norm() < 2)
            {
                currentInlierMask[i] = true;
                countInliers++;
            }
        }
        //keep the best hypothesis
        if (countInliers > bestInliers)
        {
            //TODO copy in a bettegit lor way
            inlierMask = currentInlierMask;
            bestInliers = countInliers;
            TorigBase = pose;
        }
    }
}

void Odometry::Ransac_2()
{
    assert(observationVec_2.size() == cloud.size());
    int numPoints = observationVec_2.size();

    inlierMask_2.resize(numPoints);

    const int numIterMax = 500;
    const Transformation<double> initialPose = TorigBase;
    int bestInliers = 0;
    //TODO add a termination criterion
    for (unsigned int iteration = 0; iteration < numIterMax; iteration++)
    {
        Transformation<double> pose = initialPose;
        int maxIdx = observationVec_2.size();
        //choose three points at random
        int idx1m = rand() % maxIdx;
        int idx2m, idx3m;
        do
        {
                idx2m = rand() % maxIdx;
        } while (idx2m == idx1m);

        do
        {
                idx3m = rand() % maxIdx;
        } while (idx3m == idx1m or idx3m == idx2m);

        int idx1p = rand() % observationVec_2[idx1m].size();
        int idx2p = rand() % observationVec_2[idx2m].size();
        int idx3p = rand() % observationVec_2[idx3m].size();

        int index1 [3] = { idx1m, idx2m, idx3m };
        int index2 [3] = { idx1p, idx2p, idx3p };

        //solve an optimization problem

        Problem problem;
        for (int i = 0; i < 3; i++)
        {
            CostFunction * costFunc = new OdometryError(cloud[index1[i]],
                                        observationVec_2[index1[i]][index2[i]], TbaseCam, camera);
            problem.AddResidualBlock(costFunc, NULL,
                        pose.transData(), pose.rotData());
        }

        Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        Solver::Summary summary;
        options.max_num_iterations = 10;
        Solve(options, &problem, &summary);

        //count inliers
        vector<Vector3d> XcamVec(numPoints);
        Transformation<double> TorigCam = pose.compose(TbaseCam);
        TorigCam.inverseTransform(cloud, XcamVec);
        vector<Vector2d> projVec(numPoints);
        camera.projectPointCloud(XcamVec, projVec);
        vector<vector<bool> > currentInlierMask(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            vector<bool> vec(observationVec_2[i].size(), false);
            currentInlierMask[i] = vec;
        }

        int countInliers = 0;
        for (unsigned int i = 0; i < numPoints; i++)
        {
            double errNorm = 1000000;
            int best;
            for (int j = 0; j < observationVec_2[i].size(); j++)
            {
                double errNormTemp = (observationVec_2[i][j] - projVec[i]).norm();
                if (errNormTemp < errNorm)
                {
                    errNorm = errNormTemp;
                    best = j;
                }
            }
            if (errNorm < 1)
            {
                currentInlierMask[i][best] = true;
                countInliers++;
            }
        }
        //keep the best hypothesis
        if (countInliers > bestInliers)
        {
            //TODO copy in a bettegit lor way
            inlierMask_2 = currentInlierMask;
            bestInliers = countInliers;
            TorigBase = pose;
        }
    }
}

// Transformation<double> StereoCartography::estimateOdometry(const vector<Feature> & featureVec) const
// {
//     //Matching
//
//     int numLandmarks = LM.size();
//     int numActive = min(100, numLandmarks);
//     vector<Feature> lmFeatureVec;
// //    cout << "ca va" << endl;
//     for (unsigned int i = numLandmarks - numActive; i < numLandmarks; i++)
//     {
//         lmFeatureVec.push_back(Feature(Vector2d(0, 0), LM[i].d));
//     }
// //    cout << "ca va" << endl;
//     Matcher matcher;
//     vector<int> matchVec;
//     matcher.bruteForce(lmFeatureVec, featureVec, matchVec);
//
//     Odometry odometry(trajectory.back(), stereo.TbaseCam1, stereo.cam1);
// //    cout << "ca va" << endl;
//     for (unsigned int i = 0; i < numActive; i++)
//     {
//         const int match = matchVec[i];
//         if (match == -1) continue;
//         odometry.observationVec.push_back(featureVec[match].pt);
//         odometry.cloud.push_back(LM[numLandmarks  - numActive + i].X);
//     }
// //    cout << "cloud : " << odometry.cloud.size() << endl;
//     //RANSAC
//     odometry.Ransac();
// //    cout << odometry.TorigBase << endl;
//     //Final transformation computation
//     odometry.computeTransformation();
// //    cout << odometry.TorigBase << endl;
//     return odometry.TorigBase;
// }

// odometry with ransac based on fixed BF matches
Transformation<double> StereoCartography::estimateOdometry(const vector<Feature> & featureVec) const
{
    //Matching

    int nSTM = STM.size();
    int nWM = WM.size();
    int maxActive = 300;
    int numActive = 0;
    vector<int> indexVec;
    vector<Feature> lmFeatureVec;

    // add landmarks from WM
    int k = nWM;
    while (k > 0 and numActive < maxActive)
    {
        k--;
        Eigen::Vector3d Xb, Xc;
        trajectory.back().inverseTransform(WM[k].X, Xb);
        stereo.TbaseCam1.inverseTransform(Xb, Xc);
        if (Xc(2) > 0.5 and WM[k].observations.back().poseIdx == trajectory.size()-1)
        {
            lmFeatureVec.push_back(Feature(Vector2d(0, 0), WM[k].d));
            numActive++;
            indexVec.push_back(k);
        }
    }
    int nWMreprojected = numActive;

    // add landmarks from STM
    k = nSTM;
    while (k > 0 and numActive < maxActive)
    {
        k--;
        Eigen::Vector3d Xb, Xc;
        trajectory.back().inverseTransform(STM[k].X, Xb);
        stereo.TbaseCam1.inverseTransform(Xb, Xc);
        if (Xc(2) > 0.5 and STM[k].observations.back().poseIdx == trajectory.size()-1)
        {
            lmFeatureVec.push_back(Feature(Vector2d(0, 0), STM[k].d));
            numActive++;
            indexVec.push_back(k);
        }
    }

    // matching
    vector<int> matchVec;
    matcher.bruteForceOneToOne(lmFeatureVec, featureVec, matchVec);

    Odometry odometry(trajectory.back(), stereo.TbaseCam1, stereo.cam1);

    for (int i = 0; i < nWMreprojected; i++)
    {
        if (matchVec[i] != -1)
        {
            odometry.observationVec.push_back(featureVec[matchVec[i]].pt);
            odometry.cloud.push_back(WM[indexVec[i]].X);
        }
    }
    for (int i = nWMreprojected; i < numActive; i++)
    {
        if (matchVec[i] != -1)
        {
            odometry.observationVec.push_back(featureVec[matchVec[i]].pt);
            odometry.cloud.push_back(STM[indexVec[i]].X);
        }
    }

//    cout << "cloud : " << odometry.cloud.size() << endl;
    //RANSAC
    odometry.Ransac();
//    cout << odometry.TorigBase << endl;
    //Final transformation computation
    odometry.computeTransformation();
//    cout << odometry.TorigBase << endl;
    return odometry.TorigBase;
}

// odometry based on motion hypothesis and reprojection matching
Transformation<double> StereoCartography::estimateOdometry_2(const vector<Feature> & featureVec) const
{
    int nSTM = STM.size();
    int nWM = WM.size();
    int maxActive = 300;
    int numActive = 0;

    // create motion hypothesis
    Transformation<double> Tdelta;
    if (trajectory.size() > 1)
    {
        Transformation<double> tn = trajectory[trajectory.size()-2].inverseCompose(trajectory.back());
        Tdelta.setParam(tn.trans(), tn.rot());
    }
    Transformation<double> Th = trajectory.back().compose(Tdelta);

    // predict position of the landmarks based on motion hypothesis

    vector<int> indexVec;
    vector<Feature> lmFeatureVec;
//    cout << "ca va" << endl;
    int k = nWM;
    while (k > 0 and numActive < maxActive)
    {
        k--;
        if (WM[k].observations.back().poseIdx == trajectory.size()-1)
        {
            Eigen::Vector3d Xb, Xc;
            Th.inverseTransform(WM[k].X, Xb);
            stereo.TbaseCam1.inverseTransform(Xb, Xc);
            if (Xc(2) > 0.5)
            {
                Eigen::Vector2d pos;
                bool res = stereo.cam1->projectPoint(Xc, pos);
                lmFeatureVec.push_back(Feature(pos, WM[k].d));
                numActive++;
                indexVec.push_back(k);
            }
        }
    }
    int nWMreprojected = lmFeatureVec.size();

    k = nSTM;
    while (k > 0 and numActive < maxActive)
    {
        k--;
        if (STM[k].observations.back().poseIdx == trajectory.size()-1)
        {
            Eigen::Vector3d Xb, Xc;
            Th.inverseTransform(STM[k].X, Xb);
            stereo.TbaseCam1.inverseTransform(Xb, Xc);
            if (Xc(2) > 0.5)
            {
                Eigen::Vector2d pos;
                bool res = stereo.cam1->projectPoint(Xc, pos);
                lmFeatureVec.push_back(Feature(pos, STM[k].d));
                numActive++;
                indexVec.push_back(k);
            }
        }
    }

    /*cout << endl << endl << "Tdelta: " << Tdelta << endl;
    cout << "Th: " << Th << endl;
    cout << "lmFeatureVec size: " << lmFeatureVec.size() << flush;*/

    vector<int> matchVec;
    matcher.matchReprojected(lmFeatureVec, featureVec, matchVec, 20);

    Odometry odometry(trajectory.back(), stereo.TbaseCam1, stereo.cam1);
//    cout << "ca va" << endl;
    for (int i = 0; i < nWMreprojected; i++)
    {
        if (matchVec[i] != -1)
        {
            odometry.observationVec.push_back(featureVec[matchVec[i]].pt);
            odometry.cloud.push_back(WM[indexVec[i]].X);
        }
    }
    for (int i = nWMreprojected; i < numActive; i++)
    {
        if (matchVec[i] != -1)
        {
            odometry.observationVec.push_back(featureVec[matchVec[i]].pt);
            odometry.cloud.push_back(STM[indexVec[i]].X);
        }
    }
//    cout << "cloud : " << odometry.cloud.size() << endl;
    //RANSAC
    odometry.Ransac();
//    cout << odometry.TorigBase << endl;
    //Final transformation computation
    odometry.computeTransformation();
//    cout << odometry.TorigBase << endl;
    return odometry.TorigBase;
}

// odometry with ransac based on BF matches pool
Transformation<double> StereoCartography::estimateOdometry_3(const vector<Feature> & featureVec) const
{
    //Matching
    int nSTM = STM.size();
    int nWM = WM.size();
    int maxActive = 300;
    int numActive = 0;
    vector<int> indexVec;
    vector<Feature> lmFeatureVec;

    int k = nWM;
    while (k > 0 and numActive < maxActive)
    {
        k--;
        Eigen::Vector3d Xb, Xc;
        trajectory.back().inverseTransform(WM[k].X, Xb);
        stereo.TbaseCam1.inverseTransform(Xb, Xc);
        if (Xc(2) > 0.5 and WM[k].observations.back().poseIdx == trajectory.size()-1)
        {
            lmFeatureVec.push_back(Feature(Vector2d(0, 0), WM[k].d));
            numActive++;
            indexVec.push_back(k);
        }
    }
    int nWMreprojected = numActive;

    if (WM.size() < 50)
    {
        k = nSTM;
        while (k > 0 and numActive < maxActive)
        {
            k--;
            Eigen::Vector3d Xb, Xc;
            trajectory.back().inverseTransform(STM[k].X, Xb);
            stereo.TbaseCam1.inverseTransform(Xb, Xc);
            if (Xc(2) > 0.5 and STM[k].observations.back().poseIdx == trajectory.size()-1)
            {
                lmFeatureVec.push_back(Feature(Vector2d(0, 0), STM[k].d));
                numActive++;
                indexVec.push_back(k);
            }
        }
    }
//    cout << "ca va" << endl;
    vector<vector<int> > matchVec;
    matcher.bruteForce_2(lmFeatureVec, featureVec, matchVec);

    Odometry odometry(trajectory.back(), stereo.TbaseCam1, stereo.cam1);
//    cout << "ca va" << endl;
    for (int i = 0; i < nWMreprojected; i++)
    {
        vector<Vector2d> vec;
        for (int j = 0; j < matchVec[i].size(); j++)
        {
            if (matchVec[i][j] != -1)
            {
                vec.push_back(featureVec[matchVec[i][j]].pt);
            }
        }
        if (vec.size() > 0)
        {
            odometry.observationVec_2.push_back(vec);
            odometry.cloud.push_back(WM[indexVec[i]].X);
        }
    }

    if (WM.size() < 50)
    {
        for (int i = nWMreprojected; i < numActive; i++)
        {
            vector<Vector2d> vec;
            for (int j = 0; j < matchVec[i].size(); j++)
            {
                if (matchVec[i][j] != -1)
                {
                    vec.push_back(featureVec[matchVec[i][j]].pt);
                }
            }
            if (vec.size() > 0)
            {
                odometry.observationVec_2.push_back(vec);
                odometry.cloud.push_back(STM[indexVec[i]].X);
            }
        }
    }

//    cout << "cloud : " << odometry.cloud.size() << endl;
    //RANSAC
    odometry.Ransac_2();
//    cout << odometry.TorigBase << endl;
    //Final transformation computation
    odometry.computeTransformation_2();
//    cout << odometry.TorigBase << endl;
    return odometry.TorigBase;
}
