/*
The mapping system itself
*/

#ifndef _SPCMAP_CARTOGRAPHY_H_
#define _SPCMAP_CARTOGRAPHY_H_

//STL
#include <vector>

//Eigen
#include <Eigen/Eigen>

//Ceres solver
#include <ceres/ceres.h>

#include "extractor.h"
#include "matcher.h"
#include "geometry.h"
#include "vision.h"

//Structure is used to perform map improvement

using namespace std;
using Eigen::Matrix;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;

struct Observation
{
    Observation(double u, double v, unsigned int poseIdx, CameraID camId)
        : pt(u, v), poseIdx(poseIdx), cameraId(camId) {}

    Observation(Vector2d pt, unsigned int poseIdx, CameraID camId)
        : pt(pt), poseIdx(poseIdx), cameraId(camId) {}
    //observed coordinates
    Vector2d pt;

    //index of corresponding positions in StereoCartograpy::trajectory
    unsigned int poseIdx;

    //Either left or right camera
    CameraID cameraId;
};

struct LandMark
{
    //3D position in the globa frame
    Vector3d X;

    //Feature descriptor
    Matrix<float, 64, 1> d;

    //Feature angle
    float angle;

    //All Vec6drealted measurements
    vector<Observation> observations;
};

struct ReprojectionErrorStereo : public ceres::SizedCostFunction<2, 3, 3, 3>
{
    ReprojectionErrorStereo(const Vector2d pt, const Transformation<double> & TbaseCam,
            const ICamera * camera);

    // args : double lm[3], double pose[6]
    bool Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const;

    //observed coordinates
    const double u, v;
    //Transformation information
    Vector3d PcamBase;
    Matrix3d RcamBase;

    //provides projection model
    const ICamera * camera;

};

struct ReprojectionErrorFixed : public ceres::SizedCostFunction<2, 3>
{
    ReprojectionErrorFixed(const Vector2d pt, const Transformation<double> & xi,
            const Transformation<double> & camTransformation, const ICamera * camera);

    // args : double lm[3]
    bool Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const;

    //observed coordinates
    const double u, v;
    //Transformation information
    Vector3d PcamBase, PbaseOrig;
    Matrix3d RcamBase, RbaseOrig;
    //provides projection model
    const ICamera * camera;

};

struct OdometryError : public ceres::SizedCostFunction<2, 3, 3>
{
    OdometryError(const Vector3d X, const Vector2d pt,
        const Transformation<double> & TbaseCam,
        const ICamera & camera);

    // args : double lm[3], double pose[6]
    bool Evaluate(double const* const* args,
                    double* residuals,
                    double** jac) const;

    // Landmark position
    Vector3d X;
    //observed coordinates
    const double u, v;
    //Transformation information
    Vector3d PcamBase;
    Matrix3d RcamBase;

    //provides projection model
    const ICamera * camera;

};


//TODO implement camera calibration in the future
class MapInitializer
{
public:

    void addObservation(Vector3d & X, Vector2d pt, Transformation<double> & pose,
            const ICamera * cam, const Transformation<double> & TbaseCam);

    void addFixedObservation(Vector3d & X, Vector2d pt, Transformation<double> & pose,
            const ICamera * cam, const Transformation<double> & TbaseCam);
//    void addObservationRight(Vector3d & X, double u, double v, Transformation & pose,
//            const Camera & cam, Transformation & rightCamTransformation);

    void compute();

private:
    ceres::Problem problem;

};

class Odometry
{
public:
    vector<Vector2d> observationVec;
    vector<vector<Vector2d> > observationVec_2;
    vector<Vector3d> cloud;
    vector<bool> inlierMask;
    vector<vector<bool> > inlierMask_2;
    Transformation<double> TorigBase;
    const Transformation<double> TbaseCam;
    const ICamera & camera;

    Odometry(const Transformation<double> TorigBase,
            const Transformation<double> TbaseCam,
            const ICamera & camera)
            : TorigBase(TorigBase), TbaseCam(TbaseCam), camera(camera) {}

    Odometry(const Transformation<double> TorigBase,
            const Transformation<double> TbaseCam,
            const ICamera * camera)
            : TorigBase(TorigBase), TbaseCam(TbaseCam), camera(*camera) {}

    void computeTransformation();

    void computeTransformation_2();

    void Ransac();

    void Ransac_2();
};


class StereoCartography
{
public:
    StereoCartography (Transformation<double> & p1, Transformation<double> & p2,
            ICamera & c1, ICamera & c2)
            : stereo(p1, p2, c1, c2), extractor(3000, 1, 2, false, true)
    {
        //extractor.setType(FeatureType::Custom);
        matcher.initStereoBins(stereo);
        matcher.computeMaps(stereo);
        srand(0);
    }
//    virtual ~StereoCartography () { LM.clear(); trajectory.clear(); }

    StereoSystem stereo;

    Matcher matcher;

    Extractor extractor;

    void projectPointCloud(const vector<Vector3d> & src,
            vector<Vector2d> & dst1, vector<Vector2d> & dst2, int poseIdx) const;

    //performs optimization of all landmark positions wrt the actual path
    void improveTheMap(bool firstBA);

    void improveTheMap_2();

    Transformation<double> estimateOdometry(const vector<Feature> & featureVec) const;

    Transformation<double> estimateOdometry_2(const vector<Feature> & featureVec) const;

    Transformation<double> estimateOdometry_3(const vector<Feature> & featureVec) const;

    //the library of all landmarks
    //to be replaced in the future with somth smarter than a vector
    //system memory (short-term mem, working mem, long-term mem)
    vector<LandMark> STM, WM, LTM;

    //a chain of camera positions
    //first initialized with the odometry measurements
    vector<Transformation<double>> trajectory;
    //list<LandMark &> activeLM;

};

#endif
