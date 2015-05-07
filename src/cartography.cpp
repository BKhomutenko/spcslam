#include <random>

//STL
#include <vector>
//Eigen
#include <Eigen/Eigen>

//Ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "geometry.h"
#include "matcher.h"
#include "mei.h"
#include "vision.h"
#include "cartography.h"

using namespace ceres;
using Eigen::Matrix;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

using Eigen::RowMajor;

int countCalls = 0;

inline double sinc(const double x)
{
    if (x==0)
        return 1;
    return std::sin(x)/x;
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

Transformation<double> StereoCartography::estimateOdometry(const vector<Feature> & featureVec)
{
    //Matching
    
    int numLandmarks = LM.size();
    int numActive = min(300, numLandmarks);
    vector<Feature> lmFeatureVec;
    cout << "ca va" << endl;
    for (unsigned int i = numLandmarks - numActive; i < numLandmarks; i++)
    {
        lmFeatureVec.push_back(Feature(Vector2d(0, 0), LM[i].d));
    }
    cout << "ca va" << endl;       
    Matcher matcher;    
    vector<int> matchVec;    
    matcher.bruteForce(featureVec, lmFeatureVec, matchVec);
    
    Odometry odometry(trajectory.back(), stereo.pose1, stereo.cam1);
    cout << "ca va" << endl;
    for (unsigned int i = 0; i < featureVec.size(); i++)
    {
        const int match = matchVec[i];
        if (match == -1) continue;
        odometry.observationVec.push_back(featureVec[i].pt);
        odometry.cloud.push_back(LM[numLandmarks  - numActive + match].X);
    }
    cout << "cloud : " << odometry.cloud.size() << endl;
    //RANSAC
    odometry.Ransac();
    cout << odometry.TorigBase << endl;
    //Final transformation computation
    odometry.computeTransformation();
    cout << odometry.TorigBase << endl;
    return odometry.TorigBase;
}






