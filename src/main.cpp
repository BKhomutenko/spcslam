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
#include "matcher.h"
#include "extractor.h"


#define S 200
#define SIZE Size(S, S)

using namespace std;
extern int countCalls;

using Eigen::Vector3d;

void testBins(StereoSystem & stereo)
{

    uchar color = 180;

    Matcher matcher;
    matcher.initStereoBins(stereo);

    cv::Mat imageL(stereo.cam1->imageHeight, stereo.cam1->imageWidth, CV_8UC1);
    cv::Mat imageR(stereo.cam2->imageHeight, stereo.cam2->imageWidth, CV_8UC1);

    for (int i = 0; i < stereo.cam1->imageHeight; i++)
    {
        for (int j = 0; j < stereo.cam1->imageWidth; j++)
        {
            int k = abs(matcher.binMapL(i,j) % 2);

            imageL.at<uchar>(i, j) = k*color;
        }
    }

    for (int i = 0; i < stereo.cam2->imageHeight; i++)
    {
        for (int j = 0; j < stereo.cam2->imageWidth; j++)
        {
            int k = abs(matcher.binMapR(i,j) % 2);

            imageR.at<uchar>(i, j) = k*color;
        }
    }

    float resizeRatio = 1;

    cv::resize(imageL, imageL, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(imageR, imageR, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::imshow("binMapL", imageL);
    cv::imshow("binMapR", imageR);

    cv::waitKey();

}


void testStereoMatch(StereoSystem & stereo)
{

    float resizeRatio = 0.5; // resize the images only for display purposes!!!!!

    //stringstream sstm;

    cv::Mat imgL = cv::imread("dataset_stereo_1/view_left_0.png", 0);
    cv::Mat imgR = cv::imread("dataset_stereo_1/view_right_0.png", 0);

    vector<Feature> kpVecL, kpVecR;
    Extractor extr(1000, 2, 2, false, true);

    extr(imgL, kpVecL);
    extr(imgR, kpVecR);

    const int NL = kpVecL.size();
    const int NR = kpVecR.size();
    cout << endl << "NL=" << NL << " NR=" << NR << endl;

    vector<int> matches(NL, -1);

    Matcher matcher;
    matcher.initStereoBins(stereo);
    matcher.stereoMatch(kpVecL, kpVecR, matches);

    vector<cv::KeyPoint> keypointsL;
    for (int i = 0; i < NL; i++)
    {
        cv::KeyPoint kp(kpVecL[i].pt(0)*resizeRatio, kpVecL[i].pt(1)*resizeRatio, 1);
        keypointsL.push_back(kp);
    }

    vector<cv::KeyPoint> keypointsR;
    for (int i = 0; i < NR; i++)
    {
        cv::KeyPoint kp(kpVecR[i].pt(0)*resizeRatio, kpVecR[i].pt(1)*resizeRatio, 1);
        keypointsR.push_back(kp);
    }

    vector<cv::DMatch> mD;
    for (int i = 0; i < NL; i++)
    {
        if (matches[i] != -1)
        {
            //cout << "i=" << i << " matches[i]=" << matches[i] << endl;
            cv::DMatch m(i, matches[i], 0);
            mD.push_back(m);
        }
    }

    cv::resize(imgL, imgL, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(imgR, imgR, cv::Size(0,0), resizeRatio, resizeRatio);

    cv::Mat imgMatches;
    cv::drawMatches(imgL, keypointsL, imgR, keypointsR, mD, imgMatches);
    cv::imshow("Matches", imgMatches);

    cv::waitKey();

}


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


    // Tests of the Feautre module 
    
    MeiCamera * cam1mei;
    MeiCamera * cam2mei;
    cam1mei = new MeiCamera(1296, 966, 1.39200313135677, 8.9661425437872310e+02, 8.9401437200675650e+02, 6.5205449530360681e+02, 4.7218039655643264e+02);
    cam2mei = new MeiCamera(1296, 966, 1.7009089913682820, 1.0102335582115433e+03, 1.0075016763931362e+03, 6.6174475465968897e+02, 4.8641648577657963e+02);

    Matrix<double, 2, 3> J;
    
    Vector2d p1, p2;
    Vector3d X1(1, 2, 3), X2(1.01, 2.0, 3);
    cam1mei->projectPoint(X1, p1);
    cam1mei->projectPoint(X2, p2);
    cam1mei->projectionJacobian(X1, J);
    cout << p2 - p1 << endl;
    cout << J << endl;
    cout << J*(X2 - X1) << endl;

    const Quaternion qR(1.2819328761269129e-03,  // (x, y, z, w)
                        -5.0470030584528022e-02,
                        -1.9222145763661295e-03,
                        9.9872290338813241e-01);
    const Vector3d tR(-7.8463742913216261e-01,  // (x, y, z) OL-OR expressed in CR reference frame?
                      -3.1213039143325322e-03,
                      -5.2863573996665768e-02);

    Transformation T1, T2(-tR, qR);

    
    
    // init needed objects
    
    Extractor extr(1000, 2, 2, false, true);
    
    StereoCartography cartograph(T1, T2, cam1mei, cam2mei);
    
    Matcher matcher;
    matcher.initStereoBins(cartograph.stereo);
    
    // ## THE INIT STEP ##
    // load first pair
    
    string name1 = "/home/bogdan/projects/icars/img_left/frame000";
    string name2 = "/home/bogdan/projects/icars/img_right/frame000";
    cv::Mat img1 = cv::imread(name1 + "0.jpg", 0);
    cv::Mat img2 = cv::imread(name2 + "0.jpg", 0);
   
    // extract and match features
    vector<Feature> kpVec1, kpVec2;
    extr(img1, kpVec1);
    extr(img2, kpVec2);
    
    cout << "features : " << kpVec1.size() << " "  << kpVec2.size() << endl;
    
    vector<int> matchVec;
    
    matcher.stereoMatch(kpVec1, kpVec2, matchVec);
    
    // init the cloud
    vector<Vector2d> pVec1, pVec2;
    vector< Matrix<float, 64, 1> > descVec;
    for (unsigned int i = 0; i < kpVec1.size(); i++)
    {
        const int match = matchVec[i];
        if (match == -1) continue;
        descVec.push_back(kpVec1[i].desc);
        pVec1.push_back(kpVec1[i].pt);
        pVec2.push_back(kpVec2[match].pt);        
    }
    
    vector<Vector3d> cloud;
    cartograph.stereo.reconstructPointCloud(pVec1, pVec2, cloud);
    for (unsigned int i = 0; i < cloud.size(); i++)
    {
        //TODO put filtering here according to z
        cartograph.LM.push_back(LandMark());
        LandMark & lm = cartograph.LM.back();
        lm.X = cloud[i];
        lm.d = descVec[i];
        Observation obs1(pVec1[i](0), pVec1[i](1), 0, LEFT);
        Observation obs2(pVec2[i](0), pVec2[i](1), 0, RIGHT);
        lm.observations.push_back(obs1);
        lm.observations.push_back(obs2);
        
    }
    cartograph.trajectory.push_back(Transformation(0, 0, 0, 0, 0, 0));
    cout << "cloud size : " << cloud.size() << endl;
    
    // ## ODOMETRY ## 
     
    img1 = cv::imread(name1 + "5.jpg", 0);   
    extr(img1, kpVec1);
    Transformation xi = cartograph.estimateOdometry(kpVec1);

    cout << xi << endl;
    
   // testStereoMatch(stereo);
    
    //---------------------------------


/*
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
   
////////////////////////////////////////////


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


 */
}


