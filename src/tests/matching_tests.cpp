#include <iostream>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <random>

#include "tests/matching_tests.h"
#include "geometry.h"
#include "mei.h"
#include "vision.h"
#include "matcher.h"
#include "extractor.h"

using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

vector<testPoint> initCloud()
{
    const double xMin = -10;
    const double xMax = 10;
    const double yMin = -10;
    const double yMax = 10;
    const double zMin = 10;
    const double zMax = 30;
    const int cloudSize = 1000;

    // initialize random distributions
    default_random_engine generator(1);
    // for point coordinates
    uniform_real_distribution<double> pX(xMin, xMax);
    uniform_real_distribution<double> pY(yMin, yMax);
    uniform_real_distribution<double> pZ(zMin, zMax);
    // for descriptors
    uniform_real_distribution<float> pD(0, 0.01);
    // for keypoint size
    uniform_real_distribution<float> pS(10, 100);
    // for keypoint angle
    uniform_real_distribution<float> pA(0, 360);

    // create testPoint cloud
    vector<testPoint> cloud;
    for (int pointIdx = 0; pointIdx < cloudSize; pointIdx++)
    {
        testPoint point;

        double x = pX(generator);
        double y = pY(generator);
        double z = pZ(generator);
        point.pt << x, y, z;

        for (int j = 0; j < 64; j++)
        {
            float q = pD(generator);
            point.desc(j) = q;
        }

        float s = pS(generator);
        point.size = s;

        float a = pA(generator);
        point.angle = a;

        cloud.push_back(point);
    }

    return cloud;
}

void testMatching()
{
    testBruteForce();

    testStereoMatch();

    testMatchReprojected();
}

void testBruteForce()
{

    cout << "### Brute Force Test ### " << flush;

    const double xMin = 0;
    const double xMax = 1000;
    const double yMin = 0;
    const double yMax = 1000;
    const int N = 1000;

    // initialize random distributions
    default_random_engine generator(1);
    // for point coordinates
    uniform_real_distribution<double> pX(xMin, xMax);
    uniform_real_distribution<double> pY(yMin, yMax);
    // for descriptors
    uniform_real_distribution<float> pD(0, 0.01);
    // for keypoint size
    uniform_real_distribution<float> pS(10, 100);
    // for keypoint angle
    uniform_real_distribution<float> pA(0, 360);

    vector<Feature> fVec1, fVec2;

    for (int i = 0; i < N; i++)
    {
        Eigen::Matrix<float,64,1> desc;
        for (int j = 0; j < 64; j++)
        {
            float q = pD(generator);
            desc(j) = q;
        }

        float s = pS(generator); //size
        float a = pA(generator); //angle

        double x = pX(generator);
        double y = pY(generator);
        Feature f1(x, y, desc, s, a);
        fVec1.push_back(f1);

        x = pX(generator);
        y = pY(generator);
        Feature f2(x, y, desc, s, a);
        fVec2.push_back(f2);
    }

    vector<int> matches(N, -1);

    Matcher matcher;
    matcher.bruteForceOneToOne(fVec1, fVec2, matches);

    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        if (i != matches[i])
        {
            errors ++;
            cout << endl << "match for " << i << ": " << matches[i] << endl << endl;
        }
    }
    if (errors == 0)
        cout << "OK" << endl;
    else
        cout << "Test Failed" << endl;

}

void testStereoMatch()
{
    cout << "### Stereo Match Test ### " << flush;

    double params[6]{0.3, 0.2, 375, 375, 650, 470};
    MeiCamera cam1mei(1296, 966, params);   
    MeiCamera cam2mei(1296, 966, params);

    // estrinsic parameters
    const Vector3d r(5*3.1415926/180, 2*3.1415926/180, -3*3.1415926/180); // (x, y, z), vector for 2R1
    const Vector3d tR(1, 0.1, -0.05); // (x, y, z), 2t1-2
    Transformation<double> T1, T2(tR, r);

    // create stereo system
    StereoSystem stereo(T1, T2, cam1mei, cam2mei);

    //displayBins(stereo);

    vector<testPoint> cloud = initCloud();

    int N = cloud.size();

    // create vectors to use stereo.projectPointCloud
    vector<Eigen::Vector2d> pt2Vec1, pt2Vec2;
    vector<Eigen::Vector3d> pt3Vec;
    for (int i = 0; i < N; i++)
    {
        pt3Vec.push_back(cloud[i].pt);
    }

    // project point cloud
    stereo.projectPointCloud(pt3Vec, pt2Vec1, pt2Vec2);

    // create feature vectors
    vector<Feature> fVec1, fVec2;
    // for the left camera
    for (int i = 0; i < N; i++)
    {
        Eigen::Vector2d point = pt2Vec1[i];
        Feature f(point, cloud[i].desc);
        f.size = cloud[i].size;
        f.angle = cloud[i].angle;
        fVec1.push_back(f);
        //cout << endl << "left camera projection: " << point << endl << endl;
    }
    // for the right camera
    for (int i = 0; i < N; i++)
    {
        Eigen::Vector2d point = pt2Vec2[i];
        Feature f(point, cloud[i].desc);
        f.size = cloud[i].size;
        f.angle = cloud[i].angle;
        fVec2.push_back(f);
        //cout << endl << "right camera projection: " << point << endl << endl;
    }

    /*cout << endl << "i=56 coordinates: " << endl << pt3Vec[56] << endl;
    cout << "left projection:" << endl << pt2Vec1[56] << endl;
    cout << "right projection:" << endl << pt2Vec2[56] << endl;
    cout << endl << "i=88 coordinates: " << endl << pt3Vec[88] << endl;
    cout << "left projection:" << endl << pt2Vec1[88] << endl;
    cout << "right projection:" << endl << pt2Vec2[88] << endl;*/

    vector<int> matches(N, -1);

    Matcher matcher;
    matcher.initStereoBins(stereo);
    matcher.stereoMatch(fVec1, fVec2, matches);

    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        if (i != matches[i])
        {
            errors ++;
            cout << endl << "match for " << i << ": " << matches[i] << endl << endl;
        }
    }
    if (errors == 0)
        cout << "OK" << endl;
    else
        cout << "Test Failed" << endl;
}

void testMatchReprojected()
{

    cout << "### Reprojected Match Test ### " << flush;

    const double xMin = 0;
    const double xMax = 1000;
    const double yMin = 0;
    const double yMax = 1000;
    const int N = 1000;

    // initialize random distributions
    default_random_engine generator(1);
    // for point coordinates
    uniform_real_distribution<double> pX(xMin, xMax);
    uniform_real_distribution<double> pY(yMin, yMax);
    // for descriptors
    uniform_real_distribution<float> pD(0, 0.01);
    // for keypoint size
    uniform_real_distribution<float> pS(10, 100);
    // for keypoint angle
    uniform_real_distribution<float> pA(0, 360);
    // for reprojection displacement
    uniform_real_distribution<float> pR(-10, 10);

    vector<Feature> fVec1, fVec2;

    for (int i = 0; i < N; i++)
    {
        Eigen::Matrix<float,64,1> desc;
        for (int j = 0; j < 64; j++)
        {
            float q = pD(generator);
            desc(j) = q;
        }

        float s = pS(generator); //size
        float a = pA(generator); //angle

        double x = pX(generator);
        double y = pY(generator);
        Feature f1(x, y, desc, s, a);
        fVec1.push_back(f1);

        double rX = pR(generator);
        double rY = pR(generator);
        Feature f2(x+rX, y+rY, desc, s, a);
        fVec2.push_back(f2);
    }

    vector<int> matches(N, -1);

    Matcher matcher;
    matcher.bruteForceOneToOne(fVec1, fVec2, matches);

    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        if (i != matches[i])
        {
            errors ++;
            cout << endl << "match for " << i << ": " << matches[i] << endl << endl;
        }
    }
    if (errors == 0)
        cout << "OK" << endl;
    else
        cout << "Test Failed" << endl;

}

void displayBruteForce()
{

    float resizeRatio = 0.5;

    //stringstream sstm;

    cv::Mat img1 = cv::imread("../datasets/dataset_stereo_1/view_left_0.png", 0);
    cv::Mat img2 = cv::imread("../datasets/dataset_stereo_1/view_right_0.png", 0);

    cv::resize(img1, img1, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(img2, img2, cv::Size(0,0), resizeRatio, resizeRatio);

    vector<Feature> kpVec1, kpVec2;
    Extractor extr(1000, 2, 2, false, true);

    extr(img1, kpVec1);
    extr(img2, kpVec2);

    const int N1 = kpVec1.size();
    const int N2 = kpVec2.size();
    cout << endl << "N1=" << N1 << " N2=" << N2 << endl;

    vector<int> matches(N1, -1);

    Matcher matcher;
    matcher.bruteForceOneToOne(kpVec1, kpVec2, matches);

    vector<cv::KeyPoint> keypoints1;
    for (int i = 0; i < N1; i++)
    {
        cv::KeyPoint kp(kpVec1[i].pt(0), kpVec1[i].pt(1), 1);
        keypoints1.push_back(kp);
    }

    vector<cv::KeyPoint> keypoints2;
    for (int i = 0; i < N2; i++)
    {
        cv::KeyPoint kp(kpVec2[i].pt(0), kpVec2[i].pt(1), 1);
        keypoints2.push_back(kp);
    }

    vector<cv::DMatch> mD;
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] != -1)
        {
            //cout << "i=" << i << " matches[i]=" << matches[i] << endl;
            cv::DMatch m(matches[i], i, 0);
            mD.push_back(m);
        }
    }

    cv::Mat imgMatches;
    cv::drawMatches(img2, keypoints2, img1, keypoints1, mD, imgMatches);
    cv::imshow("Matches", imgMatches);

    cv::waitKey();

}

void displayStereoMatch(const StereoSystem & stereo)
{
    float resizeRatio = 0.5; // resize the images only for display purposes!!!!!

    //stringstream sstm;

    cv::Mat img1 = cv::imread("../datasets/dataset_stereo_1/view_left_0.png", 0);
    cv::Mat img2 = cv::imread("../datasets/dataset_stereo_1/view_right_0.png", 0);

    vector<Feature> fVec1, fVec2;
    Extractor extr(1000, 2, 2, false, true);

    extr(img1, fVec1);
    extr(img2, fVec2);

    const int N1 = fVec1.size();
    const int N2 = fVec2.size();

    cout << endl << "N1=" << N1 << " N2=" << N2 << endl;

    vector<int> matches(N1, -1);

    Matcher matcher;
    matcher.initStereoBins(stereo);

    matcher.stereoMatch(fVec1, fVec2, matches);

    for (int i = 0; i < N1; i++)
    {
        cout << "match for " << i << ": " << matches[i] << endl;
    }

    vector<cv::KeyPoint> keypoints1;
    for (int i = 0; i < N1; i++)
    {
        cv::KeyPoint kp(fVec1[i].pt(0)*resizeRatio, fVec1[i].pt(1)*resizeRatio, 1);
        cout << "i=" << i << " size=" << fVec1[i].size << " angle=" << fVec1[i].angle << endl;
        keypoints1.push_back(kp);
    }

    vector<cv::KeyPoint> keypoints2;
    for (int i = 0; i < N2; i++)
    {
        cv::KeyPoint kp(fVec2[i].pt(0)*resizeRatio, fVec2[i].pt(1)*resizeRatio, 1);
        cout << "i=" << i << " size=" << fVec2[i].size << " angle=" << fVec2[i].angle << endl;
        keypoints2.push_back(kp);
    }

    vector<cv::DMatch> mD;
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] != -1)
        {
            //cout << "i=" << i << " matches[i]=" << matches[i] << endl;
            cv::DMatch m(i, matches[i], 0);
            mD.push_back(m);
        }
    }

    cout << endl << "N. matches: " << mD.size() << endl << endl;

    cv::resize(img1, img1, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(img2, img2, cv::Size(0,0), resizeRatio, resizeRatio);

    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, mD, imgMatches);
    cv::imshow("Matches", imgMatches);

    cv::waitKey();

}

void displayStereoMatch_2(const StereoSystem & stereo)
{

    float resizeRatio = 0.5; // resize the images only for display purposes!!!!!

    stringstream sstm;

    cv::Mat imgL = cv::imread("dataset_stereo_1/view_left_10.png", 0);
    cv::Mat imgR = cv::imread("dataset_stereo_1/view_right_10.png", 0);

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

    cout << endl << "Number of stereo matches: " << mD.size() << endl << endl;

    cv::resize(imgL, imgL, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(imgR, imgR, cv::Size(0,0), resizeRatio, resizeRatio);

    vector<cv::DMatch> mD2;
    vector<cv::KeyPoint> kL, kR;
    for (int i = 0; i < mD.size(); i++)
    {
        mD2.clear();
        kL.clear();
        kR.clear();

        int tr = mD[i].trainIdx;
        int qu = mD[i].queryIdx;
        cv::DMatch dm(0, 0, 1);
        mD2.push_back(dm);
        kL.push_back(keypointsL[qu]);
        kR.push_back(keypointsR[tr]);

        cv::Mat imgMatches;
        cv::drawMatches(imgL, kL, imgR, kR, mD2, imgMatches);
        cv::imshow("Matches", imgMatches);

        cv::waitKey();
    }

}

void displayBins(const StereoSystem & stereo)
{

    uchar color = 180;

    Matcher matcher;
    matcher.initStereoBins(stereo);

    cv::Mat imageL(stereo.cam1.height, stereo.cam1.width, CV_8UC1);
    cv::Mat imageR(stereo.cam2.height, stereo.cam2.width, CV_8UC1);

    for (int i = 0; i < stereo.cam1.height; i++)
    {
        for (int j = 0; j < stereo.cam1.width; j++)
        {
            int k = abs(matcher.binMapL(i,j) % 2);

            imageL.at<uchar>(i, j) = k*color;
        }
    }

    for (int i = 0; i < stereo.cam2.height; i++)
    {
        for (int j = 0; j < stereo.cam2.width; j++)
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
