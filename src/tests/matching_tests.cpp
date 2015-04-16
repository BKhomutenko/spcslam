#include <iostream>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "tests/matching_tests.h"
#include "geometry.h"
#include "mei.h"
#include "vision.h"
#include "matcher.h"
#include "extractor.h"
#include <random>

using namespace std;

void testBins(const StereoSystem & stereo)
{

    uchar color = 180;

    Matcher matcher;
    matcher.initStereoBins(stereo);

    cv::Mat imageL(stereo.cam1->height, stereo.cam1->width, CV_8UC1);
    cv::Mat imageR(stereo.cam2->height, stereo.cam2->width, CV_8UC1);

    for (int i = 0; i < stereo.cam1->height; i++)
    {
        for (int j = 0; j < stereo.cam1->width; j++)
        {
            int k = abs(matcher.binMapL(i,j) % 2);

            imageL.at<uchar>(i, j) = k*color;
        }
    }

    for (int i = 0; i < stereo.cam2->height; i++)
    {
        for (int j = 0; j < stereo.cam2->width; j++)
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

void testBruteForce()
{

    float resizeRatio = 1;

    //stringstream sstm;

    cv::Mat img1 = cv::imread("dataset_stereo_1/view_left_0.png", 0);
    cv::Mat img2 = cv::imread("dataset_stereo_1/view_right_0.png", 0);

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
    matcher.bruteForce(kpVec1, kpVec2, matches);

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

void testStereoMatch(const StereoSystem & stereo, const vector<testPoint> & cloud)
{

    /*float resizeRatio = 0.5; // resize the images only for display purposes!!!!!

    //stringstream sstm;

    cv::Mat imgL = cv::imread("../datasets/dataset_stereo_1/view_left_0.png", 0);
    cv::Mat imgR = cv::imread("../datasets/dataset_stereo_1/view_right_0.png", 0);

    vector<Feature> kpVecL, kpVecR;
    Extractor extr(1000, 2, 2, false, true);

    extr(imgL, kpVecL);
    extr(imgR, kpVecR);

    const int NL = kpVecL.size();
    const int NR = kpVecR.size();

    cout << endl << "NL=" << NL << " NR=" << NR << endl;

    vector<int> matches(NL, -1);
    */

    int N = cloud.size();

    // create vectors to use stereo.projectPointCloud
    vector<Eigen::Vector3d> pt3Vec;
    for (int i = 0; i < N; i++)
    {
        pt3Vec.push_back(cloud[i].pt);
    }
    vector<Eigen::Vector2d> pt2Vec1, pt2Vec2;

    // project point cloud
    stereo.projectPointCloud(pt3Vec, pt2Vec1, pt2Vec2);

    // vectors of indexes to check the algorithm
    vector<int> v1, v2;

    // create feature vector
    vector<Feature> fVec1, fVec2;
    // for left camera
    for (int i = 0; i < N; i++)
    {
        Eigen::Vector2d point = pt2Vec1[i];
        // include keypoints only if they are in the field of view of the camera
        if (point(0) >= 0 and point(0) < stereo.cam1->width and
            point(1) >= 0 and point(1) < stereo.cam1->width)
        {
            Feature f(point, cloud[i].desc);
            f.size = cloud[i].size;
            f.angle = cloud[i].angle;
            fVec1.push_back(f);
            v1.push_back(i);
        }
    }
    // for right camera
    for (int i = 0; i < N; i++)
    {
        Eigen::Vector2d point = pt2Vec2[i];
        // include keypoints only if they are in the field of view of the camera
        if (point(0) >= 0 and point(0) < stereo.cam2->width and
            point(1) >= 0 and point(1) < stereo.cam2->width)
        {
            Feature f(point, cloud[i].desc);
            f.size = cloud[i].size;
            f.angle = cloud[i].angle;
            fVec2.push_back(f);
            v2.push_back(i);
        }
    }

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

    /*
    vector<cv::KeyPoint> keypointsL;
    for (int i = 0; i < NL; i++)
    {
        cv::KeyPoint kp(kpVecL[i].pt(0)*resizeRatio, kpVecL[i].pt(1)*resizeRatio, 1);
        cout << "i=" << i << " size=" << kpVecL[i].size << " angle=" << kpVecL[i].angle << endl;
        keypointsL.push_back(kp);
    }

    vector<cv::KeyPoint> keypointsR;
    for (int i = 0; i < NR; i++)
    {
        cv::KeyPoint kp(kpVecR[i].pt(0)*resizeRatio, kpVecR[i].pt(1)*resizeRatio, 1);
        cout << "i=" << i << " size=" << kpVecR[i].size << " angle=" << kpVecR[i].angle << endl;
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

    cout << endl << "N. matches: " << mD.size() << endl << endl;

    cv::resize(imgL, imgL, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(imgR, imgR, cv::Size(0,0), resizeRatio, resizeRatio);

    cv::Mat imgMatches;
    cv::drawMatches(imgL, keypointsL, imgR, keypointsR, mD, imgMatches);
    cv::imshow("Matches", imgMatches);

    cv::waitKey();
    */

}

void testStereoMatch_2(StereoSystem & stereo)
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

void testMatchReprojected()
{

    float resizeRatio = 0.6;

    stringstream sstm;

    cv::Mat img1 = cv::imread("dataset_reprojected_1/view_0.png", 0);
    cv::Mat img2 = cv::imread("dataset_reprojected_1/view_1.png", 0);

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
    matcher.matchReprojected(kpVec1, kpVec2, matches);

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

void testMatching()
{
    const double xMin = -100;
    const double xMax = 100;
    const double yMin = -100;
    const double yMax = 100;
    const double zMin = 0;
    const double zMax = 100;
    const int cloudSize = 10000;

    // intrinsic parameters
    Camera * cam1 = new MeiCamera(1296, 966, 0.5, 1, 1000, 1000, 648, 483);
    Camera * cam2 = new MeiCamera(1296, 966, 0.5, 1, 1000, 1000, 648, 483);

    // estrinsic parameters
    const Quaternion qR(0.01, -0.05, -0.02, 0.95); // (x, y, z, w), 2R1
    const Vector3d tR(1, 0.03, 0.1); // (x, y, z), 2t1-2
    Transformation T1, T2(tR, qR);

    // create stereo system
    const StereoSystem stereo(T1, T2, cam1, cam2);

    // initialize random distributions
    default_random_engine generator;
    // for point coordinates
    uniform_real_distribution<double> pX(xMin, xMax);
    uniform_real_distribution<double> pY(yMin, yMax);
    uniform_real_distribution<double> pZ(zMin, zMax);
    // for descriptors
    uniform_real_distribution<float> pD(0, 1);
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

    //testBins(stereo);

    //testBruteForce();

    testStereoMatch(stereo, cloud);

    //testStereoMatch_2(stereo);

    //testMatchReprojected();

}
