#include <iostream>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <random>

#include "geometry.h"
#include "mei.h"
#include "vision.h"
#include "matcher.h"
#include "extractor.h"
#include "cartography.h"

using namespace std;
using namespace cv;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

//TODO check this
struct testPoint
{
    Eigen::Vector3d pt;
    Descriptor desc;

    float size;
    float angle;
};

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

        for (int j = 0; j < point.desc.innerSize(); j++)
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

    vector<Feature> featureVec1, featureVec2;

    for (int i = 0; i < N; i++)
    {
        Descriptor desc;
        for (int j = 0; j < desc.innerSize(); j++)
        {
            float q = pD(generator);
            desc(j) = q;
        }

        float s = pS(generator); //size
        float a = pA(generator); //angle

        double x = pX(generator);
        double y = pY(generator);
        Feature f1(x, y, desc, s, a);
        featureVec1.push_back(f1);

        x = pX(generator);
        y = pY(generator);
        Feature f2(x, y, desc, s, a);
        featureVec2.push_back(f2);
    }

    vector<int> matches(N, -1);

    Matcher matcher;
    matcher.bruteForceOneToOne(featureVec1, featureVec2, matches);

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
    vector<Feature> featureVec1, featureVec2;
    // for the left camera
    for (int i = 0; i < N; i++)
    {
        Eigen::Vector2d point = pt2Vec1[i];
        Feature f(point, cloud[i].desc);
        f.size = cloud[i].size;
        f.angle = cloud[i].angle;
        featureVec1.push_back(f);
        //cout << endl << "left camera projection: " << point << endl << endl;
    }
    // for the right camera
    for (int i = 0; i < N; i++)
    {
        Eigen::Vector2d point = pt2Vec2[i];
        Feature f(point, cloud[i].desc);
        f.size = cloud[i].size;
        f.angle = cloud[i].angle;
        featureVec2.push_back(f);
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
    matcher.stereoMatch(featureVec1, featureVec2, matches);

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

    vector<Feature> featureVec1, featureVec2;

    for (int i = 0; i < N; i++)
    {
        Descriptor desc;
        for (int j = 0; j < desc.innerSize(); j++)
        {
            float q = pD(generator);
            desc(j) = q;
        }

        float s = pS(generator); //size
        float a = pA(generator); //angle

        double x = pX(generator);
        double y = pY(generator);
        Feature f1(x, y, desc, s, a);
        featureVec1.push_back(f1);

        double rX = pR(generator);
        double rY = pR(generator);
        Feature f2(x+rX, y+rY, desc, s, a);
        featureVec2.push_back(f2);
    }

    vector<int> matches(N, -1);

    Matcher matcher;
    matcher.bruteForceOneToOne(featureVec1, featureVec2, matches);

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

    Mat img1 = imread("/home/bogdan/projects/icars/img_difficult/l00.jpg", 0);
    Mat img2 = imread("/home/bogdan/projects/icars/img_difficult/l02.jpg", 0);

    vector<Feature> featureVec1, featureVec2;
    FeatureExtractor extr(500);

    extr.compute(img1, featureVec1);
    extr.compute(img2, featureVec2);

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();
    cout << endl << "N1=" << N1 << " N2=" << N2 << endl;

    vector<int> matches(N1, -1);

    Matcher matcher;
    matcher.bruteForceOneToOne(featureVec1, featureVec2, matches);

    vector<KeyPoint> keypoints1;
    for (int i = 0; i < N1; i++)
    {
        KeyPoint kp(featureVec1[i].pt(0)*resizeRatio, featureVec1[i].pt(1)*resizeRatio, 1);
        keypoints1.push_back(kp);
    }

    vector<KeyPoint> keypoints2;
    for (int i = 0; i < N2; i++)
    {
        KeyPoint kp(featureVec2[i].pt(0)*resizeRatio, featureVec2[i].pt(1)*resizeRatio, 1);
        keypoints2.push_back(kp);
    }

    vector<DMatch> mD;
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] != -1)
        {
            //cout << "i=" << i << " matches[i]=" << matches[i] << endl;
            DMatch m(matches[i], i, 0);
            mD.push_back(m);
        }
    }
    cout << "number of matches " << mD.size() << endl;
    resize(img1, img1, Size(0,0), resizeRatio, resizeRatio);
    resize(img2, img2, Size(0,0), resizeRatio, resizeRatio);

    Mat imgMatches;
    drawMatches(img2, keypoints2, img1, keypoints1, mD, imgMatches);
    imshow("Matches", imgMatches);

    waitKey();

}

void displayStereoMatch(const StereoSystem & stereo)
{
    float resizeRatio = 0.5; // resize the images only for display purposes!!!!!

    //stringstream sstm;

    Mat img1 = imread("/home/bogdan/projects/icars/img_left/frame0000.jpg", 0);
    Mat img2 = imread("/home/bogdan/projects/icars/img_right/frame0000.jpg", 0);

    vector<Feature> featureVec1, featureVec2;
    FeatureExtractor extr(500);

    extr.compute(img1, featureVec1);
    extr.compute(img2, featureVec2);

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();

    cout << endl << "N1=" << N1 << " N2=" << N2 << endl;

    vector<int> matches(N1, -1);

    Matcher matcher;
    matcher.initStereoBins(stereo);
    matcher.computeMaps(stereo);

    matcher.stereoMatch(featureVec1, featureVec2, matches);

    for (int i = 0; i < N1; i++)
    {
        //cout << "match for " << i << ": " << matches[i] << endl;
    }

    vector<KeyPoint> keypoints1;
    for (int i = 0; i < N1; i++)
    {
        KeyPoint kp(featureVec1[i].pt(0)*resizeRatio, featureVec1[i].pt(1)*resizeRatio, 1);
        //cout << "i=" << i << " size=" << featureVec1[i].size << " angle=" << featureVec1[i].angle << endl;
        keypoints1.push_back(kp);
    }

    vector<KeyPoint> keypoints2;
    for (int i = 0; i < N2; i++)
    {
        KeyPoint kp(featureVec2[i].pt(0)*resizeRatio, featureVec2[i].pt(1)*resizeRatio, 1);
        //cout << "i=" << i << " size=" << featureVec2[i].size << " angle=" << featureVec2[i].angle << endl;
        keypoints2.push_back(kp);
    }

    vector<DMatch> mD;
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] != -1)
        {
            //cout << "i=" << i << " matches[i]=" << matches[i] << endl;
            DMatch m(i, matches[i], 0);
            mD.push_back(m);
        }
    }

    cout << endl << "N. matches: " << mD.size() << endl << endl;
cout << 111 << endl;
    resize(img1, img1, Size(0,0), resizeRatio, resizeRatio);
    resize(img2, img2, Size(0,0), resizeRatio, resizeRatio);
    cout << 111 << endl;
    Mat imgMatches;
    drawMatches(img1, keypoints1, img2, keypoints2, mD, imgMatches);
    imshow("Matches", imgMatches);

    waitKey();

}

void displayStereoMatch_2(const StereoSystem & stereo)
{

    float resizeRatio = 0.75; // resize the images only for display purposes!!!!!

    stringstream sstm;

    /*Mat img1 = imread("/home/bogdan/projects/icars/img_left/frame0000.jpg", 0);
    Mat img2 = imread("/home/bogdan/projects/icars/img_right/frame0000.jpg", 0);*/
    Mat img1 = imread("/home/bogdan/projects/icars/img_difficult/l00.jpg", 0);
    Mat img2 = imread("/home/bogdan/projects/icars/img_difficult/r00.jpg", 0);
    vector<Feature> featureVec1, featureVec2;
    FeatureExtractor extr(500);

    extr.compute(img1, featureVec1);
    extr.compute(img2, featureVec2);

    const int NL = featureVec1.size();
    const int NR = featureVec2.size();
    cout << endl << "NL=" << NL << " NR=" << NR << endl;

    vector<int> matches(NL, -1);

    Matcher matcher;
    matcher.initStereoBins(stereo);
    matcher.computeMaps(stereo);
    matcher.stereoMatch(featureVec1, featureVec2, matches);

    vector<KeyPoint> keypoints1;
    for (int i = 0; i < NL; i++)
    {
        KeyPoint kp(featureVec1[i].pt(0)*resizeRatio, featureVec1[i].pt(1)*resizeRatio, 1);
        keypoints1.push_back(kp);
    }

    vector<KeyPoint> keypoints2;
    for (int i = 0; i < NR; i++)
    {
        KeyPoint kp(featureVec2[i].pt(0)*resizeRatio, featureVec2[i].pt(1)*resizeRatio, 1);
        keypoints2.push_back(kp);
    }

    vector<DMatch> mD;
    for (int i = 0; i < NL; i++)
    {
        if (matches[i] != -1)
        {
            //cout << "i=" << i << " matches[i]=" << matches[i] << endl;
            DMatch m(matches[i], i, 0);
            mD.push_back(m);
        }
    }

    cout << endl << "Number of stereo matches: " << mD.size() << endl << endl;

    resize(img1, img1, Size(0,0), resizeRatio, resizeRatio);
    resize(img2, img2, Size(0,0), resizeRatio, resizeRatio);

    vector<DMatch> mD2;
    vector<KeyPoint> k1, k2;
    for (int i = 0; i < mD.size(); i++)
    {
        mD2.clear();
        k1.clear();
        k2.clear();

        int tr = mD[i].trainIdx;
        int qu = mD[i].queryIdx;
        DMatch dm(0, 0, 1);
        mD2.push_back(dm);
        k1.push_back(keypoints1[tr]);
        k2.push_back(keypoints2[qu]);

        /*cout << "P1 (" << featureVec1[tr].pt(0) << "," << featureVec1[tr].pt(1) << ")   P2 ("
             << featureVec2[qu].pt(0) << "," << featureVec2[qu].pt(1) << ")   beta1: "
             << matcher.betaMap1(round(featureVec1[tr].pt(1)), round(featureVec1[tr].pt(0)))
             << "   beta2: " << matcher.betaMap2(round(featureVec2[qu].pt(1)), round(featureVec2[qu].pt(0)))
             << endl;*/
        /*cout << "P1 (" << featureVec1[tr].pt(0) << "," << featureVec1[tr].pt(1) << ")   P2 ("
             << featureVec2[qu].pt(0) << "," << featureVec2[qu].pt(1) << ")   alfa1: "
             << matcher.alfaMap1(round(featureVec1[tr].pt(1)), round(featureVec1[tr].pt(0)))
             << "   alfa2: " << matcher.alfaMap2(round(featureVec2[qu].pt(1)), round(featureVec2[qu].pt(0)))
             << endl;*/

        double dist = computeDist(featureVec1[tr].desc, featureVec2[qu].desc);
        cout << "Displaying match " <<  i
             << "  distance: " << dist << " Size 1: " << featureVec1[tr].size
             << "  Size 2: " << featureVec2[qu].size << endl ;
                 
        Mat imgMatches;
        drawMatches(img1, k1, img2, k2, mD2, imgMatches);
        imshow("Matches", imgMatches);

        int u1 = keypoints1[tr].pt.x;
        int v1 = keypoints1[tr].pt.y;
        int u2 = keypoints2[qu].pt.x;
        int v2 = keypoints2[qu].pt.y;
        
        Mat leftPatch;
        resize(img1(Rect(u1-8, v1-8, 16, 16)), leftPatch, Size(50, 50));
        Mat rightPatch;
        resize(img2(Rect(u2-8, v2-8, 16, 16)), rightPatch, Size(50, 50));
        imshow("left", leftPatch);
        imshow("right", rightPatch);
        waitKey();
    }

}

void displayBins(const StereoSystem & stereo)
{

    uchar color = 180;

    Matcher matcher;
    matcher.initStereoBins(stereo);

    Mat imageL(stereo.cam1->height, stereo.cam1->width, CV_8UC1);
    Mat imageR(stereo.cam2->height, stereo.cam2->width, CV_8UC1);

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

    resize(imageL, imageL, Size(0,0), resizeRatio, resizeRatio);
    resize(imageR, imageR, Size(0,0), resizeRatio, resizeRatio);
    imshow("binMapL", imageL);
    imshow("binMapR", imageR);

    waitKey();

}

void testDistancesBF()
{

    double bfDistTh = 0.15;

    float resizeRatio = 0.5;

    //stringstream sstm;

    Mat img1 = imread("/home/bogdan/projects/icars/img_left/frame0000.jpg", 0);
    Mat img2 = imread("/home/bogdan/projects/icars/img_right/frame0000.jpg", 0);

    vector<Feature> featureVec1, featureVec2;
    FeatureExtractor extr(200);

    extr.compute(img1, featureVec1);
    extr.compute(img2, featureVec2);

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();
    cout << endl << "N1=" << N1 << " N2=" << N2 << endl;

    vector<int> matches(N1, -1);
    vector<int> matches2(N2, -1);
    vector<double> distances(N1, 0);

    for (int i = 0; i < N1; i++)
    {
        matches[i] = -1;
        int tempMatch = -1;
        double bestDist = 1000000;

        for (int j = 0; j < N2 ; j++)
        {
            double dist = (featureVec1[i].desc - featureVec2[j].desc).norm();

            if (dist < bestDist)
            {
                bestDist = dist;
                tempMatch = j;
                distances[i] = bestDist;
            }
        }
        if (bestDist < bfDistTh)
        {
            matches[i] = tempMatch;
        }
    }
    for (int i = 0; i < N2; i++)
    {
        matches2[i] = -1;
        int tempMatch = -1;
        double bestDist = 1000000;

        for (int j = 0; j < N1 ; j++)
        {
            double dist = (featureVec2[i].desc - featureVec1[j].desc).norm();

            if (dist < bestDist)
            {
                bestDist = dist;
                tempMatch = j;
            }
        }
        if (bestDist < bfDistTh)
        {
            matches2[i] = tempMatch;
        }
    }
    int counter = 0;
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] > -1 && matches2[matches[i]] != i)
        {
            matches[i] = -1;
        }
    }
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] > -1)
        {
            counter++;
        }
    }

    vector<KeyPoint> keypoints1;
    for (int i = 0; i < N1; i++)
    {
        KeyPoint kp(featureVec1[i].pt(0)*resizeRatio, featureVec1[i].pt(1)*resizeRatio, 1);
        keypoints1.push_back(kp);
    }

    vector<KeyPoint> keypoints2;
    for (int i = 0; i < N2; i++)
    {
        KeyPoint kp(featureVec2[i].pt(0)*resizeRatio, featureVec2[i].pt(1)*resizeRatio, 1);
        keypoints2.push_back(kp);
    }

    vector<DMatch> mD;
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] != -1)
        {
            //cout << "i=" << i << " matches[i]=" << matches[i] << endl;
            DMatch m(matches[i], i, distances[i]); //DMatch(query, train, distance), LEFT is train
            mD.push_back(m);
        }
    }

    resize(img1, img1, Size(0,0), resizeRatio, resizeRatio);
    resize(img2, img2, Size(0,0), resizeRatio, resizeRatio);

    vector<DMatch> mD2;
    vector<KeyPoint> k1, k2;
    cout << "Counter: " << counter << endl;
    for (int i = 0; i < mD.size(); i++)
    {
        mD2.clear();
        k1.clear();
        k2.clear();

        int tr = mD[i].trainIdx;
        int qu = mD[i].queryIdx;
        DMatch dm(0, 0, 1);
        mD2.push_back(dm);
        k1.push_back(keypoints1[tr]);
        k2.push_back(keypoints2[qu]);

        Mat imgMatches;
        drawMatches(img1, k1, img2, k2, mD2, imgMatches);
        imshow("Matches", imgMatches);

        /*cout << i << "/" << mD.size() << "  distance: " << mD[i].distance
             << "  size 1: " << featureVec1[tr].size << " size 2: "
             << featureVec2[qu].size << "  angle diff: "
             << abs(featureVec1[tr].angle-featureVec2[qu].angle) << endl;*/

        waitKey();
    }

}

void displayBruteForce_2 ()
{
    // testing function for bruteForce_2()

    float resizeRatio = 0.75;

    //stringstream sstm;

//    Mat img1 = imread("/home/bogdan/projects/icars/img_left/frame0000.jpg", 0);
//    Mat img2 = imread("/home/bogdan/projects/icars/img_left/frame0002.jpg", 0);
    Mat img1 = imread("/home/bogdan/projects/icars/img_difficult/l00.jpg", 0);
    Mat img2 = imread("/home/bogdan/projects/icars/img_difficult/r00.jpg", 0);
    vector<Feature> featureVec1, featureVec2;
    FeatureExtractor extr(500);

    extr.compute(img1, featureVec1);
    extr.compute(img2, featureVec2);

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();
    cout << endl << "N1=" << N1 << " N2=" << N2 << endl;

    vector<vector<int>> matches;

    Matcher matcher;
    matcher.bruteForce_2(featureVec1, featureVec2, matches);

    vector<KeyPoint> keypoints1;
    int nMatches = 0;
    for (int i = 0; i < N1; i++)
    {
        KeyPoint kp(featureVec1[i].pt(0)*resizeRatio, featureVec1[i].pt(1)*resizeRatio, 1);
        keypoints1.push_back(kp);
        if (matches[i].size() > 0) nMatches++;
    }
    cout << "Total number of matched features is " << nMatches << endl;
    vector<KeyPoint> keypoints2;
    for (int i = 0; i < N2; i++)
    {
        KeyPoint kp(featureVec2[i].pt(0)*resizeRatio, featureVec2[i].pt(1)*resizeRatio, 1);
        keypoints2.push_back(kp);
    }
    cout << N1 << " " << N2 << endl;
    resize(img1, img1, Size(0,0), resizeRatio, resizeRatio);
    resize(img2, img2, Size(0,0), resizeRatio, resizeRatio);

    for (int i = 0; i < N1; i++)
    {

        vector<DMatch> mD;
        for (int j = 0; j < matches[i].size(); j++)
        {
            cout << matches[i][j] << endl;
            if (matches[i][j] != -1)
            {
//                cout << "i = " << i;
//                cout << "; matches[i]=" << matches[i][j] << endl;
                DMatch m(matches[i][j], i, 0);
                mD.push_back(m);
            }
        }

        
        for (int j = 0; j < mD.size(); j++)
        {
            vector<DMatch> mD2;
        vector<KeyPoint> k1, k2;
            int tr = mD[j].trainIdx;
            int qu = mD[j].queryIdx;
            DMatch dm(k1.size(), k2.size(), 1);
            mD2.push_back(dm);
            k1.push_back(keypoints1[tr]);
            k2.push_back(keypoints2[qu]);


            

            double dist = computeDist(featureVec1[tr].desc, featureVec2[qu].desc);

            cout << "Displaying match " << j+1 << "/" << mD.size() << " for feature " << i
                 << "  distance: " << dist << " Size 1: " << featureVec1[tr].size
                 << "  Size 2: " << featureVec2[qu].size << endl ;
             

            const int deltaN = 2;
            const Descriptor & d1 = featureVec1[tr].desc, d2 = featureVec2[qu].desc;
            Matrix<float, N*N, 1> f(d1.data()), g(d2.data());
            Matrix<float, N*N, 1> err = g-f;
            for (unsigned int i = 0; i < N*N; i++)
            {
                if (abs(err[i]/(f[i]/250 + 1)) > 10) err[i] = 0;
                
            }
            Matrix<float, N*N, deltaN> df(d1.data() + N*N);
            Matrix<float, N*N, deltaN> dg(d2.data() + N*N);

            Matrix<float, N*N, deltaN> G = 0.5*(df + dg);
            Matrix<float, deltaN, 1> delta = (G.transpose()*G).inverse()*(G.transpose()*err);
             
            f += G*delta;     
            err = g-f;     
            Matrix<float, N, N> D1(d1.data());
            Matrix<float, N, N> D2(d2.data());
            
            Matrix<float, N, N> D3(f.data());
            
            Matrix<float, N, N> errSq(err.data());
             
            cout << "delta : " << endl << delta.transpose() << endl; 
            cout << "diff : " << endl << D2 - D1 << endl; 
            cout << "D1 : " << endl << D1 << endl;
            cout << "D2 : " << endl << D2 << endl;
            cout << "D3 : " << endl << D3 << endl;
            cout << "diff : " << endl << errSq << endl;
            Mat imgMatches;
            drawMatches(img1, k1, img2, k2, mD2, imgMatches, 255);
            imshow("Matches", imgMatches);
            
            int u1 = keypoints1[tr].pt.x;
            int v1 = keypoints1[tr].pt.y;
            int u2 = keypoints2[qu].pt.x;
            int v2 = keypoints2[qu].pt.y;
            
            Mat leftPatch;
            resize(img1(Rect(u1-8, v1-8, 16, 16)), leftPatch, Size(50, 50));
            Mat rightPatch;
            resize(img2(Rect(u2-8, v2-8, 16, 16)), rightPatch, Size(50, 50));
            imshow("left", leftPatch);
            imshow("right", rightPatch);
            waitKey();
        }
        
    }
}

int main(int argc, char** argv)
{
//    testBruteForce();

//    testStereoMatch();

//    testMatchReprojected();

    array<double, 6> paramsL{0.571, 1.180, 378.304, 377.960, 654.923, 474.835};
    array<double, 6> paramsR{0.570, 1.186, 377.262, 376.938, 659.914, 489.024};
    MeiCamera camL(1296, 966, paramsL.data());
    MeiCamera camR(1296, 966, paramsR.data());
    Transformation<double> tL(0, 0, 0, 0, 0, 0);
    Transformation<double> tR(0.788019, 0.00459233, -0.0203431, -0.00243736, 0.0859855, 0.000375454);
    StereoCartography cartograph(tL, tR, camL, camR);

    // displayBins(map.stereo);

//    displayStereoMatch(cartograph.stereo);
//   displayStereoMatch_2(cartograph.stereo);
//    displayBruteForce();
    displayBruteForce_2();

    // testDistancesBF();

    // displayBruteForce();

    return 0;
}
