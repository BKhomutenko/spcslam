#include <fstream>
#include <sstream>
#include <ctime>

#include "matcher.h"
#include "cartography.h"
#include "mei.h"
#include "utils.h"

using namespace std;

void testOdometry()
{
    float resizeRatio = 1;

    //stringstream sstm;

    cv::Mat img1L = cv::imread("/home/bogdan/projects/icars/img_difficult/l00.jpg", 0);
    cv::Mat img1R = cv::imread("/home/bogdan/projects/icars/img_difficult/r00.jpg", 0);
    cv::Mat img2L = cv::imread("/home/bogdan/projects/icars/img_difficult/l02.jpg", 0);

    cv::resize(img1L, img1L, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(img1R, img1R, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(img2L, img2L, cv::Size(0,0), resizeRatio, resizeRatio);

    array<double, 6> paramsL{0.571, 1.180, 378.304, 377.960, 654.923, 474.835};
    array<double, 6> paramsR{0.570, 1.186, 377.262, 376.938, 659.914, 489.024};
    MeiCamera camL(1296, 966, paramsL.data());
    MeiCamera camR(1296, 966, paramsR.data());
    Transformation<double> tL(0, 0, 0, 0, 0, 0);
    Transformation<double> tR(0.788019, 0.00459233, -0.0203431, -0.00243736, 0.0859855, 0.000375454);
    StereoCartography cartograph(tL, tR, camL, camR);

    vector<Feature> fVec1, fVec2, fVec3, fVec1m, fVec2m;

    FeatureExtractor extr(500);
    extr.compute(img1L, fVec1);
    extr.compute(img1R, fVec2);
    extr.compute(img2L, fVec3);
    const int N1 = fVec1.size();
    const int N2 = fVec2.size();
    const int N3 = fVec3.size();
    cout << endl << "N1=" << N1 << " N2=" << N2 << " N3=" << N3 << endl;

    vector<int> matches(N1, -1);

    cartograph.matcher.stereoMatch(fVec1, fVec2, matches);

    //vector<Eigen::Vector2d> vec1, vec2;
    //vector<Eigen::Vector3d> vecR;

    // create vectors of matched features and 2D points from matched features
    vector<Eigen::Vector2d> ptVec1, ptVec2;
    for (int i = 0; i < N1; i++)
    {
        if (matches[i] != -1)
        {
            ptVec1.push_back(fVec1[i].pt);
            ptVec2.push_back(fVec2[matches[i]].pt);
            fVec1m.push_back(fVec1[i]);
            fVec2m.push_back(fVec2[matches[i]]);
        }
    }

    // reconstruct point cloud
    vector<Eigen::Vector3d> PC;
    cartograph.stereo.reconstructPointCloud(ptVec1, ptVec2, PC);

    //create landmarks
    for (int i = 0; i < PC.size(); i++)
    {
        Observation o1(fVec1m[i].pt, 0, CameraID::LEFT);
        Observation o2(fVec2m[i].pt, 0, CameraID::RIGHT);
        vector<Observation> oVec;
        oVec.push_back(o1);
        oVec.push_back(o2);

        LandMark L;
        L.X = PC[i];
        L.observations = oVec;
        L.d = fVec1m[i].desc;

        cartograph.WM.push_back(L);
    }

    cout << "Number of matches= " << cartograph.WM.size() << endl;

    Transformation<double> TfirstSecond, t0;

    cartograph.trajectory.push_back(t0);

    vector<Eigen::Vector3d> PCsecond;

    for (int i = 0; i < 10; i++)
    {

        TfirstSecond = cartograph.estimateOdometry_3(fVec3);

        TfirstSecond.inverseTransform(PC, PCsecond);

        vector<Feature> featureVec;
        
       
                          
        //cout << endl << "Estimation completed" << endl;

        cout << endl << " Odometry: " << TfirstSecond << endl << endl;


        /*
        // save point cloud to text file
        ofstream myfile("cloud.txt");
        if (myfile.is_open())
        {
            for (int i = 0; i < cartograph.LM.size(); i++)
            {
                myfile << cartograph.LM[i].X << "\n";
            }
            myfile.close();
            cout << endl << "Point cloud saved to file" << endl;
        }
        */

        // reproject point cloud (first)
        vector<Eigen::Vector2d> pc1, pc2;
        cartograph.stereo.projectPointCloud(PC, pc1, pc2);

        // reproject point cloud (second)
        vector<Eigen::Vector2d> pc1second, pc2second;
        cartograph.stereo.projectPointCloud(PCsecond, pc1second, pc2second);

        for (unsigned int i = 0; i < pc1second.size(); i++)
        {
            featureVec.push_back(Feature(pc1second[i], cartograph.WM[i].d));
        }
        
        vector<int> matches;
        cartograph.matcher.matchReprojected(featureVec, fVec3, matches, 3);

        pc2second.clear();
        for (unsigned int i = 0; i < featureVec.size(); i++)
        {
            if (matches[i] != -1)
            {
                pc2second.push_back(featureVec[i].pt);            
            }
        }
        /*
        // display the projected points along with the original features (one at the time)
        cv::Mat outL, outR;
        for (int i = 0; i < cartograph.WM.size(); i++)
        {
            vector<cv::KeyPoint> k;
            k.push_back(cv::KeyPoint(ptVec1[i](0), ptVec1[i](1), 5));
            cv::drawKeypoints(img1L, k, outL, cv::Scalar(0, 255, 0),
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            k.clear();
            k.push_back(cv::KeyPoint(pc1[i](0), pc1[i](1), 10));
            cv::drawKeypoints(outL, k, outL, cv::Scalar(0, 0, 255),
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            k.clear();
            k.push_back(cv::KeyPoint(ptVec2[i](0), ptVec2[i](1), 5));
            cv::drawKeypoints(img1R, k, outR, cv::Scalar(0, 255, 0),
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            k.clear();
            k.push_back(cv::KeyPoint(pc2[i](0), pc2[i](1), 10));
            cv::drawKeypoints(outR, k, outR, cv::Scalar(0, 0, 255),
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            cv::imshow("Left", outL);
            cv::imshow("Right", outR);

            cv::waitKey();
        }
        */
        cv::Mat img1out, img2out;
        cv::cvtColor(img1L, img1out, CV_GRAY2BGR);
        cv::cvtColor(img2L, img2out, CV_GRAY2BGR);
        drawPoints(ptVec1, ptVec2, img1out);
        drawPoints(fVec3, pc1second, img2out);
        cv::imshow("Initial", img1out);
        cv::imshow("Following", img2out);

        cv::waitKey();
    }
}


/* 5: erase on, new landmarks at each step
 * 6: erase off, new landmarks at first step
 * 7: erase off, new landmarks at each step
 *
 *
 *
 */

int main()
{

    testOdometry();

//    initializeMap();

    cout << endl;

}


/*
int main( int argc, char** argv )
{


  cv::Mat img_1 = cv::imread("../datasets/dataset_odometry/view_left_0.png", 0);
  cv::resize(img_1, img_1, cv::Size(0,0), 0.5, 0.5);

  int minHessian = 1000;

  cv::SurfFeatureDetector detector(1000, 2, 2, false, true);

  vector<cv::KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );

  //-- Draw keypoints
  cv::Mat img_keypoints_1;

  cv::drawKeypoints( img_1, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  cv::imshow("Keypoints 1", img_keypoints_1 );

  cout << keypoints_1.size() << endl;

  cv::waitKey(0);

  return 0;
}
*/


