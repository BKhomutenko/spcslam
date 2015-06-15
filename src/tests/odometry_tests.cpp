#include <fstream>
#include <sstream>
#include <ctime>

#include "matcher.h"
#include "cartography.h"
#include "mei.h"

using namespace std;

// display the projected points along with the original features (all at once)
void drawPoints(const vector<Eigen::Vector2d> & ptVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out)
{
    for (int i = 0; i < ptVec1.size(); i++)
    {
        cv::circle(out, cv::Point(ptVec1[i](0), ptVec1[i](1)), 5, cv::Scalar(0, 255, 0));
        cv::circle(out, cv::Point(ptVec2[i](0), ptVec2[i](1)), 10, cv::Scalar(0, 0, 255));
    }

}

void drawPoints(const vector<Feature> & fVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out)
{
    for (int i = 0; i < fVec1.size(); i++)
    {
      //  cv::circle(out, cv::Point(fVec1[i].pt(0), fVec1[i].pt(1)), 5, cv::Scalar(0, 255, 0));

    }
    for (int i = 0; i < ptVec2.size(); i++)
    {

        cv::circle(out, cv::Point(ptVec2[i](0), ptVec2[i](1)), 10, cv::Scalar(0, 0, 255));
    }
}

void drawPoints(const vector<Feature> & fVec1,
                const vector<Feature> & fVec2,
                cv::Mat & out, cv::Scalar color1, cv::Scalar color2)
{
    for (int i = 0; i < fVec1.size(); i++)
    {
        cv::circle(out, cv::Point(fVec1[i].pt(0), fVec1[i].pt(1)), 5, color1);

    }
    for (int i = 0; i < fVec2.size(); i++)
    {

        cv::circle(out, cv::Point(fVec2[i].pt(0), fVec2[i].pt(1)), 10, color2);
    }
}

void testOdometry()
{
    float resizeRatio = 1;

    //stringstream sstm;

    cv::Mat img1L = cv::imread("../datasets/dataset_odometry/view_left_0.png", 0);
    cv::Mat img1R = cv::imread("../datasets/dataset_odometry/view_right_0.png", 0);
    cv::Mat img2L = cv::imread("../datasets/dataset_odometry/view_left_1.png", 0);

    cv::resize(img1L, img1L, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(img1R, img1R, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(img2L, img2L, cv::Size(0,0), resizeRatio, resizeRatio);

    array<double, 6> paramsL{0.571, 1.180, 378.304, 377.960, 654.923, 474.835};
    array<double, 6> paramsR{0.570, 1.186, 377.262, 376.938, 659.914, 489.024};
    MeiCamera camL(1296, 966, paramsL.data());
    MeiCamera camR(1296, 966, paramsR.data());
    Transformation<double> tL(0, 0, 0, 0, 0, 0);
    Transformation<double> tR(0.788019, 0.00459233, -0.0203431, -0.00243736, 0.0859855, 0.000375454);
    StereoCartography map(tL, tR, camL, camR);

    vector<Feature> fVec1, fVec2, fVec3, fVec1m, fVec2m;

    Extractor extr(1000, 2, 2, false, true);
    extr(img1L, fVec1);
    extr(img1R, fVec2);
    extr(img2L, fVec3);
    const int N1 = fVec1.size();
    const int N2 = fVec2.size();
    const int N3 = fVec3.size();
    cout << endl << "N1=" << N1 << " N2=" << N2 << " N3=" << N3 << endl;

    vector<int> matches(N1, -1);

    map.matcher.stereoMatch(fVec1, fVec2, matches);

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
    map.stereo.reconstructPointCloud(ptVec1, ptVec2, PC);

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

        map.LM.push_back(L);
    }

    cout << "Number of matches= " << map.LM.size() << endl;

    Transformation<double> TfirstSecond, t0;

    map.trajectory.push_back(t0);

    vector<Eigen::Vector3d> PCsecond;

    for (int i = 0; i < 10; i++)
    {

        TfirstSecond = map.estimateOdometry(fVec3);

        TfirstSecond.inverseTransform(PC, PCsecond);

        //cout << endl << "Estimation completed" << endl;

        cout << endl << " Odometry:" << TfirstSecond << endl << endl;


        /*
        // save point cloud to text file
        ofstream myfile("cloud.txt");
        if (myfile.is_open())
        {
            for (int i = 0; i < map.LM.size(); i++)
            {
                myfile << map.LM[i].X << "\n";
            }
            myfile.close();
            cout << endl << "Point cloud saved to file" << endl;
        }
        */

        // reproject point cloud (first)
        vector<Eigen::Vector2d> pc1, pc2;
        map.stereo.projectPointCloud(PC, pc1, pc2);

        // reproject point cloud (second)
        vector<Eigen::Vector2d> pc1second, pc2second;
        map.stereo.projectPointCloud(PCsecond, pc1second, pc2second);


        /*
        // display the projected points along with the original features (one at the time)
        cv::Mat outL, outR;
        for (int i = 0; i < map.LM.size(); i++)
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

        /*drawPoints(ptVec1, pc1, img1L);
        drawPoints(fVec3, pc1second, img2L);

        cv::imshow("Left", img1L);
        cv::imshow("Right", img2L);

        cv::waitKey();*/
    }
}

// divide in functions initEmptyMap, addObservation, addLandmark
void initializeMap()
{

    int Nsteps = 130;
    int firstImage = 385;

    bool display = false;

    string datasetPath("/home/valerio/projects/datasets/dataset_odometry/");
    string prefix1("view_left_");
    string prefix2("view_right_");
    string extension(".png");

    array<double, 6> params1{0.571, 1.180, 378.304, 377.960, 654.923, 474.835};
    array<double, 6> params2{0.570, 1.186, 377.262, 376.938, 659.914, 489.024};
    MeiCamera cam1(1296, 966, params1.data());
    MeiCamera cam2(1296, 966, params2.data());
    Transformation<double> t1(0, 0, 0, 0, 0, 0);
    Transformation<double> t2(0.788019, 0.00459233, -0.0203431, -0.00243736, 0.0859855, 0.000375454);

    StereoCartography map(t1, t2, cam1, cam2);

    cv::Scalar color1(255), color2(255);

    clock_t begin, end, time1, time2, time3, time4;
    double dt;

    begin = clock();

    for (int i = 0; i < Nsteps; i++)
    {

        time1 = clock();
        // acquire images
        string imageFile1 = datasetPath + prefix1 + to_string(i + firstImage) + extension;
        string imageFile2 = datasetPath + prefix2 + to_string(i + firstImage) + extension;
        cv::Mat image1 = cv::imread(imageFile1, 0);
        cv::Mat image2 = cv::imread(imageFile2, 0);

        time4 = clock();
        dt = double(time4 - time1) / CLOCKS_PER_SEC;
        cout << endl << "i=" << i << "  Time to open images: " << dt << flush;

        // extract features
        vector<Feature> featuresVec1, featuresVec2,
                        featuresVecC1, featuresVecC2, featuresLM1, featuresLM2;
        map.extractor(image1, featuresVec1);
        map.extractor(image2, featuresVec2);

        time4 = clock();
        dt = double(time4 - time1) / CLOCKS_PER_SEC;
        cout << "   Time to extract features: " << dt << endl;

        int N1 = featuresVec1.size();
        int N2 = featuresVec2.size();

        if (display)
        {
            cv::Mat testIm1(image1);

            drawPoints(featuresVec1, featuresVec1, testIm1, color1, color1);
            imshow("Extracted features", testIm1);

            cout << endl << "wait 1" << endl;
            cv::waitKey();
        }

        vector<LandMark> tempLM;
        if (i == 0)
        {
            map.trajectory.push_back(Transformation<double>());
            featuresVecC1.swap(featuresVec1);
            featuresVecC2.swap(featuresVec2);
        }
        else
        {
            int N = map.LM.size();
            //cout << "map size " << N << endl;
            Transformation<double> newPose = map.estimateOdometry(featuresVec1);
            cout << endl << "i=" << i << "  Odometry: " << newPose << endl;
            map.trajectory.push_back(newPose);

            // project landmarks on image planes
            vector<Eigen::Vector3d> point3D, point3Daux;
            vector<Eigen::Vector2d> point2D1, point2D2;
            vector<Feature> featuresLM1, featuresLM2;
            vector<int> correspondence;

            for (int j = 0; j < N; j++)
            {
                point3Daux.push_back(map.LM[j].X);
            }
            map.trajectory[i].inverseTransform(point3Daux, point3Daux);
            for (int j = 0; j < N; j++)
            {
                if (point3Daux[j](2) > 0)
                {
                    point3D.push_back(map.LM[j].X);
                    correspondence.push_back(j);
                }
            }

            int Nprojected = point3D.size();
            map.projectPointCloud(point3D, point2D1, point2D2, i);

            for (int j = 0; j < Nprojected; j++)
            {
                Feature f1(point2D1[j], map.LM[correspondence[j]].d, 1, 1);
                Feature f2(point2D2[j], map.LM[correspondence[j]].d, 1, 1); //size and angle?
                featuresLM1.push_back(f1);
                featuresLM2.push_back(f2);
            }

            // match reprojections with extracted features
            vector<int> matchesR1, matchesR2;
            map.matcher.matchReprojected(featuresLM1, featuresVec1, matchesR1, 5);
            map.matcher.matchReprojected(featuresLM2, featuresVec2, matchesR2, 5);

            /*
            cout << matchesR1.size() << endl;
            int macc = 0;
            for (int j = 0; j < N; j++)
                if (matchesR1[j] != -1)
                    macc++;
            cout << endl << "MatchesR1: " << macc;

            macc = 0;
            for (int j = 0; j < N; j++)
                if (matchesR2[j] != -1)
                    macc++;
            cout << "   MatchesR2: " << macc << endl;
            */

            // update observations
            for(int j = 0; j < Nprojected; j++)
            {
                if (matchesR1[j] != -1)
                {
                    Observation o1(featuresVec1[matchesR1[j]].pt, i, CameraID::LEFT);
                    map.LM[correspondence[j]].observations.push_back(o1);
                    map.LM[correspondence[j]].d = featuresVec1[matchesR1[j]].desc;
                }
                if (matchesR2[j] != -1)
                {
                    Observation o2(featuresVec2[matchesR2[j]].pt, i, CameraID::RIGHT);
                    map.LM[correspondence[j]].observations.push_back(o2);
                }
            }

            // update landmark database
            int allowedGaps = 0;
            for (int j = 0; j < N; j++)
            {
                int nObs = map.LM[j].observations.size();
                if (nObs >= 6)
                {
                    tempLM.push_back(map.LM[j]);
                }
                else
                {
                    unsigned int lastOb = map.LM[j].observations[nObs-1].poseIdx;
                    if (lastOb >= i - allowedGaps)
                    {
                        tempLM.push_back(map.LM[j]);
                    }
                }
            }
            map.LM = tempLM;

            cout << "i=" << i << "  LM erased: " << N - tempLM.size();

            if (display)
            {
                drawPoints(featuresLM1, featuresLM1, image1, color1, color2);
                cv::imshow("Reprojections", image1);
                //cv::waitKey(500);
            }

            // create vectors of candidates for new landmarks
            vector<bool> candidates1(N1, true), candidates2(N2, true);
            for (int j = 0; j < Nprojected; j++)
            {
                if (matchesR1[j] != -1)
                {
                    candidates1[matchesR1[j]] = false;
                }

                if (matchesR2[j] != -1)
                {
                    candidates2[matchesR2[j]] = false;
                }
            }
            for (int j = 0; j < N1; j++)
            {
                if (candidates1[j] == true)
                {
                    featuresVecC1.push_back(featuresVec1[j]);
                }
            }
            for (int j = 0; j < N2; j++)
            {
                if (candidates2[j] == true)
                {
                    featuresVecC2.push_back(featuresVec2[j]);
                }
            }
        }

        vector<int> matches;
        map.matcher.stereoMatch_2(featuresVecC1, featuresVecC2, matches);

        // create vectors of 2D points and descriptors from matched features
        vector<Eigen::Vector2d> pointsVec1, pointsVec2;
        vector<Matrix<float,64,1> > descriptorsVec;
        for (int j = 0; j < featuresVecC1.size(); j++)
        {
            if (matches[j] != -1)
            {
                pointsVec1.push_back(featuresVecC1[j].pt);
                pointsVec2.push_back(featuresVecC2[matches[j]].pt);
                descriptorsVec.push_back(featuresVecC1[j].desc);
            }
        }

        // reconstruct point cloud (stereo frame)
        vector<Eigen::Vector3d> pointCloud;
        map.stereo.reconstructPointCloud(pointsVec1, pointsVec2, pointCloud);

        // transform to world frame
        map.trajectory[i].transform(pointCloud, pointCloud);

        //if (i == 0)
        {
            //create landmarks
            for (int j = 0; j < pointCloud.size(); j++)
            {
                Observation o1(pointsVec1[j], i, CameraID::LEFT);
                Observation o2(pointsVec2[j], i, CameraID::RIGHT);
                vector<Observation> oVec;
                oVec.push_back(o1);
                oVec.push_back(o2);

                LandMark L;
                L.X = pointCloud[j];
                L.observations = oVec;
                L.d = descriptorsVec[j];

                map.LM.push_back(L);
            }
        }

        cout << "   LM added: " << map.LM.size() - tempLM.size()
             << "   LM total: " << map.LM.size() << endl;

        time2 = clock();
        dt = double(time2 - time1) / CLOCKS_PER_SEC;
        cout << "i=" << i << "  LM time: " << dt << flush;

        if (i > 5) // and i % 2 == 0)
        {
            map.improveTheMap();
            time3 = clock();
            dt = double(time3 - time2) / CLOCKS_PER_SEC;
            cout << "   BA time: " << dt;
        }
        else
        {
            time3 = clock();
        }

        dt = double(time3 - time1) / CLOCKS_PER_SEC;
        cout << "   Total time: " << dt << endl;

        if (display)
        {
            cout << endl << "wait 2" << endl;
            cv::waitKey();
        }
    }

    int progressiveNum = 52;

    // save point cloud to text file
    ofstream myfile3("../../VM_shared/cloudInit_" + to_string(progressiveNum) + ".txt");
    if (myfile3.is_open())
    {
        for (int i = 0; i < map.LM.size(); i++)
        {
            myfile3 << map.LM[i].X << "\n";
        }
        myfile3.close();
        cout << endl << "Point cloud saved to file, dataset " << progressiveNum << endl;
    }

    // save trajectory to text file
    ofstream myfile4("../../VM_shared/trajectory_" + to_string(progressiveNum) + ".txt");
    if (myfile4.is_open())
    {
        for (int i = 0; i < map.trajectory.size(); i++)
        {
            myfile4 << map.trajectory[i].trans() << "\n";
        }
        myfile4.close();
        cout << "Trajectory saved to file, dataset " << progressiveNum << endl;
    }

    /*for (int i = 0; i < map.LM.size(); i++)
    {
        if (map.LM[i].observations.size() <= 5)
        {
            cout << i << endl;
        }
    }*/

    end = clock();
    dt = double(end - begin) / CLOCKS_PER_SEC;
    cout << endl << "DONE. Total time to complete: " << dt << endl;

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

    //testOdometry();

    initializeMap();

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


