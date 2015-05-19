#include <fstream>
#include <sstream>

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
    Matcher matcher;
    matcher.initStereoBins(map.stereo);
    matcher.stereoMatch(fVec1, fVec2, matches);

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

    TfirstSecond = map.estimateOdometry(fVec3);

    vector<Eigen::Vector3d> PCsecond;

    TfirstSecond.inverseTransform(PC, PCsecond);

    cout << endl << "Estimation completed" << endl;

    cout << endl << TfirstSecond << endl;

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

    cout << endl;

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

    drawPoints(ptVec1, pc1, img1L);
    drawPoints(fVec3, pc1second, img2L);

    cv::imshow("Left", img1L);
    cv::imshow("Right", img2L);

    cv::waitKey();
}

void initializeMap()
{

    int Nsteps = 5;

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

    map.matcher.initStereoBins(map.stereo);

/*    Transformation<double> TfirstSecond, t0;

    map.trajectory.push_back(t0);

    TfirstSecond = map.estimateOdometry(fVec3);

    vector<Eigen::Vector3d> PCsecond;

    TfirstSecond.inverseTransform(PC, PCsecond);

  */

    for (int i = 0; i < Nsteps; i++)
    {

        string imageFile1 = datasetPath + prefix1 + to_string(i) + extension;
        string imageFile2 = datasetPath + prefix2 + to_string(i) + extension;

        cv::Mat image1 = cv::imread(imageFile1, 0);
        cv::Mat image2 = cv::imread(imageFile2, 0);

        vector<Feature> featuresVec1, featuresVec2, matchedFeaturesVec1, matchedFeaturesVec2;
        map.extractor(image1, featuresVec1);
        map.extractor(image2, featuresVec2);

        Transformation<double> trajectorySegment;
        if (i > 0) trajectorySegment = map.estimateOdometry(featuresVec1);
        map.trajectory.push_back(trajectorySegment);

        vector<int> matches;
        map.matcher.stereoMatch(featuresVec1, featuresVec2, matches);

        // create vectors of 2D points and descriptors from matched features
        vector<Eigen::Vector2d> pointsVec1, pointsVec2;
        vector<Matrix<float,64,1> > descriptorsVec;
        for (int j = 0; j < featuresVec1.size(); j++)
        {
            if (matches[j] != -1)
            {
                pointsVec1.push_back(featuresVec1[j].pt);
                pointsVec2.push_back(featuresVec2[matches[j]].pt);
                descriptorsVec.push_back(featuresVec1[j].desc);
            }
        }

        // reconstruct point cloud
        vector<Eigen::Vector3d> pointCloud;
        map.stereo.reconstructPointCloud(pointsVec1, pointsVec2, pointCloud);

            //create landmarks
        /*for (int i = 0; i < PC.size(); i++)
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
*/
    }


    /*
    step iniziale:
        carica stereo pair 0, ricava i primi landmarks e inizializza trajectory con identity
    for (i)
    {
        carica stereo pair i
        usa immagine sx per calcolare l'odometry
        calcola stereo matches
        ricostruisci landmarks visibili (occhio al frame)
        riproietta landmarks in memoria sulle immagini correnti (occhio al frame)
        ricava due insiemi:
            - landmarks in memoria non confermati
            - nuovi landmarks
        elimina landmarks non confermati e aggiungi i nuovi
    }
    */
}


int main()
{

    testOdometry();

    //initializeMap();

}
