#include <fstream>
#include <sstream>
#include <ctime>

#include "cartography.h"
#include "mei.h"
#include "utils.h"

using namespace std;

bool extractorDebug = false;
bool odometryDebug = false;
bool saveDebugImages = false;
bool displayResults = false;
string debugId = "3";

extern vector<Eigen::Vector3d> oD_modelLM, oD_inlierLM, oD_outlierLM;
extern vector<Eigen::Vector2d> oD_inlierFeat, oD_outlierFeat;

void projectLandmarks(StereoCartography & cartograph, vector<Feature> & featuresLM1,
                      vector<Feature> & featuresLM2, vector<int> & indexList, int & nSTMprojected)
{
    int step = cartograph.trajectory.size() - 1;
    int nSTM = cartograph.STM.size();
    int nWM = cartograph.WM.size();
    vector<Eigen::Vector3d> point3D, point3Daux;
    vector<Eigen::Vector2d> point2D1, point2D2;

    // add 3D points for projection - STM
    for (int j = 0; j < nSTM; j++)
    {
        point3Daux.push_back(cartograph.STM[j].X);
    }
    cartograph.trajectory[step].inverseTransform(point3Daux, point3Daux);
    for (int j = 0; j < nSTM; j++)
    {
        if (point3Daux[j](2) > 0.5)
        {
            point3D.push_back(cartograph.STM[j].X);
            indexList.push_back(j);
        }
    }
    nSTMprojected = point3D.size();

    // add 3D points for projection - WM
    point3Daux.clear();
    for (int j = 0; j < nWM; j++)
    {
        point3Daux.push_back(cartograph.WM[j].X);
    }
    cartograph.trajectory[step].inverseTransform(point3Daux, point3Daux);
    for (int j = 0; j < nWM; j++)
    {
        if (point3Daux[j](2) > 0.5)
        {
            point3D.push_back(cartograph.WM[j].X);
            indexList.push_back(j);
        }
    }

    // project points
    cartograph.projectPointCloud(point3D, point2D1, point2D2, step);

    // create features from projected points - STM
    for (int j = 0; j < nSTMprojected; j++)
    {
        Feature f1(point2D1[j], cartograph.STM[indexList[j]].d, 1, 1);
        Feature f2(point2D2[j], cartograph.STM[indexList[j]].d, 1, 1); //size and angle?
        featuresLM1.push_back(f1);
        featuresLM2.push_back(f2);
    }

    // create features from projected points - WM
    for (int j = nSTMprojected; j < point3D.size(); j++)
    {
        Feature f1(point2D1[j], cartograph.WM[indexList[j]].d, 1, 1);
        Feature f2(point2D2[j], cartograph.WM[indexList[j]].d, 1, 1); //size and angle?
        featuresLM1.push_back(f1);
        featuresLM2.push_back(f2);
    }
}

void updateObservations(StereoCartography & cartograph, vector<Feature> & featuresVec1,
                        vector<Feature> & featuresVec2, vector<int> & matchesR1,
                        vector<int> & matchesR2, vector<int> & indexList, int & nSTMprojected)
{
    int step = cartograph.trajectory.size() - 1;
    // update observations - STM
    for(int j = 0; j < nSTMprojected; j++)
    {
        if (matchesR1[j] != -1)
        {
            Observation o1(featuresVec1[matchesR1[j]].pt, step, CameraID::LEFT);
            cartograph.STM[indexList[j]].observations.push_back(o1);
            cartograph.STM[indexList[j]].d = featuresVec1[matchesR1[j]].desc;
        }
        if (matchesR2[j] != -1)
        {
            Observation o2(featuresVec2[matchesR2[j]].pt, step, CameraID::RIGHT);
            cartograph.STM[indexList[j]].observations.push_back(o2);
        }
    }

    // update observations - WM
    for(int j = nSTMprojected; j < matchesR1.size(); j++)
    {
        if (matchesR1[j] != -1)
        {
            Observation o1(featuresVec1[matchesR1[j]].pt, step, CameraID::LEFT);
            cartograph.WM[indexList[j]].observations.push_back(o1);
            cartograph.WM[indexList[j]].d = featuresVec1[matchesR1[j]].desc;
        }
        if (matchesR2[j] != -1)
        {
            Observation o2(featuresVec2[matchesR2[j]].pt, step, CameraID::RIGHT);
            cartograph.WM[indexList[j]].observations.push_back(o2);
        }
    }
}

void firstMemoryUpdate(StereoCartography & cartograph, int allowedGapSize, int nObsMin)
{
    vector<LandMark> tempSTM, toWM;
    int nSTM = cartograph.STM.size();
    int nWM = cartograph.WM.size();
    int step = cartograph.trajectory.size() - 1;

    for (int j = 0; j < nSTM; j++)
    {
        int nObs = cartograph.STM[j].observations.size();
        if (nObs >= nObsMin)
        {
            toWM.push_back(cartograph.STM[j]);
        }
        else
        {
            int lastOb = cartograph.STM[j].observations.back().poseIdx;
            if (lastOb >= step - allowedGapSize)
            {
                tempSTM.push_back(cartograph.STM[j]);
            }
        }
    }
    cartograph.STM = tempSTM;

    // update memory: WM, LTM
    vector<LandMark> tempWM;
    for (int j = 0; j < nWM; j++)
    {
        int lastOb = cartograph.WM[j].observations.back().poseIdx;
        if (lastOb >= step - allowedGapSize - 5)
        {
            tempWM.push_back(cartograph.WM[j]);
        }
        else
        {
            cartograph.LTM.push_back(cartograph.WM[j]);
        }
    }
    cartograph.WM = tempWM;
    cartograph.WM.insert(cartograph.WM.end(), toWM.begin(), toWM.end());
}

void findCandidates(StereoCartography & cartograph, vector<Feature> & featuresVec1,
                    vector<Feature> & featuresVec2, vector<int> & matchesR1, vector<int> & matchesR2,
                    vector<Feature> & featuresVecC1, vector<Feature> & featuresVecC2)
{
    int N1 = featuresVec1.size();
    int N2 = featuresVec2.size();
    vector<bool> candidates1(N1, true), candidates2(N2, true);

    for (int j = 0; j < matchesR1.size(); j++)
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

void secondMemoryUpdate(vector<Feature> & featuresVecC1, vector<Feature> & featuresVecC2,
                        StereoCartography & cartograph)
{
    int step = cartograph.trajectory.size() - 1;
    vector<int> matches;
    cartograph.matcher.stereoMatch_2(featuresVecC1, featuresVecC2, matches);

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
    cartograph.stereo.reconstructPointCloud(pointsVec1, pointsVec2, pointCloud);

    // transform to world frame
    cartograph.trajectory[step].transform(pointCloud, pointCloud);

    // create landmarks and update STM (2)
    for (int j = 0; j < pointCloud.size(); j++)
    {
        Observation o1(pointsVec1[j], step, CameraID::LEFT);
        Observation o2(pointsVec2[j], step, CameraID::RIGHT);
        vector<Observation> oVec;
        oVec.push_back(o1);
        oVec.push_back(o2);

        LandMark L;
        L.X = pointCloud[j];
        L.observations = oVec;
        L.d = descriptorsVec[j];

        cartograph.STM.push_back(L);
    }
}

void saveData(StereoCartography & cartograph,  int progressiveNum)
{
    ofstream myfile1("../../VM_shared/cloud_" + to_string(progressiveNum) + ".txt");
    if (myfile1.is_open())
    {
        for (int i = 0; i < cartograph.LTM.size(); i++)
        {
            myfile1 << cartograph.LTM[i].X << "\n";
        }
        for (int i = 0; i < cartograph.WM.size(); i++)
        {
            myfile1 << cartograph.WM[i].X << "\n";
        }
        myfile1.close();
        cout << endl << endl << "Map saved to file, number: " << progressiveNum << endl;
    }

    // save trajectory to text file
    ofstream myfile2("../../VM_shared/trajectory_" + to_string(progressiveNum) + ".txt");
    if (myfile2.is_open())
    {
        for (int i = 0; i < cartograph.trajectory.size(); i++)
        {
            myfile2 << cartograph.trajectory[i].trans() << "\n";
        }
        myfile2.close();
        cout << "Trajectory saved to file, number: " << progressiveNum << endl;
    }
}

void displayOdometryDebug(StereoCartography & cartograph, int step, int firstImage,
        cv::Mat & image1_new, cv::Mat & image1_ex)
{
    double resizeRatio = 1;
    cv::Scalar orange(  0, 150, 250);
    cv::Scalar yellow(  0, 255, 255);
    cv::Scalar red   (  0,   0, 255);
    cv::Scalar green (  0, 255,   0);
    cv::Scalar blue  (255,   0,   0);
    cv::Scalar purple(255,   0, 255);

    vector<Eigen::Vector2d> oD_modelLMpt, oD_inlierLMpt, oD_outlierLMpt, aux,
                            oD_modelLMpt_ex, oD_inlierLMpt_ex, oD_outlierLMpt_ex;
    cartograph.projectPointCloud(oD_modelLM, oD_modelLMpt, aux, step);
    cartograph.projectPointCloud(oD_modelLM, oD_modelLMpt_ex, aux, step - 1);
    cartograph.projectPointCloud(oD_inlierLM, oD_inlierLMpt, aux, step);
    cartograph.projectPointCloud(oD_inlierLM, oD_inlierLMpt_ex, aux, step - 1);
    cartograph.projectPointCloud(oD_outlierLM, oD_outlierLMpt, aux, step);
    cartograph.projectPointCloud(oD_outlierLM, oD_outlierLMpt_ex, aux, step - 1);

    cv::resize(image1_new, image1_new, cv::Size(0,0), resizeRatio, resizeRatio);
    cv::resize(image1_ex, image1_ex, cv::Size(0,0), resizeRatio, resizeRatio);

    cv::cvtColor(image1_new, image1_new, CV_GRAY2BGR);
    cv::cvtColor(image1_ex, image1_ex, CV_GRAY2BGR);

    drawCircles(oD_inlierFeat, image1_new, blue, resizeRatio, 5, 1);
    drawCircles(oD_outlierFeat, image1_new, purple, resizeRatio, 5, 1);

    drawCrosses(oD_inlierLMpt, image1_new, green, resizeRatio, 10, 1);
    drawCrosses(oD_outlierLMpt, image1_new, red, resizeRatio, 10, 1);
    drawCrosses(oD_modelLMpt, image1_new, orange, resizeRatio, 10, 1);

    drawCrosses(oD_inlierLMpt_ex, image1_ex, green, resizeRatio, 10, 1);
    drawCrosses(oD_outlierLMpt_ex, image1_ex, red, resizeRatio, 10, 1);
    drawCrosses(oD_modelLMpt_ex, image1_ex, orange, resizeRatio, 10, 1);

    int cSize = oD_modelLM.size() + oD_inlierLM.size() + oD_outlierLM.size();
    double iP = double((oD_modelLM.size() + oD_inlierLM.size())/(double)cSize)*100.0;
    cout << endl << endl << "# Odometry debug #  Cloud size = " << cSize << "  inliers = " << iP << "%";

    if (saveDebugImages)
    {
        string imageFile_new = "/media/valerio/Dati/Progetti/Tesi_Emaro/debug/debug_"
                + debugId + "/debug_" + to_string(firstImage + step) + "_2.png";
        string imageFile_ex = "/media/valerio/Dati/Progetti/Tesi_Emaro/debug/debug_" 
                + debugId + "/debug_" + to_string(firstImage + step) + "_1.png";
        imwrite(imageFile_new, image1_new);
        imwrite(imageFile_ex, image1_ex);
    }
    else
    {
        imshow("Odometry debug", image1_new);
        imshow("Odometry debug ex", image1_ex);
        cv::waitKey();
    }
}



int main()
{
    int Nsteps = 200;
    int firstImage = 400;
    int firstStepBA = 2;
    int odometryType = 2;
    int allowedGapSize = 2;
    int nObsMin = 6;
    int progressiveNum = 40;

    cout.precision(4);

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

    Extractor extractor(2000, 2, 2, false, true);
    extractor.setType(FeatureType::SURF);
    
    StereoCartography cartograph(t1, t2, cam1, cam2);

    cv::Scalar color1(255), color2(255);

    clock_t begin, end, time1, time2, time3, time4;
    double dt1, dt2, dt3, dt4, dt5, dt6;

    begin = clock();

    for (int step = 0; step < Nsteps; step++)
    {
        time1 = clock();

        // acquire images
        string imageFile1 = datasetPath + prefix1 + to_string(firstImage + step) + extension;
        string imageFile2 = datasetPath + prefix2 + to_string(firstImage + step) + extension;
        cv::Mat image1 = cv::imread(imageFile1, 0);
        cv::Mat image2 = cv::imread(imageFile2, 0);

        time4 = clock();
        dt1 = double(time4 - time1) / CLOCKS_PER_SEC;

        vector<Feature> featuresVec1, featuresVec2,
                        featuresVecC1, featuresVecC2,
                        featuresLM1, featuresLM2;

        // extract features
        extractor(image1, featuresVec1, 1);
        extractor(image2, featuresVec2, 2);

        time4 = clock();
        dt2 = double(time4 - time1) / CLOCKS_PER_SEC;

        int N1 = featuresVec1.size();
        int N2 = featuresVec2.size();

        if (step == 0)
        {
            cartograph.trajectory.push_back(Transformation<double>());
            //featuresVecC1.swap(featuresVec1);
            //featuresVecC2.swap(featuresVec2);
            featuresVecC1.resize(N1);
            featuresVecC2.resize(N2);
            copy(featuresVec1.begin(), featuresVec1.end(), featuresVecC1.begin());
            copy(featuresVec2.begin(), featuresVec2.end(), featuresVecC2.begin());
        }
        else
        {
            // compute odometry
            Transformation<double> newPose;
            switch (odometryType)
            {
                case 1:
                {
                    Transformation<double> tn = cartograph.estimateOdometry(featuresVec1);
                    newPose.setParam(tn.trans(), tn.rot());
                }
                case 2:
                {
                    Transformation<double> tn = cartograph.estimateOdometry_2(featuresVec1);
                    newPose.setParam(tn.trans(), tn.rot());
                }
                case 3:
                {
                    Transformation<double> tn = cartograph.estimateOdometry_3(featuresVec1);
                    newPose.setParam(tn.trans(), tn.rot());
                }
            }
            cartograph.trajectory.push_back(newPose);

            if (odometryDebug)
            {
                string imageFile1_ex = datasetPath + prefix1 + to_string(firstImage + step - 1) + extension;
                cv::Mat image1_ex = cv::imread(imageFile1_ex, 0);
                cv::Mat image1_new = cv::imread(imageFile1, 0);
                displayOdometryDebug(cartograph, step, firstImage, image1_new, image1_ex);
            }

            // create reprojections of landmarks
            vector<Feature> featuresLM1, featuresLM2;
            vector<int> indexList;
            int nSTMprojected;
            projectLandmarks(cartograph, featuresLM1, featuresLM2, indexList, nSTMprojected);

            // match reprojections with extracted features
            vector<int> matchesR1, matchesR2;
            cartograph.matcher.matchReprojected(featuresLM1, featuresVec1, matchesR1, 2.5);
            cartograph.matcher.matchReprojected(featuresLM2, featuresVec2, matchesR2, 2.5);

            // update observations
            updateObservations(cartograph, featuresVec1, featuresVec2, matchesR1,
                               matchesR2, indexList, nSTMprojected);

            // manage landmarks that are already in memory
            firstMemoryUpdate(cartograph, allowedGapSize, nObsMin);

            // create vectors of candidates for new landmarks
            findCandidates(cartograph, featuresVec1, featuresVec2, matchesR1, matchesR2,
                           featuresVecC1, featuresVecC2);
        }

        // create new landmarks and add them to STM
        secondMemoryUpdate(featuresVecC1, featuresVecC2, cartograph);

        time2 = clock();
        dt3 = double(time2 - time4) / CLOCKS_PER_SEC;
        cout << "\n\nstep: " << step << "  Odometry: " << cartograph.trajectory.back()
             << "\nstep: " << step << "  STM: " << cartograph.STM.size()  << "   WM: "
             << cartograph.WM.size()  << "   LTM: " << cartograph.LTM.size()
             << "\nstep: " << step << "  Features left: " << featuresVec1.size()
             << "   Features right: " << featuresVec2.size()
             << "\nstep: " << step << "  Images time: " << dt1
             << "   Features time: " << dt2
             << "\nstep: " << step << "  LM time: " << dt3 << flush;

        if (step >= firstStepBA)
        {
            bool firstBA = (step == firstStepBA);
            // perform bundle adjustment
            cartograph.improveTheMap(firstBA);
            time3 = clock();
            dt4 = double(time3 - time2) / CLOCKS_PER_SEC;
            cout << "   BA time: " << dt4;
        }
        else
        {
            time3 = clock();
        }
        dt5 = double(time3 - time1) / CLOCKS_PER_SEC;
        cout << "   Total time: " << dt5 << flush;

        if (extractorDebug)
        {
            double resizeRatio = 1;
            cv::resize(image1, image1, cv::Size(0,0), resizeRatio, resizeRatio);
            cv::cvtColor(image1, image1, CV_GRAY2BGR);
            for (int j = 0; j < featuresVec1.size(); j++)
            {
                cv::circle(image1, cv::Point(featuresVec1[j].pt(0),
                    featuresVec1[j].pt(1))*resizeRatio, 6, cv::Scalar(255, 0, 0), 1);
            }
            if (saveDebugImages)
            {
                string imageFile = "/media/valerio/Dati/Progetti/Tesi_Emaro/debug/debug_" 
                        + debugId + "/debug_" + to_string(firstImage + step) + "_0.png";
                cv::imwrite(imageFile, image1);
            }
            else
            {
                imshow("image1", image1);
                cv::waitKey(100);
            }
        }
    }

    saveData(cartograph, progressiveNum);

    end = clock();
    dt6 = double(end - begin) / CLOCKS_PER_SEC;
    cout << endl << "DONE. Total time to complete: " << dt6 << endl << endl;

    if (displayResults)
    {
        float resizeRatio = 0.5;

        for (int step = 0; step < Nsteps; step++)
        {

            string imageFile1 = datasetPath + prefix1 + to_string(firstImage + step) + extension;
            string imageFile2 = datasetPath + prefix2 + to_string(firstImage + step) + extension;
            cv::Mat image1 = cv::imread(imageFile1, 0);
            cv::Mat image2 = cv::imread(imageFile2, 0);

            cv::resize(image1, image1, cv::Size(0,0), resizeRatio, resizeRatio);
            cv::resize(image2, image2, cv::Size(0,0), resizeRatio, resizeRatio);

            vector<Eigen::Vector3d> Point3D;
            vector<Eigen::Vector2d> obs1, obs2;
            for (int j = 0; j < cartograph.LTM.size(); j++)
            {
                bool added = false;
                for (int k = 0; k < cartograph.LTM[j].observations.size(); k++)
                {
                    if (cartograph.LTM[j].observations[k].poseIdx == step)
                    {
                        if (added == false)
                        {
                            Point3D.push_back(cartograph.LTM[j].X);
                            added = true;
                        }
                        if (cartograph.LTM[j].observations[k].cameraId == CameraID::LEFT)
                        {
                            obs1.push_back(cartograph.LTM[j].observations[k].pt);
                        }
                        if (cartograph.LTM[j].observations[k].cameraId == CameraID::RIGHT)
                        {
                            obs2.push_back(cartograph.LTM[j].observations[k].pt);
                        }
                    }
                }
            }

            vector<Eigen::Vector2d> repro1, repro2;
            cartograph.projectPointCloud(Point3D, repro1, repro2, step);

            cv::cvtColor(image1, image1, CV_GRAY2BGR);
            cv::cvtColor(image2, image2, CV_GRAY2BGR);

            // draw observations

            for (int j = 0; j < obs1.size(); j++)
            {
                cv::circle(image1, cv::Point(obs1[j](0), obs1[j](1))*resizeRatio,
                           6, cv::Scalar(255, 0, 0), 2);
            }
            for (int j = 0; j < obs2.size(); j++)
            {
                cv::circle(image2, cv::Point(obs2[j](0), obs2[j](1))*resizeRatio,
                           6, cv::Scalar(255, 0, 0), 2);
            }
            for (int j = 0; j < Point3D.size(); j++)
            {
                cv::circle(image1, cv::Point(repro1[j](0), repro1[j](1))*resizeRatio,
                           3, cv::Scalar(0, 255, 0), 2);
                cv::circle(image2, cv::Point(repro2[j](0), repro2[j](1))*resizeRatio,
                           3, cv::Scalar(0, 255, 0), 2);
            }

            cv::imshow("Camera 1", image1);
            cv::imshow("Camera 2", image2);

            cv::waitKey();

        }
    }
}
