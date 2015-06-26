#include <fstream>
#include <sstream>
#include <ctime>

#include "cartography.h"
#include "mei.h"
#include "utils.h"

using namespace std;

void projectLandmarks(StereoCartography & map, vector<Feature> & featuresLM1,
                      vector<Feature> & featuresLM2, vector<int> & indexList, int & nSTMprojected)
{
    int step = map.trajectory.size() - 1;
    int nSTM = map.STM.size();
    int nWM = map.WM.size();
    vector<Eigen::Vector3d> point3D, point3Daux;
    vector<Eigen::Vector2d> point2D1, point2D2;

    // add 3D points for projection - STM
    for (int j = 0; j < nSTM; j++)
    {
        point3Daux.push_back(map.STM[j].X);
    }
    map.trajectory[step].inverseTransform(point3Daux, point3Daux);
    for (int j = 0; j < nSTM; j++)
    {
        if (point3Daux[j](2) > 0.5)
        {
            point3D.push_back(map.STM[j].X);
            indexList.push_back(j);
        }
    }
    nSTMprojected = point3D.size();

    // add 3D points for projection - WM
    point3Daux.clear();
    for (int j = 0; j < nWM; j++)
    {
        point3Daux.push_back(map.WM[j].X);
    }
    map.trajectory[step].inverseTransform(point3Daux, point3Daux);
    for (int j = 0; j < nWM; j++)
    {
        if (point3Daux[j](2) > 0.5)
        {
            point3D.push_back(map.WM[j].X);
            indexList.push_back(j);
        }
    }

    // project points
    map.projectPointCloud(point3D, point2D1, point2D2, step);

    // create features from projected points - STM
    for (int j = 0; j < nSTMprojected; j++)
    {
        Feature f1(point2D1[j], map.STM[indexList[j]].d, 1, 1);
        Feature f2(point2D2[j], map.STM[indexList[j]].d, 1, 1); //size and angle?
        featuresLM1.push_back(f1);
        featuresLM2.push_back(f2);
    }

    // create features from projected points - WM
    for (int j = nSTMprojected; j < point3D.size(); j++)
    {
        Feature f1(point2D1[j], map.WM[indexList[j]].d, 1, 1);
        Feature f2(point2D2[j], map.WM[indexList[j]].d, 1, 1); //size and angle?
        featuresLM1.push_back(f1);
        featuresLM2.push_back(f2);
    }
}

void updateObservations(StereoCartography & map, vector<Feature> & featuresVec1,
                        vector<Feature> & featuresVec2, vector<int> & matchesR1,
                        vector<int> & matchesR2, vector<int> & indexList, int & nSTMprojected)
{
    int step = map.trajectory.size() - 1;
    // update observations - STM
    for(int j = 0; j < nSTMprojected; j++)
    {
        if (matchesR1[j] != -1)
        {
            Observation o1(featuresVec1[matchesR1[j]].pt, step, CameraID::LEFT);
            map.STM[indexList[j]].observations.push_back(o1);
            map.STM[indexList[j]].d = featuresVec1[matchesR1[j]].desc;
        }
        if (matchesR2[j] != -1)
        {
            Observation o2(featuresVec2[matchesR2[j]].pt, step, CameraID::RIGHT);
            map.STM[indexList[j]].observations.push_back(o2);
        }
    }

    // update observations - WM
    for(int j = nSTMprojected; j < matchesR1.size(); j++)
    {
        if (matchesR1[j] != -1)
        {
            Observation o1(featuresVec1[matchesR1[j]].pt, step, CameraID::LEFT);
            map.WM[indexList[j]].observations.push_back(o1);
            map.WM[indexList[j]].d = featuresVec1[matchesR1[j]].desc;
        }
        if (matchesR2[j] != -1)
        {
            Observation o2(featuresVec2[matchesR2[j]].pt, step, CameraID::RIGHT);
            map.WM[indexList[j]].observations.push_back(o2);
        }
    }
}

void firstMemoryUpdate(StereoCartography & map, int allowedGapSize, int nObsMin)
{
    vector<LandMark> tempSTM, toWM;
    int nSTM = map.STM.size();
    int nWM = map.WM.size();
    int step = map.trajectory.size() - 1;

    for (int j = 0; j < nSTM; j++)
    {
        int nObs = map.STM[j].observations.size();
        if (nObs >= nObsMin)
        {
            toWM.push_back(map.STM[j]);
        }
        else
        {
            int lastOb = map.STM[j].observations.back().poseIdx;
            if (lastOb >= step - allowedGapSize)
            {
                tempSTM.push_back(map.STM[j]);
            }
        }
    }
    map.STM = tempSTM;

    // update memory: WM, LTM
    vector<LandMark> tempWM;
    for (int j = 0; j < nWM; j++)
    {
        int lastOb = map.WM[j].observations.back().poseIdx;
        if (lastOb >= step - allowedGapSize)
        {
            tempWM.push_back(map.WM[j]);
        }
        else
        {
            map.LTM.push_back(map.WM[j]);
        }
    }
    map.WM = tempWM;
    map.WM.insert(map.WM.end(), toWM.begin(), toWM.end());
}

void findCandidates(StereoCartography & map, vector<Feature> & featuresVec1,
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
                        StereoCartography & map)
{
    int step = map.trajectory.size() - 1;
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
    map.trajectory[step].transform(pointCloud, pointCloud);

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

        map.STM.push_back(L);
    }
}

void saveData(StereoCartography & map,  int progressiveNum)
{
    ofstream myfile1("../../VM_shared/cloud_" + to_string(progressiveNum) + ".txt");
    if (myfile1.is_open())
    {
        for (int i = 0; i < map.LTM.size(); i++)
        {
            myfile1 << map.LTM[i].X << "\n";
        }
        for (int i = 0; i < map.WM.size(); i++)
        {
            myfile1 << map.WM[i].X << "\n";
        }
        myfile1.close();
        cout << endl << endl << "Map saved to file, number: " << progressiveNum << endl;
    }

    // save trajectory to text file
    ofstream myfile2("../../VM_shared/trajectory_" + to_string(progressiveNum) + ".txt");
    if (myfile2.is_open())
    {
        for (int i = 0; i < map.trajectory.size(); i++)
        {
            myfile2 << map.trajectory[i].trans() << "\n";
        }
        myfile2.close();
        cout << "Trajectory saved to file, number: " << progressiveNum << endl;
    }
}

int main()
{
    int Nsteps = 600;
    int firstImage = 20;
    int firstStepBA = 2;
    int odometryType = 3;
    int allowedGapSize = 2;
    int nObsMin = 6;
    int progressiveNum = 24;

    bool displayProgress = true;
    bool displayResults = false;

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

    StereoCartography map(t1, t2, cam1, cam2);

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

        // extract features
        vector<Feature> featuresVec1, featuresVec2,
                        featuresVecC1, featuresVecC2, featuresLM1, featuresLM2;
        map.extractor(image1, featuresVec1, 1);
        map.extractor(image2, featuresVec2, 2);

        time4 = clock();
        dt2 = double(time4 - time1) / CLOCKS_PER_SEC;

        int N1 = featuresVec1.size();
        int N2 = featuresVec2.size();

        if (step == 0)
        {
            map.trajectory.push_back(Transformation<double>());
            featuresVecC1.swap(featuresVec1);
            featuresVecC2.swap(featuresVec2);
        }
        else
        {
            // compute odometry
            Transformation<double> newPose;
            switch (odometryType)
            {
                case 1:
                {
                    Transformation<double> tn = map.estimateOdometry(featuresVec1);
                    newPose.setParam(tn.trans(), tn.rot());
                }
                case 2:
                {
                    Transformation<double> tn = map.estimateOdometry_2(featuresVec1);
                    newPose.setParam(tn.trans(), tn.rot());
                }
                case 3:
                {
                    Transformation<double> tn = map.estimateOdometry_3(featuresVec1);
                    newPose.setParam(tn.trans(), tn.rot());
                }
            }
            map.trajectory.push_back(newPose);

            // create reprojections of landmarks
            vector<Feature> featuresLM1, featuresLM2;
            vector<int> indexList;
            int nSTMprojected;
            projectLandmarks(map, featuresLM1, featuresLM2, indexList, nSTMprojected);

            // match reprojections with extracted features
            vector<int> matchesR1, matchesR2;
            map.matcher.matchReprojected(featuresLM1, featuresVec1, matchesR1, 5);
            map.matcher.matchReprojected(featuresLM2, featuresVec2, matchesR2, 5);

            // update observations
            updateObservations(map, featuresVec1, featuresVec2, matchesR1,
                               matchesR2, indexList, nSTMprojected);

            // manage landmarks that are already in memory
            firstMemoryUpdate(map, allowedGapSize, nObsMin);

            // create vectors of candidates for new landmarks
            findCandidates(map, featuresVec1, featuresVec2, matchesR1, matchesR2,
                           featuresVecC1, featuresVecC2);
        }

        // create new landmarks and add them to STM
        secondMemoryUpdate(featuresVecC1, featuresVecC2, map);

        time2 = clock();
        dt3 = double(time2 - time4) / CLOCKS_PER_SEC;
        cout << "\n\nstep: " << step << "  Odometry: " << map.trajectory.back()
             << "\nstep: " << step << "  STM: " << map.STM.size()  << "   WM: "
             << map.WM.size()  << "   LTM: " << map.LTM.size()
             << "\nstep: " << step << "  Features left: " << featuresVec1.size()
             << "   Features right: " << featuresVec2.size()
             << "\nstep: " << step << "  Images time: " << dt1
             << "   Features time: " << dt2
             << "\nstep: " << step << "  LM time: " << dt3 << flush;

        if (step >= firstStepBA)
        {
            bool firstBA = (step == firstStepBA);
            // perform bundle adjustment
            map.improveTheMap(firstBA);
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

        if (displayProgress)
        {
            double resizeRatio = 0.5;
            cv::resize(image1, image1, cv::Size(0,0), resizeRatio, resizeRatio);
            cv::cvtColor(image1, image1, CV_GRAY2BGR);
            for (int j = 0; j < featuresVec1.size(); j++)
            {
                cv::circle(image1, cv::Point(featuresVec1[j].pt(0),
                    featuresVec1[j].pt(1))*resizeRatio, 6, cv::Scalar(255, 0, 0), 2);
            }
            imshow("image1", image1);
            cv::waitKey(200);
        }
    }

    saveData(map, progressiveNum);

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
            for (int j = 0; j < map.LTM.size(); j++)
            {
                bool added = false;
                for (int k = 0; k < map.LTM[j].observations.size(); k++)
                {
                    if (map.LTM[j].observations[k].poseIdx == step)
                    {
                        if (added == false)
                        {
                            Point3D.push_back(map.LTM[j].X);
                            added = true;
                        }
                        if (map.LTM[j].observations[k].cameraId == CameraID::LEFT)
                        {
                            obs1.push_back(map.LTM[j].observations[k].pt);
                        }
                        if (map.LTM[j].observations[k].cameraId == CameraID::RIGHT)
                        {
                            obs2.push_back(map.LTM[j].observations[k].pt);
                        }
                    }
                }
            }

            vector<Eigen::Vector2d> repro1, repro2;
            map.projectPointCloud(Point3D, repro1, repro2, step);

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
