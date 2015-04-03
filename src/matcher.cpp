
#include <iostream>
#include <Eigen/Eigen>

#include "matcher.h"
#include "geometry.h"
#include "mei.h"
#include "vision.h"


using namespace std;
using Eigen::Matrix3d;
//using namespace cv;
void Matcher::bruteForce(const vector<Feature> & kpVec1, const vector<Feature> & kpVec2, vector<int> & matches)
{

    const int N1 = kpVec1.size();
    const int N2 = kpVec2.size();

    vector<double> bestDists(N1);

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        matches[i] = -1;
        bestDists[i] = 0.2;
    }

    for (int j = 0; j < N2; j++)
    {
        double bestDist = 0.2;
        int iTempMatch = 0;
        for (int i = 0; i < N1 ; i++)
        {
            double dist = (kpVec1[i].desc - kpVec2[j].desc).norm();
            //cout << "dist=" << dist << endl;
            if (dist < bestDist)
            {
                bestDist = dist;
                iTempMatch = i;
            }
        }
        if (bestDist < bestDists[iTempMatch])
        {
            matches[iTempMatch] = j;
            bestDists[iTempMatch] = bestDist;
        }
    }

    /*for (int i = 0; i < N1; i++)
    {
        cout << " i=" << i << " bestDists[i]=" << bestDists[i] << endl;
    }*/

}

//TODO: create the Camera objects and the StereoSystem object (with their parameters)
//            somewhere else and pass references to this function to access the parameters
//            and the member functions
void Matcher::initStereoBins(const StereoSystem & stereo)
{

    const bool debug = true;

    const double delta = 1; //degrees

    const double pi = std::atan(1)*4; // move this in geometry.h ?

/*
    // theta: rotation angle for R -> L (R reference frame)
    double theta = 2*std::acos(qR(3));

    // uR: rotation versor for R -> L (R reference frame)
    Eigen::Vector3d uR(qR(0)/std::sin(theta/2), qR(1)/std::sin(theta/2), qR(2)/std::sin(theta/2));

    // u: rotation vector for R -> L (R reference frame)
    Eigen::Vector3d u = uR*theta;

     // 2R1

    // R: rotation matrix R -> L
    R = stereo.pose1.rotationMatrix()
    //fromVtoR(u, R);*/

    Matrix3d R, RSigma, RPhi, RTot;

    // R now is rotation matrix L -> R
    R = stereo.pose1.rotMat();

    // t: translation vector from L -> R (L reference frame)
    Eigen::Vector3d t = stereo.pose1.trans();

    double sigma = std::atan2(-t(1), std::sqrt(t(0)*t(0) + t(2)*t(2)));
    double phi = std::atan2(t(2), t(0));

    Eigen::Vector3d vPhi(0, phi, 0);
    Eigen::Vector3d vSigma(0, 0, sigma);

    RPhi = rotationMatrix(vPhi);
    RSigma = rotationMatrix(vSigma);

    RTot = RSigma*RPhi;

    Eigen::Vector2d p;
    Eigen::Vector3d v;

    binMapL.resize(stereo.cam1->imageHeight, stereo.cam1->imageWidth);
    binMapR.resize(stereo.cam2->imageHeight, stereo.cam2->imageWidth);

    // compute bin map for left camera
    for (int i=0; i<stereo.cam1->imageHeight; i++)
    {
        for (int j=0; j<stereo.cam1->imageWidth; j++)
        {
            p << j, i;
            stereo.cam1->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * v;

            double alfa = std::atan2(v2(1), v2(2))*180/pi;

            int bin;
            if (debug) { bin = alfa/delta; }
            else { bin = std::floor(alfa/delta); }

            binMapL(i,j) = bin;
        }
    }

    // compute bin map for right camera
    for (int i=0; i<stereo.cam2->imageHeight; i++)
    {
        for (int j=0; j<stereo.cam2->imageWidth; j++)
        {
            p << j, i;
            stereo.cam2->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * R * v;

            double alfa = std::atan2(v2(1), v2(2))*180/pi;

            int bin;
            if (debug) { bin = alfa/delta; }
            else { bin = std::floor(alfa/delta); }

            binMapR(i,j) = bin;
        }
    }

    if (debug)
    {
        cout << endl << "phi:" << endl << phi*180/pi << endl;
        cout << endl << "sigma:" << endl << sigma*180/pi << endl;
        cout << endl << "R:" << endl << R << endl;
        cout << endl << "RPhi:" << endl << RPhi << endl;
        cout << endl << "RSigma:" << endl << RSigma << endl;
        cout << endl << "RTot:" << endl << RTot << endl;
    }
}

void Matcher::stereoMatch(const vector<Feature> & kpVecL, const vector<Feature> & kpVecR,
			  vector<int> & matches)
{

    const int NL = kpVecL.size();
    const int NR = kpVecR.size();

    const double distTh = 0.2; //TODO to the class member

    vector<double> bestDists(NL, distTh);

    matches.resize(NL);

    for (int i = 0; i < NL; i++)
    {
        matches[i] = -1;
    }

    for (int j = 0; j < NR; j++)
    {
        double bestDist = distTh;
        int iTempMatch = 0;

        for (int i = 0; i < NL ; i++)
        {
            if (abs(binMapL(kpVecL[i].pt(1), kpVecL[i].pt(0)) - binMapR(kpVecR[j].pt(1), kpVecR[j].pt(0))) <= 1)
            {
                double dist = (kpVecL[i].desc - kpVecR[j].desc).norm();

                if (dist < bestDist)
                {
                    bestDist = dist;
                    iTempMatch = i;
                }
            }
        }
        if (bestDist < bestDists[iTempMatch])
        {
            matches[iTempMatch] = j;
            bestDists[iTempMatch] = bestDist;
        }
    }

    /*for (int i = 0; i < matches.size(); i++)
    {
        cout << " i=" << i << " bestDists[i]=" << bestDists[i] << endl;
    }*/

}

void Matcher::matchReprojected(const vector<Feature> & kpVec1,
		               const vector<Feature> & kpVec2,
		               vector<int> & matches)
{

    const int N1 = kpVec1.size();
    const int N2 = kpVec2.size();

    vector<double> bestScores(N1, 2);

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        matches[i] = -1;
    }

    for (int j = 0; j < N2; j++)
    {
        double bestScore = 2;
        int iTempMatch = 0;

        double alfa = 1;
        double beta = 1;

        for (int i = 0; i < N1 ; i++)
        {

            double descDist = (kpVec1[i].desc - kpVec2[j].desc).norm();
            double spaceDist = (kpVec1[i].pt - kpVec2[j].pt).norm();
            double score = alfa * descDist + beta * spaceDist;

            if (score < bestScore)
            {
                bestScore = score;
                iTempMatch = i;
            }
        }
        if (bestScore < bestScores[iTempMatch])
        {
            matches[iTempMatch] = j;
            bestScores[iTempMatch] = bestScore;
        }
    }

    /*for (int i = 0; i < N1; i++)
    {
        cout << " i=" << i << " bestScores[i]=" << bestScores[i] << endl;
    }
    */

}
