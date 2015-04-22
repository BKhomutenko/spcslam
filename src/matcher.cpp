
#include <iostream>
#include <Eigen/Eigen>

#include "matcher.h"
#include "geometry.h"
#include "mei.h"
#include "vision.h"


using namespace std;

void Matcher::bruteForce(const vector<Feature> & fVec1,
                         const vector<Feature> & fVec2,
                         vector<int> & matches)
{

    const int N1 = fVec1.size();
    const int N2 = fVec2.size();

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        matches[i] = -1;
        int tempMatch = -1;
        double bestDist = 1000000;

        for (int j = 0; j < N2 ; j++)
        {
            double dist = (fVec1[i].desc - fVec2[j].desc).norm();

            if (dist < bestDist)
            {
                bestDist = dist;
                tempMatch = j;
            }
        }
        if (bestDist < bfDistTh)
        {
            matches[i] = tempMatch;
        }
    }

    /*for (int i = 0; i < N1; i++)
    {
        cout << " i=" << i << " bestDists[i]=" << bestDists[i] << endl;
    }*/

}

void Matcher::bruteForceOneToOne(const vector<Feature> & fVec1,
                                 const vector<Feature> & fVec2,
                                 vector<int> & matches)
{
    const int N1 = fVec1.size();
    const int N2 = fVec2.size();

    vector<int> matches2(N2, -1);

    bruteForce(fVec1, fVec2, matches);
    bruteForce(fVec2, fVec1, matches2);

    for (int i = 0; i < N1; i++)
    {
        if (matches[i] > -1 && matches2[matches[i]] != i)
        {
            matches[i] = -1;
        }
    }
}

void Matcher::initStereoBins(const StereoSystem & stereo)
{

    const bool debug = false;

    const double pi = std::atan(1)*4;

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

    // R is rotation matrix L -> R
    R = stereo.pose2.rotMat();

    // t: translation vector from L -> R (L reference frame)
    Eigen::Vector3d t = stereo.pose2.trans();

    double sigma = std::atan2(-t(1), std::sqrt(t(0)*t(0) + t(2)*t(2)));
    double phi = std::atan2(t(2), t(0));

    Eigen::Vector3d vPhi(0, phi, 0);
    Eigen::Vector3d vSigma(0, 0, sigma);

    RPhi = rotationMatrix(vPhi);
    RSigma = rotationMatrix(vSigma);

    RTot = RSigma*RPhi;

    Eigen::Vector2d p;
    Eigen::Vector3d v;

    binMapL.resize(stereo.cam1->height, stereo.cam1->width);
    binMapR.resize(stereo.cam2->height, stereo.cam2->width);

    // compute bin map for left camera
    for (int i=0; i<stereo.cam1->height; i++)
    {
        for (int j=0; j<stereo.cam1->width; j++)
        {
            p << j, i;
            stereo.cam1->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * v;

            double alfa = std::atan2(v2(1), v2(2))*180/pi;

            int bin;
            if (debug) { bin = alfa/binDelta; }
            else { bin = std::floor(alfa/binDelta); }

            binMapL(i,j) = bin;
        }
    }

    // compute bin map for right camera
    for (int i=0; i<stereo.cam2->height; i++)
    {
        for (int j=0; j<stereo.cam2->width; j++)
        {
            p << j, i;
            stereo.cam2->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * R * v;

            double alfa = std::atan2(v2(1), v2(2))*180/pi;

            int bin;
            if (debug) { bin = alfa/binDelta; }
            else { bin = std::floor(alfa/binDelta); }

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
        //cout << "binMapL(center)= " << binMapL(stereo.cam1->u0, stereo.cam1->v0) << endl;
        //cout << "binMapR(center)= " << binMapR(stereo.cam2->u0, stereo.cam2->v0) << endl;
    }
}

void Matcher::stereoMatch(const vector<Feature> & fVec1,
                          const vector<Feature> & fVec2,
			  vector<int> & matches)
{

    const bool debug = false;

    const int N1 = fVec1.size();
    const int N2 = fVec2.size();

    const double distTh = 0.2;

    vector<double> bestDists(N1, distTh);

    matches.resize(N1);

    for (int i = 0; i < N1; i++) { matches[i] = -1; }

    for (int j = 0; j < N2; j++)
    {
        double bestDist = distTh;
        int iTempMatch = 0;

        if (debug)
        {
            int binDiff = binMapL(round(fVec1[j].pt(1)), round(fVec1[j].pt(0))) -
                          binMapR(round(fVec2[j].pt(1)), round(fVec2[j].pt(0)));
            if (binDiff != 0)
                cout << "j=" << j << " binDiff=" << binDiff << endl;
        }

        for (int i = 0; i < N1 ; i++)
        {
            if (abs(binMapL(round(fVec1[i].pt(1)), round(fVec1[i].pt(0))) -
                    binMapR(round(fVec2[j].pt(1)), round(fVec2[j].pt(0)))) <= 1)
            {
                double dist = (fVec1[i].desc - fVec2[j].desc).norm();

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

void Matcher::matchReprojected(const vector<Feature> & fVec1,
		               const vector<Feature> & fVec2,
		               vector<int> & matches)
{

    const int N1 = fVec1.size();
    const int N2 = fVec2.size();

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

            double descDist = (fVec1[i].desc - fVec2[j].desc).norm();
            double spaceDist = (fVec1[i].pt - fVec2[j].pt).norm();
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
