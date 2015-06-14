
#include <iostream>
#include <Eigen/Eigen>

#include "matcher.h"
#include "geometry.h"
#include "mei.h"
#include "vision.h"


using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

void Matcher::bruteForce(const vector<Feature> & featuresVec1,
                         const vector<Feature> & featuresVec2,
                         vector<int> & matches) const
{

    const int N1 = featuresVec1.size();
    const int N2 = featuresVec2.size();

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        matches[i] = -1;
        int tempMatch = -1;
        double bestDist = 1000000;

        for (int j = 0; j < N2 ; j++)
        {
            double dist = (featuresVec1[i].desc - featuresVec2[j].desc).norm();

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

void Matcher::bruteForceOneToOne(const vector<Feature> & featuresVec1,
                                 const vector<Feature> & featuresVec2,
                                 vector<int> & matches) const
{
    const int N1 = featuresVec1.size();
    const int N2 = featuresVec2.size();

    vector<int> matches2(N2, -1);

    bruteForce(featuresVec1, featuresVec2, matches);
    bruteForce(featuresVec2, featuresVec1, matches2);

    for (int i = 0; i < N1; i++)
    {
        if (matches[i] > -1 && matches2[matches[i]] != i)
        {
            matches[i] = -1;
        }
    }
}

void Matcher::stereoMatch(const vector<Feature> & featuresVec1,
                          const vector<Feature> & featuresVec2,
			  vector<int> & matches) const
{

    const bool debug = false;

    const int N1 = featuresVec1.size();
    const int N2 = featuresVec2.size();

    const double distTh = 0.2;

    vector<double> bestDists(N1, distTh);

    matches.resize(N1);

    for (int i = 0; i < N1; i++) { matches[i] = -1; }

    for (int j = 0; j < N2; j++)
    {
        double bestDist = distTh;
        int iTempMatch = 0;

        //cout << endl << "j=" << j << endl;

        if (debug)
        {
            int binDiff = binMapL(round(featuresVec1[j].pt(1)), round(featuresVec1[j].pt(0))) -
                          binMapR(round(featuresVec2[j].pt(1)), round(featuresVec2[j].pt(0)));
            if (binDiff != 0)
                cout << "j=" << j << " binDiff=" << binDiff << endl;
        }

        for (int i = 0; i < N1 ; i++)
        {
            //cout << endl << "i=" << i << endl;
            if (abs(binMapL(round(featuresVec1[i].pt(1)), round(featuresVec1[i].pt(0))) -
                    binMapR(round(featuresVec2[j].pt(1)), round(featuresVec2[j].pt(0)))) <= 1)
            {
                double dist = (featuresVec1[i].desc - featuresVec2[j].desc).norm();

                if (dist < bestDist)
                {
                    bestDist = dist;
                    iTempMatch = i;
                    //cout << "bestDist=" << bestDist << endl;
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

void Matcher::matchReprojected(const vector<Feature> & featuresVec1,
		               const vector<Feature> & featuresVec2,
		               vector<int> & matches) const
{

    double radius = 3; // search radius in pixels

    const int N1 = featuresVec1.size();
    const int N2 = featuresVec2.size();

    matches.resize(N1);

    vector<int> matches1(N1, -1);

    for (int i = 0; i < N1; i++)
    {
        double bestScore = 1000000;
        int bestMatch = -1;
        for (int j = 0; j < N2; j++)
        {
            if ((featuresVec1.at(i).pt - featuresVec2.at(j).pt).norm() < radius)
            {
                double tempScore = (featuresVec1.at(i).desc - featuresVec2.at(j).desc).norm();
                if (tempScore < bestScore)
                {
                    bestMatch = j;
                    bestScore = tempScore;
                }
            }
        }
        matches1.at(i) = bestMatch;
    }

    vector<int> matches2(N2, -1);

    for (int i = 0; i < N2; i++)
    {
        double bestScore = 1000000;
        int bestMatch = -1;
        for (int j = 0; j < N1; j++)
        {
            if ((featuresVec1.at(j).pt - featuresVec2.at(i).pt).norm() < radius)
            {
                double tempScore = (featuresVec1.at(j).desc - featuresVec2.at(i).desc).norm();
                if (tempScore < bestScore)
                {
                    bestMatch = j;
                    bestScore = tempScore;
                }
            }
        }
        matches2.at(i) = bestMatch;
    }

    for (int i = 0; i < N1; i++)
    {
        if ((matches1.at(i) > -1) && (matches2.at(matches1.at(i)) == i))
        {
            matches.at(i) = matches1.at(i);
        }
        else
        {
            matches[i] = -1;
        }
    }


/*    vector<double> bestScores(N1, 2);

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        matches[i] = -1;
    }

    for (int j = 0; j < N2; j++)
    {
        double bestScore = 1000000;
        int iTempMatch = 0;

        double alfa = 1;
        double beta = 1;

        for (int i = 0; i < N1 ; i++)
        {

            double descDist = (featuresVec1[i].desc - featuresVec2[j].desc).norm();
            double spaceDist = (featuresVec1[i].pt - featuresVec2[j].pt).norm();
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
    Transformation<double> Tcam1cam2 = stereo.TbaseCam1.inverseCompose(stereo.TbaseCam2);
    R = Tcam1cam2.rotMat();

    // t: translation vector from L -> R (L reference frame)
    Eigen::Vector3d t = Tcam1cam2.trans();

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

void Matcher::computeMaps(const StereoSystem & stereo)
{

    const double pi = std::atan(1)*4;

    alfaMap1.resize(stereo.cam1->height, stereo.cam1->width);
    betaMap1.resize(stereo.cam1->height, stereo.cam1->width);
    alfaMap2.resize(stereo.cam2->height, stereo.cam2->width);
    betaMap2.resize(stereo.cam2->height, stereo.cam2->width);

    Matrix3d R, RSigma, RPhi, RTot;

    // R is rotation matrix 1 -> 2
    Transformation<double> Tcam1cam2 = stereo.TbaseCam1.inverseCompose(stereo.TbaseCam2);
    R = Tcam1cam2.rotMat();

    // t: translation vector from 1 -> 2 (reference frame 1)
    Eigen::Vector3d t = Tcam1cam2.trans();

    double sigma = std::atan2(-t(1), std::sqrt(t(0)*t(0) + t(2)*t(2)));
    double phi = std::atan2(t(2), t(0));

    Eigen::Vector3d vPhi(0, phi, 0);
    Eigen::Vector3d vSigma(0, 0, sigma);

    RPhi = rotationMatrix(vPhi);
    RSigma = rotationMatrix(vSigma);

    RTot = RSigma*RPhi;

    Eigen::Vector2d p;
    Eigen::Vector3d v;

    // compute maps for camera 1
    for (int i = 0; i < stereo.cam1->height; i++)
    {
        for (int j = 0; j < stereo.cam1->width; j++)
        {
            p << j, i;
            stereo.cam2->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * v;

            double alfa = std::atan2(v2(1), v2(2))*180/pi;
            double beta = std::atan2(v2(0), v2(2))*180/pi;

            alfaMap1(i, j) = alfa;
            betaMap1(i, j) = beta;
        }
    }

    // compute maps for camera 2
    for (int i = 0; i < stereo.cam2->height; i++)
    {
        for (int j = 0; j < stereo.cam2->width; j++)
        {
            p << j, i;
            stereo.cam1->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * R * v;

            double alfa = std::atan2(v2(1), v2(2))*180/pi;
            double beta = std::atan2(v2(0), v2(2))*180/pi;

            alfaMap2(i, j) = alfa;
            betaMap2(i, j) = beta;
        }
    }
}
