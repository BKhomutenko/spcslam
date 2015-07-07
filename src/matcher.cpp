
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

void Matcher::stereoMatch_2(const vector<Feature> & featuresVec1,
                            const vector<Feature> & featuresVec2,
                            vector<int> & matches) const
{

    const int N1 = featuresVec1.size();
    const int N2 = featuresVec2.size();

    matches.resize(N1);

    // compute matches 1 -> 2
    for (int i = 0; i < N1; i++)
    {
        double bestDist = stereoDistTh;
        int bestMatch = -1;

        for (int j = 0; j < N2; j++)
        {
            double alfa1 = alfaMap1(round(featuresVec1[i].pt(1)), round(featuresVec1[i].pt(0)));
            double alfa2 = alfaMap2(round(featuresVec2[j].pt(1)), round(featuresVec2[j].pt(0)));
            double beta1 = betaMap1(round(featuresVec1[i].pt(1)), round(featuresVec1[i].pt(0)));
            double beta2 = betaMap2(round(featuresVec2[j].pt(1)), round(featuresVec2[j].pt(0)));

            if (abs(alfa1 - alfa2) <= alfaTolerance and beta1 <= beta2 + betaTolerance)
            {
                double dist = (featuresVec1[i].desc - featuresVec2[j].desc).norm();
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestMatch = j;
                }
            }
        }
        matches[i] = bestMatch;
    }

    // compute matches 2 -> 1
    /*vector<int> matches2(N2, -1);
    for (int i = 0; i < N2; i++)
    {
        double bestDist = stereoDistTh;
        int bestMatch = -1;

        for (int j = 0; j < N1; j++)
        {
            double alfa1 = alfaMap1(round(featuresVec1[j].pt(1)), round(featuresVec1[j].pt(0)));
            double alfa2 = alfaMap2(round(featuresVec2[i].pt(1)), round(featuresVec2[i].pt(0)));
            double beta1 = betaMap1(round(featuresVec1[j].pt(1)), round(featuresVec1[j].pt(0)));
            double beta2 = betaMap2(round(featuresVec2[i].pt(1)), round(featuresVec2[i].pt(0)));

            if (abs(alfa1 - alfa2) <= alfaTolerance and beta1 <= beta2 + betaTolerance)
            {
                double dist = (featuresVec1[j].desc - featuresVec2[i].desc).norm();
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestMatch = j;
                }
            }
        }
        matches2[i] = bestMatch;
    }

    // filter matches
    for (int i = 0; i < N1; i++)
    {
        if (matches2[matches[i]] != i)
        {
            matches[i] = -1;
        }
    }*/
}

void Matcher::matchReprojected(const vector<Feature> & featuresVec1,
		               const vector<Feature> & featuresVec2,
		               vector<int> & matches, double radius) const
{

    const int N1 = featuresVec1.size();
    const int N2 = featuresVec2.size();

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        double bestDist = reproDistTh;
        int bestMatch = -1;
        for (int j = 0; j < N2; j++)
        {
            if ((featuresVec1[i].pt - featuresVec2[j].pt).norm() < radius)
            {
                double dist = (featuresVec1[i].desc - featuresVec2[j].desc).norm();
                if (dist < bestDist)
                {
                    bestMatch = j;
                    bestDist = dist;
                }
            }
        }
        matches[i] = bestMatch;
    }

/*    vector<int> matches2(N2, -1);

    for (int i = 0; i < N2; i++)
    {
        double bestDist = reproDistTh;
        int bestMatch = -1;
        for (int j = 0; j < N1; j++)
        {
            if ((featuresVec1[j].pt - featuresVec2[i].pt).norm() < radius)
            {
                double dist = (featuresVec1[j].desc - featuresVec2[i].desc).norm();
                if (dist < bestDist)
                {
                    bestMatch = j;
                    bestDist = dist;
                }
            }
        }
        matches2[i] = bestMatch;
    }

    for (int i = 0; i < N1; i++)
    {
        if ((matches[i] > -1) && (matches2[matches[i]] == i))
        {
            matches[i] = matches1[i];
        }
        else
        {
            matches[i] = -1;
        }
    }
*/
}

void Matcher::initStereoBins(const StereoSystem & stereo)
{

    const bool debug = false;

    const double pi = std::atan(1)*4;

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

    // compute maps for camera 1
    for (int i = 0; i < stereo.cam1->height; i++)
    {
        for (int j = 0; j < stereo.cam1->width; j++)
        {
            Eigen::Vector2d p;
            Eigen::Vector3d v;
            p << j, i;
            stereo.cam1->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * v;

            double alfa = std::atan2(v2(1), v2(2))*180/pi;
            double beta = std::atan2(v2(2), v2(0))*180/pi;

            alfaMap1(i, j) = alfa;
            betaMap1(i, j) = beta;
        }
    }

    // compute maps for camera 2
    for (int i = 0; i < stereo.cam2->height; i++)
    {
        for (int j = 0; j < stereo.cam2->width; j++)
        {
            Eigen::Vector2d p;
            Eigen::Vector3d v;
            p << j, i;
            stereo.cam2->reconstructPoint(p, v);
            Eigen::Vector3d v2;
            v2 = RTot * (R * v);

            double alfa = std::atan2(v2(1), v2(2))*180/pi;
            double beta = std::atan2(v2(2), v2(0))*180/pi;

            alfaMap2(i, j) = alfa;
            betaMap2(i, j) = beta;
        }
    }
}

void Matcher::bruteForce_2(const vector<Feature> & featuresVec1,
                           const vector<Feature> & featuresVec2,
                           vector<vector<int>> & matches) const
{

    const int N1 = featuresVec1.size();
    const int N2 = featuresVec2.size();

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        matches[i].resize(0);
        vector<bool> matched(N2, false);
        double bestDistOld = 1000;

        bool found = true;
        for (int k = 0; k < bf2matches and found; k++)
        {
            found = false;
            int bestMatch = -1;
            double bestDist = bfDistTh2;

            for (int j = 0; j < N2 ; j++)
            {
                double dist = (featuresVec1[i].desc - featuresVec2[j].desc).norm();

                if (dist < bestDist and matched[j] == false)
                {
                    bestDist = dist;
                    if (bestDist < bestDistOld)
                    {
                        bestDistOld = bestDist;
                    }
                    bestMatch = j;
                    found = true;
                }
            }

            if (found and bestDist/bestDistOld < 1.3)
            {
                matches[i].push_back(bestMatch);
                matched[bestMatch] = true;
            }
        }

        for (int j = matches[i].size(); j < bf2matches; j++)
        {
            matches[i].push_back(-1);
        }
    }
}
