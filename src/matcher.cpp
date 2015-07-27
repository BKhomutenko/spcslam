#include <vector>
#include <algorithm>
#include <iostream>

#include <Eigen/Eigen>

#include "geometry.h"
#include "mei.h"
#include "vision.h"
#include "extractor.h"
#include "matcher.h"

using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2d;

//const double thresh = (N-1)*N; //FIXME
const double thresh = 26;
const int deltaN = 2;

template<int Len>
inline float dist(const Matrix<float, Len, 1> & a, const Matrix<float, Len, 1> & b, const Matrix<float, Len, 1> & err)
{
    float res = 0;
    for (unsigned int i = 0; i < Len; i++)
    {
        float x = err[i] / (a[i]) /0.1;
        res += abs(x*x/2);
    }
    return res;
}

double computeDist(const Descriptor & d1, const Descriptor & d2)
{
    
//    Descriptor delta = d1 - d2;
    Matrix<float, N*N, 1> f(d1.data()), g(d2.data());


    Matrix<float, N*N, 1> err = g-f;
    
    
//    
//    if (dist(f, g, err) > 3*thresh)
//    {
//        return 3*thresh;
//    }
//    
//    Matrix<float, N*N, deltaN> df(d1.data()+N*N);
//    Matrix<float, N*N, deltaN> dg(d2.data()+N*N);

//    Matrix<float, N*N, deltaN> G = 0.5*(df + dg);

//    Matrix<float, deltaN, 1> delta = (G.transpose()*G).inverse()*(G.transpose()*err);
//    if (delta.norm() > 0.5)
//    {
//        delta *= 0.5/delta.norm();
//    }
//    f -= G*delta;
//    err = g - f;

    return dist(f, g, err);
//    return err.dot(err);
}

void Matcher::bruteForce(const vector<Feature> & featureVec1,
                         const vector<Feature> & featureVec2,
                         vector<int> & matches) const
{

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        matches[i] = -1;
        int tempMatch = -1;
        double bestDist = 1000000;

        for (int j = 0; j < N2 ; j++)
        {
            double dist = computeDist(featureVec1[i].desc, featureVec2[j].desc);

            
            if (dist < bestDist)
            {
                bestDist = dist;
                tempMatch = j;
            }
        }
        if (bestDist < thresh)
        {
            matches[i] = tempMatch;
        }
    }

    /*for (int i = 0; i < N1; i++)
    {
        cout << " i=" << i << " bestDists[i]=" << bestDists[i] << endl;
    }*/

}

void Matcher::bruteForceOneToOne(const vector<Feature> & featureVec1,
                                 const vector<Feature> & featureVec2,
                                 vector<int> & matches) const
{
    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();

    vector<int> matches2(N2, -1);

    bruteForce(featureVec1, featureVec2, matches);
    bruteForce(featureVec2, featureVec1, matches2);

    for (int i = 0; i < N1; i++)
    {
        if (matches[i] > -1 && matches2[matches[i]] != i)
        {
            matches[i] = -1;
        }
    }
}

void Matcher::stereoMatch(const vector<Feature> & featureVec1,
                          const vector<Feature> & featureVec2,
			  vector<int> & matches) const
{

    const bool debug = false;

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();

    vector<double> bestDists(N1, stereoDistTh);

    matches.resize(N1, -1);

    for (int i = 0; i < N1; i++)
    {
        double bestDist = thresh;
        int bestMatch = -1;
        bool matched = false;
        for (int j = 0; j < N2 ; j++)
        {
            //cout << endl << "i=" << i << endl;
            if (abs(binMapL(round(featureVec1[i].pt(1)), round(featureVec1[i].pt(0))) -
                    binMapR(round(featureVec2[j].pt(1)), round(featureVec2[j].pt(0)))) <= 1)
            {
                double dist = computeDist(featureVec1[i].desc, featureVec2[j].desc);

                if (dist < bestDist)
                {
                    if (bestDist / dist < 1.2)
                    {
                        matched = false;
                    }
                    else
                    {
                        matched = true;
                    }
                    bestDist = dist;
                    bestMatch = j;
                    
                    //cout << "bestDist=" << bestDist << endl;
                }
                else if (dist / bestDist < 1.2)
                {
                    matched = false;
                }
            }
        }
        if (matched)
        {
            matches[i] = bestMatch;
        }
    }

    /*for (int i = 0; i < matches.size(); i++)
    {
        cout << " i=" << i << " bestDists[i]=" << bestDists[i] << endl;
    }*/

}

void Matcher::stereoMatch_2(const vector<Feature> & featureVec1,
                            const vector<Feature> & featureVec2,
                            vector<int> & matches) const
{

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();

    matches.resize(N1);

    // compute matches 1 -> 2
    for (int i = 0; i < N1; i++)
    {
        double bestDist = thresh;
        int bestMatch = -1;

        for (int j = 0; j < N2; j++)
        {
            double alfa1 = alfaMap1(round(featureVec1[i].pt(1)), round(featureVec1[i].pt(0)));
            double alfa2 = alfaMap2(round(featureVec2[j].pt(1)), round(featureVec2[j].pt(0)));
            double beta1 = betaMap1(round(featureVec1[i].pt(1)), round(featureVec1[i].pt(0)));
            double beta2 = betaMap2(round(featureVec2[j].pt(1)), round(featureVec2[j].pt(0)));

            if (abs(alfa1 - alfa2) <= alfaTolerance and beta1 <= beta2 + betaTolerance)
            {
                double dist = computeDist(featureVec1[i].desc, featureVec2[j].desc);
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
            double alfa1 = alfaMap1(round(featureVec1[j].pt(1)), round(featureVec1[j].pt(0)));
            double alfa2 = alfaMap2(round(featureVec2[i].pt(1)), round(featureVec2[i].pt(0)));
            double beta1 = betaMap1(round(featureVec1[j].pt(1)), round(featureVec1[j].pt(0)));
            double beta2 = betaMap2(round(featureVec2[i].pt(1)), round(featureVec2[i].pt(0)));

            if (abs(alfa1 - alfa2) <= alfaTolerance and beta1 <= beta2 + betaTolerance)
            {
                double dist = (featureVec1[j].desc - featureVec2[i].desc).norm();
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

void Matcher::matchReprojected(const vector<Feature> & featureVec1,
		               const vector<Feature> & featureVec2,
		               vector<int> & matches, double radius) const
{

    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        double bestDist = thresh;
        int bestMatch = -1;
        for (int j = 0; j < N2; j++)
        {
            if ((featureVec1[i].pt - featureVec2[j].pt).norm() < radius)
            {
                double dist = computeDist(featureVec1[i].desc, featureVec2[j].desc);
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
            if ((featureVec1[j].pt - featureVec2[i].pt).norm() < radius)
            {
                double dist = (featureVec1[j].desc - featureVec2[i].desc).norm();
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

void Matcher::bruteForce_2(const vector<Feature> & featureVec1,
                           const vector<Feature> & featureVec2,
                           vector<vector<int>> & matches) const
{
    cout << "Brute force" << endl;
    const int N1 = featureVec1.size();
    const int N2 = featureVec2.size();

    matches.resize(N1);

    for (int i = 0; i < N1; i++)
    {
        cout << "." << flush;
        matches[i].resize(0);
        vector<bool> matched(N2, false);
        double bestDistOld = thresh;
        cout << "." << flush;
        for (int k = 0; k < 25; k++)
        {
            int bestMatch = -1;
            double bestDist = thresh;

            for (int j = 0; j < N2 ; j++)
            {
                double dist = computeDist(featureVec1[i].desc, featureVec2[j].desc);
                if (dist < bestDist and matched[j] == false)
                {
                    bestDist = dist;
                    if (bestDist < bestDistOld)
                    {
                        bestDistOld = bestDist;
                    }
                    bestMatch = j;
                }
            }
            cout << "." << flush;
            if (bestMatch > -1)
            {
                matches[i].push_back(bestMatch);
                matched[bestMatch] = true;
            }
            else
            {
                break;
            }
        }
    }
    cout << "finished" << endl;
}
