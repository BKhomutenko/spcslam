
#include <iostream>
#include <Eigen/Eigen>

#include "matcher.h"
#include "geometry.h"
#include "mei.h"
#include "vision.h"


using namespace std;
using Eigen::Matrix3d;
//using namespace cv;
void Matcher::bruteForce(const vector<KeyPoint> & kpVec1, const vector<KeyPoint> & kpVec2, vector<int> & matches) 
{
  
  const int N1 = kpVec1.size();
  const int N2 = kpVec2.size();
  
  assert(N1 == matches.size());
  
  double * bestDists;
  bestDists = new double[N1];
  
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
  
  for (int i = 0; i < N1; i++) 
  {
    cout << " i=" << i << " bestDists[i]=" << bestDists[i] << endl;
  }
  
  delete[] bestDists;
  
}

//TODO: create the Camera objects and the StereoSystem object (with their parameters)
//      somewhere else and pass references to this function to access the parameters
//      and the member functions
void Matcher::initStereoBins(const StereoParameters & param, Eigen::MatrixXi & binMapL, Eigen::MatrixXi & binMapR)
{
  
  const bool debug = true;
  
  const double delta = 1; //degrees
  
  Camera * camL;
  Camera * camR;
 
  camL = new MeiCamera(param.cL.imageWidth, param.cL.imageHeight, param.cL.xi, param.cL.fu, param.cL.fv, param.cL.u0, param.cL.v0);
  camR = new MeiCamera(param.cR.imageWidth, param.cR.imageHeight, param.cR.xi, param.cR.fu, param.cR.fv, param.cR.u0, param.cR.v0);
  
  const double pi = std::atan(1)*4; // move this in geometry.h ?
  
  Eigen::Vector4d qR;
  Eigen::Vector3d tR;
  qR = param.qR;
  tR = param.tR;
  
  // theta: rotation angle for R -> L (R reference frame)
  double theta = 2*std::acos(qR(3));
  
  // uR: rotation versor for R -> L (R reference frame)
  Eigen::Vector3d uR(qR(0)/std::sin(theta/2), qR(1)/std::sin(theta/2), qR(2)/std::sin(theta/2));
  
  // u: rotation vector for R -> L (R reference frame)
  Eigen::Vector3d u = uR*theta;
  
  Eigen::Matrix3d R, RSigma, RPhi, RTot; // 2R1
  
  // R: rotation matrix R -> L 
  fromVtoR(u, R);
  
  // R now is rotatio matrix L -> R
  R.transposeInPlace();
  
  // t: translation vector from L -> R (L reference frame)
  Eigen::Vector3d t = -R*tR;
  
  double sigma = std::atan2(-t(1), std::sqrt(t(0)*t(0) + t(2)*t(2)));
  double phi = std::atan2(t(2), t(0));
  
  Eigen::Vector3d vPhi(0, phi, 0);
  Eigen::Vector3d vSigma(0, 0, sigma);
  
  fromVtoR(vPhi, RPhi);
  fromVtoR(vSigma, RSigma);
  
  RTot = RSigma*RPhi;
 
  Eigen::Vector2d p;
  Eigen::Vector3d v;
  
  // compute bin map for left camera
  for (int i=0; i<param.cL.imageHeight; i++)
  {
    for (int j=0; j<param.cL.imageWidth; j++)
    {
      p << j, i;
      camL->reconstructPoint(p, v);
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
  for (int i=0; i<param.cR.imageHeight; i++)
  {
    for (int j=0; j<param.cR.imageWidth; j++)
    {
      p << j, i;
      camR->reconstructPoint(p, v);
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
    cout << endl << "theta= " << theta*180/pi << endl;
    cout << endl << "uR:" << endl << uR << endl;
    cout << endl << "t:" << endl << t << endl;
    cout << endl << "phi:" << endl << phi*180/pi << endl;
    cout << endl << "sigma:" << endl << sigma*180/pi << endl;
    cout << endl << "R:" << endl << R << endl;
    cout << endl << "RPhi:" << endl << RPhi << endl;
    cout << endl << "RSigma:" << endl << RSigma << endl;
    cout << endl << "RTot:" << endl << RTot << endl;
  }    
}
  
void Matcher::stereoMatch(const vector<KeyPoint> & kpVecL, const vector<KeyPoint> & kpVecR, 
			    const Eigen::MatrixXi & binMapL, const Eigen::MatrixXi & binMapR, 
			    vector<int> & matches)
{
  
  const int NL = kpVecL.size();
  const int NR = kpVecR.size();
  
  assert(NL == matches.size());
  
  double * bestDists;
  bestDists = new double[NL];
  
  for (int i = 0; i < NL; i++)
  {
    matches[i] = -1;
    bestDists[i] = 0.2;
  }

  for (int j = 0; j < NR; j++)
  {
    double bestDist = 0.2;
    int iTempMatch = 0;
    
    for (int i = 0; i < NL ; i++)
    {
      if (abs(binMapL(kpVecL[i].pt(1), kpVecL[i].pt(0)) - binMapR(kpVecR[j].pt(1), kpVecR[j].pt(0))) <= 1)
      {
	double dist = (kpVecL[i].desc - kpVecR[j].desc).norm();
	//cout << "dist=" << dist << endl;
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
  
  for (int i = 0; i < matches.size(); i++) 
  {
    cout << " i=" << i << " bestDists[i]=" << bestDists[i] << endl;
  }
  
  
  delete[] bestDists;
  
}
