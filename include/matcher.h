#ifndef _MATCHER_H_
#define _MATCHER_H_ 

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

using namespace std;

//TODO: move to Camera class
struct CamParameters
{
  int imageWidth;
  int imageHeight;
  double xi;
  double fu;
  double fv;
  double u0;
  double v0;
  
  CamParameters() {}
  
  CamParameters(int imageWidth, int imageHeight, double xi, double fu, double fv, double u0, double v0)
  : imageWidth(imageWidth), imageHeight(imageHeight), xi(xi), fu(fu), fv(fv), u0(u0), v0(v0) {}
};
  
//TODO: move to StereoSystem class
struct StereoParameters
{
  CamParameters cL;
  CamParameters cR;
  
  Eigen::Vector4d qR; // (x,y,z,w)  
  Eigen::Vector3d tR; 
  
  StereoParameters(CamParameters cL, CamParameters cR, Eigen::Vector4d qR, Eigen::Vector3d tR)
  {
    this->cL = cL;
    this->cR = cR;
    this->qR = qR;
    this->tR = tR;
  }
};

struct KeyPoint 
{
  
  Eigen::Vector2d pt; // representation: (x, y). TODO: change representation to (y, x) ?
  Eigen::Matrix<float,64,1> desc;
  
  float size, angle;
  
  KeyPoint(double x, double y, float * d): pt(x, y) , desc((float *) d) {}
  KeyPoint(double x, double y, float * d, float size, float angle)
        : pt(x, y) , desc((float *) d), size(size), angle(angle) {} 
};

class Matcher
{
  
public:
  
  void bruteForce(const vector<KeyPoint> & kpVec1, const vector<KeyPoint> & kpVec2, vector<int> & matches);
  
  void stereoMatch(const vector<KeyPoint> & kpVec1, const vector<KeyPoint> & kpVec2, 
		   const Eigen::MatrixXi & binMapL, const Eigen::MatrixXi & binMapR, 
		   vector<int> & matches);
  
  /*matchReprojected*/
  
  void initStereoBins(const StereoParameters & param, Eigen::MatrixXi & binMapL, Eigen::MatrixXi & binMapR);      
  
};
 
#endif
