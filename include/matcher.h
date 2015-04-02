#ifndef _MATCHER_H_
#define _MATCHER_H_ 

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

using namespace std;

struct KeyPoint 
{
  
  Eigen::Vector2d pt; // representation: (x, y). TODO: change representation to (y, x) ?
  Eigen::Matrix<float,64,1> desc;
  
  float size, angle;
  
  KeyPoint(double x, double y, float * d) : pt(x, y), desc(d), size(1), angle(0) {}
  KeyPoint(double x, double y, float * d, float size, float angle)
        : pt(x, y) , desc(d), size(size), angle(angle) {} 
};

class Matcher
{
  
public:
  
  void bruteForce(const vector<KeyPoint> & kpVec1, const vector<KeyPoint> & kpVec2, vector<int> & matches);
  
  void stereoMatch(const vector<KeyPoint> & kpVec1, const vector<KeyPoint> & kpVec2,
		   vector<int> & matches);
  
  /*matchReprojected*/
  
  void initStereoBins(const StereoSystem & stereo);      
private:
    Eigen::MatrixXi binMapL, binMapR;
    double distanceTh, descTh;
};
 
#endif



