#ifndef _EXTRACTOR_H_
#define _EXTRACTOR_H_

#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

using Eigen::Vector2d;
using Eigen::Matrix;

using namespace std;

enum FeatureType {SURF, Custom};

const int N = 8;

typedef cv::Mat_<float> fMat;
typedef cv::Mat_<int> iMat;
typedef Matrix<float, 4*N*N, 1> Descriptor;

struct Feature
{

    Vector2d pt; // representation: (x, y)
    Descriptor desc;

    float size, angle; // rename size -> scale

    Feature(double x, double y, float size, float angle)
                : pt(x, y), size(size), angle(angle) {}
                
    Feature(const Vector2d & p, float size, float angle)
                : pt(p), size(size), angle(angle) {}
                
    Feature(double x, double y, const Descriptor & d, float size, float angle)
                : pt(x, y) , desc(d) , size(size) , angle(angle) {}

    Feature(const Vector2d & p, const Descriptor & d)
                : pt(p) , desc(d) {}

    Feature(const Vector2d & p, const Descriptor & d, float size, float angle)
                : pt(p) , desc(d), size(size), angle(angle) {}

    Feature(double x, double y, float * d)
                : pt(x, y) , desc((float *) d) {}

    Feature(double x, double y, float * d, float size, float angle)
                : pt(x, y) , desc((float *) d), size(size), angle(angle) {}

    Feature() {}
};

template<typename T>
inline bool pairCompare(const pair<float, T> & a, const pair<float, T> & b) 
{
    return a.first < b.first;
}

class FeatureExtractor
{
public:
    FeatureExtractor(int numFeatures) : numFeatures(numFeatures), thresh(0.0001), descSize(N, N) {}
    
    virtual ~FeatureExtractor() {}
    
    void compute(const fMat & src, vector<Feature> & featureVec);
    
    void findFeatures(const fMat & src);
    
    void findMaxima();
    
    void computeDescriptor(const fMat & src, Vector2d pt, Descriptor & d);
    
    void computeGradients(const fMat & src);
    
    void computeNormImage(const fMat & src);
    
    void computeResponse(float scale);
    
    // Compute desriptors, fill the Feature structs
    void finalizeFeatures(const fMat & src, vector<Feature> & featureVec);
    
private:
    int numFeatures;
    float thresh;
    iMat mask;
    fMat normImg, response;
    fMat gradx, grady;
    fMat Ixx, Ixy, Iyy;
    cv::Size descSize;
    vector<pair<float, Vector2d>> maxVec;
}; 

/*
class Extractor //TODO make two extractors for each camera
//TODO inherit from a normal extractor one with masks and bin information
{
private:

    cv::SurfFeatureDetector det;
    cv::SurfDescriptorExtractor extr;

    FeatureType fType = FeatureType::SURF;

    fMat kernel;
    fMat mask;
    int thresh = 50;
    const int descWidth = 4;

    vector<Eigen::MatrixXi> binMaps;
    const int nDivVertical = 4;
    const int nDivHorizontal = 6;

    const vector<vector<double> > circle1 = { { 680, 462, 632 },
                                              { 1253, 328, 1092 } }; // circle definition: { u0, v0, r }
    const vector<vector<double> > circle2 = { { 292, 382, 850 },
                                              { 324, 502, 918 } };
    const vector<vector<double> > uBounds = { { 48, 1142 },
                                              { 161, 1242 } };
    const vector<double> vBounds = { 100, 616 };

    const int nFeatures = 300;
    vector<vector<int>> featureDistributions;

public:

    //Extractor() {}

    Extractor(double hessianThreshold, int nOctaves, int nOctaveLayers, bool extended, bool upright)
    : det(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
    {
        computeMaps();
    }

    Extractor()
    {
        computeMaps();
    }

    void setType(FeatureType featType);

    //TODO make general extractor and inherit stereo extractor
    void operator()(const fMat & img, std::vector<Feature> & featureVec, int camId=-1);

    void extractFeatures(const fMat & src, vector<cv::KeyPoint> points, fMat & descriptors);

    void extractDescriptor(const fMat & src, cv::Point2f pt, int patchSize, cv::Size descSize, fMat & dst);

    void findMax(const fMat & src, vector<cv::Point2f> & maxPoints, float threshold);

    void findFeatures(const fMat & src, std::vector<cv::KeyPoint> & points, int camId, float scale1,
                      float scale2=-1, int steps=3);

    void computeResponse(const fMat & src, fMat & dst, float scale);

    void cvtNormimagesmage(fMat & image);

    //void computeBinMaps();

    void computeMaps();

};*/

#endif
