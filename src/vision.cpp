/*
Abstract camera definition.
Stereo vision definition.
*/

//Standard libraries
#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
//Eigen
#include <Eigen/Eigen>

#include "geometry.h"
#include "vision.h"

using namespace std;
using namespace Eigen;

void StereoSystem::projectPointCloud(const vector<Vector3d> & src,
        vector<Vector2d> & dst1, vector<Vector2d> & dst2) const
{
    assert(src.size() == dst1.size());
    assert(src.size() == dst2.size());
    vector<Vector3d> Xc1(src.size()), Xc2(src.size());
    pose1.inverseTransform(src, Xc1);
    pose2.inverseTransform(src, Xc2);
    
    for (unsigned int i = 0; i < src.size(); i++)
    {
        cam1->projectPoint(Xc1[i], dst1[i]);
        cam2->projectPoint(Xc2[i], dst2[i]);
    }
}

//TODO not finished
bool StereoSystem::triangulate(const Vector3d & v1, const Vector3d & v2,
        const Vector3d & t, Vector3d & X)
{
    //Vector3d v1n = v1 / v1.norm(), v2n = v2 / v2.norm();
    double v1v2 = v1.dot(v2);
    double v1v1 = v1.dot(v1);
    double v2v2 = v2.dot(v2);
    double tv1 = t.dot(v1);
    double tv2 = t.dot(v2);
    double delta = -v1v1 * v2v2 + v1v2 * v1v2;
    if (abs(delta) < 1e-4) // TODO the constant to be revised
    {
        X << -1, -1, -1;
        return false;
    }
    double l1 = (-tv1 * v2v2 + tv2 * v1v2)/delta;
    double l2 = (tv2 * v1v1 - tv1 * v1v2)/delta;
    X = (v1*l1 + t + v2*l2)*0.5;
    return true;
}
     
void StereoSystem::reconstructPointCloud(const vector<Vector2d> & src1,
        const vector<Vector2d> & src2, vector<Vector3d> & dst) const
{
    assert(src1.size() == src2.size());
    assert(src1.size() == dst.size());    
    
    vector<Vector3d> vVec1(src1.size()), vVec2(src1.size());
    for (unsigned int i = 0; i < src1.size(); i++)
    {
        cam1->reconstructPoint(src1[i], vVec1[i]);
        cam2->reconstructPoint(src2[i], vVec2[i]);
    }
    
    pose1.rotate(vVec1, vVec1);
    pose2.rotate(vVec2, vVec2);
    Vector3d t = pose2.trans() - pose1.trans();
    for (unsigned int i = 0; i < src1.size(); i++)
    {
        Vector3d & X = dst[i];
        triangulate(vVec1[i], vVec2[i], t, X);
        X += pose1.trans();
    }
}

StereoSystem::~StereoSystem()
{
    if (cam1)
        delete cam1;
    if (cam2)
        delete cam2;
}

////TODO make smart constructor with calibration data passed

//StereoSystem::StereoSystem();

