#ifndef _SPCSLAM_COSTFUNCTORS_H_
#define _SPCSLAM_COSTFUNCTORS_H_

#include "vision.h"
#include <vector>
#include <Eigen/Eigen>

using namespace std;
using Eigen::Vector3d;
using Eigen::Vector2d;

template<template<typename> class Camera>
struct BoardProjection 
{
    BoardProjection(const vector<Vector2d> & proj, const vector<Vector3d> & orig)
    : _proj(proj), _orig(orig) {}
            
    template <typename T>
    bool operator()(const T * const* params,
                    T* residual) const 
    {
        Camera<T> camera(params[0]);
        Transformation<T> transfo(params[1]);
        
        vector<Vector3<T>> transformedPoints(_orig.size());
        for (int i = 0; i < _orig.size(); i++)
        {
            transformedPoints[i] = _orig[i].template cast<T>();
        }
        transfo.transform(transformedPoints, transformedPoints);
        
        vector<Vector2<T>> projectedPoints;
        camera.projectPointCloud(transformedPoints, projectedPoints);
        for (unsigned int i = 0; i < projectedPoints.size(); i++)
        {
            Vector2<T> diff = _proj[i].template cast<T>() - projectedPoints[i];
            residual[2*i] = T(diff[0]);
            residual[2*i + 1] = T(diff[1]);
        }
        return true;
    }
    
    vector<Vector2d> _proj;
    vector<Vector3d> _orig;
};
   
template<template<typename> class Camera>
struct BoardEstimate
{
    BoardEstimate(const vector<Vector2d> & proj, const vector<Vector3d> & orig,
    const vector<double> & camParams) : _proj(proj), _orig(orig), _camParams(camParams) {}
            
    template <typename T>
    bool operator()(const T * const * params,
                    T* residual) const 
    {
        vector<T> camParamsT(_camParams.size());
        for (int i = 0; i < _camParams.size(); i++)
        {
            camParamsT[i] = T(_camParams[i]);
        }
        Camera<T> camera(camParamsT.data());
        Transformation<T> transfo(params[0]);
        
        vector<Vector3<T>> transformedPoints(_orig.size());
        for (int i = 0; i < _orig.size(); i++)
        {
            transformedPoints[i] = _orig[i].template cast<T>();
        }
        transfo.transform(transformedPoints, transformedPoints);
        
        vector<Vector2<T>> projectedPoints;
        camera.projectPointCloud(transformedPoints, projectedPoints);
        for (unsigned int i = 0; i < projectedPoints.size(); i++)
        {
            Vector2<T> diff = _proj[i].template cast<T>() - projectedPoints[i];
            residual[2*i] = T(diff[0]);
            residual[2*i + 1] = T(diff[1]);
        }
        return true;
    }
    
    vector<double> _camParams;
    vector<Vector2d> _proj;
    vector<Vector3d> _orig;
};

#endif
