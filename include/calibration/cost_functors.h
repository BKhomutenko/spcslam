#ifndef _SPCSLAM_COSTFUNCTORS_H_
#define _SPCSLAM_COSTFUNCTORS_H_

#include "vision.h"
#include <vector>
#include <Eigen/Eigen>

using namespace std;
using Eigen::Vector3d;
using Eigen::Vector2d;

template<template<typename> class Projector>
struct GridProjection 
{
    GridProjection(const vector<Vector2d> & proj, const vector<Vector3d> & grid)
    : _proj(proj), _grid(grid) {}
            
    template <typename T>
    bool operator()(const T * const* params,
                    T* residual) const 
    {
        Transformation<T> TbaseGrid(params[1]);
        vector<Vector3<T>> transformedPoints(_grid.size());
        for (int i = 0; i < _grid.size(); i++)
        {
            transformedPoints[i] = _grid[i].template cast<T>();
        }
        TbaseGrid.transform(transformedPoints, transformedPoints);
        
        for (unsigned int i = 0; i < transformedPoints.size(); i++)
        {
            Vector2<T> modProj;
            Projector<T>::compute(params[0], transformedPoints[i].data(), modProj.data());
            Vector2<T> diff = _proj[i].template cast<T>() - modProj;
            residual[2*i] = T(diff[0]);
            residual[2*i + 1] = T(diff[1]);
        }
        return true;
    }
    
    const vector<Vector2d> & _proj;
    const vector<Vector3d> & _grid;
};
   
template<template<typename> class Projector>
struct GridEstimate
{
    GridEstimate(const vector<Vector2d> & proj, const vector<Vector3d> & grid,
    const vector<double> & camParams) : _proj(proj), _grid(grid), _camParams(camParams) {}
            
    template <typename T>
    bool operator()(const T * const * params,
                    T* residual) const 
    {
        Transformation<T> TbaseGrid(params[0]);
        vector<Vector3<T>> transformedPoints(_grid.size());
        for (int i = 0; i < _grid.size(); i++)
        {
            transformedPoints[i] = _grid[i].template cast<T>();
        }
        TbaseGrid.transform(transformedPoints, transformedPoints);

        vector<T> camParamsT(_camParams.begin(), _camParams.end());
        for (unsigned int i = 0; i < transformedPoints.size(); i++)
        {
            Vector2<T> modProj;
            Projector<T>::compute(camParamsT.data(), transformedPoints[i].data(), modProj.data());
            Vector2<T> diff = _proj[i].template cast<T>() - modProj;
            residual[2*i] = T(diff[0]);
            residual[2*i + 1] = T(diff[1]);
        }
        return true;
    }
    
    const vector<double> & _camParams;
    const vector<Vector2d> & _proj;
    const vector<Vector3d> & _grid;
};

#endif
