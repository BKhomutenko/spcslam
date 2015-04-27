#ifndef _SPCMAP_MEI_H_
#define _SPCMAP_MEI_H_

#include <Eigen/Eigen>

#include "vision.h"

template<typename T>
inline T logistic(T x)
{
    T ex = exp(x);
    return ex/(1 + ex);
}

template<typename T>
class MeiCamera : public Camera<T>
{
public:
    using Camera<T>::params;
    using Camera<T>::width;
    using Camera<T>::height;
    MeiCamera(int W, int H, const T * const parameters) : Camera<T>(W, H, 6)
    {  
        setParameters(parameters);
    }

    MeiCamera(const T * const parameters)  : Camera<T>(1, 1, 6)
    {  
        setParameters(parameters);
    }
    
     /// takes raw image points and apply undistortion model to them
    virtual bool reconstructPoint(const Eigen::Vector2d & src, Eigen::Vector3d & dst) const
    {
        const T & alpha = params[0];
        const T & beta = params[1];
        const T & fu = params[2];
        const T & fv = params[3];
        const T & u0 = params[4];
        const T & v0 = params[5];
        
        T xn = (src(0) - u0) / fu;
        T yn = (src(1) - v0) / fv;
        
        T u2 = xn * xn + yn * yn;
        T gamma = 1 - alpha;    
        T u = sqrt(u2);
        T A = u2 * alpha * alpha * beta - 1;
        T B = u * gamma;
        T C = u2 * (alpha * alpha - gamma * gamma);
        T D1 = B * B - A * C; 
        
        T r = B + sqrt(D1);
        
        T denom = -gamma*A + alpha*sqrt(A*A + (beta  * r * r));
        dst << xn*denom, yn*denom, -A;

        return true;
    }

    /// projects 3D points onto the original image
    virtual bool projectPoint(const Eigen::Vector3d & src, Eigen::Vector2d & dst) const
    {
        const T & alpha = params[0];
        const T & beta = params[1];
        const T & fu = params[2];
        const T & fv = params[3];
        const T & u0 = params[4];
        const T & v0 = params[5];
        
        const T & x = src(0);
        const T & y = src(1);
        const T & z = src(2);
        
        T denom = alpha * sqrt(z*z + beta*(x*x + y*y)) + (1 - alpha) * z;

        // Project the point to the mu plane
        T xn = x / denom;
        T yn = y / denom;

        // Compute image point
        dst << fu * xn + u0, fv * yn + v0;

        return true;

    }

    virtual bool projectionJacobian(const Eigen::Vector3d & src, Eigen::Matrix<T, 2, 3> & Jac) const
    {
        const T & alpha = params[0];
        const T & beta = params[1];
        const T & fu = params[2];
        const T & fv = params[3];
        const T & u0 = params[4];
        const T & v0 = params[5];
        
        const T & x = src(0);
        const T & y = src(1);
        const T & z = src(2);

        T rho = sqrt(z*z + beta*(x*x + y*y));
        T gamma = 1 - alpha;
        T d = alpha * rho + gamma * z;
        T k = 1 / d / d;
        Jac(0,0) = fu * k * (gamma * z + alpha * rho - alpha * beta * x * x / rho);
        Jac(0,1) = -fu * k * alpha * beta * x * y / rho;
        Jac(0,2) = -fu * k * x * (gamma + alpha * z / rho);
        Jac(1,0) = -fv * k * alpha * beta * x * y / rho;    
        Jac(1,1) = fv * k * (gamma * z + alpha * rho - alpha * beta * y * y / rho);
        Jac(1,2) = -fv * k * y * (gamma + alpha * z / rho);

        return true;

    }
    

    virtual void setParameters(const T * const newParams)
    {
        Camera<T>::setParameters(newParams);
        params[0] = logistic(params[0]);
        params[1] = exp(params[1]);
    }
    
    virtual MeiCamera * clone() const { return new MeiCamera(width, height, params.data()); }
    
    virtual ~MeiCamera() {}
};

#endif
