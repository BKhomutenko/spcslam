#include <mei.h>
#include <Eigen/Eigen>

#include <iostream>

#include "vision.h"

using namespace std;

bool MeiCamera::projectPoint(const Eigen::Vector3d & src, Eigen::Vector2d & dst) const
{

    const double & x = src(0);
    const double & y = src(1);
    const double & z = src(2);
    
    double denom = alpha * sqrt(z*z + beta*(x*x + y*y)) + (1 - alpha) * z;

    // Project the point to the mu plane
    double xn = x / denom;
    double yn = y / denom;

    // Compute image point
    dst << fu * xn + u0, fv * yn + v0;

    return true;

}

bool MeiCamera::reconstructPoint(const Eigen::Vector2d & src, Eigen::Vector3d & dst) const
{

    double xn = (src(0) - u0) / fu;
    double yn = (src(1) - v0) / fv;
    
    double u2 = xn * xn + yn * yn;
    double gamma = 1 - alpha;    
    double u = sqrt(u2);
    double A = u2 * alpha * alpha * beta - 1;
    double B = u * gamma;
    double C = u2 * (alpha * alpha - gamma * gamma);
    double D1 = B * B - A * C; 
    
    double r = B + sqrt(D1);
    
    double denom = -gamma*A + alpha*sqrt(A*A + (beta  * r * r));
    dst << xn*denom, yn*denom, -A;

    return true;

}

bool MeiCamera::projectionJacobian(const Eigen::Vector3d & src, Eigen::Matrix<double, 2, 3> & Jac) const
{

    const double & x = src(0);
    const double & y = src(1);
    const double & z = src(2);

    double rho = sqrt(z*z + beta*(x*x + y*y));
    double gamma = 1 - alpha;
    double d = alpha * rho + gamma * z;
    double k = 1 / d / d;
    Jac(0,0) = fu * k * (gamma*z + alpha*rho - alpha*beta*x*x/rho);
    Jac(0,1) = -fu * k * alpha*beta*x*y/rho;
    Jac(0,2) = -fu * k * x * (gamma + alpha*z/rho);
    Jac(1,0) = -fv * k * alpha*beta*x*y/rho;    
    Jac(1,1) = fv * k * (gamma*z + alpha*rho - alpha*beta*y*y/rho);
    Jac(1,2) = -fv * k * y * (gamma + alpha*z/rho);

    return true;

}
