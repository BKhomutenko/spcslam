#include <mei.h>
#include <Eigen/Eigen>

bool MeiCamera::projectPoint(const Eigen::Vector3d & src, Eigen::Vector2d & dst) const
{

    const double & x = src(0);
    const double & y = src(1);
    const double & z = src(2);
    
    double xiRho = xi * src.norm();

    // Project the point to the mu plane
    double xn = x / (z + xiRho);
    double yn = y / (z + xiRho);

    // Compute image point
    dst << fu * xn + u0, fv * yn + v0;

    return true;

}

bool MeiCamera::reconstructPoint(const Eigen::Vector2d & src, Eigen::Vector3d & dst) const
{

    double xn = (src(0) - u0) / fu;
    double yn = (src(1) - v0) / fv;

    const double k = xn * xn + yn * yn;
    const double m = (xi + std::sqrt((double)1 + (1 - xi * xi) * k)) / (k + 1);

    dst << m * xn, m * yn, m - xi;

    return true;

}

bool MeiCamera::projectionJacobian(const Eigen::Vector3d & src, Eigen::Matrix<double, 2, 3> & Jac) const
{

    const double & x = src(0);
    const double & y = src(1);
    const double & z = src(2);

    double rho = src.norm();
    double d = xi * rho + z;
    double k = 1 / (d * d);
    double xi2 = xi / rho;
    double lambda = - k * (xi2 * z + 1);

    Jac(0,0) = k * (d - xi2 * x * x);
    Jac(0,1) = - k * xi * x * y;
    Jac(0,2) = lambda * x;
    Jac(1,0) = Jac(0,1);
    Jac(1,1) = k * (d - xi2 * y * y);
    Jac(1,2) = lambda * y;

    return true;

}
