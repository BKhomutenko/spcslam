/*
All necessary geometric transformations
*/

#ifndef _SPCMAP_GEOMETRY_H_
#define _SPCMAP_GEOMETRY_H_

//STL
#include <vector>

//Eigen
#include <Eigen/Eigen>



template<typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>; 
template<typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>; 
template<typename T>
using Matrix3 = Eigen::Matrix<T, 3, 3>; 

using namespace std;
//TODO think about how to perform the combination of these transformations

template<typename T>
Matrix3<T> rotationMatrix(const Vector3<T> & v)
{
    Matrix3<T> R;
    T th = v.norm();
    if (th < 1e-5)
    {
        //Rotational part in case when th is small
        
	    R <<        T(1.),   -v(2),    v(1),
	          v(2),          T(1.),   -v(0),
	         -v(1),    v(0),          T(1.);
    }
    else
    {
        //Rodrigues formula
        T u1 = v(0) / th;
        T u2 = v(1) / th;
        T u3 = v(2) / th;
        T sinth = sin(th);
        T costhVar = T(1.) - cos(th);
        
        R(0, 0) = T(1.) + costhVar * (u1 * u1 - T(1.));
        R(1, 1) = T(1.) + costhVar * (u2 * u2 - T(1.));
        R(2, 2) = T(1.) + costhVar * (u3 * u3 - T(1.));

        R(0, 1) = -sinth*u3 + costhVar * u1 * u2;
        R(0, 2) = sinth*u2 + costhVar * u1 * u3;
        R(1, 2) = -sinth*u1 + costhVar * u2 * u3;

        R(1, 0) = sinth*u3 + costhVar * u2 * u1;
        R(2, 0) = -sinth*u2 + costhVar * u3 * u1;
        R(2, 1) = sinth*u1 + costhVar * u3 * u2;
    }
    return R;
}


template<typename T>
inline Matrix3<T> hat(const Vector3<T> & u)
{
    Matrix3<T> M;
    M << 0, -u(2), u(1),   u(2), 0, -u(0),   -u(1), u(0), 0;
    return M;
}


#include "geometry/quaternion.h"
#include "geometry/transformation.h"

#endif
