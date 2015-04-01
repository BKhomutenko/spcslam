/* 
All necessary geometric transformations
*/

#include <iostream>

//STL
#include <vector>

//Eigen
#include <Eigen/Eigen>
#include <cmath>

#include "geometry.h"

using namespace std;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;

Matrix3d rotationMatrix(const Vector3d & v)
{
    Matrix3d R;
    double th = v.norm();
    if (th < 1e-5)
    {
        //Rotational part in case when th is small
        
	    R <<        1,   -v(2),    v(1),
	          v(2),          1,   -v(0),
	         -v(1),    v(0),          1;
    }
    else
    {
        //Rodrigues formula
        double u1 = v(0) / th;
        double u2 = v(1) / th;
        double u3 = v(2) / th;
        double sinth = sin(th);
        double costhVar = 1 - cos(th);
        
        R(0, 0) = 1 + costhVar * (u1 * u1 - 1);
        R(1, 1) = 1 + costhVar * (u2 * u2 - 1);
        R(2, 2) = 1 + costhVar * (u3 * u3 - 1);

        R(0, 1) = -sinth*u3 + costhVar * u1 * u2;
        R(0, 2) = sinth*u2 + costhVar * u1 * u3;
        R(1, 2) = -sinth*u1 + costhVar * u2 * u3;

        R(1, 0) = sinth*u3 + costhVar * u2 * u1;
        R(2, 0) = -sinth*u2 + costhVar * u3 * u1;
        R(2, 1) = sinth*u1 + costhVar * u3 * u2;
    }
    return R;
}

Quaternion::Quaternion(const Vector3d & rot)
{
    double theta = rot.norm();
    if ( abs(theta) < 1e-6 )
    {
        x = rot(0)/2;
        y = rot(1)/2;
        z = rot(2)/2;
        w = 1 - theta*theta/8;  
    }
    else
    {
        Vector3d u = rot / theta;
        double s = sin(theta/2);
        x = u(0)*s;
        y = u(1)*s;
        z = u(2)*s;
        w = cos(theta/2); 
    }
} 

Vector3d Quaternion::rotate(const Vector3d & v) const
{
    double t1 =   w*x;
    double t2 =   w*y;
    double t3 =   w*z;
    double t4 =  -x*x;
    double t5 =   x*y;
    double t6 =   x*z;
    double t7 =  -y*y;
    double t8 =   y*z;
    double t9 =  -z*z;
    
    const double & v1 = v(0);
    const double & v2 = v(1);
    const double & v3 = v(2);
    
    double v1new = 2*((t7 + t9)*v1 + (t5 - t3)*v2 + (t2 + t6)*v3 ) + v1;
    double v2new = 2*((t3 + t5)*v1 + (t4 + t9)*v2 + (t8 - t1)*v3 ) + v2;
    double v3new = 2*((t6 - t2)*v1 + (t1 + t8)*v2 + (t4 + t7)*v3 ) + v3;
    
    return Vector3d(v1new, v2new, v3new);
} 


Vector3d Quaternion::toRotationVector() const
{
    double s = sqrt(x*x + y*y + z*z);
    double th = 2 * atan2(s, w);
    Vector3d u(x, y, z);
    if (th < 1e-5)
    {
        return u * 2;
    }
    else
    {
        return u / s * th;
    }
} 

Quaternion Quaternion::inv() const
{
    return Quaternion(-x, -y, -z, w);
}

Quaternion Quaternion::operator*(const Quaternion & q) const
{
    const double & x2 = q.x;
    const double & y2 = q.y;
    const double & z2 = q.z;
    const double & w2 = q.w;
    
    double wn = w*w2 - x*x2 - y*y2 - z*z2;
    double xn = w*x2 + x*w2 + y*z2 - z*y2;
    double yn = w*y2 - x*z2 + y*w2 + z*x2;
    double zn = w*z2 + x*y2 - y*x2 + z*w2;
    
    return Quaternion(xn, yn, zn, wn);
} 
        


Transformation::Transformation(double x, double y, double z,
    double qx, double qy, double qz, double qw)
    : mtrans(x, y, z) 
{
    mrot = Quaternion(qx, qy, qz, qw).toRotationVector();
}
    
Transformation::Transformation(const Vector3d & t, const Quaternion & q): mtrans(t)
{
   mrot = q.toRotationVector();
}

void Transformation::toRotTrans(Matrix3d & R, Vector3d & t) const
{
    t = mtrans;
    R = rotMat();
}

void Transformation::toRotTransInv(Matrix3d & R, Vector3d & t) const
{
    R = rotMat();
    R.transposeInPlace();
    t = -R*mtrans;
}

Transformation Transformation::compose(const Transformation & T) const
{
    Transformation res;
    Quaternion q1(mrot), q2(T.mrot);
    res.mtrans = q1.rotate(T.mtrans) + mtrans;
    Quaternion qres = q1 * q2;
    res.mrot = qres.toRotationVector();
    return res;
}


Transformation Transformation::inverseCompose(const Transformation & T) const
{
    Transformation res;
    Quaternion q1(mrot), q2(T.mrot);
    Quaternion q1inv = q1.inv();
    res.mtrans = q1inv.rotate(T.mtrans - mtrans);
    Quaternion qres = q1inv * q2;
    cout << qres << "#" << q1inv << "#" << q2 << endl;
    res.mrot = qres.toRotationVector();
    return res;
}

Quaternion Transformation::rotQuat() const { return Quaternion(mrot); }

Matrix3d Transformation::rotMat() const
{
    return rotationMatrix(mrot);
}

ostream& operator<<(ostream& os, const Quaternion& Q)
{
    os << Q.x << " " << Q.y << " " << Q.z << " " << Q.w;
    return os;
}

ostream& operator<<(ostream& os, const Transformation& T)
{
    os << T.mtrans.transpose() << " # " << T.mrot.transpose();
    return os;
}

void Transformation::transform(const vector<Vector3d> & src, vector<Vector3d> & dst) const
{
    assert(src.size() == dst.size());
    rotate(src, dst);
    for (auto & v : dst)
    {
        v += mtrans;
    }
}

void Transformation::inverseTransform(const vector<Vector3d> & src, vector<Vector3d> & dst) const
{
    assert(src.size() == dst.size());
    for (unsigned int i = 0; i < src.size(); i++)
    {
        dst[i] = src[i] - mtrans;
    }
    inverseRotate(dst, dst);
}

void Transformation::rotate(const vector<Vector3d> & src, vector<Vector3d> & dst) const
{
    assert(src.size() == dst.size());
    Matrix3d R = rotMat();
    for (unsigned int i = 0; i < src.size(); i++)
    {
        dst[i] = R * src[i];
    }
}

void Transformation::inverseRotate(const vector<Vector3d> & src, vector<Vector3d> & dst) const
{
    assert(src.size() == dst.size());
    Matrix3d R = rotMat();
    R.transposeInPlace();
    for (unsigned int i = 0; i < src.size(); i++)
    {
        dst[i] = R * src[i];
    }
}

