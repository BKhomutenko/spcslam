/*
All necessary geometric transformations
*/

#ifndef _SPCMAP_GEOMETRY_H_
#define _SPCMAP_GEOMETRY_H_

//STL
#include <vector>

//Eigen
#include <Eigen/Eigen>

#include "quaternion.h"

template<class T>
using Vector2 = Eigen::Matrix<T, 2, 1>; 
template<class T>
using Vector3 = Eigen::Matrix<T, 3, 1>; 
template<class T>
using Matrix3 = Eigen::Matrix<T, 3, 3>; 

using namespace std;
//TODO think about how to perform the combination of these transformations

template<class T>
Matrix3<T> rotationMatrix(const Vector3<T> & v)
{
    Matrix3<T> R;
    T th = v.norm();
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
        T u1 = v(0) / th;
        T u2 = v(1) / th;
        T u3 = v(2) / th;
        T sinth = sin(th);
        T costhVar = 1 - cos(th);
        
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


template<class T>
inline Matrix3<T> hat(const Vector3<T> & u)
{
    Matrix3<T> M;
    M << 0, -u(2), u(1),   u(2), 0, -u(0),   -u(1), u(0), 0;
    return M;
}


// Non-redundant transformation representation
// using translation and angle-axis

template<class T>
class Transformation
{
public:
    //FIXME
    Transformation() : mrot(0, 0, 0), mtrans(0, 0, 0) {}
    
    Transformation(Vector3<T> trans, Vector3<T> rot) : mtrans(trans), mrot(rot) {}

    Transformation(const Vector3<T> & trans, const Quaternion<T> & qrot)
    : mtrans(trans), mrot(qrot.toRotationVector()) {}
    
    Transformation(const T * const data) : mtrans(data), mrot(data + 3) {}
    
    Transformation(T x, T y, T z, T rx, T ry, T rz)
    : mtrans(x, y, z), mrot(rx, ry, rz) {}
    
    Transformation(T x, T y, T z, T qx, T qy, T qz, T qw)
    : mtrans(x, y, z), mrot(Quaternion<T>(qx, qy, qz, qw).toRotationVector()) {}
    
    

    void toRotTrans(Matrix3<T> & R, Vector3<T> & t) const
    {
        t = mtrans;
        R = rotMat();
    }

    void toRotTransInv(Matrix3<T> & R, Vector3<T> & t) const
    {
        R = rotMat();
        R.transposeInPlace();
        t = -R*mtrans;
    }

    Transformation compose(const Transformation & transfo) const
    {
        Transformation res;
        Quaternion<T> q1(mrot), q2(transfo.mrot);
        res.mtrans = q1.rotate(transfo.mtrans) + mtrans;
        Quaternion<T> qres = q1 * q2;
        res.mrot = qres.toRotationVector();
        return res;
    }
    
    Transformation inverseCompose(const Transformation & transfo) const
    {
        Transformation res;
        Quaternion<T> q1(mrot), q2(transfo.mrot);
        Quaternion<T> q1inv = q1.inv();
        res.mtrans = q1inv.rotate(transfo.mtrans - mtrans);
        Quaternion<T> qres = q1inv * q2;
        res.mrot = qres.toRotationVector();
        return res;
    }

    const Vector3<T> & trans() const { return mtrans; }

    const Vector3<T> & rot() const { return mrot; }

    Vector3<T> & trans() { return mtrans; }

    Vector3<T> & rot() { return mrot; }

    Quaternion<T> rotQuat() const { return Quaternion<T>(mrot); } 
    
    Matrix3<T> rotMat() const { return rotationMatrix<T>(mrot); }

    T * rotData() { return mrot.data(); }
    T * transData() { return mtrans.data(); }

    friend ostream& operator << (ostream & os, const Transformation & transfo)
    {
        os << transfo.mtrans.transpose() << " # " << transfo.mrot.transpose();
        return os;
    }

    void transform(const vector<Vector3<T> > & src, vector<Vector3<T> > & dst) const
    {
        dst.resize(src.size());
        rotate(src, dst);
        for (auto & v : dst)
        {
            v += mtrans;
        }
    }
    
    void inverseTransform(const vector<Vector3<T> > & src, vector<Vector3<T> > &dst) const
    {
        dst.resize(src.size());
        for (unsigned int i = 0; i < src.size(); i++)
        {
            dst[i] = src[i] - mtrans;
        }
        inverseRotate(dst, dst);
    }

    void rotate(const vector<Vector3<T> > & src, vector<Vector3<T> > & dst) const
    {
        dst.resize(src.size());
        Matrix3<T> R = rotMat();
        for (unsigned int i = 0; i < src.size(); i++)
        {
            dst[i] = R * src[i];
        }
    }

    void inverseRotate(const vector<Vector3<T> > & src, vector<Vector3<T> > &dst) const
    {
        dst.resize(src.size());
        Matrix3<T> R = rotMat();
        R.transposeInPlace();
        for (unsigned int i = 0; i < src.size(); i++)
        {
            dst[i] = R * src[i];
        }
    }
    
private:
    Vector3<T> mrot;
    Vector3<T> mtrans;

};

//template<class T>
//ostream& operator << (ostream& os, const Transformation & transfo)
//{
//    os << transfo.mtrans.transpose() << " # " << transfo.mrot.transpose();
//    return os;
//}

#endif
