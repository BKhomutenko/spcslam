/*
All necessary geometric transformations
*/

#ifndef _SPCMAP_GEOMETRY_H_
#define _SPCMAP_GEOMETRY_H_

//STL
#include <vector>

//Eigen
#include <Eigen/Eigen>

using namespace std;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
//TODO think about how to perform the combination of these transformations

Matrix3d rotationMatrix(const Vector3d & v);

class Quaternion
{
public:
    Quaternion() {}
    Quaternion(double x, double y, double z, double w) : data{x, y, z, w} {}
    Quaternion(const Vector3d & rot);
    Vector3d rotate(const Vector3d & v) const;
    Vector3d toRotationVector() const;
    Quaternion inv() const;
    Quaternion operator*(const Quaternion & q) const;
    friend ostream& operator << (ostream & os, const Quaternion & Q);
private:
    double data[4];
    double & x = data[0];
    double & y = data[1];
    double & z = data[2];
    double & w = data[3];
};

// Non-redundant transformation representation
// using translation and angle-axis
class Transformation
{
public:
    //FIXME
    Transformation() : mrot(0, 0, 0), mtrans(0, 0, 0) {}
    Transformation(Vector3d trans, Vector3d rot) : mrot(rot), mtrans(trans) {}
    Transformation(double x, double y, double z, double rx, double ry, double rz)
        : mrot(rx, ry, rz), mtrans(x, y, z) { }
    Transformation(double x, double y, double z, double qx, double qy, double qz, double qw);
    Transformation(const Vector3d & t, const Quaternion & q);

    void toRotTrans(Matrix3d & Rot, Vector3d & tr) const;

    void toRotTransInv(Matrix3d & Rot, Vector3d & tr) const;

    Transformation compose(const Transformation & T) const;

    Transformation inverseCompose(const Transformation & T) const;

    const Vector3d & trans() const { return mtrans; }

    const Vector3d & rot() const { return mrot; }

    Vector3d & trans() { return mtrans; }

    Vector3d & rot() { return mrot; }

    Quaternion rotQuat() const;

    Matrix3d rotMat() const;

    double * rotData() { return mrot.data(); }
    double * transData() { return mtrans.data(); }

    friend ostream& operator << (ostream & os, const Transformation & T);

    void transform(const vector<Vector3d> & src, vector<Vector3d> & dst) const;
    void inverseTransform(const vector<Vector3d> & src, vector<Vector3d> &dst) const;

    void rotate(const vector<Vector3d> & src, vector<Vector3d> & dst) const;
    void inverseRotate(const vector<Vector3d> & src, vector<Vector3d> &dst) const;
private:
    Vector3d mrot;
    Vector3d mtrans;

};

#endif
