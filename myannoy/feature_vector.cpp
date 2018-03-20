#include<iostream>
#include<vector>
#include<unordered_map>
#include<unordered_set>
#include<string>
#include<random>
#include<ctime>
#include<cmath>
#include<iterator>
#include<fstream>
#include<algorithm>
#include<queue>

class FeatureVector {
 public:
  FeatureVector() {}

  FeatureVector(std::vector<double> vec) :
      _components(vec) {}

  void PushBack(double feature) {
    _components.push_back(feature);
  }

  double Norm() const {
    double norm = 0;
    for (const double component : _components) {
      norm += component * component;
    }
    return std::sqrt(norm);
  }

  int Size() const {
    return _components.size();
  }

  double At(int index) const {
    return _components.at(index);
  }

  double EuclideanDistance(const FeatureVector& other) const {
    double sumd = 0;
    for (int index = 0; index < Size(); ++index) {
      sumd += (At(index) - other.At(index)) * (At(index) - other.At(index));
    }
    return std::sqrt(sumd);
  }

  double Dot(const FeatureVector& other) const {
    double sumd = 0;
    for (int index = 0; index < Size(); ++index) {
      sumd += At(index) * other.At(index);
    }
    return sumd;
  }

  double Cos(const FeatureVector& other) const {
    return Dot(other) / (Norm() * other.Norm());
  }

  const FeatureVector operator+(const FeatureVector& other) const {
    FeatureVector answer;
    for (int index = 0; index < Size(); ++index) {
      answer.PushBack(At(index) + other.At(index));
    }
    return answer;
  }

  const FeatureVector operator-(const FeatureVector& other) const {
    FeatureVector answer;
    for (int index = 0; index < Size(); ++index) {
      answer.PushBack(At(index) - other.At(index));
    }
    return answer;
  }


  double Margin(const FeatureVector& left,
                const FeatureVector& right) const {
    FeatureVector right_vec = right - left;
    FeatureVector x_vec = *this - left;
    double cos = right_vec.Cos(x_vec);
    double rv_norm = right_vec.Norm();
    double xv_norm = x_vec.Norm();
    double left_distance = xv_norm * cos;
    return left_distance - 0.5 * rv_norm;
  }

  std::vector<double>* Values() {
    return &_components;
  }   
  
 private:
  std::vector<double> _components;
};
