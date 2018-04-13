#include<iostream>
#include"annoy.cpp"
#include<random>
#include<vector>

std::vector<FeatureVector> GenerateEmbeddings(int n, int m) {
  std::mt19937 gen;
  std::normal_distribution<double> nd(0, 1);
  std::vector<FeatureVector> result;
  for (int i = 0; i < m; ++i) {
    FeatureVector vec;
    for (int j = 0; j < n; ++j) {
      vec.PushBack(nd(gen));
    }
    result.push_back(vec);
  }
  return result;
}


int main() {
  AnnoyForest forest(100, 10);
  std::vector<FeatureVector> embs = GenerateEmbeddings(1000, 10);
  forest.Fit(embs);
}
