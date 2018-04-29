#include<iostream>
#include"annoy.cpp"
#include<random>
#include<vector>

std::vector<FeatureVector> GenerateEmbeddings(int n, int m) {
  std::mt19937 gen(0);
  std::normal_distribution<double> nd(0, 1);
  std::vector<FeatureVector> result;
  for (int i = 0; i < n; ++i) {
    FeatureVector vec;
    for (int j = 0; j < m; ++j) {
      vec.PushBack(nd(gen));
    }
    result.push_back(vec);
  }
  return result;
}


int main() {
  AnnoyForest forest(30, 1);
  std::vector<FeatureVector> embs = GenerateEmbeddings(10000, 128);
  forest.Fit(embs);
  std::vector<int> result = forest.Find(embs[0], 10);
  for (int i : result) {
    std::cout << "(" << i << " " << embs[0].EuclideanDistance(embs[i]) << ") ";
  }
  std::cout << std::endl << std::endl;
  std::vector<int> result1 = forest.TreeFind(0, embs[0]);
  for (int i : result1) {
    std::cout << "(" << i << " " << embs[0].EuclideanDistance(embs[i]) << ") ";
  }
  std::cout << std::endl;
}
