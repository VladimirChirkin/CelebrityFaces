#include<vector>
#include<iostream>
#include<random>
#include<ctime>
#include<cmath>
#include<iterator>
#include<fstream>
#include<algorithm>
#include<queue>
#include<unordered_set>
#include<cassert>
#include<memory>
#ifndef ANNOY_H
#define ANNOY_H

class FeatureVector {
 public:
  FeatureVector();

  FeatureVector(std::vector<double> vec);

  void PushBack(double feature);

  double Norm() const;

  int Size() const;

  double At(int index) const;

  double EuclideanDistance(const FeatureVector& other) const;

  double Dot(const FeatureVector& other) const;

  double Cos(const FeatureVector& other) const;

  const FeatureVector operator+(const FeatureVector& other) const;

  const FeatureVector operator-(const FeatureVector& other) const;

  double Margin(const FeatureVector& left,
                const FeatureVector& right) const;

  std::vector<double>* Values();

 private:
  std::vector<double> _components;
};

void SplitFeatures(int id_left, int id_right,
                   std::vector<int>::iterator begin,
                   std::vector<int>::iterator end,
                   std::back_insert_iterator<std::vector<int>> out_left,
                   std::back_insert_iterator<std::vector<int>> out_right,
                   std::vector<FeatureVector>* embeddings,
                   std::mt19937& gen);

struct Node {
  std::shared_ptr<Node> left;
  std::shared_ptr<Node> right;
  int left_point;
  int right_point;
  std::vector<int> leaf_ids;
  bool leaf = false;
};


class AnnoyTree {
 public:
  AnnoyTree(int node_size, std::mt19937& gen,
            std::uniform_int_distribution<int>& uin);

  void Fit(std::vector<FeatureVector>* embeddings);

  std::vector<int> Find(const FeatureVector& emb);

  std::shared_ptr<Node> Root();

 private:
  void _fit(const std::shared_ptr<Node>& node, std::vector<int> ids);

  std::vector<int> _find(const std::shared_ptr<Node>& node, const FeatureVector& emb);

  int _node_size;
  std::shared_ptr<Node> _root;
  std::mt19937& _gen;
  std::uniform_int_distribution<int>& _uin;
  std::vector<FeatureVector>* _embeddings;
};

class AnnoyForest {
 public:
  AnnoyForest(int node_size, int n_trees);

  void Fit(const std::vector<FeatureVector>& embeddings);

  std::vector<int> TreeFind(int tree_id, 
      const FeatureVector& emb);

  std::vector<int> Find(const FeatureVector& emb, int n_search);

 private:
  std::vector<AnnoyTree> _trees;
  std::mt19937 _gen;
  std::uniform_int_distribution<int> _uin;
  std::vector<FeatureVector> _embeddings;
  int _n_trees;
  int _node_size;
};

#endif
