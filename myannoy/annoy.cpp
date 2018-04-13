#include "annoy.h"

FeatureVector::FeatureVector() {}

FeatureVector::FeatureVector(std::vector<double> vec) :
    _components(vec) {}

void FeatureVector::PushBack(double feature) {
  _components.push_back(feature);
}

double FeatureVector::Norm() const {
  double norm = 0;
  for (const double component : _components) {
    norm += component * component;
  }
  return std::sqrt(norm);
}

int FeatureVector::Size() const {
  return _components.size();
}

double FeatureVector::At(int index) const {
  return _components.at(index);
}

double FeatureVector::EuclideanDistance(
    const FeatureVector& other) const {
  double sumd = 0;
  for (int index = 0; index < Size(); ++index) {
    sumd += (At(index) - other.At(index)) * (At(index) - other.At(index));
  }
  return std::sqrt(sumd);
}

double FeatureVector::Dot(const FeatureVector& other) const {
  double sumd = 0;
  for (int index = 0; index < Size(); ++index) {
    sumd += At(index) * other.At(index);
  }
  return sumd;
}

double FeatureVector::Cos(const FeatureVector& other) const {
  return Dot(other) / (Norm() * other.Norm() + 0.000001);
}

const FeatureVector FeatureVector::operator+(
    const FeatureVector& other) const {
  FeatureVector answer;
  for (int index = 0; index < Size(); ++index) {
    answer.PushBack(At(index) + other.At(index));
  }
  return answer;
}

const FeatureVector FeatureVector::operator-(
    const FeatureVector& other) const {
  FeatureVector answer;
  for (int index = 0; index < Size(); ++index) {
    answer.PushBack(At(index) - other.At(index));
  }
  return answer;
}


double FeatureVector::Margin(const FeatureVector& left,
              const FeatureVector& right) const {
  FeatureVector right_vec = right - left;
  FeatureVector x_vec = *this - left;
  double cos = right_vec.Cos(x_vec);
  double rv_norm = right_vec.Norm();
  double xv_norm = x_vec.Norm();
  double left_distance = xv_norm * cos;
  return left_distance - 0.5 * rv_norm;
}

std::vector<double>* FeatureVector::Values() {
  return &_components;
}   


void SplitFeatures(int id_left, int id_right,
                   std::vector<int>::iterator begin,
                   std::vector<int>::iterator end,
                   std::back_insert_iterator<std::vector<int>> out_left,
                   std::back_insert_iterator<std::vector<int>> out_right,
                   std::vector<FeatureVector>* embeddings) {
  for (auto iter = begin; iter != end; ++iter) {
    double left = embeddings->at(*iter).EuclideanDistance(
                       embeddings->at(id_left));
    double right = embeddings->at(*iter).EuclideanDistance(
                       embeddings->at(id_right));
    if (left < right) {
      *out_left = *iter;
    } else {
      *out_right = *iter;
    }
  }
}

  
AnnoyTree::AnnoyTree(int node_size, std::mt19937& gen, 
          std::uniform_int_distribution<int>& uin) :
    _node_size(node_size), _gen(gen), _uin(uin), _root(std::shared_ptr<Node>(new Node)) {}

void AnnoyTree::Fit(std::vector<FeatureVector>* embeddings) {
  _embeddings = embeddings;
  std::vector<int> ids;
  for (int id = 0; id < embeddings->size(); ++id) {
    ids.push_back(id);
  }
  _fit(_root, ids);
}

std::vector<int> AnnoyTree::Find(const FeatureVector& emb) {
  return _find(_root, emb);
}

std::shared_ptr<Node> AnnoyTree::Root() {
  return _root;
}

void AnnoyTree::_fit(const std::shared_ptr<Node>& node, std::vector<int> ids) {
  if (ids.size() < _node_size) {
    node->leaf_ids = ids;
    node->leaf = true;
  } else {
    int left_point = 0, right_point = 0;
    while (left_point == right_point) {
      left_point = _uin(_gen) % ids.size();
      right_point = _uin(_gen) % ids.size();
    }
    std::vector<int> ids_left, ids_right;
    SplitFeatures(left_point, right_point, ids.begin(), ids.end(),
                   std::back_inserter(ids_left),
                   std::back_inserter(ids_right),
                   _embeddings);
    node->left = std::shared_ptr<Node>(new Node);
    node->right = std::shared_ptr<Node>(new Node);
    node->left_point = left_point;
    node->right_point = right_point;
    _fit(node->left, ids_left);
    _fit(node->right, ids_right);
  }
}

std::vector<int> AnnoyTree::_find(
    const std::shared_ptr<Node>& node, const FeatureVector& emb) {
  if (node->leaf) {
    std::vector<std::pair<double, int>> pairs;
    for (int id : node->leaf_ids) {
      double score = emb.EuclideanDistance(_embeddings->at(id));
      pairs.push_back(std::pair<double, int>(score, id));
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<int> answer;
    for (auto pair : pairs) {
       answer.push_back(pair.second);
    }
    return answer;
  }
  double left_dist = emb.EuclideanDistance(_embeddings->at(node->left_point));
  double right_dist = emb.EuclideanDistance(
      _embeddings->at(node->right_point));
  if (left_dist < right_dist) {
    return _find(node->left, emb);
  }
  return _find(node->right, emb);
}
  

AnnoyForest::AnnoyForest(int node_size, int n_trees) :
      _node_size(node_size), _n_trees(n_trees) {}

void AnnoyForest::Fit(const std::vector<FeatureVector>& embeddings) {
  _gen.seed(time(0));
  _embeddings = embeddings;
  _uin = std::uniform_int_distribution<int>(0, embeddings.size() - 1);
  for (int tree_id = 0; tree_id < _n_trees; ++tree_id) {
    _trees.push_back(AnnoyTree(_node_size, _gen, _uin));
  }
  #pragma omp parallel
  {
  #pragma omp for
  for (int tree_id = 0; tree_id < _n_trees; ++tree_id) {
    _trees[tree_id].Fit(&_embeddings);
  }
  }
}

std::vector<int> AnnoyForest::TreeFind(int tree_id,
    const FeatureVector& emb) {
  return _trees[tree_id].Find(emb);
}

std::vector<int> AnnoyForest::Find(
    const FeatureVector& emb, int n_search) {
  std::priority_queue<std::pair<double, std::shared_ptr<Node>>> que;
  std::unordered_set<int> neighbors;
  for (int tree_id = 0; tree_id < _n_trees; ++tree_id) {
    que.push(std::make_pair(10000, _trees[tree_id].Root()));
  }
  while (neighbors.size() < n_search) {
    std::shared_ptr<Node> node = que.top().second;
    que.pop();
    if (node->leaf) {
      for (int id : node->leaf_ids) {
        neighbors.insert(id);
      }
    } else {
      double margin = emb.Margin(_embeddings[node->left_point], 
                                 _embeddings[node->right_point]);
      que.push(std::make_pair(margin, node->right));
      que.push(std::make_pair(-margin, node->left));
    }
  }
  std::vector<std::pair<double, int>> pairs;
  for (int id : neighbors) {
     double score = emb.EuclideanDistance(_embeddings.at(id));
     pairs.push_back(std::make_pair(score, id));
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int> answer;
  
  for (auto pair : pairs) {
     answer.push_back(pair.second);
  }
  return answer;
}

