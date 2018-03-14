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


std::vector<FeatureVector> read_embeddings(
    std::istream& input_stream = std::cin) {
  std::vector<FeatureVector> result;
  int emb_count, emb_size;
  double value;
  input_stream >> emb_count >> emb_size;
  for (int emb_id = 0; emb_id < emb_count; ++emb_id) {
    FeatureVector current;
    for (int feat_id = 0; feat_id < emb_size; ++ feat_id) {
      input_stream >> value;
      current.PushBack(value);
    }
    result.push_back(current);
  }
  return result;
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


  
struct Node {
  Node* left = nullptr;
  Node* right = nullptr;
  int left_point;
  int right_point;
  std::vector<int> leaf_ids;
};


class AnnoyTree {
 public:
  AnnoyTree(int node_size, std::mt19937& gen, 
            std::uniform_int_distribution<int>& uin) :
      _node_size(node_size), _gen(gen), _uin(uin) {}

  void Fit(std::vector<FeatureVector>* embeddings) {
    _embeddings = embeddings;
    std::vector<int> ids;
    for (int id = 0; id < embeddings->size(); ++id) {
      ids.push_back(id);
    }
    _fit(&_root, ids);
  }

  std::vector<int> Find(const FeatureVector& emb) {
    return _find(&_root, emb);
  }

  Node* Root() {
    return &_root;
  }

 private:
  void _fit(Node* node, std::vector<int> ids) {
    if (ids.size() < _node_size) {
      node->leaf_ids = ids;
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
      Node* left = new Node;
      Node* right = new Node;
      node->left = left;
      node->right = right;
      node->left_point = left_point;
      node->right_point = right_point;
      _fit(left, ids_left);
      _fit(right, ids_right);
    }
  }

  std::vector<int> _find(Node* node, const FeatureVector& emb) {
    if (node->leaf_ids.size() > 0) {
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
  

  int _node_size;
  Node _root;
  std::mt19937& _gen;
  std::uniform_int_distribution<int>& _uin;
  std::vector<FeatureVector>* _embeddings;
};

class AnnoyForest {
 public:
  AnnoyForest(int node_size, int n_trees) :
      _node_size(node_size), _n_trees(n_trees), _gen(std::mt19937(time(0))) {}

  void Fit(const std::vector<FeatureVector>& embeddings) {
    _embeddings = embeddings;
    _uin = std::uniform_int_distribution<int>(0, embeddings.size());
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

  std::vector<int> Find(const FeatureVector& emb, int n_search) {
    std::priority_queue<std::pair<double, Node*>> que;
    std::unordered_set<int> neighbors;
    for (int tree_id = 0; tree_id < _n_trees; ++tree_id) {
      que.push(std::make_pair(10000, _trees[tree_id].Root()));
    }
    while (neighbors.size() < n_search) {
      Node* node = que.top().second;
      que.pop();
      if (!node->leaf_ids.empty()) {
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

 private:
  std::vector<AnnoyTree> _trees;
  std::mt19937 _gen;
  std::uniform_int_distribution<int> _uin;
  std::vector<FeatureVector> _embeddings;
  int _n_trees;
  int _node_size;
};

std::vector<FeatureVector> GenerateEmbeddings(int n_embs, int emb_size) {
  std::vector<FeatureVector> result;
  std::mt19937 gen(time(0));
  std::normal_distribution<double> nd;
  for (int index = 0; index < n_embs; ++index) {
    FeatureVector current;
    for (int feat_id = 0; feat_id < emb_size; ++feat_id) {
      current.PushBack(nd(gen));
    }
    result.push_back(current);
  }
  return result;
}


int main() {
  std::cout << "Opening embeddings..." << std::endl;
  int tim = clock();
  std::ifstream file;
  file.open("embeddings.txt");
  std::vector<FeatureVector> embs = read_embeddings(file);
  // std::vector<FeatureVector> embs = GenerateEmbeddings(1000, 10);
  std::cout << "time: " << 1000 * (clock() - tim) / CLOCKS_PER_SEC
            << std::endl;
  std::mt19937 gen(time(0));
  std::uniform_int_distribution<int> uin(0, embs.size());
  std::cout << "Fitting AnnoyForest..." << std::endl;
  tim = clock();
  AnnoyForest forest(100, 8);
  forest.Fit(embs);
  std::cout << "time: " << 1000 * (clock() - tim) / CLOCKS_PER_SEC
            << std::endl;
  tim = clock();
  std::cout << "Testing 1000 cycles" << std::endl;
  for (int counter = 0; counter < 1000; ++counter) {
    int id = uin(gen);
    std::vector<int> answer = forest.Find(embs[id], 10);
  }
  std::cout << "time: " << 1000 * (clock() - tim) / CLOCKS_PER_SEC
            << std::endl;
  /*
  AnnoyTree tree(1000, gen, uin);
  std::cout << "fitting..." << std::endl;
  tree.fit(&embs);
  std::cout << "searching..." << std::endl;
  std::vector<int> answer = tree.find(embs[10]);
  */
  std::cout << std::endl;

  return 0;
}
