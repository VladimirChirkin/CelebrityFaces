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

typedef std::vector<double> features;

double euclidean_distance(const features& first, const features& second) {
  double sumd = 0;
  for (int index = 0; index < first.size(); ++index) {
    sumd += (first[index] - second[index]) * (first[index] - second[index]);
  }
  return std::sqrt(sumd);
}

 
void split_features(int id_left, int id_right,
                    std::vector<int>::iterator begin,
                    std::vector<int>::iterator end,
                    std::back_insert_iterator<std::vector<int>> out_left,
                    std::back_insert_iterator<std::vector<int>> out_right,
                    const std::vector<features>& embeddings) {
  for (auto iter = begin; iter != end; ++iter) {
    double side = euclidean_distance(embeddings.at(*iter), embeddings.at(id_left)) -
                  euclidean_distance(embeddings.at(*iter), embeddings.at(id_right));
    if (side < 0) {
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
  AnnoyTree(int node_size) {
    _node_size = node_size;
    _gen = std::mt19937(time(0));
  }
  
  void fit(std::vector<features> embeddings) {
    _uin = std::uniform_int_distribution<int>(0, embeddings.size());
    _embeddings = embeddings;
    std::vector<int> ids;
    for (int id = 0; id < embeddings.size(); ++id) {
      ids.push_back(id);
    }
    _fit(&_root, ids);
  }

  std::vector<int> find(const features& emb) {
    return _find(&_root, emb);
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
      split_features(left_point, right_point, ids.begin(), ids.end(),
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

  std::vector<int> _find(Node* node, const features& emb) {
    if (node->leaf_ids.size() > 0) {
      std::vector<std::pair<double, int>> pairs;
      for (int id : node->leaf_ids) {
        double score = euclidean_distance(emb, _embeddings[id]);
        pairs.push_back(std::pair<double, int>(score, id));
      }
      std::sort(pairs.begin(), pairs.end());
      std::vector<int> answer;
      for (auto pair : pairs) {
         answer.push_back(pair.second);
      }
      return answer;
    }
    double left_dist = euclidean_distance(emb, _embeddings[node->left_point]);
    double right_dist = euclidean_distance(emb, _embeddings[node->right_point]);
    if (left_dist < right_dist) {
      return _find(node->left, emb);
    }
    return _find(node->right, emb);
  }

  int _node_size;
  Node _root;
  std::mt19937 _gen;
  std::vector<features> _embeddings;
  std::uniform_int_distribution<int> _uin;
};

class AnnoyForest {

};
 
std::vector<features> read_embeddings(std::istream& input_stream = std::cin) {
  std::vector<features> result;
  int emb_count, emb_size;
  double value;
  input_stream >> emb_count >> emb_size;
  for (int emb_id = 0; emb_id < emb_count; ++emb_id) {
    features current;
    for (int feat_id = 0; feat_id < emb_size; ++ feat_id) {
      input_stream >> value;
      current.push_back(value);
    }
    result.push_back(current);
  }
  return result;
}
  

int main() {
  std::ifstream file;
  file.open("embeddings.txt");
  std::cout << "sizes: " << std::endl;
  std::vector<features> embs = read_embeddings(file);
  std::cout << embs.size() << " " << embs[0].size() << std::endl;
  std::vector<int> idsall;
  for (int count = 0; count < 10; ++count) {
    AnnoyTree tree(10000);
    std::cout << "fitting..." << std::endl;
    tree.fit(embs);
    std::cout << "searching..." << std::endl;
    std::vector<int> answer = tree.find(embs[10]);
    std::cout << "answer size: " <<  answer.size() << std::endl;
    for (int idx = 0; idx < 10; ++idx) {
      std::cout << answer[idx] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
