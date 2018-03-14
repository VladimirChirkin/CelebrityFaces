#include <iostream>
#include <random>
#include <ctime>
#include<vector>
#include<iterator>

void merge(std::vector<int>::iterator begin_first,
           std::vector<int>::iterator end_first,
           std::vector<int>::iterator begin_second,
           std::vector<int>::iterator end_second,
           std::back_insert_iterator<std::vector<int>> out) {
  while ((begin_first != end_first) && (begin_second != end_second)) {
    if (*begin_first < *begin_second) {
      *out = *begin_first;
      ++begin_first;
    } else {
      *out = *begin_second;
      ++begin_second;
    }
  }
}

class Generator {
 public:
  Generator(std::mt19937& gen, std::uniform_int_distribution<int>& uin) :
      _uin(uin), _gen(gen) {}

  void generate() {
    std::cout << _uin(_gen);
  }
 private:
  std::uniform_int_distribution<int>& _uin;
  std::mt19937& _gen;
};

int main() {
  std::mt19937 gen(time(0));
  std::uniform_int_distribution<int> uin(0, 10);

  Generator generator(gen, uin);
  generator.generate();
  return 0;
}
