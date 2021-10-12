#include <cstdio>
#include <vector>
#include <algorithm>

int main(int argc, char *argv[])
{
  std::vector<float> float_vec = {1.5, -1.4, 0.4, -2.4, 2.4};
  std::vector<int> int_vec(float_vec.begin(), float_vec.end());
  
  // for (auto &v : int_vec) {
  //   printf("%d\n", v);
  // }

  std::vector<int> src = {0, 1, 2, 3, 4};
  std::vector<int> dst(src.size());
  auto it = std::copy_if(src.begin(), src.end(), dst.begin(), [](int s) {return s != 2;});
  dst.resize(std::distance(dst.begin(), it));
  for (auto &v : dst) {
    printf("%d\n", v);
  }

  return 0;
}

