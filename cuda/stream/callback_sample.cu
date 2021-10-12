#include <cstdio>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <memory>

__global__
void kernel_sleep() {
  int a = 0;
  float threshold = 1e4;
  while (a < threshold) {
    a += 1;
  }
}

class UserData {
public:
  int i = 0;
  int j = 0;
};

void callback_func(cudaStream_t stream, cudaError_t status, void *userData) {
  UserData *user_data = (UserData*)userData;
  printf("callbacked, i = %d\n", user_data->i);
//  printf("callbacked, error= %d\n", status);
}

int main(int argc, char* argv[]) {
  // Setttings
  int num_callbacks = 10;


  // Add callback
  std::vector<std::shared_ptr<UserData>> user_data_list;
  for (int i = 0; i < num_callbacks; ++i) {
    std::shared_ptr<UserData> user_data = std::make_shared<UserData>();
    user_data->i = i;
    user_data_list.push_back(user_data);
    printf("i=%d\n", user_data->i);
    cudaStreamAddCallback(0, callback_func, user_data.get(), 0);

  }

  cudaDeviceSynchronize();

  return 0;
}


