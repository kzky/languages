#include <cstdio>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

__global__
void kernel_sleep() {
  int a = 0;
  float threshold = 1e5;
  while (a < threshold) {
    a += 1;
  }
}

int main(int argc, char* argv[]) {
  // Setttings
  int num_events = 10;

  // Create evnets
  auto events = std::vector<cudaEvent_t>(num_events);
  for (int i = 0; i < num_events; ++i) {
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventBlockingSync);
    events[i] = std::move(event);
  }

  // Put kernels and Record events to default stream alternately
  for (int i = 0; i < num_events; ++i) {
    kernel_sleep<<<512, 512>>>();
    cudaEventRecord(events[i], 0);  // put it default stream in this sample
  }

  // Wait for events
  printf("Waiting for events starts\n");
  for (int i = 0; i < num_events; ++i) {
    cudaEventSynchronize(events[i]);
  }

  printf("Waiting for events finished\n");

  // Destroy Event
  for (int i = 0; i < num_events; ++i) {
    cudaEventDestroy(events[i]);
  }

  return 0;
}


