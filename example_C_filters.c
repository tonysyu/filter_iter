#include <stdlib.h>


float mean_filt(float* buf, int n)
{
  float res = 0.0f;
  int i;
  for (i=0; i<n; i++) res += buf[i];
  return res/n;
}

float rand_select(float* buf, int n)
{
  int idx;
  idx = random() % n;
  return buf[idx];
}

void* rand_select_addr = rand_select;
void* mean_filt_addr = mean_filt;
