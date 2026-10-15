// SPDX-License-Identifier: MIT
//
// kernel_check: verify and time one mlir-aie bf16 x bf16 -> f32 matmul kernel
// (whole_array: A, B, C) against a CPU reference on random data.
//
//   g++-14 -O2 -std=c++20 -I../../runtime-cpp -I/opt/xilinx/xrt/include kernel_check.cpp \
//       -L/opt/xilinx/xrt/lib -lxrt_coreutil     (cont.) -Wl,-rpath,/opt/xilinx/xrt/lib -o kernel_check
//   ./kernel_check ../kernels/mm_bf16_f32_256x2304x256.xclbin \
//       ../kernels/mm_bf16_f32_256x2304x256.insts.bin 256 2304 256
//
// Note: a kernel that passes here can still fail when co-resident with other
// kernels on the NPU (see kernels-unsafe/); npu-face validates every layer at startup.
#include "xdna_runner.hpp"
#include <cstdio>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdlib>
static uint16_t f2bf(float f){uint32_t u;memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float bf2f(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;memcpy(&f,&u,4);return f;}
int main(int, char** a){
  const int M=atoi(a[3]),K=atoi(a[4]),N=atoi(a[5]);
  xdna::RunnerConfig rc{a[1],a[2],"",{size_t(M)*K*2,size_t(K)*N*2},size_t(M)*N*4}; rc.out_pos=2;
  xdna::XdnaRunner r(rc);
  std::mt19937 g(1); std::normal_distribution<float> d;
  int bad_runs=0; double worst=0;
  std::vector<float> A(M*K),B(K*N),ref(M*N);
  for(int it=0; it<10; it++){
    auto* pa=(uint16_t*)r.input(0).data(); auto* pb=(uint16_t*)r.input(1).data();
    for(int i=0;i<M*K;i++){pa[i]=f2bf(d(g));A[i]=bf2f(pa[i]);}
    for(int i=0;i<K*N;i++){pb[i]=f2bf(d(g));B[i]=bf2f(pb[i]);}
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){float s=0;for(int k=0;k<K;k++)s+=A[i*K+k]*B[k*N+j];ref[i*N+j]=s;}
    auto o=r.run(); auto* c=(const float*)o.data();
    double e=0,mx=0; for(int i=0;i<M*N;i++){e=std::max(e,(double)std::fabs(c[i]-ref[i])); mx=std::max(mx,(double)std::fabs(ref[i]));}
    worst=std::max(worst,e/mx); if(e/mx>1e-2) bad_runs++;
  }
  { // fine-grained precision: double-precision reference on the same bf16 inputs
    auto* pa=(uint16_t*)r.input(0).data(); auto* pb=(uint16_t*)r.input(1).data();
    for(int i=0;i<M*K;i++){pa[i]=f2bf(d(g));A[i]=bf2f(pa[i]);}
    for(int i=0;i<K*N;i++){pb[i]=f2bf(d(g));B[i]=bf2f(pb[i]);}
    auto o=r.run(); auto* c=(const float*)o.data();
    double sum_rel=0, max_rel=0, scale=std::sqrt((double)K);
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){double s=0;for(int k=0;k<K;k++)s+=(double)A[i*K+k]*B[k*N+j];
      double e=std::fabs(c[i*N+j]-s)/scale; sum_rel+=e; max_rel=std::max(max_rel,e);}
    printf("  error / sqrt(K): mean %.2e max %.2e (f32 accumulation ~1e-6, bf16 accumulation ~1e-3)\n", sum_rel/(M*N), max_rel);
  }
  auto t0=std::chrono::steady_clock::now(); for(int i=0;i<200;i++) r.run();
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/200;
  printf("10 runs: %d wrong, worst rel err %.5f; %.1f us/call => %.1f GFLOP/s\n",bad_runs,worst,us, 2.0*M*K*N/us/1e3);
}
