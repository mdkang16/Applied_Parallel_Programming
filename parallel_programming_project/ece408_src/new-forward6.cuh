#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 32
#define BLOCK_SIZE 1024


#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__constant__ float A[10000]; //7200 is enough for forward2
__global__ void matrixMultiplyShared(float* __restrict__ xptr, float* __restrict__ yptr, int M, int C_len, int H, int W, int K, int B_len) //  int numAColumns, int numCRows, int numCColumns, int K, int W_out, int H, int W)

{
__shared__ float mytileM[TILE_WIDTH][TILE_WIDTH];
__shared__ float mytileN[TILE_WIDTH][TILE_WIDTH];
// int batch = blockIdx.z;
const int H_out = H - K + 1;
const int W_out = W - K + 1;

//w.dptr_, ((C*H*W*batch) + x.dptr_), batch*(H_out*W_out)*M + (y.dptr_)

int numAColumns = K * K * C_len;
int numCRows = M;
int numCColumns = H_out * W_out;

int bx = blockIdx.x;
int by = blockIdx.y;

int tx = threadIdx.x;
int ty = threadIdx.y;

int row = by * blockDim.y + ty;
int col = bx * blockDim.x + tx;


for (int batch = blockIdx.z*TILE_WIDTH; batch < B_len && batch < (blockIdx.z+1)*TILE_WIDTH; batch++) {
  float* B = ((C_len*H*W*batch) + xptr);
  float* C = batch*(H_out*W_out)*M + (yptr);
  
  float pval = 0;
  int idx, x, y;

  #pragma unroll
  for(int i = 0; i < ceil((1.0) * numAColumns/TILE_WIDTH); i++) {
    int idx_col = i*TILE_WIDTH+threadIdx.x;
    if (idx_col < numAColumns) {
      mytileM[ty][tx] = A[row * numAColumns + idx_col];
    } else {
      mytileM[ty][tx] = 0;
    }
    
    // From kernel unroll
    int idx_row = ty + i * TILE_WIDTH;
    if(idx_row < numAColumns){
      idx = col + idx_row * numCColumns;
      x = idx % (numCColumns);
      y = idx / (numCColumns);

      int q = y % K;
      y = y / K;
      int c = y / K;
      int p = y % K;
      int h = x / W_out;
      int w = x % W_out;
      mytileN[ty][tx] = B[(H * W * c) + (h+p) * (W) + w+q];
    } else {
      mytileN[ty][tx] = 0;      
    }


    __syncthreads();
    #pragma unroll (32)
    for (int x = 0; x < TILE_WIDTH; ++x) {
      pval += mytileM[ty][x] * mytileN[x][tx];
    }
    __syncthreads();

  }

  if(row<numCRows){
    if(col<numCColumns){
      atomicAdd(&C[row * numCColumns + col], pval);
    }
  }
}
}

// __global__ void kernel_unroll(int B, int C, int H, int W, int K, float* I, float* op_unroll){
//     int H_out = H - K + 1;
//     int W_out = W - K + 1;

//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if(idx >= (C * K * K * H_out * W_out)){
//         return;
//     }

//     int x = idx % (H_out * W_out);
//     int y = idx / (H_out * W_out);

//     int q = y % K;
//     y = y / K;
//     int c = y / K;
//     int p = y % K;
//     int h = x / W_out;
//     int w = x % W_out;

//     op_unroll[idx] = I[(H * W * c) + (W * (h + p)) + (w + q)];
// }


// void unroll(int B, int C, int H, int W, int K, float* I, float* op_unroll, int H_out, int W_out){
//     int dimGrid = ceil((float) (K * K * H_out * W_out * C) / BLOCK_SIZE);
//     kernel_unroll<<<dimGrid, BLOCK_SIZE>>>(B, C, H, W, K, I, op_unroll);
// }



/*
 This function is called by new-inl.h
 Any code you write should be executed by this function.
 For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

  // Use mxnet's CHECK_EQ to do assertions.
  // Remove this assertion when you do your implementation!
  // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

  // Extract the tensor dimensions into B,M,C,H,W,K

  const int B = x.shape_[0];
  const int M = y.shape_[1];
  const int C = x.shape_[1];
  const int H = x.shape_[2];
  const int W = x.shape_[3];
  const int K = w.shape_[3];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  dim3 dimgrid(ceil((float) (H*W) / TILE_WIDTH), ceil((float) M / TILE_WIDTH), ceil((float)B/TILE_WIDTH));
  dim3 dimblock(TILE_WIDTH, TILE_WIDTH);
  cudaMemcpyToSymbol(A, w.dptr_, sizeof(float)*M*C*K*K);

  matrixMultiplyShared<<<dimgrid, dimblock>>>(x.dptr_, y.dptr_, M, C, H, W, K, B);

  // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/*
  This tells mxnet how to do an op when it's not a float.
  This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
  CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
