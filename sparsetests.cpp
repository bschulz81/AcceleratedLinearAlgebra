
#include <vector>
#include <iostream>

#include "datablock.h"
#include "datablocksparseutils.h"
#include "inkernel_mathfunctions.h"
#include "gpu_mathfunctions.h"
#include "mdspan_omp.h"
#include "mdspan.hpp"
#include "mdspan_data.h"
#include "mathutilitiesdatablock.h"


using namespace std;

int main()
{


    ptrdiff_t M = 8, K = 8, N = 8;
    std::vector<double> A(M*K,0), B(K*N,0), C1(M*N,0),C2(M*N,0);

    // fill A and B with simple values
    for (ptrdiff_t i =0; i < M*K; ++i) A[i] = i;
    for (ptrdiff_t i = 0; i < K*N; ++i) B[i] = i + 1;


    // wrap dense arrays in DataBlock
    std::vector<ptrdiff_t> extA{M,K}, extB{K,N},extC{M,N};
    std::vector<ptrdiff_t> stridesA{K,1}, stridesB{N,1}, stridesC{N,1};

    auto Ad=DataBlock<double> (A.data(),M*K,2, extA.data(),stridesA.data(),DataBlockConfig{},ComputeMetadata{.ComputeLength=false});


    ptrdiff_t sub_ext[2],sub_strides[2];
    DataBlock<double> A1 = DataBlockUtilities::matrix_subspan(Ad,0, 0, 4, 4, sub_ext, sub_strides);
    DataBlock<double> A2 = DataBlockUtilities::matrix_subspan(Ad,4, 0, 4, 4, sub_ext, sub_strides);
    DataBlock<double> A3 = DataBlockUtilities::matrix_subspan(Ad,0, 4, 4, 4, sub_ext, sub_strides);
    DataBlock<double> A4 = DataBlockUtilities::matrix_subspan(Ad,4, 4, 4, 4, sub_ext, sub_strides);

// fill some blocks (e.g., leave A3 zero)
    for(ptrdiff_t i = 0; i < 4; ++i)
        for(ptrdiff_t j = 0; j < 4; ++j)
        {
            A1(i,j) = i+i;
            A2(i,j) = i;
            A3(i,j)=0;
            A4(i,j) =0;
        }


    ptrdiff_t sub_ext2[2],sub_strides2[2];

    auto Bd=DataBlock<double> (B.data(),K*N,2, extB.data(),stridesB.data(),DataBlockConfig{},ComputeMetadata{.ComputeLength=false});


    DataBlock<double> B1 =  DataBlockUtilities::matrix_subspan(Bd,0, 0, 4, 4, sub_ext2, sub_strides2);
    DataBlock<double> B2 =  DataBlockUtilities::matrix_subspan(Bd,4, 0, 4, 4, sub_ext2, sub_strides2);
    DataBlock<double> B3 =  DataBlockUtilities::matrix_subspan(Bd,0, 4, 4, 4, sub_ext2, sub_strides2);
    DataBlock<double> B4 =  DataBlockUtilities::matrix_subspan(Bd,4, 4, 4, 4, sub_ext2, sub_strides2);

// fill some blocks (e.g., leave A3 zero)
    for(ptrdiff_t i = 0; i < 4; ++i)
        for(ptrdiff_t j = 0; j < 4; ++j)
        {
            B1(i,j) = i;
            B2(i,j) = 0;
            B3(i,j)=0;
            B4(i,j) =0;
        }



    Bd.print();
cout <<"sparsity "<< DataBlockUtilities::sparsity(Bd)<<endl;


    auto C1d=DataBlock<double> (C1.data(),M*N,2, extC.data(),stridesC.data(),DataBlockConfig{},ComputeMetadata{.ComputeLength=false});

    auto C2d=DataBlock<double> (C2.data(),M*N,2, extC.data(),stridesC.data(),DataBlockConfig{},ComputeMetadata{.ComputeLength=false});
cout<<"naive matrix multiplication"<<endl;
    In_Kernel_Mathfunctions::matrix_multiply_dot(Ad, Bd, C1d);
    C1d.print();

    ptrdiff_t block_shape[2]={2,2};
    BlockedDataView<double> Ablocks(Ad, block_shape,true);
    ptrdiff_t block_shape2[2]={2,2};
    BlockedDataView<double> Bblocks(Bd, block_shape2,true);


cout<<"We now do a sparse multiplication"<<endl;
   In_Kernel_Mathfunctions::matrix_multiply_dot_sparse(Ablocks, Bblocks, C2d);
   //would also work on device
    //GPU_Math_Functions<double>::matrix_multiply_dot_sparse_g(Ablocks,Bblocks,C2d,omp_get_default_device(),true,true);

    C2d.print();

cout<<"now an example with sparse matrx multiplication and the mdspan class"<<endl;


mdspan<double, std::vector<ptrdiff_t>> Aspan(A.data(),  {M,K},DataBlockConfig{});
mdspan<double, std::vector<ptrdiff_t>> Bspan(B.data(),  {K,N},DataBlockConfig{});


mdspan_data<double, std::vector<ptrdiff_t>> Cspan({M,N},ManagedDataBlockConfig{});

cout<<"of course we offload the data first to device"<<endl;

cout<<"did the offload of A work?: "<<Aspan.device_data_upload(true)<<endl;
cout<<"did the offload of B work?: "<<Bspan.device_data_upload(true)<<endl;
cout<<"did the offload of C work?: "<<Cspan.device_data_alloc(true)<<endl;

//cout <<"sparsity "<<Bspan.sparsity()<<endl;
ptrdiff_t block_shape3[2]={2,2};
ptrdiff_t block_shape4[2]={2,2};
BlockedDataView<double> Ablocks1(Aspan, block_shape3,true);
BlockedDataView<double> Bblocks2(Bspan, block_shape4,true);


GPU_Math_Functions::matrix_multiply_dot_sparse_g(Ablocks1,Bblocks2,Cspan,GPUOptions{.device=omp_get_default_device(),.update_host=true});

Cspan.print();

Aspan.device_data_release();
Bspan.device_data_release();
Cspan.device_data_release();
    return 0;
}
