#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <cstddef>

#include "mdspan_data.h"
#include "mathfunctions.h"
#include "datablock_mpifunctions.h"
#include "mathfunctions_mpi.h"


using namespace std;

int main(int argc, char** argv)
{



//    int process_Rank, size_Of_Cluster;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
//    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

//    {
//
//        size_t rows = 4, cols = 4;
//        if(process_Rank == 0)
//        {
//
//            cout<<"this tests recursive algorithms of the library that use hybrid gpu and cpu mode with Message passing interface and OpenMP on device"<<endl;
//
//            vector<double>A2_data(16,4);
//            mdspan<double, std::vector<size_t>> A2(A2_data.data(),  {rows, cols},true);
//            DataBlock_MPI_Functions<double>::MPI_Send_DataBlock(A2,1,1,MPI_COMM_WORLD);
//            cout<<"Message Sent:\n";
//            A2.printtensor();
//        }
//        else if(process_Rank == 1)
//        {
//
//            cout<<"As a recieve buffer, mdspan_data is very useful, which allocates its own memory in the suitable size." <<endl;
//            cout<<" It can do so on a memory map, on host working memory, or on device memory, which is then accesible only with a device kernel"<<endl;
//            mdspan_data<double, std::vector<size_t>> B( {rows, cols},true);
//
//            DataBlock_MPI_Functions<double>::MPI_Recv_DataBlock(B,0,1,MPI_COMM_WORLD);
//            cout<<"Message recieved"<<endl;
//            B.printtensor();
//        }
//
//    }
//
//    {
//
//
//        size_t rows = 8, cols = 8;
//        Math_MPI_RecursiveMultiplication_Policy p(Math_Functions_Policy::GPU_ONLY,true,true);
//        p.update_host=true;
//        if(process_Rank == 0)
//        {
//            vector<double>A3_data(rows*cols,0);
//            vector<double>B3_data(rows*cols,0);
//            for (size_t i = 0; i < rows * cols; ++i)
//            {
//                A3_data[i] = i + 1;
//                B3_data[i] = i ;
//            }
//
//            mdspan<double, std::vector<size_t>> A3(A3_data.data(),  {rows, cols},true);
//            mdspan<double, std::vector<size_t>> B3(B3_data.data(), {rows, cols},true);
//
//            cout<<"We define two matrices A and B:" <<endl;
//            A3.printtensor();
//            B3.printtensor();
//
//            {
//
//                cout<<"ordinary matrix multiplication on a single node with openmp. It will decide automatically whether to compute on gpu or not"<<endl;
//
//                Math_Functions_Policy p1(Math_Functions_Policy::AUTO);
//                cout<<"supplying nullptr instead of a pointer to Math_Functions_Policy lets the library use a global default that can be configured."<<endl;
//                mdspan_data<double, std::vector<size_t>> C({rows, cols},true);
//                Math_Functions<double>::matrix_multiply_dot(A3, B3, C,&p1);
//                C.printtensor();
//
//            }
//
//            {
//
//                mdspan_data<double, std::vector<size_t>> C3({rows, cols},true);
//
//                cout<<"matrix multiplication with the Strassen algorithm over message passing interface"<<std::endl;
//                cout<<"in auto mode, the following default treshholds are set in mathfunctions.h and can be changed for convenience"<<std::endl;
//                cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded"<< std::endl;
//                cout <<" default_cubic_treshold = 256;"<< "The default number of elements at which matrices are auto offloaded in multiplication"<< std::endl;
//                cout<< " default_square_treshold = 1000;"<<"The default number of elements at which matrices are auto offloaded for addition"<< std::endl;
//                cout <<" default_linear_treshold = 1000000;"<<"The default number of elements at which vectors are auto offloaded for addition"<<std::endl<<endl;
//
//                Math_Functions_MPI<double>::strassen_multiply(A3, B3, C3,&p);
//                C3.printtensor();
//                Math_Functions_MPI<double>::MPI_recursion_helper_end(p.comm);
//            }
//        }
//        else
//        {
//            Math_Functions_MPI<double>::MPI_recursive_multiplication_helper(&p);
//        }
//
//    }


    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rootrank=0;
//    if (rank == rootrank)
//    {

//
//
//       std::vector<double> A_data =
//{
//     1,  2,  3,   4,  5,  6,   7,  8,  9,   10,
//    13, 14, 15,  16, 17, 18,  19, 20, 21,   22,
//    25, 26, 27,  28, 29, 30,  31, 32, 33,   34,
//    37, 38, 39,  40, 41, 42,  43, 44, 45,   46,
//    49, 50, 51,  52, 53, 54,  55, 56, 57,   58
//};
//      size_t extents[2] = {5,10};
//      size_t strides[2];
//
//      DataBlock<double> A1(A_data.data(),0, true,2, extents,strides,true, true, false, -1);
//
//        A1.printtensor();
//
//        DistributedDataBlock<double> block;
//        std::cout<<"scatter blocks";
//        DataBlock_MPI_Functions<double>::MPI_Scatter_matrix_to_submatrices_alloc(3,4,block,false, true , omp_get_default_device(), MPI_COMM_WORLD,0,&A1);
//        std::cout<<"printblocks from root"<<std::endl;
//        std::cout<< block.local_blocknumber();
//        block.printtensors();
//
//
//
//        DataBlock<double> A1copy;
//        DataBlock_MPI_Functions<double>::MPI_Gather_matrix_from_submatrices_alloc(block,MPI_COMM_WORLD,0,&A1copy,  false, false, -1);
//       A1copy.printtensor();
//        if(block.local_blocknumber()>0)
//        {
//          DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block);
//        }
//
//        DataBlock_MPI_Functions<double>::MPI_Free_DataBlock(A1copy,false);

    if (rank == rootrank)
    {
       std::vector<double> A_data =
{
     1,  2,  3,   4,  5,  6,   7,  8,  9,   10,
    13, 14, 15,  16, 17, 18,  19, 20, 21,   22,
    25, 26, 27,  28, 29, 30,  31, 32, 33,   34,
    37, 38, 39,  40, 41, 42,  43, 44, 45,   46,
    49, 50, 51,  52, 53, 54,  55, 56, 57,   58
};
      size_t extents[3] = {5,2,5};
      size_t strides[3];

      DataBlock<double> A1(A_data.data(),0, false,3, extents,strides,true, true, false, -1);


        A1.printtensor();

        DistributedDataBlock<double> block;
        std::cout<<"scatter block";
size_t blockrank=3;
size_t blockextents[3]={3,1,3};

        DataBlock_MPI_Functions<double>::MPI_Scatter_tensor_to_subtensors_alloc(blockrank,blockextents,block,false, false ,-1, MPI_COMM_WORLD,0,&A1);
        std::cout<<"printblocks from root"<<std::endl;
        std::cout<< block.local_blocknumber();
        block.printtensors();



        DataBlock<double> A1copy;
       DataBlock_MPI_Functions<double>::MPI_Gather_tensor_from_subtensors_alloc(block,MPI_COMM_WORLD,0,&A1copy,  false, false, -1);
      A1copy.printtensor();
        if(block.local_blocknumber()>0)
        {
          DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block);
        }

        DataBlock_MPI_Functions<double>::MPI_Free_DataBlock(A1copy,false);


    }
    else
    {


        DistributedDataBlock<double> block;

size_t blockrank=3;
size_t blockextents[3]={3,1,3};

        DataBlock_MPI_Functions<double>::MPI_Scatter_tensor_to_subtensors_alloc(blockrank,blockextents,block, false, false,-1,MPI_COMM_WORLD,0);

        std::cout<<"printblock"<<std::endl;
        std::cout<< block.local_blocknumber();

        block.printtensors();

       DataBlock_MPI_Functions<double>::MPI_Gather_tensor_from_subtensors_alloc(block,MPI_COMM_WORLD,0);

       if(block.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block);
        }

//        DistributedDataBlock<double> block;
//        DataBlock_MPI_Functions<double>::MPI_Scatter_matrix_to_submatrices_alloc(3,4,block,false, false,-1,MPI_COMM_WORLD,0);
//
//        std::cout<<"printblock"<<std::endl;
//        std::cout<< block.local_blocknumber();
//
//        block.printtensors();
//
//
//      DataBlock_MPI_Functions<double>::MPI_Gather_matrix_from_submatrices_alloc(block,MPI_COMM_WORLD,0);
//
//       if(block.local_blocknumber()>0)
//        {
//            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block);
//        }




    }


    MPI_Finalize();
    return 0;
}
