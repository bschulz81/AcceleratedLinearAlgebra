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



    int process_Rank, size_Of_Cluster;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    {

        size_t rows = 4, cols = 4;
        if(process_Rank == 0)
        {

            cout<<"this tests recursive algorithms of the library that use hybrid gpu and cpu mode with Message passing interface and OpenMP on device"<<endl;



            vector<double>A2_data(16,4);
            mdspan<double, std::vector<size_t>> A2(A2_data.data(),  {rows, cols},true);
            DataBlock_MPI_Functions<double>::MPI_Send_DataBlock(A2,1,1,MPI_COMM_WORLD);
            cout<<"Message Sent:\n";
            A2.printtensor();
        }

        else if(process_Rank == 1)
        {

            cout<<"As a recieve buffer, mdspan_data is very useful, which allocates its own memory in the suitable size." <<endl;
            cout<<" It can do so on a memory map, on host working memory, or on device memory, which is then accesible only with a device kernel"<<endl;
            mdspan_data<double, std::vector<size_t>> B( {rows, cols},true);

            DataBlock_MPI_Functions<double>::MPI_Recv_DataBlock(B,0,1,MPI_COMM_WORLD);
            cout<<"Message recieved"<<endl;
            B.printtensor();
            B(1,1)=42;
        }

    }

    {


        size_t rows = 8, cols = 8;
        Math_MPI_RecursiveMultiplication_Policy p(Math_Functions_Policy::GPU_ONLY,true,true);
        p.update_host=true;
        if(process_Rank == 0)
        {
            vector<double>A3_data(rows*cols,0);
            vector<double>B3_data(rows*cols,0);
            for (size_t i = 0; i < rows * cols; ++i)
            {
                A3_data[i] = i + 1;
                B3_data[i] = i ;
            }

            mdspan<double, std::vector<size_t>> A3(A3_data.data(),  {rows, cols},true);
            mdspan<double, std::vector<size_t>> B3(B3_data.data(), {rows, cols},true);

            cout<<"We define two matrices A and B:" <<endl;
            A3.printtensor();
            B3.printtensor();

            {

                cout<<"ordinary matrix multiplication on a single node with openmp. It will decide automatically whether to compute on gpu or not"<<endl;

                Math_Functions_Policy p1(Math_Functions_Policy::AUTO);
                cout<<"supplying nullptr instead of a pointer to Math_Functions_Policy lets the library use a global default that can be configured."<<endl;
                mdspan_data<double, std::vector<size_t>> C({rows, cols},true);
                Math_Functions<double>::matrix_multiply_dot(A3, B3, C,&p1);
                C.printtensor();

            }

            {

                mdspan_data<double, std::vector<size_t>> C3({rows, cols},true);

                cout<<"matrix multiplication with the Strassen algorithm over message passing interface"<<std::endl;
                cout<<"in auto mode, the following default treshholds are set in mathfunctions.h and can be changed for convenience"<<std::endl;
                cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded"<< std::endl;
                cout <<" default_cubic_treshold = 256;"<< "The default number of elements at which matrices are auto offloaded in multiplication"<< std::endl;
                cout<< " default_square_treshold = 1000;"<<"The default number of elements at which matrices are auto offloaded for addition"<< std::endl;
                cout <<" default_linear_treshold = 1000000;"<<"The default number of elements at which vectors are auto offloaded for addition"<<std::endl<<endl;

                Math_Functions_MPI<double>::strassen_multiply(A3, B3, C3,&p);
                C3.printtensor();
                Math_Functions_MPI<double>::MPI_recursion_helper_end(p.comm);
            }
        }
        else
        {
            Math_Functions_MPI<double>::MPI_recursive_multiplication_helper(&p);
        }

    }
    MPI_Finalize();
    return 0;
}
