#include <stdio.h>
#include <mpi.h>
#include <vector>
#include "mdspan_data.h"
#include "mathfunctions.h"
#include "datastruct_mpifunctions.h"
#include "mathfunctions_mpi.h"


using namespace std;

int main(int argc, char** argv)
{
    int process_Rank, size_Of_Cluster;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    size_t rows = 4, cols = 4;
//demonstrates recieve and send. Works if uncommented.
    if(process_Rank == 0)
    {


        vector<double>A2_data(16,4);
        mdspan<double, std::vector<size_t>> A2(A2_data.data(), true, {rows, cols});
        Datastruct_MPI_Functions<double>::MPI_Send_datastruct(A2,1,1,MPI_COMM_WORLD);
        cout<<"Message Sent:\n";
        A2.printmatrix();
    }
//
    else if(process_Rank == 1)
    {
        vector<double>B2_data(16,2);
        mdspan_data<double, std::vector<size_t>> B(B2_data.data(), true, {rows, cols});

        Datastruct_MPI_Functions<double>::MPI_Recv_datastruct(B,0,1,MPI_COMM_WORLD);
        cout<<"Message recieved";
        B.printmatrix();
        B(1,1)=42;

//
    }

//
 rows = 8, cols = 8;
////
//

    Math_MPI_RecursiveMultiplication_Policy p(Math_Functions_Policy::GPU_ONLY,true,true);
    p.update_host=true;
    if(process_Rank == 0)
    {

    cout<<"matrix multiplication with the Strassen algorithm over message passing interface. For GPU only, the variable Unified_Shared_Memory should be defined"<<std::endl;
    cout<<"Note: if GPU_ONLY is set, and Unified_Shared_Memory is not defined, then ordinary matrix multiplication on gpu will be called."<<std::endl;
    cout<<"in auto mode, the following default treshholds are set in mathfunctions.h and can be changed for convenience"<<std::endl;
    cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded"<< std::endl;
    cout <<" default_cubic_treshold = 256;"<< "The default number of elements at which matrices are auto offloaded in multiplication"<< std::endl;
    cout<< " default_square_treshold = 1000;"<<"The default number of elements at which matrices are auto offloaded for addition"<< std::endl;
    cout <<" default_linear_treshold = 1000000;"<<"The default number of elements at which vectors are auto offloaded for addition"<<std::endl;


//        // Define some data for matrix multiplication A*B=C;
        vector<double>A3_data(64,0);
        vector<double>B3_data(64,0);
        vector<double>C3_data(64,1);
////
////        // Allocate data
        for (size_t i = 0; i < rows * cols; ++i)
        {
            A3_data[i] = i + 1; // Example initialization
            B3_data[i] = i ; // Example initialization
        }
//
        mdspan<double, std::vector<size_t>> A3(A3_data.data(), true, {rows, cols});
        mdspan<double, std::vector<size_t>> B3(B3_data.data(), true, {rows, cols});
        mdspan<double, std::vector<size_t>> C3(C3_data.data(), true, {rows, cols});
//
        A3.printmatrix();
        B3.printmatrix();



        Math_Functions_MPI<double>::strassen_multiply(A3, B3, C3,&p);
        C3.printmatrix();
        Math_Functions_MPI<double>::MPI_recursion_helper_end(p.comm);
    }
    else
    {
        Math_Functions_MPI<double>::MPI_recursive_multiplication_helper(&p);
    }


    MPI_Finalize();
    return 0;
}
