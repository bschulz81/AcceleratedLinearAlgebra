#include "datablock.h"
#include "mdspan_omp.h"
#include "mathfunctions.h"
#include "mathfunctions_mpi.h"
#include "expression_templates.h"
#include "mdspan_omp.h"
using namespace std;



// Main function
int main()
{


    cout<<"This demonstrates basic mathematical abilities of the library on gpu, cpu and with the message passing interface"<<endl;



    {
        cout<<"We can also use a more simplified interface for writing expressions. Although evaluations of more than one operator are not yet supported."<<endl;
        using namespace expr;

        // --------------------------
        // Matrix initialization
        // --------------------------
        std::vector<double> A_data =
        {
            1, 2, 3,
            4, 5, 6
        };
        std::vector<double> B_data =
        {
            6, 5, 4,
            3, 2, 1
        };

        std::vector<double>C_data(6,0);
        size_t rows = 2, cols = 3;
        cout<<"define A"<<endl;
        mdspan<double, std::array<size_t,2>> A(A_data.data(), rows, cols, true); // row-major
        A.printtensor();
        cout <<"define B"<<endl;
        mdspan<double, std::array<size_t,2>> B(B_data.data(), rows, cols, true);
        B.printtensor();

        //dynamic_tag is the same as a vector container
        mdspan_data_t<double, dynamic_tag> C(rows,cols, true,false);
        cout<<"addition of A and B"<<endl;
        C =A+B;
        C.printtensor();
        //static_tag<rank> is the same as an array container of <rank>
        mdspan_data_t<double, static_tag<2>> D(rows,rows, true,false);
        cout<<"multiplication of A and transpose of B"<<endl;
        D=A*B.transpose();

        D.printtensor();

        mdspan_data_t<double, dynamic_tag> E(rows,cols, true,false);
        cout<<"Subtraction of A. one can also assign the type later, as in this example, but E=A-B would also work here"<<endl;

        cout<<"But here we set a poliy to do this on gpu"<<endl;
        Math_Functions_Policy mypol(Math_Functions_Policy::AUTO);
        auto expr=A-B;
        expr.assign_to(E,&mypol);
        E.printtensor();


        cout<<"two vectors"<<endl;

        std::vector<double> vectorA_data =
        {
            1, 2, 3,
        };

        std::vector<double> vectorB_data =
        {
            6, 5, 4,
        };

        mdspan<double, std::array<size_t,1>> vecA(vectorA_data.data(), 3, true);
        mdspan<double, std::array<size_t,1>> vecB(vectorB_data.data(), 3, true);



        vecA.printtensor();
        vecB.printtensor();

        cout<<"a scalar product between two vectors"<<endl;

        auto c=dot(vecA,vecB).eval_scalar<double>();

        cout<<c<<endl;;

        double d=dot(vecA,vecB);
        cout<< d;


    }
    //

    {
        cout<<"We define two matrices"<<endl;
        vector<double>A_data(12*12,0);
        vector<double>B_data(12*12,0);
        vector<double>C_data(12*12,1);
        size_t rowsA = 12, colsA = 12;
        A_data=
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11,
            11, 9, 7, 5, 3, 1, 12, 10, 8, 6, 4, 2,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10,
            10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9,
            9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10,
            5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12,
            12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5,
            6, 1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11,
            11, 2, 9, 4, 12, 7, 3, 10, 5, 1, 8, 6
        };
        B_data= {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10,
                 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
                 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12,
                 12, 9, 6, 3, 10, 7, 4, 1, 8, 5, 2, 11,
                 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11,
                 11, 8, 5, 2, 9, 6, 3, 12, 7, 4, 1, 10,
                 3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10,
                 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
                 4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9,
                 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10
                };


        cout<<"the same code base can have the strides and extents on heap(vector) or on the stack(array). "<<endl;
        cout <<"The library works as well with col major data but in this example, we define row-major data"<<endl;

        mdspan<double, std::vector<size_t>> A(A_data.data(), {rowsA, colsA});
        mdspan<double, std::array<size_t,2>> B(B_data.data(), {rowsA, colsA});
        mdspan_data<double, std::vector<size_t>> C({rowsA, colsA});


        cout<<"Ordinary matrix multiplication, foced on gpu with a policy object"<<std::endl;


        A.printtensor();
        B.printtensor();

        cout <<"CPU_ONLY lets it multiply on CPU.AUTO lets the library decide based on whether the data is already on gpu, the algorithm, and the data size."<<endl;

        Math_Functions_Policy p1(Math_Functions_Policy::CPU_ONLY);
        cout<<"supplying nullptr instead of a pointer to Math_Functions_Policy lets the library use a global default that can be configured."<<endl;
        Math_Functions<double>::matrix_multiply_dot(A, B, C,&p1);
        cout<<"per default update_host is set to true. If one has several calculations on gpu, this may not be desired and can be switched to false"<<endl;


        C.printtensor();

        cout<<"We can also use the Strassen algorithm or its Winograd variant for the multiplication."<<std::endl;
        cout<<"It may offload on gpu. With the Message Passing Interface enabled, it can do so in parallel. "<<std::endl;
        cout<<"otherwise it offloads sequentially. The algorithm can also work entirely on device with devicepointers to the data"<<std::endl;

        cout<<"in auto mode, the following default treshholds are set in mathfunctions.h and can be changed for convenience"<<std::endl;
        cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded"<< std::endl;
        cout <<" default_cubic_treshold = 256;"<< "The default number of elements at which matrices are auto offloaded in multiplication"<< std::endl;
        cout<< " default_square_treshold = 1000;"<<"The default number of elements at which matrices are auto offloaded for addition"<< std::endl;
        cout <<" default_linear_treshold = 1000000;"<<"The default number of elements at which vectors are auto offloaded for addition"<<std::endl;
        cout <<std::endl;

        std::fill(C_data.begin(),C_data.end(),0);

        cout<<"we now set it on gpu and set the size when to stop recursion to 2, per default, this is at 64"<<endl;

        Math_MPI_RecursiveMultiplication_Policy p(Math_Functions_Policy::GPU_ONLY,false,false);
        p.size_to_stop_recursion=2;

        Math_Functions_MPI<double>::winograd_multiply(A, B, C,&p);

        C.printtensor();

    }

    {
        size_t rows = 4, cols = 4;


        cout<< "We create a 4x4 matrix that owns its own data buffer in a memapped file and then fill the buffer and print it"<<endl;
        cout<<"usually, the own data buffer is more interesting for storing the results of the computation and for intermediary evaluations"<<endl;
        mdspan_data<double, std::vector<size_t>> O( {rows, cols},true,true);


        for (size_t i=0; i<16; i++)
        {
            O.data()[i]=(double)i;
        }


        O.printtensor();


        cout<<"now we create a 4x4 matrix with data in a separate vector"<<endl;

        vector<double>O2_data(16,2);
        mdspan<double, std::vector<size_t>> O2(O2_data.data(),  {rows, cols},true);
        O2.printtensor();



        cout<< "now we make a shallow copy of the first matrix on the second"<<endl;

        O2=O;
        O2.printtensor();


        cout<<"We test the shallow copy by setting the first element of the first matrix to 42 and then print the first and second matrix"<<endl;
        O.data()[0]=42;

        O.printtensor();
        O2.printtensor();


    }




    {
        vector<double>A_data= {210, -92, 68, -33, -34, -4, 118, -6, -92, 318, -100, 130, -153, -64, 160, 33, 68, -100, 204, -96, 41, -69, -16, -26, -33, 130, -96, 338, -152, -51, 12, 22, -34, -153, 41, -152, 346, 11, -30, -25, -4, -64, -69, -51, 11, 175, -79, 5, 118, 160, -16, 12, -30, -79, 320, 7, -6, 33, -26, 22, -25, 5, 7, 239};
        size_t rows2 = 8, cols2 = 8;
        cout <<"Now we test more advanced algorithms"<<endl;
        {





            cout<<endl<<endl<<endl<<endl;
            cout<<"Now a cholesky decomposition on CPU"<<std::endl;

            mdspan<double, std::vector<size_t>> A(A_data.data(),  {rows2, cols2},true);

            cout<<"The result is put in an mdspan_data, which allocates its own ressources";
            mdspan_data<double, std::vector<size_t>> L({rows2, cols2},true);

            cout<<"with the dataset"<<endl;

            A.printtensor();
            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);

            Math_Functions<double>::cholesky_decomposition(A,L,&p);

            L.printtensor();

            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify(  {rows2, cols2},true);

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            cout<<"We can create a transpose with the base class DataBlock, but also with mdspan"<<endl;
            size_t newext[2],newstr[2];
            DataBlock<double>m=L.transpose(newext,newstr);
            Math_Functions<double>::matrix_multiply_dot(L,m, verify,&p2);
            verify.printtensor();
        }



        {

            cout<<"Now the cholesky decomposition is entirely done on GPU"<<std::endl;
            mdspan<double, std::vector<size_t>> A(A_data.data(), {rows2, cols2},true);
            mdspan_data<double, std::vector<size_t>> L({rows2, cols2},true );

            Math_Functions_Policy p(Math_Functions_Policy::GPU_ONLY);

            Math_Functions<double>::cholesky_decomposition(A,L,&p);

            L.printtensor();

            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify( {rows2, cols2},true);

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);

            cout<<"Here we create the transpose with mdspan"<<endl;
            mdspan<double,std::vector<size_t>> m=L.transpose();
            Math_Functions<double>::matrix_multiply_dot(L,m, verify,&p2);
            verify.printtensor();

        }


        {

            cout<<"With the advanced algorithms on GPU"<<std::endl;

            mdspan<double, std::vector<size_t>> A(A_data.data(), {rows2, cols2},true);
            mdspan_data<double, std::vector<size_t>> L(  {rows2, cols2},true);

            A.printtensor();

            Math_MPI_Decomposition_Policy p(
                Math_Functions_Policy::GPU_ONLY,
                false,
                false,
                Math_MPI_Decomposition_Policy::Naive);
            p.size_to_stop_recursion=2;
            Math_Functions_MPI<double>::cholesky_decomposition(A,L,&p);
            L.printtensor();


            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify( {rows2, cols2},true);

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            size_t newext[2],newstr[2];

            DataBlock<double>m=L.transpose(newext,newstr);
            Math_Functions<double>::matrix_multiply_dot(L,m, verify,&p2);
            verify.printtensor();

        }
    }

    {

        cout<< "Now we do the same with the lu decomposition of"<<std::endl;
        vector<double>A_data= {-3,3,-3,5,2,7,4,2,-2,4,2,-10,-4,-2,-10,1,-3,0,8,6,-3,-8,-8,-10,-6,-1,-4,-2,-4,-2,-3,1,-9,-10,5,-6,-8,1,-3,-8,-10,-8,-6,4,3,-8,-10,-6,3,-4,-2,4,4,-1,2,8,-4,6,9,-7,-6,-4,2,4};
        size_t rows3 = 8, cols3 = 8;

        {


            mdspan<double, std::vector<size_t>> A(A_data.data(),  {rows3, cols3},true);
            mdspan_data<double, std::vector<size_t>> L( {rows3, cols3},true);
            mdspan_data<double, std::vector<size_t>> U( {rows3, cols3},true);

            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);
            A.printtensor();

            cout<<"on CPU"<<std::endl;

            Math_Functions<double>::lu_decomposition(A,L,U,&p);
            L.printtensor();
            U.printtensor();

            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify( {rows3, cols3},true);
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(L,U, verify,&p2);
            verify.printtensor();

        }

        {


            mdspan<double, std::vector<size_t>> A(A_data.data(),  {rows3, cols3},true);
            mdspan_data<double, std::vector<size_t>> L( {rows3, cols3},true);
            mdspan_data<double, std::vector<size_t>> U({rows3, cols3},true);

            cout<<"Entirely on gpu"<<std::endl;
            Math_Functions_Policy p(Math_Functions_Policy::GPU_ONLY);
            Math_Functions<double>::lu_decomposition(A,L,U,&p);
            L.printtensor();
            U.printtensor();

            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify({rows3, cols3},true);
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(L,U, verify,&p2);
            verify.printtensor();
        }

        {



            mdspan<double, std::vector<size_t>> A(A_data.data(), {rows3, cols3},true);
            mdspan_data<double, std::vector<size_t>> L(  {rows3, cols3},true);
            mdspan_data<double, std::vector<size_t>> U(  {rows3, cols3},true);

            cout<<"With the advanced algorithms on GPU"<<std::endl;

            Math_MPI_Decomposition_Policy p(
                Math_Functions_Policy::GPU_ONLY,
                false,
                false,
                Math_MPI_Decomposition_Policy::Strassen);

            p.size_to_stop_recursion=2;
            Math_Functions_MPI<double>::lu_decomposition(A,L,U,&p);
            L.printtensor();


            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify({rows3, cols3},true);
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(L,U, verify,&p2);
            verify.printtensor();

        }
    }

    {

        cout<< "Now we do the same with the qr decomposition"<<std::endl;
        vector<double>A_data= {-4, 9, 4, 0, -3, -4, 8, 0, 0, -7, -3, -8, -9, 1, -5, -9, -10, 1, 1, 6, -1, 5, 4, 4, 8, 1, 9, -8, -6, 8, -4, -2, -4, 7, -7, 3, 7, -2, -9, 9, 4, -4, 1, -3, 4, -8, 3, 6, -7, 7, -3, -7, -9, -5, -1, -7, 7, 1, -9, -1, -7, 3, 5, 4};
        size_t rows4 = 8, cols4 = 8;
        {


            mdspan<double, std::vector<size_t>> A(A_data.data(),  {rows4, cols4},true);
            mdspan_data<double, std::vector<size_t>> Q( {rows4, cols4},true);
            mdspan_data<double, std::vector<size_t>> R( {rows4, cols4},true);


            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);
            A.printtensor();

            cout<<"On cpu"<<std::endl;
            Math_Functions<double>::qr_decomposition(A,Q,R,&p);
            Q.printtensor();
            R.printtensor();



            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify({rows4, cols4},true);
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(Q,R, verify,&p2);
            verify.printtensor();
        }


        {


            mdspan<double, std::vector<size_t>> A(A_data.data(),  {rows4, cols4},true);
            mdspan_data<double, std::vector<size_t>> Q(  {rows4, cols4},true);
            mdspan_data<double, std::vector<size_t>> R(  {rows4, cols4},true);


            cout<<"On gpu"<<std::endl;
            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);

            Math_Functions<double>::qr_decomposition(A,Q,R,&p);
            Q.printtensor();
            R.printtensor();


            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify({rows4, cols4},true);
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(Q,R, verify,&p2);
            verify.printtensor();

        }

        {
            cout<<"with the advanced algorithms on gpu "<<std::endl;

            mdspan<double, std::vector<size_t>> A(A_data.data(),  {rows4, cols4});
            mdspan_data<double, std::vector<size_t>> Q({rows4, cols4});
            mdspan_data<double, std::vector<size_t>> R({rows4, cols4});

            Math_MPI_Decomposition_Policy p(
                Math_Functions_Policy::GPU_ONLY,
                false,
                false,
                Math_MPI_Decomposition_Policy::Naive);

            p.size_to_stop_recursion=2;


            Math_Functions_MPI<double>::qr_decomposition(A,Q,R,&p);
            Q.printtensor();
            R.printtensor();
            vector<double>verifydata(64,0);

            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            mdspan_data<double, std::vector<size_t>> verify(  {rows4, cols4},true);
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(Q,R, verify,&p2);
            verify.printtensor();

        }
    }

}






