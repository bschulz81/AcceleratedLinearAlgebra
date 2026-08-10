
#include <vector>
#include <iostream>

#include "datablock.h"
#include "datablock_mpifunctions.h"
#include "mathutilitiesdatablock.h"


#include "mathfunctions.h"
#include "mathfunctions_mpi.h"

#include "mdspan_omp.h"
#include "mdspan.hpp"
#include "mdspan_data.h"


#include "mathutilitiesmdspan.h"
#include "expression_templates.h"

using namespace std;



// Main function
int main()
{
    {



        cout<<"This demonstrates basic mathematical abilities of the library on gpu, cpu"<<endl;



        cout << "We can also use a more simplified interface for writing expressions." << endl;
        using namespace expr;

        std::vector<double> A_data = { 1, 2, 3, 4, 5, 6 };
        std::vector<double> B_data = { 6, 5, 4, 3, 2, 1 };
        std::vector<double> C_data(6, 0);
        ptrdiff_t rows = 2, cols = 3;

        cout << "define A" << endl;

        auto A = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     A_data.data(), rows, cols, DataBlockConfig{  }
                 );
        A.print();
    cout<<"We load A up"<<endl;
        A.device_data_upload(true);

        cout << "define B" << endl;

        auto B = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     B_data.data(), rows, cols, DataBlockConfig{ });
        B.print();
       cout<<"We load B up"<<endl;
        B.device_data_upload(true);

        cout << "define C as empty" << endl;

        mdspan_data_t<double,dynamic_tag> C;

        cout << "addition of A and B+A" << endl;
        C = A + B+A;
        C.print();

        cout<<"Define U as empty"<<endl;
        mdspan_data_t<double,dynamic_tag> U;

        cout<<"is empty U on device? "<<U.data_is_devptr()<<endl;
        cout << "a combined expression with matrices on device: 2.0*A+A-B*3.0" << endl;
        U=2.0*A+A-B*3.0;
        U.print();
        cout<<"is U on device? "<<U.data_is_devptr()<<endl;




        mdspan_data_t<double,dynamic_tag> D;


        std::vector<double> V_data = { 6, 5, 4, 3 };
        auto V = mdspan_utilities::create_matrix<double,dynamic_tag>(V_data.data(),
                 rows, rows, DataBlockConfig{  }  );
        V.device_data_upload(true);
        V.print();
        cout << "multiplication of A and transpose of B" << endl;

        mdspan_t<double,dynamic_tag> H = mdspan_utilities::matrix_transpose(B);
    cout << "multiplication of A and transpose(B)+V" << endl;
        D = A * H+V;
        D.print();





        auto E = mdspan_utilities::create_matrix<double, dynamic_tag>(rows, cols, ::ManagedDataBlockConfig{});


  //      cout << "Subtraction of A. one can also assign the type later, as in this example, but E=A-B would also work here" << endl;
    //    cout << "But here we set a policy to do this on gpu" << endl;
cout<<"expr=A-B"<<endl;
        expr::ExpressionExecutionPolicy mypol;

        auto expr = A - B;
        expr.assign_to(E, &mypol);
        E.print();

        cout << "two vectors" << endl;
        std::vector<double> vectorA_data = { 1, 2, 3 };
        std::vector<double> vectorB_data = { 6, 5, 4 };


        auto vecA = mdspan_utilities::create_vector<double,static_tag<1>>(vectorA_data.data(), 3, DataBlockConfig{ });
        auto vecB = mdspan_utilities::create_vector<double,static_tag<1>>(vectorB_data.data(), 3, DataBlockConfig{});

        vecA.print();
        vecB.print();

        cout << "a scalar product between two vectors" << endl;
        auto c = dot(vecA, vecB).eval_scalar<double>();
        cout << c << endl;
        double d = dot(vecA, vecB);
        cout << d << endl;
    }
    {
        cout << "\n==============================\n";
        cout << "Testing nested expression templates\n";
        cout << "==============================\n";

        using namespace expr;

        ptrdiff_t rows = 2;
        ptrdiff_t cols = 3;


// ----------------------------
// Matrices A B C D
// ----------------------------

        std::vector<double> A_data =
        {
            1,2,3,
            4,5,6
        };

        std::vector<double> B_data =
        {
            6,5,4,
            3,2,1
        };

        std::vector<double> C_data =
        {
            1,1,1,
            2,2,2
        };

        std::vector<double> D_data =
        {
            2,3,4,
            5,6,7
        };


        auto A = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     A_data.data(), rows, cols, DataBlockConfig{}
                 );

        auto B = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     B_data.data(), rows, cols, DataBlockConfig{}
                 );

        auto C = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     C_data.data(), rows, cols, DataBlockConfig{}
                 );

        auto D = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     D_data.data(), rows, cols, DataBlockConfig{}
                 );


        cout << "A" << endl;
        A.print();

        cout << "B" << endl;
        B.print();

        cout << "C" << endl;
        C.print();

        cout << "D" << endl;
        D.print();



        mdspan_data_t<double,dynamic_tag> U;


// ----------------------------
// A+B+C+D
// ----------------------------

        cout << "\nTesting U=A+B+C+D" << endl;

        U = A+B+C+D;

        U.print();



// ----------------------------
// 2*(A+B)-3*(C-D)
// ----------------------------

        cout << "\nTesting U=2*(A+B)-3*(C-D)" << endl;

        U = 2.0*(A+B)-3.0*(C-D);

        U.print();



// ----------------------------
// repeated scaling
// ----------------------------

        cout << "\nTesting U=5*A-2*B+C" << endl;

        U = 5.0*A - 2.0*B + C;

        U.print();



// ----------------------------
// Matrix multiplication
// ----------------------------

        cout << "\nTesting matrix multiplication expression" << endl;


// make a compatible 3x2 matrix

        std::vector<double> M_data =
        {
            1,2,
            3,4,
            5,6
        };


        auto M = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     M_data.data(), cols, rows, DataBlockConfig{}
                 );


      mdspan_data_t<double,dynamic_tag> MM;


        cout << "Testing MM=A*M" << endl;

        MM = A*M;

        MM.print();



// ----------------------------
// Matrix multiplication + addition
// ----------------------------

        cout << "\nTesting MM=A*M + identity-like matrix" << endl;


        std::vector<double> I_data =
        {
            1,0,
            0,1
        };


        auto I = mdspan_utilities::create_matrix<double,dynamic_tag>(
                     I_data.data(), rows, rows, DataBlockConfig{}
                 );


        MM = A*M + I;

        MM.print();




        cout << "\nTesting dot product as scalar multiplier" << endl;


        std::vector<double> v1_data = {1,2,3};
        std::vector<double> v2_data = {4,5,6};


        auto v1 =
            mdspan_utilities::create_vector<double,static_tag<1>>(
                v1_data.data(),3,DataBlockConfig{}
            );

        auto v2 =
            mdspan_utilities::create_vector<double,static_tag<1>>(
                v2_data.data(),3,DataBlockConfig{}
            );
        cout<<"v1 and v2";
        v1.print();
        v2.print();

        double alpha = dot(v1,v2).eval_scalar<double>();

        cout << "dot(v1,v2)=" << alpha << endl;


// if scalar expression is supported:
        cout << "\nTesting U=dot(v1,v2)*A" << endl;

        U = alpha*A;

        U.print();



// ----------------------------
// nested scalar expression
// ----------------------------

        cout << "\nTesting U=(dot(v1,v2)-1)*A+B" << endl;

        U = (alpha-1.0)*A+B;

        U.print();

        cout << "\nTesting U=A+(B+(C+D))" << endl;

        U = A + (B + (C + D));

        U.print();



        cout << "\nTesting U=((A+B)+C)+D" << endl;

        U = ((A + B) + C) + D;

        U.print();



// ------------------------------------------------------------
// Nested scaling
// ------------------------------------------------------------

        cout << "\nTesting U=2*(A+B+C)" << endl;

        U = 2.0 * (A + B + C);

        U.print();



// ------------------------------------------------------------
// Addition/Subtraction combinations
// ------------------------------------------------------------

        cout << "\nTesting U=(A+B)-(C+D)" << endl;

        U = (A + B) - (C + D);

        U.print();



        cout << "\nTesting U=(A+B)+(C-D)" << endl;

        U = (A + B) + (C - D);

        U.print();



// ------------------------------------------------------------
// Matrix multiplication inside larger expressions
// ------------------------------------------------------------

        cout << "\nTesting MM=2*(A*M+I)" << endl;

        MM = 2.0 * (A * M + I);

        MM.print();



// ------------------------------------------------------------
// Deeply nested expression
// ------------------------------------------------------------

        cout << "\nTesting U=3*(A+B)-2*(C-D)+A" << endl;

        U = 3.0 * (A + B) - 2.0 * (C - D) + A;

        U.print();



// ------------------------------------------------------------
// Multiple scalings
// ------------------------------------------------------------

        cout << "\nTesting U=2*A+3*B-4*C+D" << endl;

        U = 2.0 * A + 3.0 * B - 4.0 * C + D;

        U.print();



// ------------------------------------------------------------
// Redundant parentheses
// ------------------------------------------------------------

        cout << "\nTesting U=(((((A+B)))))" << endl;

        U = (((((A + B)))));

        U.print();



// ------------------------------------------------------------
// Dot product reused as scalar
// ------------------------------------------------------------

        cout << "\nTesting U=(dot(v1,v2)+5)*A" << endl;

        U = (alpha + 5.0) * A;

        U.print();



        cout << "\nTesting U=(2*dot(v1,v2)-10)*A+B" << endl;

        U = (2.0 * alpha - 10.0) * A + B;

        U.print();


        cout << "\nTesting U=(A+B)*(M+I2)" << endl;

// I2 is a 3x2 matrix, same shape as M
std::vector<double> I2_data = {
    1,0,
    0,1,
    0,0
};

auto I2 = mdspan_utilities::create_matrix<double,dynamic_tag>(
    I2_data.data(), cols, rows, DataBlockConfig{}
);


MM = (A+B) * (M+I2);
MM.print();

    }

    {
        using namespace expr;
        cout << "now we test complex numbers" << endl;
        std::vector<std::complex<double>> vectorA_data = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
        std::vector<std::complex<double>> vectorB_data = { {5.0, 2.0}, {3.0, 4.0}, {1.0, 6.0} };


        auto vecA = mdspan_utilities::create_vector<std::complex<double>, static_tag<1>>(vectorA_data.data(), 3, DataBlockConfig{ });
        auto vecB = mdspan_utilities::create_vector<std::complex<double>, static_tag<1>>(vectorB_data.data(), 3, DataBlockConfig{  });

        vecA.print();
        vecB.print();

        cout << "conjugate of A" << endl;
        DataBlock<std::complex<double>> con = DataBlockUtilities::conjugate(vecA);

        con.print();


        mdspan_data_t<std::complex<double>, dynamic_tag> C =
            mdspan_utilities::create_vector<std::complex<double>, dynamic_tag>(3, ::ManagedDataBlockConfig{  });

        cout << "addition of A and B" << endl;
        C = vecA + vecB;
        C.print();

        mdspan_data_t<std::complex<double>, dynamic_tag> U =
            mdspan_utilities::create_vector<std::complex<double>, dynamic_tag>(3, ::ManagedDataBlockConfig{  });
        std::complex<double>x= {2,2};
        cout<<"scaling of A by "<< x<<endl;
        U=vecA*x;
        U.print();


//        mdspan_data_t<std::complex<double>, dynamic_tag> D =
//            mdspan_utilities::create_vector<std::complex<double>, dynamic_tag>(3, ManagedDataBlockConfig{  });
//        Math_Functions_Policy mypol(Math_Functions_Policy::CPU_ONLY);
//
//        cout << "subtraction" << endl;
//        auto expr = vecA - vecB;
//        expr.assign_to(D, &mypol);
//        D.print();
    }
    //

    {
        cout << "We define two matrices" << endl;
        vector<double> A_data(12 * 12, 0);
        vector<double> B_data(12 * 12, 0);
        ptrdiff_t rowsA = 12, colsA = 12;

        A_data =
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 11, 9, 7, 5, 3, 1, 12, 10, 8, 6, 4, 2,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9, 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10,
            5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5,
            6, 1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 11, 2, 9, 4, 12, 7, 3, 10, 5, 1, 8, 6
        };

        B_data =
        {
            12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12, 12, 9, 6, 3, 10, 7, 4, 1, 8, 5, 2, 11,
            2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 11, 8, 5, 2, 9, 6, 3, 12, 7, 4, 1, 10,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9, 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10
        };

        cout << "the same code base can have the strides and extents on heap(vector) or on the stack(array). " << endl;
        cout << "The library works as well with col major data but in this example, we define row-major data" << endl;

        auto A = mdspan_utilities::create_matrix<double, static_tag<2>>(
                     A_data.data(), rowsA, colsA, DataBlockConfig{  }
                 );
        auto B = mdspan_utilities::create_matrix<double,  static_tag<2>>(
                     B_data.data(), rowsA, colsA, DataBlockConfig{ }
                 );

        cout << "Ordinary matrix multiplication, forced on gpu with a policy object" << std::endl;
        A.print();
        B.print();

        cout << "the header In_Kernel_mathfunctions executes math functions either on the host or can run them in parallel. Abbreviations v just with simd, s without parallel loops" << endl;

        auto C0 = mdspan_utilities::create_matrix<double, dynamic_tag>(
                      rowsA, colsA, ManagedDataBlockConfig{}
                  );
        In_Kernel_Mathfunctions::matrix_multiply_dot(A, B, C0);

        cout << "per default update_host is set to true. If one has several calculations on gpu, this may not be desired and can be switched to false" << endl;
        C0.print();

        cout << "the header In_Kernel_mathfunctions executes math functions either on the host or can run them in parallel. Abbreviations w mean with parallel for" << endl;

        auto C1 = mdspan_utilities::create_matrix<double, dynamic_tag>(rowsA, colsA, ManagedDataBlockConfig{});
        In_Kernel_Mathfunctions::matrix_multiply_dot<OpenMPVariant::Sequential>(A, B, C1);

        cout << "per default update_host is set to true. If one has several calculations on gpu, this may not be desired and can be switched to false" << endl;
        C1.print();

        auto C2 = mdspan_utilities::create_matrix<double,dynamic_tag>(
                      rowsA, colsA, ManagedDataBlockConfig{}
                  );

        cout << "CPU_ONLY lets it multiply on CPU. GPU_ONLY executes on gpu. AUTO lets the library decide based on whether the data is already on gpu, the algorithm, and the data size." << endl;
        Math_Functions_Policy p1(Math_Functions_Policy::GPU_ONLY);

        cout << "supplying nullptr instead of a pointer to Math_Functions_Policy lets the library use a global default that can be configured." << endl;
        Math_Functions::matrix_multiply_dot(A, B, C2, &p1);

        cout<<"the algorithms work on gpu even if the data was not offloaded before. In that case, they offload at the beginning of the call and then download if update_host is set to true"<<endl;
        cout<<"if update host is set to false, then one msut download the data by oneself.  If one has several calculations on gpu, this may not be desired and can be switched to false"<<endl;

        C2.print();
    }
    {

        cout<<"one can, however, especially for repeated calculations, offload the data on gpu at the beginning and download it at the end"<<endl;

        cout<<"this would work as follows"<<endl;


        cout << "We define two matrices" << endl;
        vector<double> A_data(12 * 12, 0);
        vector<double> B_data(12 * 12, 0);
        ptrdiff_t rowsA = 12, colsA = 12;

        A_data =
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 11, 9, 7, 5, 3, 1, 12, 10, 8, 6, 4, 2,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9, 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10,
            5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5,
            6, 1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 11, 2, 9, 4, 12, 7, 3, 10, 5, 1, 8, 6
        };

        B_data =
        {
            12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12, 12, 9, 6, 3, 10, 7, 4, 1, 8, 5, 2, 11,
            2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 11, 8, 5, 2, 9, 6, 3, 12, 7, 4, 1, 10,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9, 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10
        };

        cout << "the same code base can have the strides and extents on heap(vector) or on the stack(array). " << endl;
        cout << "The library works as well with col major data but in this example, we define row-major data" << endl;

        auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(
                     A_data.data(), rowsA, colsA, DataBlockConfig{  }
                 );
        auto B = mdspan_utilities::create_matrix<double, static_tag<2>>(
                     B_data.data(), rowsA, colsA, DataBlockConfig{ }
                 );
        A.device_data_upload(true);
        B.device_data_upload(true);

        cout << "Ordinary matrix multiplication, forced on gpu with a policy object" << std::endl;
        A.print();
        B.print();

        cout << "the header In_Kernel_mathfunctions executes math functions either on the host or can run them in parallel. Abbreviations v just with simd, s without parallel loops" << endl;

        auto C0 = mdspan_utilities::create_matrix<double,dynamic_tag>(rowsA, colsA, ManagedDataBlockConfig{.data_ondevice=true,.default_device=true});

        Math_Functions_Policy p1(Math_Functions_Policy::GPU_ONLY);


        Math_Functions::matrix_multiply_dot(A, B, C0, &p1);
        cout<<"the printer function is able to print data on gpu directly"<<endl;

        C0.print();

        cout<<"another method would have been to allocate the data for C on host and then offload it"<<endl;
        cout<<"the deconstructors take care of the release of the data when they go out of their scope"<<endl;
    }
//
    {
        vector<double> A_data(12 * 12, 0);
        vector<double> B_data(12 * 12, 0);
        ptrdiff_t rowsA = 12, colsA = 12;

        A_data =
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 11, 9, 7, 5, 3, 1, 12, 10, 8, 6, 4, 2,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9, 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10,
            5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5,
            6, 1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 11, 2, 9, 4, 12, 7, 3, 10, 5, 1, 8, 6
        };

        B_data =
        {
            12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12, 12, 9, 6, 3, 10, 7, 4, 1, 8, 5, 2, 11,
            2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 11, 8, 5, 2, 9, 6, 3, 12, 7, 4, 1, 10,
            3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10, 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
            4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9, 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10
        };

        cout << "the same code base can have the strides and extents on heap(vector) or on the stack(array). " << endl;
        cout << "The library works as well with col major data but in this example, we define row-major data" << endl;

        auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(
                     A_data.data(), rowsA, colsA, DataBlockConfig{  }
                 );
        auto B = mdspan_utilities::create_matrix<double, dynamic_tag>(
                     B_data.data(), rowsA, colsA, DataBlockConfig{ }
                 );

        cout << "We can also use the Strassen algorithm or its Winograd variant for the multiplication." << std::endl;
        cout << "It may offload on gpu. With the Message Passing Interface enabled, it can do so in parallel. " << std::endl;
        cout << "otherwise it offloads sequentially. The algorithm can also work entirely on device with devicepointers to the data" << std::endl;
        cout << "in auto mode, the following default thresholds are set in mathfunctions.h and can be changed for convenience" << std::endl;
        cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded" << std::endl;
        cout << " default_cubic_threshold = 256;" << "The default number of elements at which matrices are auto offloaded in multiplication" << std::endl;
        cout << " default_square_threshold = 1000;" << "The default number of elements at which matrices are auto offloaded for addition" << std::endl;
        cout << " default_linear_threshold = 1000000;" << "The default number of elements at which vectors are auto offloaded for addition" << std::endl;
        cout << std::endl;

        mdspan_data_t<double,dynamic_tag> C3 = mdspan_utilities::create_matrix<double, dynamic_tag>(rowsA, colsA, ManagedDataBlockConfig{});

        cout << "we now set it on gpu and set the size when to stop recursion to 2, per default, this is at 64" << endl;
        Math_MPI_RecursiveMultiplication_Policy p(false,Math_Functions_Policy::GPU_ONLY);
        p.size_to_stop_recursion = 2;
        Math_Functions_MPI::strassen_multiply(A, B, C3,MPI_COMM_WORLD, &p);
        C3.print();
    }
//
    {
        ptrdiff_t rows = 4, cols = 4;
        cout << "We create a 4x4 matrix that owns its own data buffer in a memapped file and then fill the buffer and print it" << endl;
        cout << "usually, the own data buffer is more interesting for storing the results of the computation and for intermediary evaluations" << endl;

        auto O = mdspan_utilities::create_matrix<double, dynamic_tag>(rows, cols, ManagedDataBlockConfig{ .memmap = true });


        for (ptrdiff_t i = 0; i < 16; i++)
        {
            O.data()[i] = (double)i;
        }
        O.print();

        cout << "now we create a 4x4 matrix with data in a separate vector" << endl;
        vector<double> O2_data(16, 2);
        auto O2 = mdspan_utilities::create_matrix<double, dynamic_tag>(O2_data.data(), rows, cols, DataBlockConfig{ });
        O2.print();

        cout << "now we make a shallow copy of the first matrix on the second" << endl;
        O2 = O;
        O2.print();

        cout << "We test the shallow copy by setting the first element of the first matrix to 42 and then print the first and second matrix" << endl;
        O.data()[0] = 42;
        O.print();
        O2.print();
    }




    {
        vector<double>A_data= {210, -92, 68, -33, -34, -4, 118, -6, -92, 318, -100, 130, -153, -64, 160, 33, 68, -100, 204, -96, 41, -69, -16, -26, -33, 130, -96, 338, -152, -51, 12, 22, -34, -153, 41, -152, 346, 11, -30, -25, -4, -64, -69, -51, 11, 175, -79, 5, 118, 160, -16, 12, -30, -79, 320, 7, -6, 33, -26, 22, -25, 5, 7, 239};
        ptrdiff_t rows2 = 8, cols2 = 8;
        cout <<"Now we test more advanced algorithms"<<endl;
        {

            cout<<endl<<endl<<endl<<endl;
            cout<<"Now a cholesky decomposition on CPU"<<std::endl;

            auto A = mdspan_utilities::create_matrix<double,dynamic_tag>(A_data.data(), rows2, cols2, DataBlockConfig{});

            auto L = mdspan_utilities::create_matrix<double, dynamic_tag>(rows2, cols2, ManagedDataBlockConfig{});


            cout<<"with the dataset"<<endl;

            A.print();
            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);

            Math_Functions::cholesky_decomposition(A,L,&p);

            L.print();

            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double, dynamic_tag>(rows2, cols2, ManagedDataBlockConfig{});

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            cout<<"We can create a transpose with the base class DataBlock, but also with mdspan"<<endl;
            ptrdiff_t newext[2];
            ptrdiff_t newstr[2];
            DataBlock<double>m=DataBlockUtilities::matrix_transpose(L,newext,newstr);
            Math_Functions::matrix_multiply_dot(L,m, verify,&p2);
            verify.print();
        }
//
        {

            cout<<"Now the cholesky decomposition is entirely done on GPU"<<std::endl;

            auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(A_data.data(), rows2, cols2, DataBlockConfig{});

            auto L = mdspan_utilities::create_matrix<double,dynamic_tag>(rows2, cols2, ManagedDataBlockConfig{});

            Math_Functions_Policy p(Math_Functions_Policy::GPU_ONLY);

            Math_Functions::cholesky_decomposition(A,L,&p);

            L.print();

            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double, dynamic_tag>(rows2, cols2, ManagedDataBlockConfig{});

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);

            cout<<"Here we create the transpose with mdspan"<<endl;
            mdspan_t<double,dynamic_tag> m=mdspan_utilities::matrix_transpose(L);
            Math_Functions::matrix_multiply_dot(L,m, verify,&p2);
            verify.print();

        }


        {

            cout<<"With the advanced algorithms on GPU"<<std::endl;

            auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(A_data.data(), rows2, cols2, DataBlockConfig{});
            auto L = mdspan_utilities::create_matrix<double, dynamic_tag>(rows2, cols2, ManagedDataBlockConfig{});

            A.print();

            Math_MPI_Decomposition_Policy p(false,
                Math_Functions_Policy::GPU_ONLY,Math_MPI_RecursiveMultiplication_Policy::Naive);
            p.size_to_stop_recursion=2;
            Math_Functions_MPI::cholesky_decomposition(A,L,MPI_COMM_WORLD,&p);
            L.print();


            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double,dynamic_tag>(rows2, cols2, ManagedDataBlockConfig{});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            ptrdiff_t newext[2];
            ptrdiff_t newstr[2];

            DataBlock<double>m=DataBlockUtilities::matrix_transpose(L,newext,newstr);
            Math_Functions::matrix_multiply_dot(L,m, verify,&p2);
            verify.print();

        }
    }

    {

        cout<< "Now we do the same with the lu decomposition of"<<std::endl;
        vector<double>A_data= {-3,3,-3,5,2,7,4,2,-2,4,2,-10,-4,-2,-10,1,-3,0,8,6,-3,-8,-8,-10,-6,-1,-4,-2,-4,-2,-3,1,-9,-10,5,-6,-8,1,-3,-8,-10,-8,-6,4,3,-8,-10,-6,3,-4,-2,4,4,-1,2,8,-4,6,9,-7,-6,-4,2,4};
        ptrdiff_t rows3 = 8, cols3 = 8;

        {


            auto A = mdspan_utilities::create_matrix<double,dynamic_tag>(A_data.data(), rows3, cols3, DataBlockConfig{});
            auto L = mdspan_utilities::create_matrix<double,dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});
            auto U = mdspan_utilities::create_matrix<double,dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});

            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);
            A.print();

            cout<<"on CPU"<<std::endl;

            Math_Functions::lu_decomposition(A,L,U,&p);
            L.print();
            U.print();

            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double,dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions::matrix_multiply_dot(L,U, verify,&p2);
            verify.print();

        }

        {


            auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(A_data.data(), rows3, cols3, DataBlockConfig{});
            auto L = mdspan_utilities::create_matrix<double, dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});
            auto U = mdspan_utilities::create_matrix<double, dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});

            cout<<"Entirely on gpu"<<std::endl;
            Math_Functions_Policy p(Math_Functions_Policy::GPU_ONLY);
            Math_Functions::lu_decomposition(A,L,U,&p);
            L.print();
            U.print();

            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double, dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions::matrix_multiply_dot(L,U, verify,&p2);
            verify.print();
        }

        {



            auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(A_data.data(), rows3, cols3, DataBlockConfig{});
            auto L = mdspan_utilities::create_matrix<double, dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});
            auto U = mdspan_utilities::create_matrix<double, dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});

            cout<<"With the advanced algorithms on GPU"<<std::endl;

            Math_MPI_Decomposition_Policy p(false,
                Math_Functions_Policy::GPU_ONLY,
                Math_MPI_RecursiveMultiplication_Policy::Strassen);

            p.size_to_stop_recursion=2;
            Math_Functions_MPI::lu_decomposition(A,L,U,MPI_COMM_WORLD,&p);
            L.print();


            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double,dynamic_tag>(rows3, cols3, ManagedDataBlockConfig{});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions::matrix_multiply_dot(L,U, verify,&p2);
            verify.print();

        }
    }
//
    {
//
        cout<< "Now we do the same with the qr decomposition"<<std::endl;
        vector<double>A_data= {-4, 9, 4, 0, -3, -4, 8, 0, 0, -7, -3, -8, -9, 1, -5, -9, -10, 1, 1, 6, -1, 5, 4, 4, 8, 1, 9, -8, -6, 8, -4, -2, -4, 7, -7, 3, 7, -2, -9, 9, 4, -4, 1, -3, 4, -8, 3, 6, -7, 7, -3, -7, -9, -5, -1, -7, 7, 1, -9, -1, -7, 3, 5, 4};
        ptrdiff_t rows4 = 8, cols4 = 8;
        {



            auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(A_data.data(), rows4, cols4, DataBlockConfig{});
            auto Q = mdspan_utilities::create_matrix<double, dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});
            auto R = mdspan_utilities::create_matrix<double, dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});

            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);
            A.print();

            cout<<"On cpu"<<std::endl;
            Math_Functions::qr_decomposition(A,Q,R,&p);
            Q.print();
            R.print();

            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double, dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions::matrix_multiply_dot(Q,R, verify,&p2);
            verify.print();
        }


        {


            auto A = mdspan_utilities::create_matrix<double,dynamic_tag>(A_data.data(), rows4, cols4, DataBlockConfig{});
            auto Q = mdspan_utilities::create_matrix<double,dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});
            auto R = mdspan_utilities::create_matrix<double, dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});


            cout<<"On gpu"<<std::endl;
            Math_Functions_Policy p(Math_Functions_Policy::GPU_ONLY);

            Math_Functions::qr_decomposition(A,Q,R,&p);
            Q.print();
            R.print();


            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double,dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions::matrix_multiply_dot(Q,R, verify,&p2);
            verify.print();

        }

        {
            cout<<"with the advanced algorithms on gpu "<<std::endl;


            auto A = mdspan_utilities::create_matrix<double, dynamic_tag>(A_data.data(), rows4, cols4, DataBlockConfig{});
            auto Q = mdspan_utilities::create_matrix<double, dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});
            auto R = mdspan_utilities::create_matrix<double, dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});

            Math_MPI_Decomposition_Policy p(false,
                Math_Functions_Policy::GPU_ONLY,
                Math_MPI_Decomposition_Policy::Strassen);

            p.size_to_stop_recursion=2;


            Math_Functions_MPI::qr_decomposition(A,Q,R,MPI_COMM_WORLD,&p);
            Q.print();
            R.print();
            vector<double>verifydata(64,0);

            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            auto verify = mdspan_utilities::create_matrix<double,dynamic_tag>(rows4, cols4, ManagedDataBlockConfig{});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions::matrix_multiply_dot(Q,R, verify,&p2);
            verify.print();

        }
    }

}






