
#include <iostream>
#include <vector>
#include <climits>

#include "datablock.h"
#include "mathutilitiesdatablock.h"

using std::cout;
using std::endl;
using std::vector;

int main(int argc, char** argv)
{
    cout << "Now some tests whether the library accepts row and column major data and can extract rows and columns with the same code. " << endl;
    cout << " Note that this tests only the DataBlock class, which can be offloaded to gpu. it is non owning, " << endl;
    cout << " compared to the mdspan class which owns strides and extents and mdspan_data, which owns the data as well" << endl << endl;

    {
        // ====================================================================
        // TEST MATRIX A (Row-Major)
        // ====================================================================
        vector<double> A_data(3 * 7, 0);
        A_data =
        {
            1, 2, 3, 4, 5, 6, 7,
            8, 9,10,11,12,13,14,
            15,16,17,18,19,20,21
        };
        ptrdiff_t extaa[2] = {3, 7};
        ptrdiff_t straa[2];
        cout<<"construction with the create_matrix function"<<endl;

        DataBlock<double> A = DataBlockUtilities::create_matrix(
                                  A_data.data(), 3, 7, extaa, straa,
                                  DataBlockConfig{},StridesCalculation::Compute );

        cout << "A" << A.datalength() << endl;


        A.print();
        cout<<"construction with the constructor of DataBlock, which is less fast if it should generate strides and length since it works for general tensors"<<endl;
        ptrdiff_t extaab[2] = {3, 7};
        ptrdiff_t straab[2];
        auto A2=DataBlock<double> (A_data.data(),0,2, extaab,straab,DataBlockConfig{},ComputeMetadata{});

        cout << "A" << A2.datalength() << endl;
        A2.print();

        cout << "column" << endl;
        ptrdiff_t exta[1];
        ptrdiff_t stra[1];
        DataBlock<double> Aa = DataBlockUtilities::matrix_column(A, 1, exta, stra);
        cout << "C" << endl << "Rank" << Aa.rank() << endl;
        Aa.print();


        cout << "row" << endl;
        ptrdiff_t extar[2];
        ptrdiff_t strar[2];
        DataBlock<double> Aa1 = DataBlockUtilities::matrix_row(A, 1, extar, strar);
        cout << "C" << endl;
        Aa1.print();
        cout << "Rank" << Aa1.rank() << endl;



        ptrdiff_t exta2[2];
        ptrdiff_t stra2[2];
        DataBlock<double> Ab = DataBlockUtilities::matrix_subspan(A, 1, 1, 2, 4, exta2, stra2);
        cout << "matrix_subspanA" << endl;
        Ab.print();



        ptrdiff_t exta5[2];
        ptrdiff_t stra5[2];
        DataBlock<double> Ae = DataBlockUtilities::matrix_transpose(A, exta5, stra5);
        cout << "transpose" << endl;
        Ae.print();




        std::vector<double> data_rowmajor =
        {
            // block 0 (first 3x4 matrix)
            1,2,3,4, 5,6,7,8, 9,10,11,12,
            // block 1 (second 3x4 matrix)
            13,14,15,16, 17,18,19,20, 21,22,23,24
        };
        ptrdiff_t extents[3] = {2, 3, 4};
        ptrdiff_t strides[3]; // will be computed

        // The general tensor constructor uses the persistent config and the instructions struct
        DataBlock<double> T_row(
            data_rowmajor.data(),
            data_rowmajor.size(),
            3, // rank
            extents,
            strides,
            DataBlockConfig{},ComputeMetadata{});

        ptrdiff_t offsets[3] = {1, 0, 0}; // start at block 1
        ptrdiff_t sub_extents[3] = {1, 3, 4}; // take 1 block of full 3x4

        ptrdiff_t newextT[2];
        ptrdiff_t newstrT[2];
        DataBlock<double> subT_view = DataBlockUtilities::tensor_subspan(T_row, offsets, sub_extents, newextT, newstrT);
        std::cout << "now a Tensor" << endl;
        T_row.print();
        cout << "Rank" << T_row.rank() << endl;
        std::cout << "Subtensor view (row-major):\n";
        subT_view.print();
        cout << "Rank" << subT_view.rank() << endl;




        vector<double> B_data_colmajor =
        {
            1, 8, 15, 2, 9, 16, 3, 10, 17, 4, 11, 18, 5, 12, 19, 6, 13, 20, 7, 14, 21
        };

        ptrdiff_t extbb[2] = {3, 7};
        ptrdiff_t strbb[2];


        DataBlock<double> B = DataBlockUtilities::create_matrix(
                                  B_data_colmajor.data(), 3, 7, extbb, strbb,
                                  DataBlockConfig{.dprowmajor = false}, StridesCalculation::Compute );
        cout << "B" << B.datalength() << endl << endl;
        B.print();
        cout << "B" << B.datalength() << endl << endl;

        ptrdiff_t extbbb[2];
        ptrdiff_t strbbb[2];

        // Call factory with compute_strides = false
        DataBlock<double> Bbbb = DataBlockUtilities::create_matrix(
                                     B_data_colmajor.data(), 3, 7, extbbb, strbbb,
                                     DataBlockConfig
        {.dprowmajor = false  },  StridesCalculation::Compute  );

        Bbbb.print();

        cout << "column 1" << endl;
        ptrdiff_t extb[2];
        ptrdiff_t strb[2];
        DataBlock<double> Ba = DataBlockUtilities::matrix_column(B, 1, extb, strb);
        Ba.print();


        cout << "row 1" << endl;
        ptrdiff_t extb35[2];
        ptrdiff_t strb35[2];
        DataBlock<double> Ba2 = DataBlockUtilities::matrix_row(B, 1, extb35, strb35);
        Ba2.print();
        cout << "Rank" << Ba2.rank() << endl;


        ptrdiff_t extb2[2];
        ptrdiff_t strb2[2];
        cout << "subspanmatrx B" << endl;
        DataBlock<double> Bb = DataBlockUtilities::matrix_subspan(B, 1, 1, 2, 4, extb2, strb2);
        Bb.print();
        cout << "Rank" << Bb.rank() << endl;


        ptrdiff_t extb5[2];
        ptrdiff_t strb5[2];
        DataBlock<double> Be = DataBlockUtilities::matrix_transpose(B, extb5, strb5);
        cout << "transpose" << endl;
        Be.print();




        std::vector<double> data_colmajor =
        {
            1,13, 2,14, 3,15, 4,16, 5,17, 6,18, 7,19, 8,20, 9,21, 10,22, 11,23, 12,24
        };
        ptrdiff_t extentsC[3] = {2, 3, 4};
        ptrdiff_t stridesC[3];


        DataBlock<double> T_col(
            data_colmajor.data(),
            data_colmajor.size(),
            3,
            extentsC,
            stridesC,
            DataBlockConfig
        {.dprowmajor = false},ComputeMetadata{});


        std::cout << "A tensor in colmajor \n";
        T_col.print();

        ptrdiff_t offsetsC[3] = {1, 0, 0};
        ptrdiff_t sub_extentsC[3] = {1, 3, 4};
        ptrdiff_t newext[2];
        ptrdiff_t newstr[2];
        DataBlock subC_view = DataBlockUtilities::tensor_subspan(T_col, offsetsC, sub_extentsC, newext, newstr);
        std::cout << "Subtensor view (col-major):\n";
        subC_view.print();

    }
}
