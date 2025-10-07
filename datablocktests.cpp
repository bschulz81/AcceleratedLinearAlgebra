

#include <iostream>
#include <vector>

#include "datablock.h"
#include "mdspan_omp.h"
#include "mdspan_data.h"

using namespace std;

int main(int argc, char** argv)
{


    cout<< "Now some tests whether the library accepts row and column major data and can extract rows and columns with the same code. "<<endl;
    cout<<" Note that this tests only the DataBlock class, which can be offloaded to gpu. it is non owning, "<<endl;
    cout<<" compared to the mdspan class which owns strides and extents and mdspan_data, which owns the data as well"<<endl<<endl;
    {
//
        vector<double>A_data(3*7,0);
        A_data = {1,2,3,4,5,6,7,
                  8,9,10,11,12,13,14,
                  15,16,17,18,19,20,21
                 };


        size_t extaa[2]= {3,7};
        size_t straa[2];
        DataBlock<double>A(A_data.data(),0, true,2,extaa,straa,true,true,false);
        cout<<"A"<<A.datalength()<<endl;
        A.printtensor();

        cout<<"column"<<endl;
        size_t exta[1];
        size_t stra[1];

        DataBlock<double>Aa=A.column(1,exta,stra);
        cout <<"C"<<endl<<"Rank"<<Aa.rank()<<endl;
        Aa.printtensor();



        size_t exta3[2],stra3[2];
        double newda3[7];
        cout<<"column 1 of A with data c"<<endl;
        DataBlock<double>Ac=A.column_copy_s(1, exta3,stra3,newda3);
        Ac.printtensor();
        cout <<"Rank: "<<Ac.rank()<<endl;
        cout<<"row"<<endl;
        size_t extar[2];
        size_t strar[2];
        DataBlock<double>Aa1=A.row(1,extar,strar);
        cout <<"C"<<endl;
        Aa1.printtensor();
        cout <<"Rank"<<Aa1.rank()<<endl;

        size_t exta3r[2],stra3r[2];
        double newda3r[7];
        cout<<"row 1 of A with data c"<<endl;
        DataBlock<double>Ac2=A.row_copy_s(1, exta3r,stra3r,newda3r);
        Ac2.printtensor();

        size_t exta2[2],stra2[2];
        DataBlock<double>Ab= A.subspanmatrix(1,1,2,4,exta2,stra2);
        cout<<"subspanmatrixA"<<endl;
        Ab.printtensor();


        size_t exta4[2],stra4[2];
        double newda4[8];
        DataBlock<double>Ad= A.subspanmatrix_copy_s(1,1,2,4,exta4,stra4,newda4);
        cout<<"subspanmatrixA with data copy"<<endl;
        Ad.printtensor();

        size_t exta5[2],stra5[2];
        DataBlock<double>Ae= A.transpose(exta5,stra5);
        cout<<"transpose"<<endl;
        Ae.printtensor();

        size_t exta6[2],stra6[2];
        double dataa6[21];
        DataBlock<double>Af= A.transpose_copy_s(exta6,stra6,dataa6);
        cout<<"transpose with data copy"<<endl;
        Af.printtensor();



        std::vector<double> data_rowmajor =
        {
            //                block 0 (first 3x4 matrix)
            1,2,3,4,
            5,6,7,8,
            9,10,11,12,
            //           block 1 (second 3x4 matrix)
            13,14,15,16,
            17,18,19,20,
            21,22,23,24
        };

        size_t extents[3] = {2,3,4};
        size_t strides[3]; // will be computed

        DataBlock<double> T_row(data_rowmajor.data(),
                                 data_rowmajor.size(),
                                 true, // row-major
                                 3,    // rank
                                 extents,
                                 strides,
                                 true, // compute_datalength
                                 true, // compute_strides
                                 false // data is not device ptr
                                );

        size_t offsets[3]    = {1,0,0}; // start at block 1
        size_t sub_extents[3]= {1,3,4}; // take 1 block of full 3x4



        size_t newextT[2],newstrT[2];
        DataBlock<double> subT_view =T_row.subspan(offsets, sub_extents, newextT,newstrT);

        std::cout<<"now a Tensor"<<endl;

        T_row.printtensor();
        cout <<"Rank"<<T_row.rank()<<endl;
        std::cout << "Subtensor view (row-major):\n";

        subT_view.printtensor();
        cout <<"Rank"<<subT_view.rank()<<endl;

        double buffer1[12];
        size_t newxtCa[2],newstrCA[2];
        DataBlock<double> subC_view2A =T_row.subspan_copy(offsets, sub_extents, newxtCa, newstrCA,buffer1);

        std::cout << "Subtensor view (row-major) with buffer:\n";
        subC_view2A.printtensor();
        cout <<"Rank"<<subC_view2A.rank()<<endl;


        vector<double> B_data_colmajor =
        {
            1, 8, 15,
            2, 9, 16,
            3, 10, 17,
            4, 11, 18,
            5, 12, 19,
            6, 13, 20,
            7, 14, 21
        };


        size_t extbb[2]= {3,7};
        size_t strbb[2];
        DataBlock<double>B(B_data_colmajor.data(),21, false,2,extbb,strbb,true,true,false);
        cout<<"B"<< B.datalength()<<endl<<endl;
        B.printtensor();
        cout<<"B"<< B.datalength()<<endl<<endl;
        size_t extbbb[2];
        size_t strbbb[2];
        DataBlock<double>Bbbb(B_data_colmajor.data(),0, false,3,7,extbbb,strbbb,true,true,false);
        Bbbb.printtensor();

        cout<<"column 1"<<endl;

        size_t extb[2],strb[2];

        DataBlock<double>Ba= B.column(1,extb,strb);
        Ba.printtensor();
        size_t extb2[2],strb2[2];


        size_t extb3[2],strb3[2];
        double newdb3[7];
        cout<< "column1  of B with data copy"<<endl;

        DataBlock<double> Bc1=B.column_copy_s(1, extb3,strb3,  newdb3);
        Bc1.printtensor();

        cout<<"row 1"<<endl;

        size_t extb35[2],strb35[2];
        DataBlock<double>Ba2= B.row(1,extb35,strb35);
        Ba2.printtensor();
        cout <<"Rank"<<Ba2.rank()<<endl;

        size_t extb34[2],strb34[2];
        double newdb3a[7];
        cout<< "row 1  of B with data copy"<<endl;

        DataBlock<double> Bc3=B.row_copy_s(1, extb34,strb34,  newdb3a);
        Bc3.printtensor();
        cout <<"Rank"<<Bc3.rank()<<endl;

        cout<<"subspanmatrx B"<<endl;
        DataBlock<double>Bb= B.subspanmatrix(1,1,2,4,extb2,strb2);
        Bb.printtensor();
        cout <<"Rank"<<Bb.rank()<<endl;

        size_t extb4[2],strb4[2];
        double newdb4[8];
        DataBlock<double>Bd= B.subspanmatrix_copy_s(1,1,2,4,extb4,strb4,newdb4);
        cout<<"subspanmatrixB with data copy"<<endl;
        Bd.printtensor();

        size_t extb5[2],strb5[2];
        DataBlock<double>Be= B.transpose(extb5,strb5);
        cout<<"transpose"<<endl;
        Be.printtensor();

        size_t extb6[2],strb6[2];
        double datab6[21];
        DataBlock<double>Bf= B.transpose_copy_s(extb6,strb6,datab6);
        cout<<"transpose with data copy"<<endl;
        Bf.printtensor();


        std::vector<double> data_colmajor =
        {
            1,13,
            2,14,
            3,15,
            4,16,

            5,17,
            6,18,
            7,19,
            8,20,

            9,21,
            10,22,
            11,23,
            12,24
        };

        size_t extentsC[3] = {2,3,4};
        size_t stridesC[3];

        DataBlock<double> T_col(data_colmajor.data(),
                                 data_colmajor.size(),
                                 false, // column-major
                                 3,
                                 extentsC,
                                 stridesC,
                                 true,
                                 true,
                                 false);

        size_t offsetsC[3]     = {1,0,0};
        size_t sub_extentsC[3] = {1,3,4};


        size_t newext[2],newstr[2];
        DataBlock<double> subC_view =
            T_col.subspan(offsetsC, sub_extentsC, newext,newstr);
        std::cout << "Subtensor view (col-major):\n";
        subC_view.printtensor();

        double buffer4[12];
        size_t newextC[2],sub_stridesC[2];
        DataBlock<double> subC_view2 =T_col.subspan_copy(offsetsC, sub_extentsC,newextC, sub_stridesC,buffer4);
        std::cout << "Subtensor view (col-major) with buffer:\n";
        subC_view2.printtensor();
        cout <<"Rank"<<subC_view2.rank()<<endl;



    }

}
