#ifndef GPUMATHFUNCTIONS
#define GPUMATHFUNCTIONS

#include "datablock.h"
#include "mathutilitiesdatablock.h"




class GPU_Math_Functions
{
public:
    template <typename T>
    inline static void matrix_multiply_dot_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_dot_kahan_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_add_g(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_subtract_g( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_dot_accumulate_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_dot_accumulate_kahan_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_add_accumulate_g( DataBlock<T>& A,const DataBlock<T>& B,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_subtract_accumulate_g(  DataBlock<T>& A,const  DataBlock<T>& B,int dev,bool update_host=true);

    template <typename T>
    inline static void matrix_multiply_scalar_g (const  DataBlock<T>& M, const T V, DataBlock<T>& C, int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_scalar_accumulate_g ( DataBlock<T>& M, const T V, int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_g(const  DataBlock<T>&M, const DataBlock<T> &V, DataBlock<T>&C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_g(const  DataBlock<T>&M, const T*V, DataBlock<T> &C, int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_kahan_g(const  DataBlock<T>&M, const DataBlock<T> &V, DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_kahan_g(const  DataBlock<T>&M, const T*V, DataBlock<T> & C, int dev,bool update_host=true);

    template <typename T>
    inline static void vector_multiply_scalar_g(const DataBlock<T>& vec, const T scalar,DataBlock<T>& res,int dev,bool update_host=true);
    template <typename T>
    inline static void vector_multiply_scalar_accumulate_g(DataBlock<T>& vec, const T scalar,int dev,bool update_host=true);
    template <typename T>
    inline static void vector_add_g(const  DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,int dev,bool update_host=true);
    template <typename T>
    inline static void vector_add_accumulate_g(  DataBlock<T>& vec1, const DataBlock<T>& vec2,int dev,bool update_host=true);

    template <typename T>
    inline static void vector_subtract_g( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res,  int dev,bool update_host=true);
    template <typename T>
    inline static void vector_subtract_accumulate_g(  DataBlock<T>& vec1,const  DataBlock<T>& vec2,  int dev,bool update_host=true);


    template <typename T>
    inline static T dot_product_g( const DataBlock<T> &vec1,const  DataBlock<T> &vec2, int dev);
    template <typename T>
    inline static T dot_product_g_kahan( const DataBlock<T> &vec1,const  DataBlock<T> &vec2,int dev, int nteams, int nthreads_per_team );
    template <typename T>
    inline static void cholesky_decomposition_g(const DataBlock<T>& A, DataBlock<T> & L, int dev,bool update_host=true, bool initialize_output_to_zero=true);
    template <typename T>
    inline static void lu_decomposition_g(const DataBlock<T> &A,  DataBlock<T> & L,DataBlock<T> & U, int dev,bool update_host=true,bool initialize_output_to_zero=true);
    template <typename T>
    inline static void qr_decomposition_g(const DataBlock<T> &A,DataBlock<T>& Q, DataBlock<T> & R,  int dev,bool update_host=true,bool initialize_output_to_zero=true,bool memmaptempfiles=false);
    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const BlockedDataView<T>& Bblocks, DataBlock<T>& C,int dev,bool update_host=true,bool initialize_output_to_zero=true );
    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const DataBlock<T>& Bblocks, DataBlock<T>& C,int dev,bool update_host=true,bool initialize_output_to_zero=true );
    template <typename T>
    inline static void matrix_vector_multiply_sparse_g(const BlockedDataView<T>& A, const DataBlock<T>& x,          DataBlock<T>& y,int dev,bool update_host=true,bool initialize_output_to_zero=true ) ;
    template <typename T>
    inline static void matrix_vector_multiply_sparse_g(const BlockedDataView<T>& A, const BlockedDataView<T>& x,    DataBlock<T>& y,int dev,bool update_host=true,bool initialize_output_to_zero=true );

};

#include "gpu_mathfunctions.hpp"

#endif

