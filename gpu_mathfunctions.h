#ifndef GPUMATHFUNCTIONS
#define GPUMATHFUNCTIONS

#include "datablock.h"
#include "mathutilitiesdatablock.h"

struct GPUOptions
{
    int  device      = omp_get_default_device();
    bool update_host = true;
};

class GPU_Math_Functions
{
public:

    template <typename T>
    inline static void matrix_multiply_dot_g(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,const T CoefficientB = T(1),const T CoefficientC  = T(0),GPUOptions opt= {});

    template <typename T>
    inline static void matrix_multiply_dot_g(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,GPUOptions opt)
    {
        matrix_multiply_dot_g(A,B,C,T(1),T(0),opt);
    }

    template <typename T>
    inline static void matrix_multiply_dot_kahan_g(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,const T CoefficientB = T(1),const T CoefficientC  = T(0),
            GPUOptions opt= {});

    template <typename T>
    inline static void matrix_multiply_dot_kahan_g(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,GPUOptions opt)
    {
        matrix_multiply_dot_kahan_g(A,B,C,T(1),T(0),opt);
    }

    template <typename T>
    inline static void matrix_multiply_vector_g(const DataBlock<T>& A,const DataBlock<T>& x,DataBlock<T>& y,const T Coefficientx = T(1),const T Coefficienty  = T(0),
            GPUOptions opt= {});



    template <typename T>
    inline static void matrix_multiply_vector_g(const DataBlock<T>& A,const T* x,DataBlock<T>& y,const T Coefficientx = T(1),const T Coefficienty  = T(0),
            GPUOptions opt= {});

    template <typename T>
    inline static void matrix_multiply_vector_g(const DataBlock<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
            GPUOptions opt)
    {
        matrix_multiply_vector_g(A,x,y,T(1),T(0),opt);
    }

    template <typename T>
    inline static void matrix_multiply_vector_kahan_g(const DataBlock<T>& A,const DataBlock<T>& x,DataBlock<T>& y,const T Coefficientx = T(1),const T Coefficienty  = T(0),
            GPUOptions opt= {});


    template <typename T>
    inline static void matrix_multiply_vector_kahan_g(const DataBlock<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
            GPUOptions opt)
    {
        matrix_multiply_vector_kahan_g(A,x,y,T(1),T(0),opt);
    }

    template <typename T>
    inline static void  matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const T*V, DataBlock<T>&C,const T CoeffV=T(1),const T CoeffC=T(0),
            GPUOptions opt={});

    template <typename T>
    inline static void  matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const T*V, DataBlock<T>&C,
            GPUOptions opt)
    {
        matrix_multiply_vector_kahan_g(M,V,C,T(1),T(0),opt);
    }




    template <typename T>
    inline static void matrix_linear_combination_g(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
            const T CoefficientA = T(1),const T CoefficientB  = T(1),const T CoefficientC = T(0),
            GPUOptions opt= {});

    template <typename T>
    inline static void matrix_linear_combination_g(const DataBlock<T>& A,DataBlock<T>& C,
            const T CoefficientA = T(1),const T CoefficientC = T(0),
            GPUOptions opt= {});

    template <typename T>
    inline static void matrix_add(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,  GPUOptions opt)
    {
        matrix_linear_combination_g(A,B,C,T(1),T(1),T(0),opt);
    }

    template <typename T>
    inline static void matrix_subtract(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
                                       GPUOptions opt)
    {
        matrix_linear_combination_g(A,B,C,T(1),-T(1),T(0),opt);
    }


    template <typename T>
    inline static void matrix_add(const DataBlock<T>& A,DataBlock<T>& C,  GPUOptions opt)
    {
        matrix_linear_combination_g(A,C,T(1),T(1),opt);
    }
    template <typename T>

    inline static void matrix_subtract(const DataBlock<T>& A,DataBlock<T>& C,
                                       GPUOptions opt)
    {
        matrix_linear_combination_g(A,C,-T(1),T(1),opt);
    }

    template <typename T>
    inline static void matrix_multiply_scalar_g(const DataBlock<T>& M,const T scalar,DataBlock<T>& C,
            GPUOptions opt= {});

    template <typename T>
    inline static void matrix_multiply_scalar_g(DataBlock<T>& M,const T scalar,GPUOptions opt= {});

    template <typename T>
    inline static T dot_product_g(const DataBlock<T>& vec1,const DataBlock<T>& vec2,GPUOptions opt= {});

    template <typename T>
    inline static T dot_product_kahan_g(const DataBlock<T>& vec1,const DataBlock<T>& vec2,GPUOptions opt= {});

    template <typename T>
    inline static T dot_product_g_kahan(const DataBlock<T> &vec1, const DataBlock<T> &vec2, int dev, int nteams, int nthreads_per_team);

    template <typename T>
    inline static void vector_linear_combination_g(const DataBlock<T>& vecA,const DataBlock<T>& vecB,DataBlock<T>& vecC,
            const T CoefficientA = T(1),const T CoefficientB  = T(1),const T CoefficientC = T(0),
            GPUOptions opt= {});

    template <typename T>
    inline static void vector_add_g(const DataBlock<T>& vecA,const DataBlock<T>& vecB,DataBlock<T>& vecC,
                                    GPUOptions opt)
    {
        vector_linear_combination_g(vecA,vecB,vecC,T(1),T(1),T(0),opt);
    }

    template <typename T>
    inline static void vector_subtract_g(const DataBlock<T>& vecA,const DataBlock<T>& vecB,DataBlock<T>& vecC,
                                         GPUOptions opt)
    {
        vector_linear_combination_g(vecA,vecB,vecC,T(1),-T(1),T(0),opt);
    }

    template <typename T>
    inline static void vector_linear_combination_g(const DataBlock<T>& vecA,DataBlock<T>& vecC,const T CoefficientA = T(1),const T CoefficientC = T(1),
            GPUOptions opt= {});


    template <typename T>
    inline static void vector_add_g(const DataBlock<T>& vecA,DataBlock<T>& vecC,GPUOptions opt)
    {
        vector_linear_combination_g(vecA,vecC,T(1),T(1),opt);
    }

    template <typename T>
    inline static void vector_subtract_g(const DataBlock<T>& vecA,DataBlock<T>& vecC,
                                         GPUOptions opt)
    {
        vector_linear_combination_g(vecA,vecC,-T(1),T(1),opt);
    }

    template <typename T>
    inline static void vector_multiply_scalar_g(const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,
            GPUOptions opt= {});

    template <typename T>
    inline static void vector_multiply_scalar_g(DataBlock<T>& vec,const T scalar,
            GPUOptions opt= {});



    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const BlockedDataView<T>& Bblocks, DataBlock<T>& C,
            const T CoefficientB = T(1),const T CoefficientC = T(0),
            GPUOptions opt= {} );

    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const BlockedDataView<T>& Bblocks, DataBlock<T>& C,
            GPUOptions opt)
    {
        matrix_multiply_dot_sparse_g(Ablocks,Bblocks,C,T(1),T(0),opt);
    }


    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const DataBlock<T>& Bblocks, DataBlock<T>& C,
            const T CoefficientB  = T(1),const T CoefficientC = T(0),
            GPUOptions opt= {});

    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const DataBlock<T>& Bblocks, DataBlock<T>& C,
            GPUOptions opt)
    {
        matrix_multiply_dot_sparse_g(Ablocks,Bblocks,C,T(1),T(0),opt);
    }


    template <typename T>
    inline static void matrix_multiply_vector_sparse_g(const BlockedDataView<T>& A, const DataBlock<T>& x,DataBlock<T>& y,
            const T CoefficientX = T(1),const T Coefficienty = T(0),
            GPUOptions opt= {} ) ;

    template <typename T>
    inline static void matrix_multiply_vector_sparse_g(const BlockedDataView<T>& A, const DataBlock<T>& x, DataBlock<T>& y,
            GPUOptions opt )
    {
        matrix_multiply_vector_sparse_g(A,x,y,T(1),T(0),opt);
    }

    template <typename T>
    inline static void matrix_multiply_vector_sparse_g(const BlockedDataView<T>& A, const BlockedDataView<T>& x, DataBlock<T>& y,
            const T CoefficientX = T(1),const T Coefficienty = T(0),
            GPUOptions opt= {} );

    template <typename T>
    inline static void matrix_multiply_vector_sparse_g(const BlockedDataView<T>& A,const BlockedDataView<T>& x,DataBlock<T>& y,
            GPUOptions opt)
    {
        matrix_multiply_vector_sparse_g(A,x,y,T(1),T(0),opt);
    }

    template <typename T>
    inline static void cholesky_decomposition_g(const DataBlock<T>& A, DataBlock<T> & L, bool initialize_output_to_zero=true,
            GPUOptions opt= {});

    template <typename T>
    inline static void lu_decomposition_g(const DataBlock<T> &A,  DataBlock<T> & L,DataBlock<T> & U,bool initialize_output_to_zero=true,
                                          GPUOptions opt= {});
    template <typename T>
    inline static void qr_decomposition_g(const DataBlock<T> &A,  DataBlock<T>& Q, DataBlock<T> & R,  bool initialize_output_to_zero=true, bool memmap_tempfiles=false,
                                          GPUOptions opt= {});

};

#include "gpu_mathfunctions.hpp"

#endif

