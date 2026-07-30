
#ifndef DATABLOCKUTILITIES
#define DATABLOCKUTILITIES
#include "datablock.h"
#include "mathutilitiesdatablock.h"

#pragma omp begin declare target
enum class StridesCalculation
{
    NoComputation,
    Compute
};
#pragma omp end declare target

template <typename T>
class DataBlock;

#pragma omp begin declare target
class DataBlockUtilities
{
public:

    template<typename T>
    inline static DataBlock<T>conjugate(const  DataBlock<T>&d);

     template<typename T>
    inline static bool same_extents(const DataBlock<T>&d1,const DataBlock<T>&d2);

    template<typename T>
    inline static DataBlock<T>matrix_subspan(const  DataBlock<T>&d, const ptrdiff_t row, const ptrdiff_t col,const  ptrdiff_t tile_rows,const  ptrdiff_t tile_cols,  ptrdiff_t *    psub_extents,  ptrdiff_t *   psub_strides);

    template <typename T>
    inline static DataBlock<T> matrix_transpose(const  DataBlock<T>&d,ptrdiff_t*    newextents, ptrdiff_t*    newstrides);

    template<typename T>
    inline static DataBlock<T> matrix_hermitian_transpose(const  DataBlock<T>&d,ptrdiff_t*    newextents, ptrdiff_t*    newstrides);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd, typename T>
    inline static DataBlock<T> tensor_subspan(const  DataBlock<T>&d,const ptrdiff_t * poffsets, const ptrdiff_t * psub_extents,ptrdiff_t* newextents, ptrdiff_t*    new_strides);

    template<typename T>
    inline static DataBlock<T> matrix_row(const  DataBlock<T>&d,const ptrdiff_t row_index, ptrdiff_t*    newextents, ptrdiff_t*    newstrides);

    template<typename T>
    inline static DataBlock<T> matrix_column(const  DataBlock<T>&d,const ptrdiff_t col_index, ptrdiff_t*    newextents, ptrdiff_t*    newstrides);


    template<typename T>
    inline static ptrdiff_t count_noncollapsed_dims(const  DataBlock<T>&d) ;

    template<typename T>
    inline static DataBlock<T>  collapsed_view(const  DataBlock<T>&d,const ptrdiff_t num_non_collapsed_dims,ptrdiff_t* extents, ptrdiff_t* strides) ;


    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd, typename T>
    inline static float sparsity(const  DataBlock<T>&d);

    template<typename T>
    inline static DataBlock<T> create_vector(T* data, ptrdiff_t* extents, ptrdiff_t* strides, DataBlockConfig config, const StridesCalculation computestrides);

    template<typename T>
    inline static DataBlock<T>create_matrix(T* data,  const ptrdiff_t rows,  const ptrdiff_t cols,  ptrdiff_t* extents,  ptrdiff_t* strides,  DataBlockConfig config,   const StridesCalculation computestrides);

    template<typename T>
    inline static void copy(DataBlock<T>&target,DataBlock<T> source,DataBlockConfig targetcfg);
};
#pragma omp end declare target

#include "mathutilitiesdatablock.hpp"
#endif
