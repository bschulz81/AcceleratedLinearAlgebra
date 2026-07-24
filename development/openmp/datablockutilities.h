#ifndef DATABLOCKUTILITIES
#define DATABLOCKUTILITIES
#include "datablock.h"
#include "indiceshelperfunctions.h"

#pragma omp begin declare target
enum class StridesCalculation {
    NoComputation,       // Use the provided strides array as-is
    Compute       // Compute strides automatically from the extents
};
#pragma omp end declare target

#pragma omp begin declare target
class DataBlockUtilities
{
public:

    template<typename T>
    inline static DataBlock<T>conjugate(const  DataBlock<T>&d);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd,typename T>
    inline static DataBlock<T>tensor_subspan(const  DataBlock<T>&d, const size_t *    poffsets,const size_t *   psub_extents, size_t* new_extents, size_t*    psub_strides);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd,typename T>
    inline static DataBlock<T>tensor_subspan_copy(const  DataBlock<T>&d,const size_t *    poffsets,const size_t *   psub_extents, size_t* new_extents, size_t*   psub_strides, T*    sub_data);

    template<typename T>
    inline static DataBlock<T>matrix_subspan(const  DataBlock<T>&d, const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd,typename T>
    inline static DataBlock<T>matrix_subspan_copy(const  DataBlock<T>&d, const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data);

    template <typename T>
    inline static DataBlock<T> matrix_transpose(const  DataBlock<T>&d,size_t*    newextents, size_t*    newstrides);

    template<typename T>
    inline static DataBlock<T> matrix_hermitian_transpose(const  DataBlock<T>&d,size_t*    newextents, size_t*    newstrides);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd,typename T>
    inline static DataBlock<T> matrix_transpose_copy(const  DataBlock<T>&d,size_t*    newextents, size_t*    newstrides,T* newdata);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd,typename T>
    inline static DataBlock<T> matrix_hermitian_transpose_copy(const  DataBlock<T>&d,size_t*    newextents, size_t*    newstrides,T* newdata);

    template<typename T>
    inline static DataBlock<T> matrix_row(const  DataBlock<T>&d,const size_t row_index, size_t*    newextents, size_t*    newstrides);

    template<typename T>
    inline static DataBlock<T> matrix_column(const  DataBlock<T>&d,const size_t col_index, size_t*    newextents, size_t*    newstrides);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd,typename T>
    inline static DataBlock<T> matrix_column_copy(const  DataBlock<T>&d,const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata);

    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd,typename T>
    inline static DataBlock<T> matrix_row_copy(const  DataBlock<T>&d,const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata);

    template<typename T>
    inline static size_t count_noncollapsed_dims(const  DataBlock<T>&d) ;

    template<typename T>
    inline static DataBlock<T>  collapsed_view(const  DataBlock<T>&d,const size_t num_non_collapsed_dims,size_t* extents, size_t* strides) ;


    template <OpenMPVariant Policy = OpenMPVariant::ParallelSimd, typename T>
    inline static float sparsity(const  DataBlock<T>&d);

    template<typename T>
    inline static DataBlock<T> create_vector(T* data, size_t* extents, size_t* strides, DataBlockConfig config, const StridesCalculation computestrides);

    template<typename T>
    inline static DataBlock<T>create_matrix(T* data,  const size_t rows,  const size_t cols,  size_t* extents,  size_t* strides,  DataBlockConfig config,   const StridesCalculation computestrides);

};
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
DataBlock<T>DataBlockUtilities::create_vector(
    T* data,
    size_t* extents,
    size_t* strides,
    DataBlockConfig config,
    const StridesCalculation stride_mode)
    {
    config.dprowmajor =true;
    size_t calculated_length = 0;
    if (extents!=nullptr && strides!=nullptr)
    {
        if(stride_mode == StridesCalculation::Compute)
            strides[0] = 1;

        calculated_length = (extents[0] - 1) * strides[0] + 1;
    }

    return DataBlock<T>(data, calculated_length, 1, extents, strides, config);
    }


#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
inline DataBlock<T>DataBlockUtilities::create_matrix(
    T* data,
    const size_t rows,
    const size_t cols,
    size_t* extents,
    size_t* strides,
    DataBlockConfig config,
    const StridesCalculation stride_mode) {
    size_t final_rank = 0;
    size_t final_length = 0;
    const bool rowm = config.dprowmajor;


    if (rows > 1 && cols > 1) {
        final_rank = 2;
        if (extents) {
            extents[0] = rows;
            extents[1] = cols;
        }
        if (strides!=nullptr) {
            if (stride_mode == StridesCalculation::Compute) {
                strides[0] = rowm ? cols : 1;
                strides[1] = rowm ? 1 : rows;
            } else {
                config.dprowmajor = (strides[1] < strides[0]) ? true : false;
            }
            if (extents!=nullptr) {
                final_length = (rows - 1) * strides[0] + (cols - 1) * strides[1] + 1;
            }
        }
    }

    else if (rows == 0 && cols == 0) {
        final_rank = 0;
        final_length = 0;
        config.dprowmajor = true;
    }

    else {
        final_rank = 1;
        config.dprowmajor =true;
        const size_t length = (rows > 1) ? rows : cols;

        if (extents!=nullptr)
            extents[0] = length;

        if (strides!=nullptr) {
            if (stride_mode == StridesCalculation::Compute)
                strides[0] = 1;
            if (extents!=nullptr)
                final_length = (extents[0] - 1) * strides[0] + 1;
        }
    }

    return DataBlock<T>(data, final_length, final_rank, extents, strides, config);
}

#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
DataBlock<T>DataBlockUtilities::conjugate(const  DataBlock<T>&d)
{
    return DataBlock<T>(
               d.dpdata,
               d.dpdatalength,
               d.dprank,
               d.dpextents,
               d.dpstrides,
               DataBlockConfig{
        .dprowmajor       = d.dpconfig.dprowmajor,
        .data_ondevice = d.dpconfig.data_ondevice,
        .devicenum = d.dpconfig.devicenum,
        .dpconjugate      = !d.dpconfig.dpconjugate}
           );
}
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>
inline DataBlock<T> DataBlockUtilities::matrix_transpose(const  DataBlock<T>&d,size_t*    newextents, size_t *newstrides)
{

    newextents[0]=d.dpextents[1];
    newextents[1]=d.dpextents[0];
    newstrides[0]=d.dpstrides[1];
    newstrides[1]=d.dpstrides[0];

    return DataBlock(d.dpdata,d.dpdatalength,2,newextents,newstrides,d.dpconfig);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
inline DataBlock<T> DataBlockUtilities::matrix_hermitian_transpose(const  DataBlock<T>&d,size_t*    newextents, size_t *newstrides)
{

    newextents[0]=d.dpextents[1];
    newextents[1]=d.dpextents[0];
    newstrides[0]=d.dpstrides[1];
    newstrides[1]=d.dpstrides[0];

    return DataBlock(d.dpdata,d.dpdatalength,2,newextents,newstrides, DataBlockConfig{
        .dprowmajor       = d.dpconfig.dprowmajor,
        .data_ondevice = d.dpconfig.data_ondevice,
        .devicenum = d.dpconfig.devicenum,
        .dpconjugate      = !d.dpconfig.dpconjugate});
}
#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
inline DataBlock<T> DataBlockUtilities::matrix_transpose_copy(const  DataBlock<T>&d,size_t*    newextents, size_t *newstrides, T* newdata)
{

    newextents[0]=d.dpextents[1];
    newextents[1]=d.dpextents[0];

    newstrides[0]=d.dpstrides[1];
    newstrides[1]=d.dpstrides[0];
    T* pd=d.dpdata;

    const size_t rows=d.dpextents[0];
    const size_t cols=d.dpextents[1];
    const size_t old_s0=d.dpstrides[0];
    const size_t old_s1=d.dpstrides[1];
    const size_t new_s0=newstrides[0];
    const size_t new_s1=newstrides[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {


        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {

            #pragma omp target parallel for simd collapse(2) device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i=0; i<rows; i++)
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i=0; i<rows; i++)
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
        }
    }
    else if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target simd collapse(2) device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i=0; i<rows; i++)
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
        }
        else
        {
            #pragma omp simd collapse(2)
            for (size_t i=0; i<rows; i++)
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
        }
    }
    else
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target  device(d.dpconfig.devicenum) is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i=0; i<rows; i++)
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
        }
        else
        {

            for (size_t i=0; i<rows; i++)
                #pragma omp unroll partial
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
        }

    }

    return DataBlock(newdata,d.dpdatalength,2,newextents,newstrides,d.dpconfig);

}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
inline DataBlock<T> DataBlockUtilities::matrix_hermitian_transpose_copy(const  DataBlock<T>&d,size_t*    newextents, size_t *newstrides, T* newdata)
{

    newextents[0]=d.dpextents[1];
    newextents[1]=d.dpextents[0];

    newstrides[0]=d.dpstrides[1];
    newstrides[1]=d.dpstrides[0];
    T* pd=d.dpdata;

    const size_t rows=d.dpextents[0];
    const size_t cols=d.dpextents[1];
    const size_t old_s0=d.dpstrides[0];
    const size_t old_s1=d.dpstrides[1];
    const size_t new_s0=newstrides[0];
    const size_t new_s1=newstrides[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {

            #pragma omp target parallel for simd collapse(2) device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i=0; i<rows; i++)
            {
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
            }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i=0; i<rows; i++)
            {
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
            }
        }


    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target simd collapse(2) device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i=0; i<rows; i++)
            {
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
            }
        }
        else
        {
            #pragma omp simd collapse(2)
            for (size_t i=0; i<rows; i++)
            {
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
            }
        }
    }
    else
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target  device(d.dpconfig.devicenum) is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i=0; i<rows; i++)
            {
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
            }
        }
        else
        {

            for (size_t i=0; i<rows; i++)
            {
                #pragma omp unroll partial
                for (size_t j=0; j<cols; j++)
                {
                    size_t dst_index = j*new_s0+i*new_s1;
                    size_t src_index = i*old_s0+ j*old_s1;
                    newdata[dst_index] = pd[src_index];
                }
            }
        }


    }

    return DataBlock(newdata,d.dpdatalength,2,newextents,newstrides,d.dpconfig);

}
#pragma omp end declare target



#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
DataBlock<T>DataBlockUtilities::tensor_subspan(const  DataBlock<T>&d,const size_t * poffsets, const size_t * psub_extents,size_t* newextents, size_t*    new_strides)
{
    const size_t r = d.dprank;
    size_t offset_index = 0;
    size_t length_index = 0;
    size_t rank_out = 0;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for simd reduction(+:offset_index,length_index)
        for (size_t i = 0; i < r; ++i)
        {
            offset_index  += poffsets[i] * d.dpstrides[i];
            length_index  += (psub_extents[i] - 1) * d.dpstrides[i];
        }


        #pragma omp parallel for simd reduction(+:rank_out)
        for (size_t i = 0; i < r; ++i)
            if (psub_extents[i] > 1)
                ++rank_out;
    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(+:offset_index,length_index)
        for (size_t i = 0; i < r; ++i)
        {
            offset_index  += poffsets[i] * d.dpstrides[i];
            length_index  += (psub_extents[i] - 1) * d.dpstrides[i];
        }

        #pragma omp simd reduction(+:rank_out)
        for (size_t i = 0; i < r; ++i)
            if (psub_extents[i] > 1)
                ++rank_out;
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < r; ++i)
        {
            offset_index  += poffsets[i] * d.dpstrides[i];
            length_index  += (psub_extents[i] - 1) * d.dpstrides[i];
        }

        #pragma omp unroll partial
        for (size_t i = 0; i < r; ++i)
            if (psub_extents[i] > 1)
                ++rank_out;

    }


    if (rank_out == 0) rank_out = 1;


    if (rank_out != r)
    {
        size_t idx = 0;
        for (size_t i = 0; i < r; ++i)
        {
            if (psub_extents[i] > 1)
            {
                newextents[idx] = psub_extents[i];
                new_strides[idx] = d.dpstrides[i] ;
                ++idx;
            }
        }

        if (idx == 0)   // scalar case
        {
            newextents[0] = 1;
            new_strides[0] = 1;
        }
    }

    return DataBlock(
               d.dpdata + offset_index,
               length_index + 1,
               rank_out,
               newextents,
               new_strides,
               d.dpconfig
           );
}
#pragma omp end  declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
DataBlock<T> DataBlockUtilities::tensor_subspan_copy(const  DataBlock<T>&d,
        const size_t* poffsets,
        const size_t* psub_extents,
        size_t* new_extents,
        size_t* new_strides,
        T* sub_data)
{
    const size_t r = d.dprank;

    size_t* tempstr = new size_t[r];
    fill_strides(psub_extents, tempstr, r, d.dpconfig.dprowmajor);

    // Allocate index arrays
    size_t* indices        = new size_t[r]();
    size_t* global_indices = new size_t[r];

    bool tmcpy = omp_is_initial_device() && d.dpconfig.data_ondevice;

    // Copy loop
    while (true)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for simd
            for (size_t i = 0; i < r; ++i)
                global_indices[i] = poffsets[i] + indices[i];
        }
        else  if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp  simd
            for (size_t i = 0; i < r; ++i)
                global_indices[i] = poffsets[i] + indices[i];
        }
        else
        {
            #pragma omp unroll partial
            for (size_t i = 0; i < r; ++i)
                global_indices[i] = poffsets[i] + indices[i];
        }

        size_t original_index = compute_offset<Policy>(global_indices, d.dpstrides, r);
        size_t buffer_index   = compute_offset<Policy>(indices, tempstr, r);

        if (tmcpy)
            omp_target_memcpy(sub_data,
                              d.dpdata,
                              sizeof(T),
                              sizeof(T) * buffer_index,
                              sizeof(T) * original_index,
                              d.dpconfig.devicenum,
                              d.dpconfig.devicenum);
        else
            sub_data[buffer_index] = d.dpdata[original_index];

        // Increment multi-dimensional indices
        size_t dim = r;
        while (dim-- > 0)
        {
            if (++indices[dim] < psub_extents[dim])
                break;

            indices[dim] = 0;
        }
        if (dim == size_t(-1)) break;
    }

    delete[] indices;
    delete[] global_indices;

    // Collapse trivial dimensions
    size_t rank_out = 0;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd  reduction(+:rank_out)
        for (size_t i = 0; i < r; ++i)
            if (psub_extents[i] > 1) ++rank_out;
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd  reduction(+:rank_out)
        for (size_t i = 0; i < r; ++i)
            if (psub_extents[i] > 1) ++rank_out;
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < r; ++i)
            if (psub_extents[i] > 1) ++rank_out;
    }

    if (rank_out == 0) rank_out = 1; // scalar

    size_t idx = 0;
    for (size_t i = 0; i < r; ++i)
    {
        if (psub_extents[i] > 1)
        {
            new_extents[idx] = psub_extents[i];
            new_strides[idx] = tempstr[i];
            ++idx;
        }
    }

    if (rank_out == 1 && idx == 0) // scalar case
    {
        new_extents[0] = 1;
        new_strides[0] = 1;
    }

    delete[] tempstr;

    size_t pl = compute_data_length<Policy>(new_extents, new_strides, rank_out);

    return DataBlock(
               sub_data,
               pl,
               rank_out,
               new_extents,
               new_strides,
               d.dpconfig
           );
}
#pragma omp end declare target







#pragma omp begin declare target
template<typename T>
DataBlock<T>  DataBlockUtilities::matrix_subspan( const  DataBlock<T>&d,const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides)
{
    psub_strides[0] = d.dpstrides[0];
    psub_strides[1] = d.dpstrides[1];
    psub_extents[0] = tile_rows;
    psub_extents[1] = tile_cols;

    size_t offset = row * d.dpstrides[0] + col * d.dpstrides[1];
    T* data_ptr = d.dpdata + offset;

    if (tile_rows == 1 && tile_cols == 1)
    {
        psub_extents[0] = 1;
        psub_strides[0]=1;
        return DataBlock<T>(data_ptr, 1,  1, psub_extents, psub_strides, d.dpconfig);
    }
    else if (tile_rows == 1)
    {
        psub_extents[0] = tile_cols;
        psub_strides[0] = d.dpstrides[1];
        return DataBlock<T>(data_ptr, tile_cols, 1, psub_extents, psub_strides, d.dpconfig);
    }
    else if (tile_cols == 1)
    {
        psub_extents[0] = tile_rows;
        psub_strides[0] = d.dpstrides[0];
        return DataBlock<T>(data_ptr, tile_rows,  1, psub_extents, psub_strides, d.dpconfig);
    }
    else
    {
        size_t pl = (tile_rows-1) * d.dpstrides[0] + (tile_cols-1) * d.dpstrides[1] + 1;
        return DataBlock<T>(data_ptr, pl,2, psub_extents, psub_strides, d.dpconfig);
    }
}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
DataBlock<T>  DataBlockUtilities::matrix_subspan_copy(const  DataBlock<T>&d, const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)
{

    const size_t s0 = d.dpstrides[0];
    const size_t s1 = d.dpstrides[1];
    const T* pd = d.dpdata;

    // Set extents and strides
    psub_extents[0] = tile_rows;
    psub_extents[1] = tile_cols;
    psub_strides[0] = d.dpconfig.dprowmajor ? tile_cols : 1;
    psub_strides[1] = d.dpconfig.dprowmajor ? 1 : tile_rows;
    const size_t ps_str0=psub_strides[0];
    const size_t ps_str1=psub_strides[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        if (omp_is_initial_device() &&d.dpconfig.data_ondevice)
        {
            #pragma omp target parallel for simd collapse(2) device(d.dpconfig.devicenum) is_device_ptr(sub_data) is_device_ptr(pd)
            for (size_t i = 0; i < tile_rows; ++i)
                for (size_t j = 0; j < tile_cols; ++j)
                    sub_data[i * ps_str0 + j * ps_str1] = pd[(row+i)*s0 + (col+j)*s1];
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < tile_rows; ++i)
                for (size_t j = 0; j < tile_cols; ++j)
                    sub_data[i * ps_str0 + j * ps_str1] = pd[(row+i)*s0 + (col+j)*s1];
        }
    }
    else    if constexpr (Policy == OpenMPVariant::Simd)
    {
        if (omp_is_initial_device() && d.dpconfig.data_ondevice)
        {
            #pragma omp target  simd collapse(2) device(d.dpconfig.devicenum) is_device_ptr(sub_data) is_device_ptr(pd)
            for (size_t i = 0; i < tile_rows; ++i)
                for (size_t j = 0; j < tile_cols; ++j)
                    sub_data[i * ps_str0 + j * ps_str1] = pd[(row+i)*s0 + (col+j)*s1];
        }
        else
        {
            #pragma omp simd collapse(2)
            for (size_t i = 0; i < tile_rows; ++i)
                for (size_t j = 0; j < tile_cols; ++j)
                    sub_data[i *ps_str0+ j * ps_str1] = pd[(row+i)*s0 + (col+j)*s1];
        }
    }
    else
    {
        if (omp_is_initial_device() && d.dpconfig.data_ondevice)
        {
            #pragma omp target device(d.dpconfig.devicenum) is_device_ptr(sub_data) is_device_ptr(pd)
            for (size_t i = 0; i < tile_rows; ++i)
                for (size_t j = 0; j < tile_cols; ++j)
                    sub_data[i * ps_str0 + j * ps_str1] = pd[(row+i)*s0 + (col+j)*s1];

        }
        else
        {

            for (size_t i = 0; i < tile_rows; ++i)
                #pragma omp unroll partial
                for (size_t j = 0; j < tile_cols; ++j)
                    sub_data[i * ps_str0 + j * ps_str1] = pd[(row+i)*s0 + (col+j)*s1];
        }
    }

    // Determine rank
    size_t rank_out = 2;
    size_t length = tile_rows * tile_cols;
    if (tile_rows == 1 && tile_cols == 1)
    {
        rank_out = 1;
        psub_extents[0] = 1;
        length = 1;
    }
    else if (tile_rows == 1)
    {
        rank_out = 1;
        psub_extents[0] = tile_cols;
        psub_strides[0] = 1;
        length = tile_cols;
    }
    else if (tile_cols == 1)
    {
        rank_out = 1;
        psub_extents[0] = tile_rows;
        psub_strides[0] = 1;
        length = tile_rows;
    }
    return DataBlock<T>(sub_data, length, rank_out, psub_extents, psub_strides, d.dpconfig);

}
#pragma omp end declare target












#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlockUtilities::matrix_row(const  DataBlock<T>&d,const size_t row_index, size_t*    extents,size_t *    new_strides)
{
    extents[0] = d.dpextents[1];
    new_strides[0]=d.dpstrides[1];

    return DataBlock<T>( d.dpdata + row_index * d.dpstrides[0],  d.dpstrides[1] * extents[0],   1, extents,    new_strides, d.dpconfig);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlockUtilities::matrix_column(const  DataBlock<T>&d,const size_t col_index, size_t*    extents,size_t *   new_strides)
{
    extents[0] = d.dpextents[0];
    new_strides[0]=d.dpstrides[0];
    return DataBlock(d.dpdata + col_index * d.dpstrides[1], d.dpstrides[0] * extents[0],  1, extents,   new_strides,d.dpconfig );
}
#pragma omp end declare target



#pragma omp begin declare target

template <OpenMPVariant Policy, typename T>
DataBlock<T> DataBlockUtilities::matrix_row_copy(const  DataBlock<T>&d,const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)
{
    const size_t pl=d.dpextents[1];
    extents[0] = pl;
    new_strides[0] =1;

    const size_t s0=d.dpstrides[0];
    const size_t s1=d.dpstrides[1];
    const T*    pd=d.dpdata;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {

            #pragma omp target parallel for simd device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t j = 0; j < pl; ++j)
                newdata[j] = pd[row_index*s0+j*s1];
        }
        else
        {
            #pragma omp parallel for simd
            for (size_t j = 0; j < pl; ++j)
                newdata[j] = pd[row_index*s0+j*s1];
        }
    }

    else     if constexpr (Policy == OpenMPVariant::Simd)
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target simd device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t j = 0; j < pl; ++j)
                newdata[j] = pd[row_index*s0+j*s1];
        }
        else
        {
            #pragma omp simd
            for (size_t j = 0; j < pl; ++j)
                newdata[j] = pd[row_index*s0+j*s1];
        }
    }
    else
    {

        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target  device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t j = 0; j < pl; ++j)
                newdata[j] = pd[row_index*s0+j*s1];
        }
        else
        {
            #pragma omp unroll partial
            for (size_t j = 0; j < pl; ++j)
                newdata[j] = pd[row_index*s0+j*s1];
        }

    }

    return DataBlock<T>(newdata,  pl,   1, extents,    new_strides, d.dpconfig);
}
#pragma omp end declare target



#pragma omp begin declare target
template< OpenMPVariant Policy, typename T>
DataBlock<T> DataBlockUtilities::matrix_column_copy(const  DataBlock<T>&d,const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)
{

    const size_t pl=d.dpextents[0];
    extents[0] = pl;

    new_strides[0] = 1;

    const size_t s0=d.dpstrides[0];
    const size_t s1=d.dpstrides[1];
    const T*    pd=d.dpdata;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target parallel for simd device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i = 0; i < pl; ++i)
                newdata[i] = pd[i*s0+col_index*s1];
        }
        else
        {
            #pragma omp parallel for simd
            for (size_t i = 0; i < pl; ++i)
                newdata[i] = pd[i*s0+col_index*s1];
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target simd device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i = 0; i < pl; ++i)
                newdata[i] = pd[ i*s0+col_index*s1];
        }
        else
        {
            #pragma omp simd
            for (size_t i = 0; i < pl; ++i)
                newdata[i] = pd[ i*s0+col_index*s1];
        }
    }
    else
    {
        if(omp_is_initial_device()&&d.dpconfig.data_ondevice)
        {
            #pragma omp target device(d.dpconfig.devicenum)is_device_ptr(newdata) is_device_ptr(pd)
            for (size_t i = 0; i < pl; ++i)
                newdata[i] = pd[ i*s0+col_index*s1];
        }
        else
        {

            #pragma omp unroll partial
            for (size_t i = 0; i < pl; ++i)
                newdata[i] = pd[ i*s0+col_index*s1];
        }
    }


    return DataBlock<T>(newdata,  pl,   1, extents,    new_strides, d.dpconfig);
}
#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
inline  float DataBlockUtilities::sparsity(const  DataBlock<T>&d)
{
    size_t count=0;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {


        if(omp_is_initial_device()&& d.dpconfig.data_ondevice)
        {
            #pragma omp target teams distribute parallel for simd map(tofrom: count) shared(count)  device(d.dpconfig.devicenum)
            for(size_t i=0; i<d.dpdatalength; i++)
            {
                if(d.dpdata[i]==0)
                {
                    #pragma omp atomic update
                    count++;
                }
            }
        }
        else
        {
            #pragma omp parallel for  shared(count)
            for(size_t i=0; i<d.dpdatalength; i++)
            {
                if(d.dpdata[i]==0)
                {
                    #pragma omp atomic update
                    count++;
                }
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        if(omp_is_initial_device()&& d.dpconfig.data_ondevice)
        {
            #pragma omp target simd map(tofrom: count)  device(d.dpconfig.devicenum)
            for(size_t i=0; i<d.dpdatalength; i++)
            {
                if(d.dpdata[i]==0)
                {
                    #pragma omp atomic update
                    count++;
                }
            }
        }
        else
        {
            #pragma omp simd
            for(size_t i=0; i<d.dpdatalength; i++)
            {
                if(d.dpdata[i]==0)
                {
                    #pragma omp atomic update
                    count++;
                }
            }
        }
    }
    else
    {

        if(omp_is_initial_device()&& d.dpconfig.data_ondevice)
        {
            #pragma omp target map(tofrom: count)  device(d.dpconfig.devicenum)
            for(size_t i=0; i<d.dpdatalength; i++)
            {
                if(d.dpdata[i]==0)
                {
                    count++;
                }
            }
        }
        else
        {
            #pragma omp unroll partial
            for(size_t i=0; i<d.dpdatalength; i++)
            {
                if(d.dpdata[i]==0)
                {
                    count++;
                }
            }
        }
    }
    return (float)count/(float)d.dpdatalength;
}
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>
size_t DataBlockUtilities::count_noncollapsed_dims(const  DataBlock<T>&d)
{
    size_t count = 0;

    for (size_t i = 0; i < d.dprank; ++i)
        if (d.dpextents[i] > 1) ++count;
    return count == 0 ? 1 : count;
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
DataBlock<T> DataBlockUtilities::collapsed_view(const  DataBlock<T>&d,const size_t num_non_collapsed_dims,size_t *extents, size_t *strides)
{

    size_t idx = 0;
    for (size_t i = 0; i < d.dprank; ++i)
    {
        if (d.dpextents[i] > 1)
        {
            extents[idx] = d.dpextents[i];
            strides[idx] = d.dpstrides[i];
            ++idx;
        }
    }
    // handle scalar case
    if (idx == 0)
    {
        extents[0] = 1;
        strides[0] = 1;
    }


    DataBlock<T> view(
        d.dpdata,
        d.dpdatalength,
        num_non_collapsed_dims,
        extents,
        strides,
        d.dpconfig );


    return view;
}
#pragma omp end declare target


#endif
