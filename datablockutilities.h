#ifndef DATABLOCKUTILITIES
#define DATABLOCKUTILITIES
#include "datablock.h"
#include "indiceshelperfunctions.h"

#pragma omp begin declare target
enum class StridesCalculation
{
    NoComputation,
    Compute
};
#pragma omp end declare target

#pragma omp begin declare target
class DataBlockUtilities
{
public:

    template<typename T>
    inline static DataBlock<T>conjugate(const  DataBlock<T>&d);


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

#pragma omp begin declare target
template<typename T>
DataBlock<T>DataBlockUtilities::create_vector(
    T* data,
    ptrdiff_t* extents,
    ptrdiff_t* strides,
    DataBlockConfig config,
    const StridesCalculation stride_mode)
{
    config.dprowmajor =true;
    ptrdiff_t calculated_length = 0;
    if (extents!=nullptr && strides!=nullptr)
    {
        if(stride_mode == StridesCalculation::Compute)
            strides[0] = 1;

        calculated_length = (abs(extents[0]) - 1) * strides[0] + 1;
    }

    return DataBlock<T>(data, calculated_length, 1, extents, strides, config);
}


#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
inline DataBlock<T>DataBlockUtilities::create_matrix(
    T* data,
    const ptrdiff_t rows,
    const ptrdiff_t cols,
    ptrdiff_t* extents,
    ptrdiff_t* strides,
    DataBlockConfig config,
    const StridesCalculation stride_mode)
{
    ptrdiff_t final_rank = 0;
    ptrdiff_t final_length = 0;
    const bool rowm = config.dprowmajor;


    if (rows > 1 && cols > 1)
    {
        final_rank = 2;
        if (extents)
        {
            extents[0] = abs(rows);
            extents[1] = abs(cols);
        }
        if (strides!=nullptr)
        {
            if (stride_mode == StridesCalculation::Compute)
            {
                strides[0] = rowm ? abs(cols) : 1;
                strides[1] = rowm ? 1 : abs(rows);
            }
            else
            {
                config.dprowmajor = (abs(strides[1]) < abs(strides[0])) ? true : false;
            }
            if (extents!=nullptr)
            {
                final_length = (abs(rows) - 1) * strides[0] + (abs(cols) - 1) * strides[1] + 1;
            }
        }
    }

    else if (rows == 0 && cols == 0)
    {
        final_rank = 0;
        final_length = 0;
        config.dprowmajor = true;
    }

    else
    {
        final_rank = 1;
        config.dprowmajor =true;
        const ptrdiff_t length = (abs(rows) > 1) ? abs(rows) : abs(cols);

        if (extents!=nullptr)
            extents[0] = length;

        if (strides!=nullptr)
        {
            if (stride_mode == StridesCalculation::Compute)
                strides[0] = 1;
            if (extents!=nullptr)
                final_length = (abs(extents[0]) - 1) * strides[0] + 1;
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
               DataBlockConfig
    {
        .dprowmajor       = d.dpconfig.dprowmajor,
        .dpconjugate      = !d.dpconfig.dpconjugate,
        .pmemmap=        d.dpconfig.pmemmap,
        .data_is_devptr = d.dpconfig.data_is_devptr,
        .devicenum = d.dpconfig.devicenum,
    }
           );
}
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>
inline DataBlock<T> DataBlockUtilities::matrix_transpose(const  DataBlock<T>&d,ptrdiff_t*    newextents, ptrdiff_t *newstrides)
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
inline DataBlock<T> DataBlockUtilities::matrix_hermitian_transpose(const  DataBlock<T>&d,ptrdiff_t*    newextents, ptrdiff_t *newstrides)
{

    newextents[0]=d.dpextents[1];
    newextents[1]=d.dpextents[0];
    newstrides[0]=d.dpstrides[1];
    newstrides[1]=d.dpstrides[0];

    return DataBlock(d.dpdata,d.dpdatalength,2,newextents,newstrides, DataBlockConfig
    {
        .dprowmajor       = d.dpconfig.dprowmajor,
        .dpconjugate      = !d.dpconfig.dpconjugate,
        .pmemmap=        d.dpconfig.pmemmap,
        .data_is_devptr = d.dpconfig.data_is_devptr,
        .devicenum = d.dpconfig.devicenum,});
}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
DataBlock<T>DataBlockUtilities::tensor_subspan(const  DataBlock<T>&d,const ptrdiff_t * poffsets, const ptrdiff_t * psub_extents,ptrdiff_t* newextents, ptrdiff_t*    new_strides)
{
    const ptrdiff_t r = d.dprank;
    ptrdiff_t offset_index = 0;
    ptrdiff_t length_index = 0;
    ptrdiff_t rank_out = 0;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for simd reduction(+:offset_index,length_index)
        for (ptrdiff_t i = 0; i < r; ++i)
        {
            offset_index  += abs(poffsets[i]) * d.dpstrides[i];
            length_index  += (abs(psub_extents[i]) - 1) * d.dpstrides[i];
        }


        #pragma omp parallel for simd reduction(+:rank_out)
        for (ptrdiff_t i = 0; i < r; ++i)
            if (abs(psub_extents[i]) > 1)
                ++rank_out;
    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(+:offset_index,length_index)
        for (ptrdiff_t i = 0; i < r; ++i)
        {
            offset_index  += abs(poffsets[i]) * d.dpstrides[i];
            length_index  += (abs(psub_extents[i]) - 1) * d.dpstrides[i];
        }

        #pragma omp simd reduction(+:rank_out)
        for (ptrdiff_t i = 0; i < r; ++i)
            if (abs(psub_extents[i]) > 1)
                ++rank_out;
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < r; ++i)
        {
            offset_index  += abs(poffsets[i]) * d.dpstrides[i];
            length_index  += (abs(psub_extents[i]) - 1) * d.dpstrides[i];
        }

        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < r; ++i)
            if (abs(psub_extents[i]) > 1)
                ++rank_out;

    }


    if (rank_out == 0) rank_out = 1;


    if (rank_out != r)
    {
        ptrdiff_t idx = 0;
        for (ptrdiff_t i = 0; i < r; ++i)
        {
            if (abs(psub_extents[i]) > 1)
            {
                newextents[idx] = abs(psub_extents[i]);
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
template<typename T>
void DataBlockUtilities::copy(DataBlock<T>&target,DataBlock<T> source,DataBlockConfig targetcfg)
{

    int targetdev, sourcedev;
    bool useomptargetmemcpy = false;
    if(omp_is_initial_device())
    {
        if(targetcfg.data_is_devptr && source.dpconfig.data_is_devptr)
        {
            targetdev = targetcfg.devicenum;
            sourcedev = source.dpconfig.devicenum;
            useomptargetmemcpy = true;
        }
        else if(targetcfg.data_is_devptr && !source.dpconfig.data_is_devptr)
        {
            targetdev = targetcfg.devicenum;
            sourcedev = omp_get_initial_device();
            useomptargetmemcpy = true;
        }
        else if(!targetcfg.data_is_devptr &&source.dpconfig.data_is_devptr)
        {
            targetdev = omp_get_initial_device();
            sourcedev = source.dpconfig.devicenum;
            useomptargetmemcpy = true;
        }

        if(useomptargetmemcpy)
            omp_target_memcpy(target.dpdata, source.dpdata, sizeof(T) * source.dpdatalength, 0, 0, targetdev, sourcedev);
        else
            memcpy(target.dpdata, source.dpdata, sizeof(T) * source.dpdatalength);
    }
    else
    {
        #pragma omp parallel for simd
        for(ptrdiff_t i=0; i<source.dpdatalength; i++)
            target.dpdata[i]=source.dpdata[i];
    }
    target.dpdatalength=source.dpdatalength;
    target.dprank=source.dprank;
    memcpy(target.dpextents, source.dpextents,sizeof(ptrdiff_t)*source.dprank);
    memcpy(target.dpstrides, source.dpstrides,sizeof(ptrdiff_t)*source.dprank);
    target.dpconfig.conjugate=source.dpconfig.dpconjugate;
    target.dpconfig.dprowmajor=source.dpconfig.dprowmajor;
    target.dpconfig.data_is_devptr=targetcfg.data_is_devptr;
    target.dpconfig.devicenum=targetcfg.devicenum;
    target.dpconfig.pmemmap=targetcfg.data_is_devptr? false: targetcfg.pmemmap;
    return;
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
DataBlock<T>  DataBlockUtilities::matrix_subspan( const  DataBlock<T>&d,const ptrdiff_t row, const ptrdiff_t col,const  ptrdiff_t tile_rows,const  ptrdiff_t tile_cols,  ptrdiff_t *    psub_extents,  ptrdiff_t *   psub_strides)
{
    psub_strides[0] = d.dpstrides[0];
    psub_strides[1] = d.dpstrides[1];
    psub_extents[0] = abs(tile_rows);
    psub_extents[1] =abs(tile_cols);

    ptrdiff_t offset = abs(row) * d.dpstrides[0] + abs(col) * d.dpstrides[1];
    T* data_ptr = d.dpdata + offset;

    if (abs(tile_rows) == 1 && abs(tile_cols) == 1)
    {
        psub_extents[0] = 1;
        psub_strides[0]=1;
        return DataBlock<T>(data_ptr, 1,  1, psub_extents, psub_strides, d.dpconfig);
    }
    else if (abs(tile_rows) == 1)
    {
        psub_extents[0] = abs(tile_cols);
        psub_strides[0] = d.dpstrides[1];
        return DataBlock<T>(data_ptr, abs(tile_cols), 1, psub_extents, psub_strides, d.dpconfig);
    }
    else if (abs(tile_cols) == 1)
    {
        psub_extents[0] = abs(tile_rows);
        psub_strides[0] = d.dpstrides[0];
        return DataBlock<T>(data_ptr, abs(tile_rows),  1, psub_extents, psub_strides, d.dpconfig);
    }
    else
    {
        ptrdiff_t pl = (abs(tile_rows)-1) * d.dpstrides[0] + (abs(tile_cols)-1) * d.dpstrides[1] + 1;
        return DataBlock<T>(data_ptr, pl,2, psub_extents, psub_strides, d.dpconfig);
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlockUtilities::matrix_row(const  DataBlock<T>&d,const ptrdiff_t row_index, ptrdiff_t*    extents,ptrdiff_t *    new_strides)
{
    extents[0] = d.dpextents[1];
    new_strides[0]=d.dpstrides[1];

    return DataBlock<T>( d.dpdata + abs(row_index) * d.dpstrides[0],  d.dpstrides[1] * extents[0],   1, extents,    new_strides, d.dpconfig);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlockUtilities::matrix_column(const  DataBlock<T>&d,const ptrdiff_t col_index, ptrdiff_t*    extents,ptrdiff_t *   new_strides)
{
    extents[0] = d.dpextents[0];
    new_strides[0]=d.dpstrides[0];
    return DataBlock(d.dpdata + abs(col_index) * d.dpstrides[1], d.dpstrides[0] * extents[0],  1, extents,   new_strides,d.dpconfig );
}
#pragma omp end declare target



#
#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
inline  float DataBlockUtilities::sparsity(const  DataBlock<T>&d)
{
    ptrdiff_t count=0;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {


        if(omp_is_initial_device()&& d.dpconfig.data_is_devptr)
        {
            #pragma omp target teams distribute parallel for simd map(tofrom: count) shared(count)  device(d.dpconfig.devicenum)
            for(ptrdiff_t i=0; i<d.dpdatalength; i++)
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
            for(ptrdiff_t i=0; i<d.dpdatalength; i++)
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
        if(omp_is_initial_device()&& d.dpconfig.data_is_devptr)
        {
            #pragma omp target simd map(tofrom: count)  device(d.dpconfig.devicenum)
            for(ptrdiff_t i=0; i<d.dpdatalength; i++)
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
            for(ptrdiff_t i=0; i<d.dpdatalength; i++)
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

        if(omp_is_initial_device()&& d.dpconfig.data_is_devptr)
        {
            #pragma omp target map(tofrom: count)  device(d.dpconfig.devicenum)
            for(ptrdiff_t i=0; i<d.dpdatalength; i++)
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
            for(ptrdiff_t i=0; i<d.dpdatalength; i++)
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
ptrdiff_t DataBlockUtilities::count_noncollapsed_dims(const  DataBlock<T>&d)
{
    ptrdiff_t count = 0;

    for (ptrdiff_t i = 0; i < d.dprank; ++i)
        if (d.dpextents[i] > 1) ++count;
    return count == 0 ? 1 : count;
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
DataBlock<T> DataBlockUtilities::collapsed_view(const  DataBlock<T>&d,const ptrdiff_t num_non_collapsed_dims,ptrdiff_t *extents, ptrdiff_t *strides)
{

    ptrdiff_t idx = 0;
    for (ptrdiff_t i = 0; i < d.dprank; ++i)
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
