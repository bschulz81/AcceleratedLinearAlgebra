#ifndef DATASTRUCT
#define DATASTRUCT

#include <omp.h>

#include "indiceshelperfunctions.h"



#if defined(Unified_Shared_Memory)
#pragma omp requires unified_shared_memory
#else
#pragma omp requires unified_address
#endif



template<typename T>
class Datastruct_GPU_Memory_Functions;

template<typename T>
class Datastruct_Host_Memory_Functions;

template<typename T>
class Datastruct_MPI_Functions;


template<typename T>
class In_Kernel_Mathfunctions;

template<typename T>
class Math_Functions;

template<typename T>
class Math_Functions_MPI;

template<typename T>
class GPU_Math_Functions;
template<typename U, typename Container>
class mdspan;

#pragma omp begin declare target
template <typename T>
class datastruct
{
public:
    friend class Datastruct_GPU_Memory_Functions<T>;
    friend class Datastruct_Host_Memory_Functions<T>;
    friend class Datastruct_MPI_Functions<T>;
    friend class In_Kernel_Mathfunctions<T>;
    friend class GPU_Math_Functions<T>;
    friend class Math_Functions<T>;
    friend class Math_Functions_MPI<T>;

    template<typename U, typename C>
    friend class ::mdspan;

    datastruct() {};

    // Constructors
    datastruct(T*  data, size_t datalength, bool rowm, size_t rank,size_t*   extents, size_t*   strides,
               bool compute_datalength,    bool compute_strides_from_extents,bool data_is_devptr );

    datastruct(T*  data,size_t datalength,bool rowm,size_t rows, size_t cols,  size_t*  extents, size_t*  strides,
               bool compute_datalength, bool compute_strides_from_extents,  bool data_is_devptr);

    datastruct(T*  data, size_t datalength, bool rowm,  size_t rank, size_t*  extents, size_t*  strides, bool data_is_devptr );

    inline size_t datalength() const
    {
        return dpdatalength;
    }

    inline size_t rank() const
    {
        return dprank;
    }

    inline bool rowmajor() const
    {
        return dprowmajor;
    }

    inline bool data_is_devptr() const
    {
        return dpdata_is_devptr;
    }

    inline T& data(size_t i)
    {
        return dpdata[i];
    }

    inline const T& data(size_t i) const
    {
        return dpdata[i];
    }

    inline size_t& extent(size_t i)
    {
        return dpextents[i];
    }

    inline const size_t& extent(size_t i) const
    {
        return dpextents[i];
    }

    inline size_t& stride(size_t i)
    {
        return dpstrides[i];
    }

    inline const size_t& stride(size_t i) const
    {
        return dpstrides[i];
    }


    inline T* data()
    {
        return dpdata;
    }

    inline const T* data() const
    {
        return dpdata;
    }

    inline size_t* extents()
    {
        return dpextents;
    }

    inline const size_t* extents() const
    {
        return dpextents;
    }

    inline size_t* strides()
    {
        return dpstrides;
    }

    inline const size_t* strides() const
    {
        return dpstrides;
    }




    // Operator overloads
    inline T& operator()(const size_t* indices)
    {
        return dpdata[compute_offset_s(indices, dpstrides, dprank)];
    };

    inline const T operator()(const size_t*    indices) const
    {
        return dpdata[compute_offset_s(indices, dpstrides, dprank)];
    };

    // Operator overloads
    inline T& operator()(const size_t row,  const size_t col)
    {
        return dpdata[row*dpstrides[0]+col*dpstrides[1]];
    };

    inline const T operator()( const size_t row, const size_t col) const
    {
        return dpdata[row*dpstrides[0]+col*dpstrides[1]];
    };

    // Operator overloads
    inline T& operator()(const size_t i)
    {
        return dpdata[i*dpstrides[0]];
    };

    inline const T operator()(const size_t i) const
    {
        return dpdata[i*dpstrides[0]];
    };

    inline datastruct<T>subspan_w(const size_t *    poffsets, size_t *   psub_extents, size_t*    psub_strides)const;
    inline datastruct<T>subspan_v(const size_t *    poffsets, size_t *   psub_extents, size_t*    psub_strides)const;
    inline datastruct<T>subspan_s(const size_t *    poffsets, size_t *   psub_extents, size_t*    psub_strides)const;

    inline datastruct<T>subspan_w(const size_t *    poffsets, size_t *   psub_extents, size_t*   psub_strides, T*    sub_data)const;
    inline datastruct<T>subspan_v(const size_t *    poffsets, size_t *   psub_extents, size_t*   psub_strides, T*    sub_data)const;
    inline datastruct<T>subspan_s(const size_t *    poffsets, size_t *   psub_extents, size_t*   psub_strides, T*    sub_data)const;

    inline datastruct<T>subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides)const;

    inline datastruct<T>subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const;
    inline datastruct<T>subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const ;
    inline datastruct<T>subspanmatrix_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const;

    inline datastruct<T> transpose(size_t*    newextents, size_t*    newstrides)const;

    inline datastruct<T> transpose_v(size_t*    newextents, size_t*    newstrides,T* newdata)const;
    inline datastruct<T> transpose_w(size_t*    newextents, size_t*    newstrides,T* newdata)const;
    inline datastruct<T> transpose_s(size_t*    newextents, size_t*    newstrides,T* newdata)const;


    inline datastruct<T> row(const size_t row_index, size_t*    newextents, size_t*    newstrides)const;
    inline datastruct<T> column(const size_t col_index, size_t*    newextents, size_t*    newstrides)const;

    inline datastruct<T> column_w(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline datastruct<T> column_v(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline datastruct<T> column_s(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;

    inline datastruct<T> row_w(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline datastruct<T> row_v(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline datastruct<T> row_s(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;

    size_t count_noncollapsed_dims() const;
    datastruct<T>collapsed_view(size_t num_non_collapsed_dims,size_t* extents, size_t* strides) const;
    inline bool is_contiguous()const;
    inline void printtensor()const;

protected:
    void printtensor_recursive(size_t* indices, size_t depth) const;

    T*          dpdata = nullptr;
    size_t*  dpextents = nullptr;
    size_t*  dpstrides = nullptr;
    size_t      dpdatalength = 0;
    size_t      dprank = 0;
    bool         dprowmajor = true;
    bool    dpdata_is_devptr=false;

};
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose(size_t*    newextents, size_t *newstrides)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];
    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];

    return datastruct(dpdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose_w(size_t*    newextents, size_t *newstrides, T* newdata)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];
    T* pd=this->dpdata;
    #pragma omp parallel for simd collapse(2)
    for (size_t i=0; i<dpextents[0]; i++)
        for (size_t j=0; j<dpextents[1]; j++)
        {
            size_t dst_index = compute_offset(j, i, newstrides[0], newstrides[1]);
            size_t src_index = compute_offset(i, j, dpstrides[0], dpstrides[1]);
            newdata[dst_index] = pd[src_index];
        }

    return datastruct(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose_v(size_t*    newextents, size_t *newstrides, T* newdata)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];
    T* pd=this->dpdata;
    #pragma omp simd collapse(2)
    for (size_t i=0; i<dpextents[0]; i++)
        for (size_t j=0; j<dpextents[1]; j++)
        {
            size_t dst_index = compute_offset(j, i, newstrides[0], newstrides[1]);
            size_t src_index = compute_offset(i, j, dpstrides[0], dpstrides[1]);
            newdata[dst_index] = pd[src_index];
        }
    return datastruct(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
inline datastruct<T> datastruct<T>::transpose_s(size_t*    newextents, size_t *newstrides, T* newdata)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];
    T* pd=this->dpdata;
    #pragma omp unroll
    for (size_t i=0; i<dpextents[0]; i++)
        for (size_t j=0; j<dpextents[1]; j++)
        {
            size_t dst_index = compute_offset(j, i, newstrides[0], newstrides[1]);
            size_t src_index = compute_offset(i, j, dpstrides[0], dpstrides[1]);
            newdata[dst_index] = pd[src_index];
        }

    return datastruct(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
bool datastruct<T>::is_contiguous() const
{
    if (dprank == 0)
    {
        return dpdatalength == 1;
    }

    size_t expected_stride = 1;
    if (dprowmajor)
    {

        for (int i = (int)dprank - 1; i >= 0; --i)
        {
            if (dpstrides[i] != expected_stride)return false;
            expected_stride *= dpextents[i];
        }
    }
    else
    {

        for (size_t i = 0; i < dprank; ++i)
        {
            if (dpstrides[i] != expected_stride)return false;
            expected_stride *= dpextents[i];
        }
    }

    // check total size matches
    return expected_stride == dpdatalength;
}
#pragma omp end declare target






#pragma omp begin declare target
inline void fill_strides(const size_t*    extents,size_t*    strides, const size_t rank, const bool rowmajor)
{
    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[rank - 1] = 1;
        #pragma omp unroll
        for (int i = rank - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
        #pragma omp unroll
        for (size_t i = 1; i < rank; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];
        }
    }
}

#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void datastruct<T>::printtensor() const
{
    size_t* indices= new size_t[dprank];
    #pragma omp simd
    for (size_t i = 0; i < dprank; ++i)
        indices[i] = 0;

    printtensor_recursive(indices, 0);
    delete []indices;

    printf("\n");
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void datastruct<T>::printtensor_recursive(size_t* indices, size_t depth) const
{
    if (depth == dprank)
    {
        printf("%g", (*this)(indices)); // element access via operator()(size_t*)
        return;
    }

    printf("[");

    for (size_t i = 0; i < dpextents[depth]; ++i)
    {
        indices[depth] = i;
        printtensor_recursive(indices, depth + 1);

        if (i + 1 < dpextents[depth])
        {
            printf(", ");
            if (depth < dprank - 1)
            {
                printf("\n");
                for (size_t k = 0; k < depth + 1; ++k)
                    printf(" ");
            }
        }
    }
    printf("]");
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T> datastruct<T>::datastruct(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t*    extents,
    size_t*    strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr

) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false)
#else
    dpdata_is_devptr(data_is_devptr)
#endif

{
    if(compute_strides_from_extents==true && extents!=nullptr && strides!=nullptr && rank !=0)
    {
        fill_strides(dpextents,dpstrides,rank,rowm);
    }
    if(compute_datalength==true && extents!=nullptr && strides!=nullptr && rank !=0)
    {
        dpdatalength=compute_data_length_s(extents,strides,rank);
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T> datastruct<T>::datastruct(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t*    extents,
    size_t*    strides,
    bool data_is_devptr
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false)
#else
    dpdata_is_devptr(data_is_devptr)
#endif
{}
#pragma omp end declare target





#pragma omp begin declare target
template<typename T> datastruct<T>::datastruct(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rows,
    size_t cols,
    size_t*    extents,
    size_t*    strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(2),
    dprowmajor(rowm),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false)
#else
    dpdata_is_devptr(data_is_devptr)
#endif
{
    if(extents!=nullptr)
    {
        dpextents[0]=rows;
        dpextents[1]=cols;
    }
    if(strides!=nullptr && compute_strides_from_extents)
    {
        dpstrides[0]=(rowm==true)? cols:1;
        dpstrides[1]=(rowm==true)?1: rows;
    }
    if(compute_datalength==true && extents!=nullptr && strides!=nullptr)
    {
        dpdatalength=(rows-1) * strides[0]+(cols-1)*strides[1]+1;
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
size_t datastruct<T>::count_noncollapsed_dims() const
{
    size_t count = 0;
    for (size_t i = 0; i < dprank; ++i)
        if (dpextents[i] > 1) ++count;
    return count == 0 ? 1 : count;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
datastruct<T> datastruct<T>::collapsed_view(size_t num_non_collapsed_dims,size_t *extents, size_t *strides) const
{

    size_t idx = 0;
    for (size_t i = 0; i < dprank; ++i)
    {
        if (dpextents[i] > 1)
        {
            extents[idx] = dpextents[i];
            strides[idx] = dpstrides[i];
            ++idx;
        }
    }
    // handle scalar case
    if (idx == 0)
    {
        extents[0] = 1;
        strides[0] = 1;
    }

    // Create non-owning datastruct
    datastruct<T> view(
        dpdata,
        dpdatalength,
        dprowmajor,
        num_non_collapsed_dims,
        extents,
        strides,
        dpdata_is_devptr
    );

    // User must manage extents/strides memory if needed
    return view;
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::subspan_w(const size_t *    poffsets,  size_t *   psub_extents, size_t*    psub_strides)const
{
    const size_t r = dprank;
    size_t offset_index = 0;
    size_t length_index = 0; // for computing pl


    #pragma omp parallel for simd reduction(+:offset_index,length_index)
    for (size_t i = 0; i < r; ++i)
    {
        offset_index  += poffsets[i]      * dpstrides[i];
        psub_strides[i] = dpstrides[i];
        length_index  += (psub_extents[i] - 1) * dpstrides[i];
    }

    return datastruct(dpdata + offset_index,
                      length_index + 1,
                      dprowmajor,
                      r,
                      psub_extents,
                      psub_strides,
                      dpdata_is_devptr);


}
#pragma omp end  declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::subspan_v(const size_t *    poffsets,  size_t *   psub_extents, size_t*    psub_strides)const
{
    const size_t r = dprank;
    size_t offset_index = 0;
    size_t length_index = 0; // for computing pl


    #pragma omp simd reduction(+:offset_index,length_index)
    for (size_t i = 0; i < r; ++i)
    {
        offset_index  += poffsets[i]      * dpstrides[i];
        psub_strides[i] = dpstrides[i];
        length_index  += (psub_extents[i] - 1) * dpstrides[i];
    }

    return datastruct(dpdata + offset_index,
                      length_index + 1,
                      dprowmajor,
                      r,
                      psub_extents,
                      psub_strides,
                      dpdata_is_devptr);

}
#pragma omp end  declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::subspan_s(const size_t * poffsets, size_t * psub_extents, size_t*    psub_strides)const
{
    const size_t r = dprank;
    size_t offset_index = 0;
    size_t length_index = 0; // for computing pl

    for (size_t i = 0; i < r; ++i)
    {
        offset_index  += poffsets[i]      * dpstrides[i];
        psub_strides[i] = dpstrides[i];
        length_index  += (psub_extents[i] - 1) * dpstrides[i];
    }

    return datastruct(dpdata + offset_index,
                      length_index + 1,
                      dprowmajor,
                      r,
                      psub_extents,
                      psub_strides,
                      dpdata_is_devptr);

}
#pragma omp end  declare target

#pragma omp begin declare target
template<typename T>
datastruct<T> datastruct<T>::subspan_s(
   const size_t* poffsets,
    size_t* psub_extents,
    size_t* psub_strides,
    T* sub_data)const
{
    size_t r = dprank;


    fill_strides(psub_extents, psub_strides, r, dprowmajor);


    size_t* indices        = new size_t[r];

    for(size_t i=0; i<r; i++)
        indices[i]=0;

    size_t* global_indices = new size_t[r];


    while (true)
    {
        for (size_t i = 0; i < r; ++i)
            global_indices[i] = poffsets[i] + indices[i];

        size_t original_index = compute_offset_s(global_indices, dpstrides, r);
        size_t buffer_index   = compute_offset_s(indices, psub_strides, r);

        sub_data[buffer_index] = dpdata[original_index];

        // Increment multi-dimensional indices
        size_t dim = r;
        while (dim-- > 0)
        {
            if (++indices[dim] < psub_extents[dim]) break;
            indices[dim] = 0;
        }
        if (dim == size_t(-1)) break;
    }

    delete[] indices;
    delete[] global_indices;


    size_t pl=compute_data_length_s(psub_extents,psub_strides,r);

    // Return non-owning datastruct: just pass pointers
    return datastruct(sub_data,
                      pl,
                      dprowmajor,
                      r,               // full rank
                      psub_extents,
                      psub_strides,
                      dpdata_is_devptr);
}
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>
datastruct<T> datastruct<T>::subspan_v(const size_t* poffsets,  size_t* psub_extents, size_t* psub_strides,  T* sub_data)const
{
    size_t r = dprank;

    fill_strides(psub_extents, psub_strides, r, dprowmajor);


    size_t* indices        = new size_t[r];

    #pragma omp simd
    for(size_t i=0; i<r; i++)
        indices[i]=0;

    size_t* global_indices = new size_t[r];


    while (true)
    {
        #pragma omp simd
        for (size_t i = 0; i < r; ++i)
            global_indices[i] = poffsets[i] + indices[i];

        size_t original_index = compute_offset_v(global_indices, dpstrides, r);
        size_t buffer_index   = compute_offset_v(indices, psub_strides, r);

        sub_data[buffer_index] = dpdata[original_index];

        // Increment multi-dimensional indices
        size_t dim = r;
        while (dim-- > 0)
        {
            if (++indices[dim] < psub_extents[dim]) break;
            indices[dim] = 0;
        }
        if (dim == size_t(-1)) break;
    }

    delete[] indices;
    delete[] global_indices;


    size_t pl=compute_data_length_v(psub_extents,psub_strides,r);

    // Return non-owning datastruct: just pass pointers
    return datastruct(sub_data,
                      pl,
                      dprowmajor,
                      r,               // full rank
                      psub_extents,
                      psub_strides,
                      dpdata_is_devptr);
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
datastruct<T> datastruct<T>::subspan_w(
   const size_t* poffsets,
     size_t* psub_extents,
    size_t* psub_strides,
    T* sub_data)const
{
    size_t r = dprank;


    fill_strides(psub_extents, psub_strides, r, dprowmajor);


    size_t* indices        = new size_t[r];

    #pragma omp parallel for simd
    for(size_t i=0; i<r; i++)
        indices[i]=0;

    size_t* global_indices = new size_t[r];


    while (true)
    {
        #pragma omp parallel for simd
        for (size_t i = 0; i < r; ++i)
            global_indices[i] = poffsets[i] + indices[i];

        size_t original_index = compute_offset_w(global_indices, dpstrides, r);
        size_t buffer_index   = compute_offset_w(indices, psub_strides, r);

        sub_data[buffer_index] = dpdata[original_index];

        // Increment multi-dimensional indices
        size_t dim = r;
        while (dim-- > 0)
        {
            if (++indices[dim] < psub_extents[dim]) break;
            indices[dim] = 0;
        }
        if (dim == size_t(-1)) break;
    }

    delete[] indices;
    delete[] global_indices;


    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);

    // Return non-owning datastruct: just pass pointers
    return datastruct(sub_data,
                      pl,
                      dprowmajor,
                      r,               // full rank
                      psub_extents,
                      psub_strides,
                      dpdata_is_devptr);
}
#pragma omp end declare target






#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides)const
{
    psub_strides[0]=dpstrides[0];
    psub_strides[1]=dpstrides[1];
    psub_extents[0]=tile_rows;
    psub_extents[1]=tile_cols;
    size_t pl=(psub_extents[0]-1) * psub_strides[0]+ (psub_extents[1]-1) * psub_strides[1]+1;
    size_t offset=+row * dpstrides[0]+col * dpstrides[1];
    return datastruct(dpdata+offset,pl,dprowmajor,2,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const
{

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    T*    pd=dpdata;
    psub_strides[0]=(dprowmajor==true)? tile_cols:1;
    psub_strides[1]=(dprowmajor==true)?1: tile_rows;
    psub_extents[0]=tile_rows;
    psub_extents[1]=tile_cols;
    #pragma omp parallel for simd collapse(2)
    for (size_t i = 0; i < tile_rows; ++i)
    {
        for (size_t j = 0; j < tile_cols; ++j)
        {
            sub_data[i * psub_strides[0] + j*psub_strides[1]] = pd[(row + i)*s0+( col + j)*s1];
        }
    }
    return datastruct(sub_data,tile_rows*tile_cols,dprowmajor,2,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target








#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const
{


    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    T*    pd=dpdata;
    psub_strides[0]=(dprowmajor==true)? tile_cols:1;
    psub_strides[1]=(dprowmajor==true)?1: tile_rows;
    psub_extents[0]=tile_rows;
    psub_extents[1]=tile_cols;

    #pragma omp simd collapse(2)
    for (size_t i = 0; i < tile_rows; ++i)
    {
        for (size_t j = 0; j < tile_cols; ++j)
        {
            sub_data[i * psub_strides[0] + j*psub_strides[1]] = pd[(row + i)*s0+( col + j)*s1];
        }
    }

    return datastruct(sub_data,tile_rows*tile_cols,dprowmajor,2,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const
{

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    T*    pd=dpdata;
    psub_strides[0]=(dprowmajor==true)? tile_cols:1;
    psub_strides[1]=(dprowmajor==true)?1: tile_rows;
    psub_extents[0]=tile_rows;
    psub_extents[1]=tile_cols;

    #pragma omp unroll
    for (size_t i = 0; i < tile_rows; ++i)
    {
        for (size_t j = 0; j < tile_cols; ++j)
        {
            sub_data[i * psub_strides[0] + j*psub_strides[1]] = pd[(row + i)*s0+( col + j)*s1];
        }
    }

    return datastruct(sub_data,tile_rows*tile_cols,dprowmajor,2,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target






#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row(const size_t row_index, size_t*    extents,size_t *    new_strides)const
{
    extents[0] = dpextents[1];
    new_strides[0]=dpstrides[1];

    return datastruct<T>( dpdata + row_index * dpstrides[0],  dpstrides[1] * extents[0],dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row_w(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    size_t pl;
    pl=dpextents[1];
    extents[0] = pl;
    new_strides[0]=1;
    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    #pragma omp parallel for simd
    for (size_t j = 0; j < pl; ++j)
        newdata[j] = pd[row_index*s0+j*s1];


    return datastruct<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row_v(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{
    size_t pl;
    pl=dpextents[1];
    extents[0] = pl;
    new_strides[0]=1;
    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    #pragma omp simd
    for (size_t i = 0; i < pl; ++i)
        newdata[i] = pd[row_index*s0+i*s1];

    return datastruct<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row_s(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{
    size_t pl;
    pl=dpextents[1];
    extents[0] = pl;
    new_strides[0]=1;
    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    #pragma omp unroll
    for (size_t i = 0; i < pl; ++i)
        newdata[i] = pd[row_index*s0+i*s1];

    return datastruct<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column_v(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    size_t pl;
    pl=dpextents[0];
    extents[0] = pl;
    new_strides[0]=1;
    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    #pragma omp simd
    for (size_t i = 0; i < pl; ++i)
        newdata[i] = pd[ i*s0+col_index*s1];

    return datastruct<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column_s(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    size_t pl;
    pl=dpextents[0];
    extents[0] = pl;
    new_strides[0]=1;
    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    #pragma omp unroll
    for (size_t i = 0; i < pl; ++i)
        newdata[i] = pd[ i*s0+col_index*s1];

    return datastruct<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column_w(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    size_t pl;
    pl=dpextents[0];
    extents[0] = pl;
    new_strides[0]=1;
    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    #pragma omp parallel for simd
    for (size_t i = 0; i < pl; ++i)
        newdata[i] = pd[i*s0+col_index*s1];

    return datastruct<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target







#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column(const size_t col_index, size_t*    extents,size_t *   new_strides)const
{
    extents[0] = dpextents[0];
    new_strides[0]=dpstrides[0];
    return datastruct(dpdata + col_index * dpstrides[1], dpstrides[0] * extents[0],dprowmajor,  1, extents,   new_strides,dpdata_is_devptr );
}
#pragma omp end declare target



#endif
