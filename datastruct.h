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
    datastruct() {};

    // Constructors
    datastruct(T*  data, size_t datalength, bool rowm, size_t rank ,size_t*   extents, size_t*   strides,
          bool compute_datalength,    bool compute_strides_from_extents,bool data_is_devptr );

    datastruct(T*  data, size_t datalength,  bool rowm, size_t*  extents,  size_t*   strides,
        bool compute_datalength, bool compute_strides_from_extents, bool data_is_devptr );

    datastruct(T*  data,size_t datalength,bool rowm,size_t rows, size_t cols,  size_t*  extents, size_t*  strides,
        bool compute_datalength, bool compute_strides_from_extents,  bool data_is_devptr);

    datastruct(T*  data,  size_t datalength,  bool rowm,  bool rowvector,  size_t rank,  size_t*  extents,  size_t*  strides,
        bool compute_datalength, bool compute_strides_from_extents, bool data_is_devptr);

    datastruct(T*  data, size_t datalength, bool rowm,  size_t rank, size_t*  extents, size_t*  strides, bool data_is_devptr );

    inline size_t datalength() const { return dpdatalength; }

    inline size_t rank() const{ return dprank;  }

    inline bool rowmajor() const  {  return dprowmajor; }

    inline bool data_is_devptr() const  {  return dpdata_is_devptr;  }

    inline T& data(size_t i)  {  return dpdata[i];  }

    inline const T& data(size_t i) const  {  return dpdata[i];  }

    inline size_t& extent(size_t i) {  return dpextents[i];  }

    inline const size_t& extent(size_t i) const  {  return dpextents[i];  }

    inline size_t& stride(size_t i)  {   return dpstrides[i];  }

    inline const size_t& stride(size_t i) const  {  return dpstrides[i];  }


    inline T* data()  { return dpdata;    }

    inline const T* data() const  {  return dpdata;  }

    inline size_t* extents()  {  return dpextents;  }

    inline const size_t* extents() const  {  return dpextents;  }

    inline size_t* strides()  {  return dpstrides;  }

    inline const size_t* strides() const {   return dpstrides;  }




    // Operator overloads
    inline T& operator()(const size_t*    indices)
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

    inline datastruct<T>substruct_w(const size_t *    poffsets,const size_t *   psub_extents, size_t*    psub_strides);
    inline datastruct<T>substruct_v(const size_t *    poffsets,const size_t *   psub_extents, size_t*    psub_strides);
    inline datastruct<T>substruct_s(const size_t *    poffsets,const size_t *   psub_extents, size_t*    psub_strides);

    inline datastruct<T>substruct_w(const size_t *    poffsets,const size_t *   psub_extents, size_t*   psub_strides, T*    sub_data);
    inline datastruct<T>substruct_v(const size_t *    poffsets,const size_t *   psub_extents, size_t*   psub_strides, T*    sub_data);
    inline datastruct<T>substruct_s(const size_t *    poffsets,const size_t *   psub_extents, size_t*   psub_strides, T*   sub_data);

    inline datastruct<T>subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides);

    inline datastruct<T>subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data);
    inline datastruct<T>subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data);
    inline datastruct<T>subspanmatrix_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data);


    inline datastruct<T> transpose(size_t*    newextents, size_t*    newstrides);

    inline datastruct<T> transpose_v(size_t*    newextents, size_t*    newstrides,T* newdata);
    inline datastruct<T> transpose_w(size_t*    newextents, size_t*    newstrides,T* newdata);
    inline datastruct<T> transpose_s(size_t*    newextents, size_t*    newstrides,T* newdata);


    inline datastruct<T> row(const size_t row_index, size_t*    newextents, size_t*    newstrides);
    inline datastruct<T> column(const size_t col_index, size_t*    newextents, size_t*    newstrides);

    inline datastruct<T> column_w(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata);
    inline datastruct<T> column_v(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata);
    inline datastruct<T> column_s(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata);

    inline datastruct<T> row_w(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata);
    inline datastruct<T> row_v(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata);
    inline datastruct<T> row_s(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata);


    inline bool is_contiguous()const;

    inline void printmatrix()const;
    inline void printvector()const;

protected:
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
template<typename T>inline datastruct<T> datastruct<T>::transpose(size_t*    newextents, size_t *newstrides)
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];
    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];
    return datastruct(dpdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose_w(size_t*    newextents, size_t *newstrides, T* newdata)
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[0];
    newstrides[1]=dpstrides[1];
    T* pd=this->dpdata;
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i<dpextents[0]; i++)
        for (size_t j=0; j<dpextents[1]; j++)
            newdata[compute_offset(j, i, newstrides[0], newstrides[1])] = pd[compute_offset(i, j, newstrides[0], newstrides[1])];

    return datastruct(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose_v(size_t*    newextents, size_t *newstrides, T* newdata)
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[0];
    newstrides[1]=dpstrides[1];
    T* pd=this->dpdata;

    #pragma omp simd collapse(2)
    for (size_t i=0; i<dpextents[0]; i++)
        for (size_t j=0; j<dpextents[1]; j++)
        {
            newdata[compute_offset(j, i, newstrides[0], newstrides[1])] = pd[compute_offset(i, j, newstrides[0], newstrides[1])];
        }

    return datastruct(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose_s(size_t*    newextents, size_t *newstrides, T* newdata)
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[0];
    newstrides[1]=dpstrides[1];
    T* pd=this->dpdata;
    for (size_t i=0; i<dpextents[0]; i++)
        for (size_t j=0; j<dpextents[1]; j++)
            newdata[compute_offset(j, i, newstrides[0], newstrides[1])] = pd[compute_offset(i, j, newstrides[0], newstrides[1])];

    return datastruct(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
bool datastruct<T>::is_contiguous() const
{
    if (dprank == 0) {
        return dpdatalength == 1;
    }

    size_t expected_stride = 1;
    if (dprowmajor)
    {
        for (int i = (int)dprank - 1; i >= 0; --i)
        {
            if (dpstrides[i] != expected_stride)
            {
                return false;
            }
            expected_stride *= dpextents[i];
        }
    }
    else
    {
        for (size_t i = 0; i < dprank; ++i)
        {
            if (dpstrides[i] != expected_stride)
            {
                return false;
            }
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
void datastruct<T>::printmatrix()const
{
    const size_t rows= this->dpextents[0];
    const size_t cols=this->dpextents[1];
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            printf("%f ",(*this)(i, j));
        }
        printf("%s \n","");
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void datastruct<T>::printvector()const
{
    const size_t rows= this->dpextents[0];
    for (size_t i = 0; i <rows; ++i)
    {
        printf("%f\n",(*this)(i));
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
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr

) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
    #if !defined(Unified_Shared_Memory)
    dpdata_is_devptr(data_is_devptr)
    #else
    dpdata_is_devptr(false)
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
    #if !defined(Unified_Shared_Memory)
    dpdata_is_devptr(data_is_devptr)
    #else
    dpdata_is_devptr(false)
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
    #if !defined(Unified_Shared_Memory)
    dpdata_is_devptr(data_is_devptr)
    #else
    dpdata_is_devptr(false)
    #endif
{
    if(extents!=nullptr)
    {
        dpextents[0]=(rowm==true)?rows:cols;
        dpextents[1]=(rowm==true)?cols:rows;
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
template<typename T> datastruct<T>::datastruct(
    T*    data,
    size_t datalength,
    bool rowm,
    bool rowvector,
    size_t noelements,
    size_t*    extents,
    size_t*    strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(1),
    dprowmajor(true),
    #if !defined(Unified_Shared_Memory)
    dpdata_is_devptr(data_is_devptr)
    #else
    dpdata_is_devptr(false)
    #endif
{
    if(extents!=nullptr)
    {
        dpextents[0]=noelements;
    }
    if(dpstrides!=nullptr && compute_strides_from_extents)
    {
        if(rowvector)
            dpstrides[0]=(rowm==true)? 1:noelements;
        else
            dpstrides[0]=(rowm==true)? noelements:1;
    }
    if(compute_datalength==true && strides!=nullptr)
    {
        dpdatalength=(noelements-1) * strides[0]+1;
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_w(const size_t *    poffsets,const size_t *   psub_extents, size_t*    psub_strides)
{
    size_t offset_index = 0;
    const size_t r=dprank;
    #pragma omp parallel for simd reduction( + : offset_index )
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += poffsets[i] * dpstrides[i];
        psub_strides[i]=dpstrides[i];
    }
    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
    return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr);

}
#pragma omp end  declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_v(const size_t *    poffsets,const size_t *   psub_extents, size_t*    psub_strides)
{
    size_t offset_index = 0;
    const size_t r=dprank;
    #pragma omp simd reduction( + : offset_index )
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += poffsets[i] * dpstrides[i];
        psub_strides[i]=dpstrides[i];
    }
    size_t pl=compute_data_length_v(psub_extents,psub_strides,r);
    return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr);

}
#pragma omp end  declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_s(const size_t *    poffsets,const size_t *   psub_extents, size_t*    psub_strides)
{
    size_t offset_index = 0;
    const size_t r=dprank;
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += poffsets[i] * dpstrides[i];
        psub_strides[i]=dpstrides[i];
    }
    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
    return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr);

}
#pragma omp end  declare target





#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_s(const size_t *    poffsets,const size_t *   psub_extents, size_t*   psub_strides, T*   sub_data)
{
    size_t offset_index = 0;
    const size_t r=dprank;
    if(sub_data==nullptr)
    {
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += poffsets[i] * dpstrides[i];
            psub_strides[i]=dpstrides[i];
        }
        size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
        return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr);
    }
    else
    {
        // Compute the new strides for the subspan
        size_t *    indices;
        size_t *   global_indices;

        indices=new size_t[r];
        global_indices= new size_t[r];


        for (size_t i=0; i<r; i++)
        {
            indices[i]=0;
        }

        size_t largest_buffer_index=0;
        // Fill the supplied buffer with subspan data
        fill_strides(psub_extents,psub_strides,r,dprowmajor);
        while (true)
        {
            // Compute the current global indices
            for (size_t i = 0; i < r; ++i)
            {
                global_indices[i] = poffsets[i] + indices[i];
            }

            // Compute the offsets for the original data and the new buffer
            size_t original_index = compute_offset_s(global_indices, dpstrides, dprowmajor);
            size_t buffer_index = compute_offset_s(indices,psub_strides, dprowmajor);

            // Copy the data from the original tensor to the sub-buffer
            sub_data[buffer_index] = dpdata[original_index];

            if(buffer_index>largest_buffer_index)
                largest_buffer_index=buffer_index;

            // Increment the indices for the Cartesian product
            size_t dim = r;
            while (dim-- > 0)
            {
                if (++indices[dim] < psub_extents[dim])
                {
                    break; // If no overflow, stop carrying
                }
                indices[dim] = 0; // Reset the current dimension and carry to the next
            }

            // If all dimensions have overflowed, we're done
            if (dim == size_t(-1))
            {
                break;
            }

        }
        size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
        // Create and return a new mdspan with the updated pointer, extents, and strides
        delete[] global_indices;
        delete[] indices;

        return datastruct (sub_data,pl,dprowmajor,psub_extents, psub_strides,dpdata_is_devptr);

    }

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_v(const size_t *    poffsets,const size_t *   psub_extents, size_t*   psub_strides, T*    sub_data)
{
    // Compute the new strides for the subspan
    size_t *    indices;
    size_t *   global_indices;
    const size_t r=dprank;
    indices=new size_t[r];
    global_indices= new size_t[r];

    #pragma omp simd
    for (size_t i=0; i<r; i++)
    {
        indices[i]=0;
    }

    size_t largest_buffer_index=0;
    // Fill the supplied buffer with subspan data
    fill_strides(psub_extents,psub_strides,r,dprowmajor);
    while (true)
    {
        // Compute the current global indices
        #pragma omp simd
        for (size_t i = 0; i < r; ++i)
        {
            global_indices[i] = poffsets[i] + indices[i];
        }

        // Compute the offsets for the original data and the new buffer
        size_t original_index = compute_offset_v(global_indices, dpstrides, dprowmajor);
        size_t buffer_index = compute_offset_v(indices,psub_strides, dprowmajor);

        // Copy the data from the original tensor to the sub-buffer
        sub_data[buffer_index] = dpdata[original_index];

        if(buffer_index>largest_buffer_index)
            largest_buffer_index=buffer_index;

        // Increment the indices for the Cartesian product
        size_t dim = r;
        while (dim-- > 0)
        {
            if (++indices[dim] < psub_extents[dim])
            {
                break; // If no overflow, stop carrying
            }
            indices[dim] = 0; // Reset the current dimension and carry to the next
        }

        // If all dimensions have overflowed, we're done
        if (dim == size_t(-1))
        {
            break;
        }

    }
    // Create and return a new mdspan with the updated pointer, extents, and strides
    delete[] global_indices;
    delete[] indices;

    size_t pl=compute_data_length_v(psub_extents,psub_strides,r);
    return datastruct (sub_data,pl,dprowmajor,psub_extents, psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target







#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_w(const size_t *    poffsets,const size_t *   psub_extents, size_t*   psub_strides, T*    sub_data)
{
    // Compute the new strides for the subspan
    size_t *    indices;
    size_t *   global_indices;
    const size_t r=dprank;
    indices=new size_t[r];
    global_indices= new size_t[r];

    #pragma omp parallel for simd
    for (size_t i=0; i<r; i++)
    {
        indices[i]=0;
    }

    size_t largest_buffer_index=0;
    // Fill the supplied buffer with subspan data
    fill_strides(psub_extents,psub_strides,r,dprowmajor);
    while (true)
    {
        // Compute the current global indices
        #pragma omp parallel for simd
        for (size_t i = 0; i < r; ++i)
        {
            global_indices[i] = poffsets[i] + indices[i];
        }

        // Compute the offsets for the original data and the new buffer
        size_t original_index = compute_offset_w(global_indices, dpstrides, dprowmajor);
        size_t buffer_index = compute_offset_w(indices,psub_strides, dprowmajor);

        // Copy the data from the original tensor to the sub-buffer
        sub_data[buffer_index] = dpdata[original_index];

        if(buffer_index>largest_buffer_index)
            largest_buffer_index=buffer_index;

        // Increment the indices for the Cartesian product
        size_t dim = r;
        while (dim-- > 0)
        {
            if (++indices[dim] < psub_extents[dim])
            {
                break; // If no overflow, stop carrying
            }
            indices[dim] = 0; // Reset the current dimension and carry to the next
        }

        // If all dimensions have overflowed, we're done
        if (dim == size_t(-1))
        {
            break;
        }

    }
    // Create and return a new mdspan with the updated pointer, extents, and strides
    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);

    delete[] global_indices;
    delete[] indices;

    return datastruct (sub_data,pl,dprowmajor,psub_extents, psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides)
{
    psub_strides[0]=dpstrides[0];
    psub_strides[1]=dpstrides[1];
    psub_extents[0]=(dprowmajor==true)?tile_rows:tile_cols;
    psub_extents[1]=(dprowmajor==true)?tile_cols:tile_rows;
    size_t pl=(psub_extents[0]-1) * psub_strides[0]+ (psub_extents[1]-1) * psub_strides[1]+1;
    size_t offset=+row * dpstrides[0]+col * dpstrides[1];
    return datastruct(dpdata+offset,pl,dprowmajor,2,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)
{
    if (dprowmajor)
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp parallel for simd collapse(2) shared(sub_data,pd,tile_rows,tile_cols,row,col,s0,s1)
        for (size_t i = 0; i < tile_rows; ++i)
        {
            for (size_t j = 0; j < tile_cols; ++j)
            {
                sub_data[i * tile_cols + j] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    else
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp parallel for simd collapse(2) shared(sub_data,pd,tile_rows,tile_cols,row,col,s0,s1)
        for (size_t j = 0; j < tile_cols; ++j)
        {
            for (size_t i = 0; i < tile_rows; ++i)
            {
                sub_data[j * tile_rows + i] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    fill_strides(psub_extents,psub_strides,2,dprowmajor);
    size_t pl=(tile_rows-1) * psub_strides[0]+(tile_cols-1)*psub_strides[1]+1;
    return datastruct(sub_data,pl,dprowmajor,tile_rows, tile_cols,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target








#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)
{
    if (dprowmajor)
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp simd collapse(2)
        for (size_t i = 0; i < tile_rows; ++i)
        {
            for (size_t j = 0; j < tile_cols; ++j)
            {
                sub_data[i * tile_cols + j] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    else
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp simd collapse(2)
        for (size_t j = 0; j < tile_cols; ++j)
        {
            for (size_t i = 0; i < tile_rows; ++i)
            {
                sub_data[j * tile_rows + i] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    fill_strides(psub_extents,psub_strides,2,dprowmajor);
    size_t pl=(tile_rows-1) * psub_strides[0]+(tile_cols-1)*psub_strides[1]+1;
    return datastruct(sub_data,pl,dprowmajor,tile_rows, tile_cols,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)
{
    if (dprowmajor)
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        // Row-major layout: fill row by row

        for (size_t i = 0; i < tile_rows; ++i)
        {
            for (size_t j = 0; j < tile_cols; ++j)
            {
                sub_data[i * tile_cols + j] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    else
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        // Column-major layout: fill column by column

        for (size_t j = 0; j < tile_cols; ++j)
        {
            for (size_t i = 0; i < tile_rows; ++i)
            {
                sub_data[j * tile_rows + i] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    fill_strides(psub_extents,psub_strides,2,dprowmajor);
    size_t pl=(tile_rows-1) * psub_strides[0]+(tile_cols-1)*psub_strides[1]+1;
    return datastruct(sub_data,pl,dprowmajor,tile_rows, tile_cols,psub_extents,psub_strides,dpdata_is_devptr);
}
#pragma omp end declare target






#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row(const size_t row_index, size_t*    extents,size_t *    new_strides)
{
    T*    row_data;
    size_t pl;

    if (dprowmajor)
    {
        row_data = dpdata + row_index * dpstrides[0];
        extents[0] = dpextents[1];
        new_strides[0]=dpstrides[0];
        pl=dpstrides[1] * extents[0];
    }
    else
    {
        row_data = dpdata + row_index * dpstrides[1];
        extents[0] = dpextents[0];
        new_strides[0]=dpstrides[0];
        pl=dpstrides[0] * extents[0];
    }
    return datastruct<T>(row_data,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row_w(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)
{
    T*    row_data;
    size_t pl;
    if (dprowmajor)
    {
        pl=dpextents[1];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp parallel for simd
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[compute_offset(row_index, j, s0, s1)];
    }
    else
    {
        pl=dpextents[0];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp parallel for simd
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[compute_offset(j,row_index, s0, s1)];
    }

    datastruct<T>m(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row_v(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)
{
    T*    row_data;
    size_t pl;
    if (dprowmajor)
    {
        pl=dpextents[1];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp simd
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(row_index, i, s0, s1)];
    }
    else
    {
        pl=dpextents[0];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp simd
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(i,row_index, s0, s1)];
    }

    datastruct<T>m(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row_s(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)
{
    T*    row_data;
    size_t pl;
    if (dprowmajor)
    {
        pl=dpextents[1];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(row_index, i, s0, s1)];
    }
    else
    {
        pl=dpextents[0];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(i,row_index, s0, s1)];
    }

    datastruct<T>m(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column_v(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)
{
    T*    row_data;
    size_t pl;
    if (!dprowmajor)
    {
        pl=dpextents[1];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp simd
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(col_index, i, s0, s1)];
    }
    else
    {
        pl=dpextents[0];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp simd
        for (size_t i= 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(i,col_index, s0, s1)];
    }

    datastruct<T>m(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column_s(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)
{
    T*    row_data;
    size_t pl;
    if (!dprowmajor)
    {
        pl=dpextents[1];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(col_index, i, s0, s1)];
    }
    else
    {
        pl=dpextents[0];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        for (size_t i= 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(i,col_index,  s0, s1)];
    }

    datastruct<T>m(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column_w(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)
{
    T*    row_data;
    size_t pl;
    if (!dprowmajor)
    {
        pl=dpextents[1];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp parallel for simd
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(col_index, i, s0, s1)];
    }
    else
    {
        pl=dpextents[0];
        extents[0] = pl;
        new_strides[0]=1;
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T*    pd=dpdata;
        #pragma omp parallel for simd
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[compute_offset(i,col_index, s0, s1)];
    }

    datastruct<T>m(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target







#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column(const size_t col_index, size_t*    extents,size_t *   new_strides)
{
    T*    col_data;
    size_t pl;
    if (dprowmajor)
    {
        col_data = dpdata + col_index * dpstrides[1];
        extents[0] = dpextents[0];
        new_strides[0]=dpstrides[0];
        pl=dpstrides[0] * extents[0];
    }
    else
    {
        col_data = dpdata + col_index * dpstrides[0];
        extents[0] = dpextents[1];
        new_strides[0]=dpstrides[0];
        pl=dpstrides[1] * extents[0];
    }
    return datastruct(col_data, pl,dprowmajor,  1, extents,   new_strides,dpdata_is_devptr );
}
#pragma omp end declare target



#endif
