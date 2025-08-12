#ifndef MDSPANH
#define MDSPANH

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>


#include <iostream>
#include <array>
#include <vector>
#include <map>
#include <numeric>
#include <cmath>
#include <numbers>
#include <memory>


#include <omp.h>
#include <iostream>
#include <thread>
#include <mpi.h>


#if (Unified_Shared_Memory==True)
#pragma omp requires unified_shared_memory
#endif


enum Matrix_Multiplication_Algorithm
{
    Naive=0,
    Strassen=1,
    WinogradVariant=2
};

enum MessageType
{
    COMMAND_STRASSEN,
    COMMAND_WINOGRAD,
    COMMAND_SENDMATRIX,
    COMMAND_END_LISTENER
};

struct matrix_multiplication_parameters
{
    size_t algorithm_version{Matrix_Multiplication_Algorithm::Naive};
    size_t size_for_naive_algorithm=2;
    bool memmapped_files=true;
    bool gpu_offload=false;
    bool default_device=true;
    int devicenum=0;
    bool omp=true;
    bool mpi=false;
    MPI_Comm comm=MPI_COMM_WORLD;
    bool size_for_mpi=2;
};

#pragma omp begin declare target
inline size_t compute_offset(const size_t row, const size_t col, const size_t* __restrict  strides)
{
    return row * strides[0]+col*strides[1];
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_offset(const size_t row, const size_t col, const size_t matrixstride0, const size_t matrixstride1)
{
    return row * matrixstride0+col*matrixstride1;
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_offset_w(const size_t * __restrict indices, const size_t* __restrict strides,const size_t r, bool rowmajor=true)
{
    size_t offset = 0;

    if (rowmajor)
    {
        // Row-major layout: iterate outermost to innermost
        #pragma omp parallel for simd reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        // Column-major layout: iterate innermost to outermost
        #pragma omp parallel for simd reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[r - 1 - i] * strides[r - 1 - i];
        }
    }

    return offset;
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_offset_s(const size_t * __restrict indices, const size_t* __restrict strides,const size_t r, bool rowmajor=true)
{
    size_t offset = 0;

    if (rowmajor)
    {
        // Row-major layout: iterate outermost to innermost
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        // Column-major layout: iterate innermost to outermost
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[r - 1 - i] * strides[r - 1 - i];
        }
    }

    return offset;
}
#pragma omp end declare target




#pragma omp begin declare target
inline size_t compute_offset_v(const size_t * __restrict indices, const size_t* __restrict strides,const size_t r, bool rowmajor=true)
{
    size_t offset = 0;

    if (rowmajor)
    {
        // Row-major layout: iterate outermost to innermost
        #pragma omp simd reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        // Column-major layout: iterate innermost to outermost
        #pragma omp simd   reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[r - 1 - i] * strides[r - 1 - i];
        }
    }

    return offset;
}
#pragma omp end declare target



#pragma omp begin declare target
inline size_t compute_data_length_w(const size_t*__restrict  extents, const size_t*__restrict  strides,const size_t rank)
{
    size_t offset=0;
    #pragma omp parallel for simd reduction(+:offset)
    for (size_t i = 0; i < rank; ++i)
    {
        offset += (extents[i]-1) * strides[i];
    }
    return offset+1;
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_data_length_v(const size_t*__restrict  extents, const size_t*__restrict  strides,const size_t rank)
{
    size_t offset=0;
    #pragma omp simd reduction(+:offset)
    for (size_t i = 0; i < rank; ++i)
    {
        offset += (extents[i]-1) * strides[i];
    }
    return offset+1;
}
#pragma omp end declare target




#pragma omp begin declare target
inline size_t compute_data_length_s(const size_t*__restrict  extents, const size_t*__restrict  strides,const size_t rank)
{
    size_t offset=0;
    for (size_t i = 0; i < rank; ++i)
    {
        offset += (extents[i]-1) * strides[i];
    }
    return offset+1;
}
#pragma omp end declare target







#pragma omp begin declare target
template <typename T>
struct datastruct
{
    T*   dpdata = nullptr;
    size_t*  dpextents = nullptr;
    size_t*   dpstrides = nullptr;
    size_t dpdatalength = 0;
    size_t dprank = 0;
    bool dprowmajor = true;
    bool dpdata_is_devptr=false;
    bool dpstrides_is_devptr=false;
    bool dpextents_is_devptr=false;

    datastruct() {}

    // Constructors
    datastruct(
        T* __restrict data,
        size_t pdatalength,
        bool rowm,
        size_t rank,
        size_t*  __restrict extents,
        size_t* __restrict  strides,
        bool compute_datalength,
        bool compute_strides_from_extents,
        bool data_is_devptr,
        bool strides_is_devptr,
        bool extents_is_devptr
    );

    datastruct(
        T* __restrict data,
        size_t pdatalength,
        bool rowm,
        size_t* __restrict extents,
        size_t* __restrict  strides,
        bool compute_datalength,
        bool compute_strides_from_extents,
        bool data_is_devptr,
        bool strides_is_devptr,
        bool extents_is_devptr
    );

    datastruct(
        T* __restrict data,
        size_t datalength,
        bool rowm,
        size_t rows,
        size_t cols,
        size_t* __restrict extents,
        size_t* __restrict strides,
        bool compute_datalength,
        bool compute_strides_from_extents,
        bool data_is_devptr,
        bool strides_is_devptr,
        bool extents_is_devptr
    );

    datastruct(
        T* __restrict data,
        size_t datalength,
        bool rowm,
        bool rowvector,
        size_t length,
        size_t* __restrict extents,
        size_t* __restrict strides,
        bool compute_datalength,
        bool compute_strides_from_extents,
        bool data_is_devptr,
        bool strides_is_devptr,
        bool extents_is_devptr
    );

    datastruct(
        T* __restrict data,
        size_t datalength,
        bool rowm,
        size_t rank,
        size_t* __restrict extents,
        size_t* __restrict strides,
        bool data_is_devptr,
        bool strides_is_devptr,
        bool extents_is_devptr
    );

    // Operator overloads
    T& operator()(const size_t* __restrict indices)
    {
        return dpdata[compute_offset_s(indices, dpstrides, dprank)];
    };

    T operator()(const size_t* __restrict indices) const
    {
        return dpdata[compute_offset_s(indices, dpstrides, dprank)];
    };

    T& operator()(const size_t row, const size_t col)
    {
        const size_t strides0=dpstrides[0];
        const size_t strides1=dpstrides[1];
        return dpdata[row * strides0 + col *strides1];
    };

    T operator()(const size_t row, const size_t col) const
    {
        const size_t strides0=dpstrides[0];
        const size_t strides1=dpstrides[1];
        return dpdata[row * strides0 + col *strides1];
    };

    T& operator()(const size_t i)
    {
        const size_t stride=dpstrides[0];
        return dpdata[i * stride];
    };

    T operator()(const size_t i) const
    {
        const size_t stride=dpstrides[0];
        return dpdata[i * stride];
    };


    datastruct<T>substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides);
    datastruct<T>substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides);
    datastruct<T>substruct_s(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides);
    datastruct<T>substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>substruct_s(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T*__restrict sub_data);

    datastruct<T>subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides);
    datastruct<T>subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>subspanmatrix_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data);


    datastruct<T> transpose(size_t* __restrict newextents, size_t* __restrict newstrides);
    datastruct<T> row(const size_t row_index, size_t* __restrict newextents, size_t* __restrict newstrides);
    datastruct<T> column(const size_t col_index, size_t* __restrict newextents, size_t* __restrict newstrides);
};
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose(size_t* __restrict newextents, size_t *newstrides)
{
    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];
    return datastruct(dpdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);

}
#pragma omp end declare target

#pragma omp begin declare target
inline void fill_strides(const size_t* __restrict extents,size_t* __restrict strides, const size_t rank, const bool rowmajor)
{
    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[rank - 1] = 1;
        #pragma omp ordered
        for (int i = rank - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
        #pragma omp ordered
        for (size_t i = 1; i < rank; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];

        }
    }
}

#pragma omp end declare target

#pragma omp begin declare target
template<typename T> datastruct<T>::datastruct(
    T* __restrict data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t* __restrict extents,
    size_t* __restrict strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr,
    bool strides_is_devptr,
    bool extents_is_devptr
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
    dpdata_is_devptr(data_is_devptr),
    dpstrides_is_devptr(strides_is_devptr),
    dpextents_is_devptr(extents_is_devptr)
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
    T* __restrict data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t* __restrict extents,
    size_t* __restrict strides,
    bool data_is_devptr,
    bool strides_is_devptr,
    bool extents_is_devptr
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
    dpdata_is_devptr(data_is_devptr),
    dpstrides_is_devptr(strides_is_devptr),
    dpextents_is_devptr(extents_is_devptr)
{}
#pragma omp end declare target





#pragma omp begin declare target
template<typename T> datastruct<T>::datastruct(
    T* __restrict data,
    size_t datalength,
    bool rowm,
    size_t rows,
    size_t cols,
    size_t* __restrict extents,
    size_t* __restrict strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr,
    bool strides_is_devptr,
    bool extents_is_devptr
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(2),
    dprowmajor(rowm),
    dpdata_is_devptr(data_is_devptr),
    dpstrides_is_devptr(strides_is_devptr),
    dpextents_is_devptr(extents_is_devptr)
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
    T* __restrict data,
    size_t datalength,
    bool rowm,
    bool rowvector,
    size_t noelements,
    size_t* __restrict extents,
    size_t* __restrict strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr,
    bool strides_is_devptr,
    bool extents_is_devptr
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(1),
    dprowmajor(true),
    dpdata_is_devptr(data_is_devptr),
    dpstrides_is_devptr(strides_is_devptr),
    dpextents_is_devptr(extents_is_devptr)
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
datastruct<T>datastruct<T>::substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides)
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
    return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);

}
#pragma omp end  declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides)
{
    size_t offset_index = 0;
    const size_t r=dprank;
    #pragma omp simd reduction( + : offset_index )
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += poffsets[i] * dpstrides[i];
        psub_strides[i]=dpstrides[i];
    }
    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
    return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);

}
#pragma omp end  declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_s(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides)
{
    size_t offset_index = 0;
    const size_t r=dprank;
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += poffsets[i] * dpstrides[i];
        psub_strides[i]=dpstrides[i];
    }
    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
    return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);

}
#pragma omp end  declare target





#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_s(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T*__restrict sub_data)
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
        return datastruct(dpdata + offset_index,pl,dprowmajor,r, psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
    }
    else
    {
        // Compute the new strides for the subspan
        size_t * __restrict indices;
        size_t *__restrict global_indices;

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

        return datastruct (sub_data,pl,dprowmajor,psub_extents, psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);

    }

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data)
{
    // Compute the new strides for the subspan
    size_t * __restrict indices;
    size_t *__restrict global_indices;
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
    delete[] global_indices;
    delete[] indices;

    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
    return datastruct (sub_data,pl,dprowmajor,psub_extents, psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data)
{
    // Compute the new strides for the subspan
    size_t * __restrict indices;
    size_t *__restrict global_indices;
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

    return datastruct (sub_data,pl,dprowmajor,psub_extents, psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides)
{
    psub_strides[0]=dpstrides[0];
    psub_strides[1]=dpstrides[1];
    psub_extents[0]=(dprowmajor==true)?tile_rows:tile_cols;
    psub_extents[1]=(dprowmajor==true)?tile_cols:tile_rows;
    size_t pl=(psub_extents[0]-1) * psub_strides[0]+ (psub_extents[1]-1) * psub_strides[1]+1;
    size_t offset=+row * dpstrides[0]+col * dpstrides[1];
    return datastruct(dpdata+offset,pl,dprowmajor,2,psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
}
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data)
{
    if (dprowmajor)
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T* __restrict pd=dpdata;
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
        const T* __restrict pd=dpdata;
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
    return datastruct(sub_data,pl,dprowmajor,tile_rows, tile_cols,psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data)
{
    if (dprowmajor)
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T* __restrict pd=dpdata;
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
        const T* __restrict pd=dpdata;
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
    return datastruct(sub_data,pl,dprowmajor,tile_rows, tile_cols,psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data)
{
    if (dprowmajor)
    {
        const size_t s0=dpstrides[0];
        const size_t s1=dpstrides[1];
        const T* __restrict pd=dpdata;
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
        const T* __restrict pd=dpdata;
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
    return datastruct(sub_data,pl,dprowmajor,tile_rows, tile_cols,psub_extents,psub_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
}
#pragma omp end declare target



template<typename T>
inline void update_device(datastruct<T>& dL,int devicenum)
{
#if (Unified_Shared_Memory==false)
    size_t l=dL.dpdatalength;
    size_t r=dL.dprank;

    #pragma omp target update to (dL) device(devicenum)
    if(!dL.dpdata_is_devptr)
    {
        #pragma omp target update to (dL.dpdata[0:l])device(devicenum)
    }
    if(!dL.dpextents_is_devptr)
    {
        #pragma omp target update to (dL.dpextents[0:r])device(devicenum)
    }
    if(!dL.dpstrides_is_devptr)
    {
        #pragma omp target update to (dL.dpstrides[0:r])device(devicenum)
    }
#endif
}

template<typename T>
inline void update_host(datastruct<T>& dL,int devicenum)
{
#if (Unified_Shared_Memory==false)
    size_t l=dL.dpdatalength;
    size_t r=dL.dprank;
    #pragma omp target update from (dL) device(devicenum)
    if(!dL.dpdata_is_devptr)
    {
        #pragma omp target update from (dL.dpdata[0:l])device(devicenum)
    }
    if(!dL.dpextents_is_devptr)
    {
        #pragma omp target update from (dL.dpextents[0:r])device(devicenum)
    }
    if(!dL.dpstrides_is_devptr)
    {
        #pragma omp target update from (dL.dpstrides[0:r])device(devicenum)
    }
#endif
}

template<typename T>
void inline create_out_struct(datastruct<T>& dA,int devicenum)
{
#if (Unified_Shared_Memory==false)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    #pragma omp target enter data map(to: dA) device(devicenum)
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target enter data map(alloc: dA.dpdata[0:l])device(devicenum)
    }
    if(!dA.dpextents_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)
    }
    if(!dA.dpstrides_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)
    }

#endif
}



template<typename T>
void inline create_in_struct(datastruct<T>& dA,int devicenum)
{
#if (Unified_Shared_Memory==false)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    #pragma omp target enter data map(to: dA)device(devicenum)
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpdata[0:l])device(devicenum)
    }
    if(!dA.dpstrides_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)
    }
    if(!dA.dpextents_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)
    }
#endif
}
template<typename T>
inline void exit_struct(datastruct<T> &dA,int devicenum)
{
#if (Unified_Shared_Memory==false)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target exit data map(delete:dA.dpdata[0:l])device(devicenum)
    }
    if(!dA.dpstrides_is_devptr)
    {
        #pragma omp target exit data map(delete: dA.dpstrides[0:r])device(devicenum)
    }

    if(!dA.dpextents_is_devptr)
    {
        #pragma omp target exit data map(delete:dA.dpextents[0:r])device(devicenum)
    }
    #pragma omp target exit data map(delete:dA)device(devicenum)

#endif
}


template<typename T>
inline void release_struct(datastruct<T> &dA,int devicenum)
{
#if (Unified_Shared_Memory==false)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target exit data map(release:dA.dpdata[0:l])device(devicenum)
    }
    if(!dA.dpstrides_is_devptr)
    {
        #pragma omp target exit data map(release: dA.dpstrides[0:r])device(devicenum)
    }

    if(!dA.dpextents_is_devptr)
    {
        #pragma omp target exit data map(release:dA.dpextents[0:r])device(devicenum)
    }
    #pragma omp target exit data map(release:dA)device(devicenum)

#endif
}





#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row(const size_t row_index, size_t* __restrict extents,size_t *__restrict  new_strides)
{

    // Offset the data pointer to the start of the row
    T* __restrict row_data = dpdata + row_index * dpstrides[0];
    // Fill the extents array with the appropriate values for the row
    extents[0] = dpextents[1]; // Extent for a row is the number of columns
    new_strides[0]=dpstrides[0];
    return datastruct<T>(row_data,  dpstrides[1] * extents[0],dprowmajor,   1, &extents[0],    new_strides, dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr);
}
#pragma omp end declare target





#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column(const size_t col_index, size_t*__restrict  extents,size_t *__restrict new_strides)
{
    // Offset the data pointer to the start of the column
    T* __restrict col_data = dpdata + col_index * dpstrides[1];
    // Fill the extents array with the appropriate values for the column
    extents[0] = dpextents[0]; // Extent for a column is the number of rows
    new_strides[0]=dpstrides[0];
    return datastruct(col_data, dpstrides[0] * extents[0],dprowmajor,  1, &extents[0],   new_strides,dpdata_is_devptr,dpstrides_is_devptr,dpextents_is_devptr );
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void printmatrix(const datastruct<T>&span)
{
    const size_t rows= span.dpextents[0];
    const size_t cols=span.dpextents[1];
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            printf("%f ",span(i, j));
        }
        printf("%s \n","");
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void printvector(const datastruct<T>&span)
{
    const size_t rows= span.pextents[0];
    for (size_t i = 0; i <rows; ++i)
    {
        printf("%f\n",span(i));
    }

}
#pragma omp end declare target



template<typename T>
T* create_temp_mmap(const size_t array_size)
{
    size_t file_size = array_size * sizeof(T);

    // Create a temporary file using std::tmpfile()
    FILE* tmpf = tmpfile();
    if (!tmpf)
    {
        perror("tmpfile");
        return NULL;
    }

    // Get the file descriptor from the FILE*
    int fd = fileno(tmpf);
    if (fd == -1)
    {
        perror("fileno");
        fclose(tmpf);
        return NULL;
    }

    // Resize the file to the required size
    if (ftruncate(fd, file_size) == -1)
    {
        perror("ftruncate");
        fclose(tmpf);
        return NULL;
    }

    // Memory map the file
    T* mmap_ptr = (T*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED)
    {
        perror("mmap");
        fclose(tmpf);
        return NULL;
    }

    // Close the FILE* but keep the memory mapping valid
    fclose(tmpf);

    // Return the pointer to the mapped memory
    return mmap_ptr;
}

template<typename T>
void delete_temp_mmap(double* mmap_ptr,const size_t array_size)
{
    size_t file_size = array_size * sizeof(T);
    if (munmap(mmap_ptr, file_size) == -1)
    {
        perror("munmap");
    }
}

using namespace std;


// Concept definitions
template <typename Container>
concept StaticContainer =
    requires(Container c, size_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<size_t>;
    {
        c[i]
    }
    -> std::convertible_to<typename Container::value_type>;
    (!requires(Container c, size_t i)
    {
        c.reserve(i);
    });
};

template <typename Container>
concept DynamicContainer =
    requires(Container c, size_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<size_t>;
    {
        c[i]
    }
    -> std::convertible_to<typename Container::value_type>;
    c.reserve(i);  // Require reserve() for dynamic containers
};




// Concept to check if two containers are of the same type and have matching size
template <typename ExtentsContainer>
concept Container =
    (StaticContainer<ExtentsContainer>   ||  // Same size for static containers
     (DynamicContainer<ExtentsContainer>));  // Same size for dynamic containers
// Class template for mdspan



template <typename T, typename Container>
class mdspan
{
public:
    // Constructors
    // Simplified constructors
    mdspan(T* __restrict data, const size_t datalength,const bool rowm, const Container& extents, const Container& strides);
    mdspan(T* __restrict data, const bool rowm, const Container& extents, const Container& strides);
    mdspan(T* __restrict data, const bool rowm, const Container& extents);
    mdspan(T* __restrict data, const bool rowm,const size_t rows,const size_t cols);
    mdspan(const size_t datalength, const bool rowm, const bool memmap, const Container& extents, const Container& strides);
    mdspan(const bool rowm,const  bool memmap, const Container& extents, const Container& strides);
    mdspan(const bool rowm,const  bool memmap, const Container& extents);
    mdspan(const bool rowm, const bool memmap, const size_t rows, const size_t cols);

    mdspan(const bool memmap,  int source, int tag,  MPI_Comm pcomm);

    // Access operators
    inline T& operator()(const Container& extents);
    inline T operator()(const Container& extents)const;

    inline T& operator()(const size_t row, const size_t col)
    {
        const size_t stride0=pstrides[0];
        const size_t stride1=pstrides[1];
        return pdatastruct.dpdata[row * stride0 + col * stride1];
    };
    inline T operator()(const size_t row, const size_t col)const
    {
        const size_t stride0=pstrides[0];
        const size_t stride1=pstrides[1];
        return pdatastruct.dpdata[row * stride0 + col * stride1];
    };

    inline T operator()(const size_t i)const
    {
        const size_t stride=pstrides[0];
        return pdatastruct.dpdata[i * stride];
    };
    inline T& operator()(const size_t i)
    {
        const size_t stride=pstrides[0];
        return pdatastruct.dpdata[i * stride];
    };
    // Subspan methods
    mdspan<T, Container> subspan(const Container& offsets, const Container& sub_extents, T* __restrict sub_data=nullptr) const;


    mdspan<T, Container> subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*sub_data=nullptr)const;
    mdspan<T, Container> row(const size_t row_index);
    mdspan<T, Container> column(const size_t col_index);
    mdspan<T, Container> transpose(T* __restrict sub_data=nullptr);

    bool device_update(bool default_device=true, int devicenum=0);
    bool device_update_async(bool default_device=true, int devicenum=0);
    bool host_update(bool default_device=true, int devicenum=0);
    bool host_update_async(bool default_device=true, int devicenum=0);
    bool device_download_release(bool default_device=true, int devicenum=0);
    bool device_upload(bool default_device=true,int devicenum=0);
    bool device_alloc(bool default_device=true,int devicenum=0);
    bool device_release(bool default_device=true,int devicenum=0);
    bool device_delete(bool default_device=true,int devicenum=0);


    void MPI_Recv_mdspan_pdata(int dest, int tag,MPI_Comm pcomm);
    void MPI_Irecv_mdspan_pdata( int source, int tag,  MPI_Comm pcomm, MPI_Request& request);

    void MPI_Send_mdspan_pdata(int dest, int tag,MPI_Comm pcomm);
    void MPI_Isend_mdspan_pdata( int dest,  int tag, MPI_Comm pcomm, MPI_Request &request);

    void MPI_Send_mdspan(int dest, int tag, MPI_Comm pcomm);
    void MPI_Bcast_mdspan(int root, MPI_Comm pcomm);

    // Other utility methods
    size_t extent(const size_t dim) const
    {
        return pdatastruct.dpextents[dim];
    };
    size_t rank() const
    {
        return pdatastruct.dprank;
    };
    size_t stride(const size_t dim) const
    {
        return pstrides[dim];
    };

    // Member function declarations
    Container& extents()const
    {
        return pextents;
    };
    Container& strides()const
    {
        return pstrides;
    };

    size_t datalength() const
    {
        return pdatastruct.dpdatalength;
    };


    bool is_offloaded(const int devicenum)const
    {
        return pis_offloaded.contains(devicenum);
    };

    map<int,bool> is_offloaded() const
    {
        return pis_offloaded;
    };

    const datastruct<T> & get_datastruct()const
    {
        return pdatastruct;
    };

    datastruct<T> & get_datastruct()
    {
        return pdatastruct;
    };

    bool ownsdata() const
    {
        return pownsdata;
    };

protected:
    datastruct<T> pdatastruct;
    shared_ptr<T> p_refctr;

    bool pownsdata=false;
    bool pwith_memmap=false;
    // Private member variables
    map<int,bool> pis_offloaded;

    void initialize_extents_and_strides(const Container&extents,const Container & strides);
    void initialize_extents(const Container&extents);
    void allocate_data(const bool memmap,const size_t datalength);
    void addrefcounter(const mdspan<T, Container> *other);

    Container pextents;  // Use the ExtentContainer type
    Container pstrides;  // Use the StrideContainer type
};




// Access operator for multidimensional indices
template <typename T, typename Container>
inline T& mdspan<T, Container>::operator()(const Container& indices)
{


    size_t offset = 0;
    #pragma omp simd reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * pdatastruct.dpstrides[i];
    }

    return pdatastruct.dpdata[offset];
}


// Access operator for multidimensional indices
template <typename T, typename Container>
T mdspan<T, Container>::operator()(const Container& indices)const
{

    size_t offset = 0;
    #pragma omp simd reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * pdatastruct.dpstrides[i];
    }

    return pdatastruct.dpdata[offset];
}


template <typename Container>
void compute_strides(const Container& extents, Container& strides,const bool rowmajor)
{
    const size_t n = extents.size();

    if constexpr (StaticContainer<Container>)
    {
        strides = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        strides.resize(n); // Resize dynamic container
    }

    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
        for (size_t i = 1; i < n; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];
        }
    }
}

template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents_and_strides(const Container& extents, const Container& strides)
{
    const size_t r = extents.size();

    if constexpr (StaticContainer<Container>)
    {
        pextents = {};
        pstrides = {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r);
        pstrides.resize(r);
    }
    #pragma omp simd
    for (size_t i = 0; i < r; ++i)
    {
        pextents[i] = extents[i];
        pstrides[i] = strides[i];
    }
    // Assign to datastruct
    pdatastruct.dpextents = pextents.data();
    pdatastruct.dpstrides = pstrides.data();
}
template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents(const Container& extents)
{
    const size_t r = extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents = {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r);

    }

    #pragma omp simd
    for (size_t i = 0; i < r; ++i)
    {
        pextents[i] = extents[i];
    }
    // Assign to datastruct
    pdatastruct.dpextents = pextents.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  size_t datalength,const  bool rowm, const Container& extents, const Container& strides)
    :pdatastruct(data,datalength,rowm,extents.size(),nullptr,nullptr,false,false,false,false,false),
     pownsdata(false),
     pwith_memmap(false)
{
    initialize_extents_and_strides(extents,strides);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm, const Container& extents, const Container& strides )
    : pdatastruct(data, 0,rowm,extents.size(),nullptr,nullptr,false,false,false,false,false),
      pownsdata(false),
      pwith_memmap(false)
      // Initialize pdatastruct with placeholders
{
    initialize_extents_and_strides(extents,strides);
    pdatastruct.dpdatalength=compute_data_length_w(pdatastruct.dpextents,pdatastruct.dpstrides,pdatastruct.dprank);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const bool rowm,const  Container& extents)
    :  pdatastruct(data,0,rowm,extents.size(),nullptr,nullptr,false,false,false,false,false),
       pownsdata(false),
       pwith_memmap(false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    pdatastruct.dpstrides = pstrides.data();
    pdatastruct.dpdatalength=compute_data_length_w(pdatastruct.dpextents,pdatastruct.dpstrides,pdatastruct.dprank);
}







template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm,  const size_t rows, const size_t cols)
    :  pdatastruct(data,0,rowm,2,nullptr,nullptr,false,false,false,false,false),
       pownsdata(false),
       pwith_memmap(false)
{
    const size_t r=2;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }
    // Resize and copy extents from container


    pextents[0]=(rowm==true)?rows:cols;
    pextents[1]=(rowm==true)?cols:rows;
    compute_strides(pextents,pstrides,rowm);

    pdatastruct.dpextents = pextents.data();
    pdatastruct.dpstrides = pstrides.data();
    pdatastruct.dpdatalength=compute_data_length_w(pdatastruct.dpextents,pdatastruct.dpstrides,pdatastruct.dprank);
}





template <typename T, typename Container>
mdspan<T, Container>::mdspan( const size_t datalength,  const bool rowm,const bool memmap, const Container& extents, const Container& strides)
    :pdatastruct(nullptr, datalength,rowm,extents.size(),nullptr,nullptr,false,false,false,false,false)
{
    initialize_extents_and_strides(extents,strides,rowm);
    allocate_data(memmap,pdatastruct.dpdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan( const bool rowm,const bool memmap, const Container& extents, const Container& strides )
    : pdatastruct(nullptr, 0,rowm,extents.size(),nullptr,nullptr,false,false,false,false,false)
{
    initialize_extents_and_strides(extents,strides,rowm);
    pdatastruct.dpdatalength=compute_data_length_w(pdatastruct.dpextents,pdatastruct.dpstrides,pdatastruct.dprank);
    allocate_data(memmap,pdatastruct.dpdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(const bool rowm,const bool memmap,const  Container& extents)
    :  pdatastruct(nullptr,0,rowm,extents.size(),nullptr,nullptr,false,false,false,false,false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    pdatastruct.dpextents = pextents.data();
    pdatastruct.dpstrides = pstrides.data();
    pdatastruct.dpdatalength=compute_data_length_w(pdatastruct.dpextents,pdatastruct.dpstrides,pdatastruct.dprank);
    allocate_data(memmap,pdatastruct.dpdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(const bool rowm,const bool memmap, const size_t rows, const size_t cols)
    :  pdatastruct(nullptr,0,rowm,2,nullptr,nullptr,false,false,false,false,false)

{
    const size_t r=2;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }
    // Resize and copy extents from container
    pextents[0]=(rowm==true)?rows:cols;
    pextents[1]=(rowm==true)?cols:rows;
    compute_strides(pextents,pstrides,rowm);

    pdatastruct.dpextents = pextents.data();
    pdatastruct.dpstrides = pstrides.data();
    pdatastruct.dpdatalength=compute_data_length_w(pdatastruct.dpextents,pdatastruct.dpstrides,pdatastruct.dprank);
    allocate_data(memmap,pdatastruct.dpdatalength);

}





template <typename T, typename Container>
void mdspan<T, Container>::allocate_data(bool memmap, size_t dl)
{
    pownsdata = true;
    if (memmap)
    {
        pdatastruct.dpdata = create_temp_mmap<T>(dl);
        pwith_memmap = true;
    }

    else
    {
        pdatastruct.dpdata = new T[dl];
        pwith_memmap = false;
    }

    p_refctr=shared_ptr<T>(pdatastruct.dpdata,[this](T* p)
    {
        for(auto& p : pis_offloaded)
            if(p.second==true)
                exit_struct(pdatastruct,p.first);


        if(pownsdata==true)
        {
            if (pwith_memmap==true)
                munmap(p,sizeof(T)*pdatastruct.dpdatalength);
            else
                delete[] p;
        }
    });

}

template <typename T, typename Container>
void mdspan<T, Container>::addrefcounter(const mdspan<T, Container> *other)
{
    if(other->pownsdata)
    {
        p_refctr=shared_ptr<T>(other->p_refctr);
        pownsdata=true;
    }
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan(const Container&offsets, const Container &sub_extents, T* __restrict sub_data)const
{
    const size_t r=pdatastruct.dprank;

    if (sub_data==nullptr)
    {
        // Compute the offset to the starting point
        size_t offset_index = 0;

        #pragma omp simd reduction( + : offset_index )
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += offsets[i] * pdatastruct.dpstrides[i];
        }

        // Create a new mdspan_dynamic with the updated pointer, extents, and the same strides
        mdspan md= mdspan(pdatastruct.dpdata + offset_index,pdatastruct.dprowmajor, sub_extents, pstrides);
        md.addrefcounter(this);
        return md;

    }
    else
    {
        // Compute the new strides for the subspan
        Container sub_strides;
        compute_strides(sub_extents, sub_strides, pdatastruct.dprowmajor);
        vector<size_t> indices(r,0);
        vector<size_t> global_indices(r,0);
        while (true)
        {
            // Compute the current global indices
            #pragma omp simd
            for (size_t i = 0; i < r; ++i)
            {
                global_indices[i] = offsets[i] + indices[i];
            }

            // Compute the offsets for the original data and the new buffer
            size_t original_index = compute_offset_v(global_indices.data(), pdatastruct.dpstrides,global_indices.size(), pdatastruct.dprowmajor);
            size_t buffer_index = compute_offset_v(indices.data(),sub_strides.data(),indices.size(), pdatastruct.dprowmajor);

            // Copy the data from the original tensor to the sub-buffer
            sub_data[buffer_index] = pdatastruct.dpdata[original_index];

            // Increment the indices for the Cartesian product
            size_t dim = r;
            while (dim-- > 0)
            {
                if (++indices[dim] < sub_extents[dim])
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
        return mdspan(sub_data, pdatastruct.dprowmajor, sub_extents, sub_strides );
    }
}

template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*__restrict  sub_data)const
{

    if(sub_data==nullptr)
    {

        const size_t offset=row * pdatastruct.dpstrides[0]+col * pdatastruct.dpstrides[1];
        const Container ext= {tile_rows,tile_cols};
        mdspan md=mdspan(pdatastruct.dpdata +offset,pdatastruct.dprowmajor,ext,pstrides);
        md.addrefcounter(this);
        return md;
    }
    else
    {
        if (pdatastruct.dprowmajor)
        {
            // Row-major layout: fill row by row
            const size_t str0= pdatastruct.dpstrides[0];
            const size_t str1=pdatastruct.dpstrides[1];
            T *pd=pdatastruct.dpdata;
            #pragma omp parallel for simd collapse (2) shared(sub_data,pd,tile_cols,row,col,str0,str1)
            for (size_t i = 0; i < tile_rows; ++i)
            {
                for (size_t j = 0; j < tile_cols; ++j)
                {
                    sub_data[i * tile_cols + j] =pd[ compute_offset(row + i, col + j, str0,str1) ];
                }
            }
        }
        else
        {
            const size_t str0= pdatastruct.dpstrides[0];
            const size_t str1=pdatastruct.dpstrides[1];
            T *pd=pdatastruct.dpdata;
            #pragma omp parallel for simd collapse (2) shared(sub_data,pd,tile_rows,tile_cols,row,col,str0,str1)
            for (size_t j = 0; j < tile_cols; ++j)
            {
                for (size_t i = 0; i < tile_rows; ++i)
                {
                    sub_data[j * tile_rows + i] = pd[ compute_offset(row + i, col + j,str0,str1 )  ];
                }
            }
        }
        const Container sub_extents = {tile_rows, tile_cols};

        const Container sub_strides = (pdatastruct.dprowmajor==true)? Container{tile_cols, 1} :
                                      Container{1,tile_rows};

        return mdspan(sub_data,pdatastruct.dprowmajor, sub_extents, sub_strides );
    }
}




template<typename TargetSpan, typename SourceSpan>
bool glue_matrices(TargetSpan target, const vector<SourceSpan>& spans,
                   const vector<pair<size_t, size_t>>& offsets)
{

    #pragma omp parallel for
    for (size_t idx = 0; idx < spans.size(); ++idx)
    {
        const SourceSpan& span = spans[idx];
        const size_t row_offset = offsets[idx].first;
        const size_t col_offset = offsets[idx].second;
        const size_t ext0=span.get_datastruct().pextents[0];
        const size_t ext1=span.get_datastruct().pextents[1];
        const size_t sstr0=span.get_datastruct().pstrides[0];
        const size_t sstr1=span.get_datastruct().pstrides[1];
        const size_t tstr0=target.get_datastruct().pstrides[0];
        const size_t tstr1=target.get_datastruct().pstrides[1];
        // Copy the current span into the target at the given offset
        #pragma omp simd collapse(2)
        for (size_t i = 0; i < ext0; ++i)    // Rows of the span
        {
            for (size_t j = 0; j < ext1; ++j)    // Columns of the span
            {
                target(row_offset + i, col_offset + j,tstr0,tstr1) = span(i, j,sstr0,sstr1);
            }
        }
    }

    return true;
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::column(const size_t col_index)
{

    const size_t num_rows = pdatastruct.dpextents[0];
    const Container column_extents = {num_rows};
    const Container column_strides = {pdatastruct.dpstrides[0]};

    return mdspan<T, Container>( pdatastruct.dpdata + col_index * pdatastruct.dpstrides[1], pdatastruct.dprowmajor, column_extents, column_strides );
}

template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::row(const size_t row_index)
{

    const size_t num_cols = pdatastruct.dpextents[1];
    const Container row_extents = {num_cols};
    const Container row_strides = {pdatastruct.dpstrides[1]};

    return mdspan<T, Container>( pdatastruct.dpdata + row_index * pdatastruct.dpstrides[0], pdatastruct.dprowmajor, row_extents, row_strides );
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_upload(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices()) return false;

    this->pis_offloaded.try_emplace(devicenum, true);

    create_in_struct(this->get_datastruct(),devicenum);
    return true;
}


template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_alloc(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();

    if(devicenum>omp_get_num_devices()) return false;
    this->pis_offloaded.try_emplace(devicenum, true);
    create_out_struct(this->get_datastruct(),devicenum);
    return true;
}




template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_download_release(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    update_host(this->get_datastruct(),devicenum);
    release_struct(this->get_datastruct(),devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}


template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_delete(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    exit_struct(this->get_datastruct(),devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_release(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    release_struct(this->get_datastruct(),devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: host_update(bool default_device,int devicenum)
{
    if (default_device)  devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices()) return false;
    if(this->pis_offloaded.contains(devicenum)==false) return false;
    update_host(this->get_datastruct(),devicenum);
    return true;

}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_update(bool default_device,int devicenum)
{

    if (default_device)  devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices()) return false;
    if(this->pis_offloaded.contains(devicenum)==false) return false;

    update_device(this->get_datastruct(),devicenum);
    return true;

}



template <typename T, typename Container>
void mdspan<T, Container>:: MPI_Send_mdspan(int dest, int tag, MPI_Comm pcomm)
{
    MPI_Send(&pdatastruct.dpdatalength, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&pdatastruct.dprank, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&pdatastruct.dprowmajor, 1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(&pdatastruct.dpdata_is_devptr,1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(&pdatastruct.dpextents_is_devptr,1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(&pdatastruct.dpstrides_is_devptr,1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(pdatastruct.dpextents, pdatastruct.dprank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(pdatastruct.dpstrides,pdatastruct.dprank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(pdatastruct.dpdata,sizeof(T)* pdatastruct.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}


template <typename T, typename Container>
void mdspan<T, Container>::MPI_Isend_mdspan_pdata( int dest,  int tag, MPI_Comm pcomm, MPI_Request &request)
{
    MPI_Isend(&pdatastruct.dpdata[0],sizeof(T)* pdatastruct.dpdatalength, MPI_BYTE, dest, tag, pcomm,&request);
}
template <typename T, typename Container>
void mdspan<T, Container>::MPI_Send_mdspan_pdata( int dest,  int tag,MPI_Comm pcomm)
{
    MPI_Send(&pdatastruct.dpdata[0],sizeof(T)* pdatastruct.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}



template <typename T, typename Container>
void mdspan<T, Container>::MPI_Irecv_mdspan_pdata( int source, int tag, MPI_Comm pcomm, MPI_Request& request)
{
    MPI_Irecv(&pdatastruct.dpdata[0],sizeof(T)* pdatastruct.dpdatalength, MPI_BYTE, source, tag, pcomm,&request);
}
template <typename T, typename Container>


void mdspan<T, Container>::MPI_Recv_mdspan_pdata( int source,  int tag, MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(&pdatastruct.dpdata[0],sizeof(T)* pdatastruct.dpdatalength, MPI_BYTE, source, tag, pcomm,&status);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(const bool memmap,  int source, int tag,  MPI_Comm pcomm)
{
    MPI_Status status;
    size_t pl, pr;
    bool prm;
    bool dataisdevptr,stridesisdevptr,extentsisdevptr;

    MPI_Recv(&pl, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&pr, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&prm, 1, MPI_C_BOOL, source, tag, pcomm, &status);


    MPI_Recv(&dataisdevptr,1, MPI_C_BOOL, source, tag, pcomm,&status);
    MPI_Recv(&extentsisdevptr,1, MPI_C_BOOL, source, tag, pcomm,&status);
    MPI_Recv(&stridesisdevptr,1, MPI_C_BOOL, source, tag, pcomm,&status);

    pextents.resize(pr);
    pstrides.resize(pr);

    pdatastruct.dpextents=pextents.data();
    pdatastruct.dpstrides=pstrides.data();
    pdatastruct.dprowmajor=prm;
    pdatastruct.dpdatalength=pl;
    pdatastruct.dprank=pr;

//    if (pextentsisdevptr)
//    {
//        size_t* pr= (size_t*)omp_alloc(sizeof(size_t)*prank,omp_get_default_device());
//        MPI_Recv(&pr,prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
//    }
//    else
//    {
    MPI_Recv(&pextents[0],pr, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
//    }

//    if (pstridesisdevptr)
//    {
//        size_t*pr= (size_t*) omp_target_alloc(sizeof(size_t)*prank,omp_get_default_device());
//        MPI_Recv(&pr,prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
//    }
//    else
//    {
    MPI_Recv(&pstrides[0],pr, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
//    }

    allocate_data(memmap,pdatastruct.dpdatalength);

//    if(pdataisdevptr)
//    {
//        size_t*pr= (size_t*) omp_target_alloc(sizeof(size_t)*pdatalength,omp_get_default_device());
//        MPI_Recv(&pr,pdatalength, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
//    }
//    else
//    {
    MPI_Recv(&pdatastruct.dpdata[0],sizeof(size_t)*pl, MPI_BYTE, source, tag, pcomm, &status);
//    }
}





template <typename T>
void MPI_send_datastruct(datastruct<T> &m, int dest, int tag, MPI_Comm pcomm)
{
    MPI_Send(&m.dpdatalength, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.dprank, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.dprowmajor, 1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(&m.dpdata_is_devptr,1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(&m.dpextents_is_devptr,1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(&m.dpstrides_is_devptr,1, MPI_C_BOOL, dest, tag, pcomm);
    MPI_Send(&m.dpextents[0], m.dprank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.dpstrides[0], m.dprank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.dpdata[0],sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}

template <typename T>
void MPI_Isend_datastruct_pdata(datastruct<T> &m,const int dest,const  int tag,const MPI_Comm pcomm)
{
    MPI_Request request;
    MPI_Isend(&m.dpdata[0],sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm,&request);
}
template <typename T>
void MPI_send_datastruct_pdata(datastruct<T> &m,const int dest,const int tag,const MPI_Comm pcomm)
{
    MPI_Send(&m.dpdata[0],sizeof(T)* m.pdatalength, MPI_BYTE, dest, tag, pcomm);
}


template <typename T>
datastruct<T> MPI_recv_alloc_datastruct(bool with_memmap, bool on_device,  bool defaultdevice,int devicenum, const int source, int tag, MPI_Comm pcomm)
{
    MPI_Status status;
    size_t pdatalength, prank;
    bool prowmajor;
    bool pdataisdevptr,pstridesisdevptr,pextentsisdevptr;

    MPI_Recv(&pdatalength, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&prank, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&prowmajor, 1, MPI_C_BOOL, source, tag, pcomm, &status);

    MPI_Recv(&pdataisdevptr,1, MPI_C_BOOL, source, tag, pcomm,&status);
    MPI_Recv(&pextentsisdevptr,1, MPI_C_BOOL, source, tag, pcomm,&status);
    MPI_Recv(&pstridesisdevptr,1, MPI_C_BOOL, source, tag, pcomm,&status);

    pdataisdevptr=on_device;
    pextentsisdevptr=on_device;
    pstridesisdevptr=on_device;

    if (defaultdevice)
        devicenum=omp_get_default_device();

    size_t *pextents,*pstrides;

    T* pdata;

    if (pextentsisdevptr)
    {
#if (Unified_Shared_Memory==True)
        pextents= (size_t*)malloc(sizeof(size_t)*prank);
        MPI_Recv(&pextents[0],prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
#else
        pextents= (size_t*)omp_target_alloc(sizeof(size_t)*prank,devicenum);
        MPI_Recv(&pextents,prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
#endif
    }
    else
    {
        pextents= (size_t*)malloc(sizeof(size_t)*prank);
        MPI_Recv(&pextents[0],prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    }

    if (pstridesisdevptr)
    {
#if (Unified_Shared_Memory==True)
        pstrides= (size_t*)malloc(sizeof(size_t)*prank);
        MPI_Recv(&pstrides[0],prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
#else
        pstrides=(size_t*) omp_target_alloc(sizeof(size_t)*prank,devicenum);
        MPI_Recv(&pstrides,prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
#endif
    }
    else
    {
        pstrides= (size_t*)malloc(sizeof(size_t)*prank);
        MPI_Recv(&pstrides[0],prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    }

    if(pdataisdevptr)
    {
#if (Unified_Shared_Memory==True)
        pdata=(T*)malloc(sizeof(T)*pdatalength);
        MPI_Recv(&pdata[0],sizeof(size_t)*pdatalength, MPI_BYTE, source, tag, pcomm, &status);
#else
        pdata= (size_t*) omp_target_alloc(sizeof(size_t)*pdatalength,devicenum);
        MPI_Recv(pdata,pdatalength, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
#endif
    }
    else
    {
        if(with_memmap)
        {
            pdata=create_temp_mmap<T>(pdatalength);
        }
        else
        {
            pdata=(T*)malloc(sizeof(T)*pdatalength);
        }
        MPI_Recv(&pdata[0],sizeof(size_t)*pdatalength, MPI_BYTE, source, tag, pcomm, &status);
    }

    return datastruct<T>(&pdata[0],pdatalength,prowmajor,prank,pextents,pstrides,false,false,pdataisdevptr,pstridesisdevptr,pextentsisdevptr);
}


template <typename T>
void datastruct_free(datastruct<T> &m,bool created_with_memmap, bool on_device,  bool defaultdevice,int devicenum)
{
    if (defaultdevice)
        devicenum=omp_get_default_device();

    if (m.dpextents_is_devptr)
#if (Unified_Shared_Memory==True)
        free(m.dpextents);
#else
        omp_target_free(m.dpextents, devicenum);
#endif
    else
        free(m.dpextents);

    if (m.dpstrides_is_devptr)
#if (Unified_Shared_Memory==True)
        free(m.dpstrides);
#else
        omp_target_free(m.dpstrides, devicenum);
#endif


    else
        free(m.dpstrides);

    if (m.dpdata_is_devptr)
#if (Unified_Shared_Memory==True)
        free(m.dpdata);
#else
        omp_target_free(m.pdata, devicenum);
#endif

    else
    {
        if (created_with_memmap)
            delete_temp_mmap<T>(m.dpdata,m.dpdatalength);
        else
            free(m.dpdata);
    }
}

template <typename T>
datastruct<T>& datastruct_allocate_heap(T* data,size_t datalength, bool rowmajor, size_t *extents,size_t* strides,size_t rank,bool create_memmap, bool on_device,  bool defaultdevice,int devicenum)
{
    size_t*pextents;
    size_t*pstrides;
    size_t* pdata;
    if (on_device)
    {
        if (defaultdevice)
            devicenum=omp_get_default_device();

        if (extents==nullptr )
#if (Unified_Shared_Memory==True)
            pextents=(size_t*)malloc(sizeof(T)*rank);
#else
            pextents=omp_target_alloc(sizeof(T)*rank,devicenum);
#endif

        else
        {
#if (Unified_Shared_Memory==True)
            pextents=(size_t*) malloc(sizeof(T)*rank);
            memcpy(pextents,extents,sizeof(T)*rank);
#else
            pextents=omp_target_alloc(sizeof(T)*rank,devicenum);
            omp_target_memcpy(pextents,extents,sizeof(T)*rank,0,0,devicenum,omp_get_initial_device());
#endif
        }

        if (strides==nullptr )
#if (Unified_Shared_Memory==True)
            pstrides=(T*) malloc(sizeof(T)*rank);
#else
            pstrides=omp_target_alloc(sizeof(T)*rank,devicenum);
#endif
        else
        {
#if (Unified_Shared_Memory==True)
            pstrides=(size_t*) malloc(sizeof(T)*rank);
            memcpy(pstrides,extents,sizeof(T)*rank);
#else
            pstrides=omp_target_alloc(sizeof(T)*rank,devicenum);
            omp_target_memcpy(pstrides,extents,sizeof(T)*rank,0,0,devicenum,omp_get_initial_device());
#endif
        }
        if (data==nullptr )
#if (Unified_Shared_Memory==True)
            pdata=(T*) malloc(sizeof(T)*datalength);
#else
            pdata=omp_target_alloc(sizeof(T)*datalength,devicenum);
#endif
        else
        {
#if (Unified_Shared_Memory==True)
            pdata=(size_t*) malloc(sizeof(T)*datalength);
            memcpy(pdata,data,sizeof(T)*datalength);
#else
            pdata=omp_target_alloc(sizeof(T)*datalength,devicenum);
            omp_target_memcpy(pdata,data,sizeof(T)*datalength,0,0,devicenum,omp_get_initial_device());
#endif


        }
        return datastruct<T>(pdata,datalength,rowmajor,rank,pextents,pstrides,true,true,true);
    }

    else
    {
        if (extents==nullptr )
            pextents=(size_t*) malloc(sizeof(T)*rank);
        else
        {
            pextents=(size_t*) malloc(sizeof(T)*rank);
            memcpy(pextents,extents,sizeof(T)*rank);
        }
        if (strides==nullptr )
            pstrides=(size_t*) malloc(sizeof(T)*rank);
        else
        {
            pstrides=(size_t*) malloc(sizeof(T)*rank);
            memcpy(pstrides,extents,sizeof(T)*rank);
        }

        if (create_memmap)
        {
            pdata=create_temp_mmap<T>(datalength);
        }
        else
        {
            if(data==nullptr)
                pdata=(size_t*) malloc(sizeof(T)*datalength);
            else
            {
                pdata=(size_t*) malloc(sizeof(T)*datalength);
                memcpy(pdata,data,sizeof(T)*rank);
            }

        }
        return datastruct<T>(pdata,datalength,rowmajor,rank,pextents,pstrides,false, false, false);
    }

}





template <typename T>
void MPI_Irecv_datastruct_pdata(datastruct<T> &mds, const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Request request;
    MPI_Irecv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &request);
}


template <typename T>
void MPI_recv_datastruct_pdata(datastruct<T>& mds,const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);
}








void MPI_recursion_helper_end(MPI_Comm pcomm)
{
    int commsize=0;
    MPI_Comm_size(pcomm, &commsize);
    int message=COMMAND_END_LISTENER;
    for (int i=0; i<commsize; i++)
    {
        MPI_Send(&message,1,MPI_INT,i,0,pcomm);
    }
}


template <typename T>
void MPI_recursive_multiplication_helper(matrix_multiplication_parameters &par)
{

    MPI_Status status;
    int message;
    for(;;)
    {
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, 0, par.comm, &status);
        switch (message)
        {
        case COMMAND_STRASSEN:
        {
            mdspan<T, vector<size_t>> A1(true,status.MPI_SOURCE, 1, par.comm);

            mdspan<T, vector<size_t>> B1(true,status.MPI_SOURCE, 2, par.comm);

            size_t rowsC=A1.get_datastruct().dpextents[0],
                   colsC=B1.get_datastruct().dpextents[1];

            mdspan<T, std::vector<size_t>> C1(true,par.memmapped_files, rowsC, colsC);

            strassen_multiply(A1,B1,C1,par);
            C1.MPI_Send_mdspan_pdata(status.MPI_SOURCE,3,par.comm);
            break;
        }
        case COMMAND_WINOGRAD:
        {
            mdspan<T, vector<size_t>> A(true,status.MPI_SOURCE, 1, par.comm);
            mdspan<T, vector<size_t>> B(true,status.MPI_SOURCE, 2, par.comm);
            size_t rowsC=A.get_datastruct().dpextents[0],
                   colsC=B.get_datastruct().dpextents[1];

            mdspan<T, std::vector<size_t>> C1(true,par.memmapped_files, rowsC, colsC);
            winograd_multiply(A,B,C1,par);

            C1.MPI_Send_mdspan_pdata(status.MPI_SOURCE,3,par.comm);
            break;
        }
        case COMMAND_END_LISTENER:
            goto endloop;
        }
    }

endloop:
    return;

}











template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::transpose(T* __restrict pdata)
{
    Container transposed_extents = {pdatastruct.dpextents[1], pdatastruct.dpextents[0]};
    Container transposed_strides = {pdatastruct.dpstrides[1], pdatastruct.dpstrides[0]};
    if (pdata==nullptr)
    {
        mdspan md=mdspan(pdatastruct.dpdata,pdatastruct.dpdatalength, pdatastruct.dprowmajor,  transposed_extents,   transposed_strides);
        md.addrefcounter(this);
        return md;
    }
    else
    {
        size_t s=pdatastruct.dpextents[1]* pdatastruct.dpextents[0];
        for (size_t i=0; i<s; i++)
        {
            pdata[i]=pdatastruct.dpdata[i];
        }
        return mdspan(pdata,pdatastruct.dpdatalength, pdatastruct.dprowmajor,  transposed_extents,   transposed_strides);
    }

}

#pragma omp begin declare target
template <typename T>
void gpu_cholesky_decomposition_w(const datastruct<T>& A, datastruct<T>& L, T* __restrict buffer=nullptr, size_t step_size=0)
{
    const size_t n = A.dpextents[0];
    size_t z = 0; // Zero-based indexing, starts at the first column

    step_size=(size_t)pow(n,0.8385);

    const size_t tempsize = (n - step_size) * (n - step_size);


    size_t pext3[2];
    size_t pstrides3[2];

    const size_t nn=n*n;
    // Allocate memory for S on the device
    T* __restrict sdata;
    T* __restrict adata;

    if (buffer==(T*) nullptr)
    {
        sdata=(T*) omp_alloc(sizeof(T*)*tempsize,omp_large_cap_mem_alloc);
        adata=(T*) omp_alloc(sizeof(T*)*nn,omp_large_cap_mem_alloc);
    }
    else
    {
        sdata=buffer;
        adata=buffer+tempsize;
    }

    datastruct<T> tempA(adata, 0,A.dprowmajor,n, n,pext3, pstrides3,true,true);
    #pragma omp parallel for simd shared(adata,A,L)
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=A.dpdata[i];
        L.dpdata[i]=0;
    }


    for (size_t c = 0; c < n; ++c)
    {
        if (c == z + step_size)
        {
            size_t u=n-c;
            size_t v=c-z;


            size_t pSstrides[2];
            size_t pSext[2];
            size_t pRext[2];
            size_t pRstr[2];

            datastruct<T> R =L.subspanmatrix(c, z, u, v,pRext,pRstr);

            datastruct<T> S(sdata, 0, A.dprowmajor,u, u, pSext, pSstrides,true,true);

            const size_t rows=R.dpextents[0];
            const size_t cols=R.dpextents[0];
            const size_t inner_dim=R.dpextents[1];



            #pragma omp parallel for collapse(2) shared(R,S,rows,cols,inner_dim)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += R(i,k) *R(k,j);
                    }
                    S(i,j)= sum;
                }
            }

            #pragma omp parallel for collapse(2) shared(c,tempA)
            for (size_t i = c; i < n; ++i)
            {
                for (size_t j = c; j < n; ++j)
                {
                    tempA(i,j) -=S(i-c,j-c);
                }
            }

            z = c;
        }

        T tmp=tempA(c, c);
        #pragma omp parallel for simd reduction(-:tmp) shared(L,c,z)
        for (size_t k = z; k < c; ++k)
        {
            const T tmp3=L(c,k);
            tmp-= tmp3 * tmp3;
        }
        L(c, c) =sqrt(tmp);

        T tmp4=L(c, c);
        #pragma omp parallel for shared(tempA,L,c,z,tmp4)
        for (size_t i = c + 1; i < n; ++i)
        {
            T tmp2 = tempA(i, c);
            #pragma omp simd reduction(-:tmp2)
            for (size_t k = z; k < c; ++k)
            {
                tmp2 -= L(i, k) * L(c, k);
            }
            L(i, c)=tmp2/tmp4;
        }
    }

    if(buffer==nullptr)
    {
        omp_free(sdata,omp_large_cap_mem_alloc);
        omp_free(adata,omp_large_cap_mem_alloc);
    }
}
#pragma omp end declare target





#pragma omp begin declare target
template <typename T>
inline  void gpu_lu_decomposition_w(const  datastruct<T>& dA, datastruct<T>& dL, datastruct<T>& dU, T* __restrict buffer=nullptr, size_t step_size=0)
{

    const size_t n = dA.dpextents[0];
    size_t z = 0; // Zero-based indexing, starts at the first column

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    const size_t tempsize = (n - step_size) * (n - step_size);
    size_t pext3[2];
    size_t pstrides3[2];
    const size_t nn=n*n;


    T* __restrict sdata;
    T* __restrict adata;

    if (buffer==nullptr)
    {
        sdata=(T*)omp_alloc(sizeof(T)*tempsize,omp_large_cap_mem_alloc);
        adata=(T*)omp_alloc(sizeof(T)*nn,omp_large_cap_mem_alloc);
    }
    else
    {
        sdata=buffer;
        adata=buffer+tempsize;
    }
    datastruct<T> tempA(adata,  0, dA.dprowmajor,n, n,pext3, pstrides3,true,true);

    #pragma omp parallel for simd shared(adata,dA,dL,dU)
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=dA.dpdata[i];
        dL.dpdata[i]=0;
        dU.dpdata[i]=0;
    }

    for (size_t c = 0; c < n; ++c)
    {
        if (c == z + step_size)
        {
            const size_t u=n-c;
            const size_t v=c-z;
            size_t pRLext[2];
            size_t pRLstr[2];
            size_t pRUext[2];
            size_t pRUstr[2];
            size_t pSstrides[2];
            size_t pSext[2];
            datastruct<T> RL = dL.subspanmatrix(c, z, u, v,pRLext,pRLstr);
            datastruct<T> RU = dU.subspanmatrix(z, c, v, u,pRUext,pRUstr);

            datastruct<T> S(sdata,  0, dA.dprowmajor,u, u,pSext, pSstrides,true,true);

            const size_t rows=RL.dpextents[0];
            const size_t cols=RU.dpextents[1];
            const size_t inner_dim=RL.dpextents[1];



            #pragma omp parallel for  collapse(2) shared(RL,RU,S,rows,cols,inner_dim)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    #pragma omp simd reduction(+: sum)
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += RL(i,k) *RU(k,j);
                    }
                    S(i,j)= sum;
                }
            }
            const size_t h=c;
            #pragma omp parallel for simd collapse(2) shared(h,S,tempA)
            for (size_t i = c; i < n; ++i)
            {
                for (size_t j = c; j < n; ++j)
                {
                    tempA(i,j) -= S(i - h, j - h);
                }
            }
            z = c;
        }

        #pragma omp parallel for shared(c,z,dU,dL,tempA)
        for (size_t i = c; i < n; ++i)
        {
            T temp=tempA(c,i);
            #pragma omp simd reduction(-:temp)
            for (size_t k = z; k < c; ++k)
            {
                temp-= dU( k,i) * dL( c,k);
            }
            dU(c,i)=temp;
        }
        const T temp4=dU(c,c);

        #pragma omp parallel for shared(c,z,dU,dL,tempA,temp4)
        for (size_t i = c; i < n; ++i)
        {
            T temp= tempA(i,c);
            #pragma omp simd reduction (-:temp)
            for (size_t k = z; k < c; ++k)
            {
                temp -= dU(k,c) * dL( i,k);
            }
            dL(i,c)=temp/temp4;
        }
    }

    if(buffer==nullptr)
    {
        omp_free(sdata,omp_large_cap_mem_alloc);
        omp_free(adata,omp_large_cap_mem_alloc);
    }

}
#pragma omp end declare target







#pragma omp begin declare target
template <typename T >
inline void gpu_qr_decomposition_w( const datastruct<T>&A, datastruct<T> Q, datastruct<T> &R, T* __restrict buffer=nullptr, size_t step_size=0)
{
    const size_t n = A.dpextents[0];
    const size_t m = A.dpextents[1];

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    const size_t nm=n*m;
    T* __restrict tempC;
    T* __restrict tempS;
    T* __restrict tempM;

    if(buffer==nullptr)
    {
        tempC=(T*)omp_alloc(sizeof(T)*m*m,omp_large_cap_mem_alloc);
        tempS=(T*)omp_alloc(sizeof(T)*nm,omp_large_cap_mem_alloc);
        tempM=(T*)omp_alloc(sizeof(T)*nm,omp_large_cap_mem_alloc);
    }
    else
    {
        tempC=buffer;
        tempS=buffer+m*m;
        tempM=tempS+nm;
    }

    size_t Mext[2];
    Mext[0]= A.dpextents[1];
    Mext[1]=A.dpextents[0];
    size_t Mstrides[2];
    Mstrides[0]= A.dpstrides[0];
    Mstrides[1]=A.dpstrides[1];



    #pragma omp parallel for simd shared(A,tempM)
    for (size_t i=0; i<nm; i++)
    {
        tempM[i]=A.dpdata[i];
    }


    #pragma omp parallel for simd shared(Q)
    for (size_t i=0; i<Q.dpdatalength; i++)
    {
        Q.dpdata[i]=0;
    }

    #pragma omp parallel for simd shared(R)
    for (size_t i=0; i<R.dpdatalength; i++)
    {
        R.dpdata[i]=0;
    }


    datastruct<T> M(tempM,A.dpdatalength,A.dprowmajor,A.dprank,Mext,Mstrides,false,false); //Copy of A
    size_t z = 0;



    for (size_t c = 0; c < m; ++c)
    {
        if (c == z +step_size)
        {
            // Extract submatrices
            size_t cz=c-z;
            size_t mc=m-c;

            size_t extbq[2];
            size_t extbm[2];
            size_t strbq[2];
            size_t strbm[2];
            size_t extc[2];
            size_t strc[2];
            size_t extbqt[2];
            size_t strbqt[2];
            size_t exts[2];
            size_t strs[2];
            datastruct<T> BQ = Q.subspanmatrix(0, z, n, cz,extbq,strbq);
            datastruct<T> BM = M.subspanmatrix(0, c, n, mc,extbm,strbm);

            // Compute C = BQ^T * BM
            datastruct<T> C(tempC,0, BM.prowmajor,cz, mc,extc,strc,true,true);

            datastruct<T> BQT=BQ.transpose(extbqt,strbqt);


            const size_t rows=BQT.dpextents[0];
            const size_t cols=BM.dpextents[1];
            const size_t inner_dim=BQT.dpextents[1];

            #pragma omp parallel for collapse(2) shared(BQT,BM,C,rows,cols,inner_dim)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += BQT(i,k) *BM(k,j);
                    }
                    C(i,j)= sum;
                }
            }
            // Compute S = BQ * C
            datastruct<T>S(tempS, 0,BQ.prowmajor,n, mc,exts,strs,true,true);

            const size_t rows2=BQ.dpextents[0];
            const size_t cols2=C.dpextents[1];
            const size_t inner_dim2=BQ.dpextents[1];



            #pragma omp parallel for collapse(2) shared(BQ,C,S,rows2,cols2,inner_dim2)
            for (size_t i = 0; i < rows2; ++i)
            {
                for (size_t j = 0; j < cols2; ++j)
                {
                    T sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < inner_dim2; ++k)
                    {
                        sum += BQ(i,k) *C(k,j);
                    }
                    S(i,j)= sum;
                }
            }


            // Update M: M[:, c:] -= S
            const size_t h=c;
            #pragma omp parallel for shared(M,S,h)
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = h; j <n; ++j)
                {
                    M(i,  j) -= S(i, j-h);
                }
            }
            z = c;
        }
        // Extract column c of M
        size_t pextv[1];
        size_t pstrv[1];
        datastruct<T>  v = M.column(c,pextv,pstrv);
        const size_t pext0=pextv[0];

        for (size_t j = z; j < c; ++j)
        {
            size_t pextu[1];
            size_t pstru[1];
            const datastruct<T> u = Q.column(j,pextu,pstru);
            T dot_pr=0;

            #pragma omp parallel for simd shared(u,v) reduction(+:dot_pr)
            for (size_t i = 0; i < pext0; ++i)
            {
                dot_pr += u(i) * v(i);
            }

            const T cdot_pr=dot_pr;
            #pragma omp parallel for simd shared(v,u,cdot_pr)
            for (size_t i = 0; i < pext0; ++i)
            {
                v(i) -= cdot_pr * u(i);
            }
        }
        // Normalize v
        T norm=0;
        #pragma omp parallel for simd shared(v) reduction(+: norm)
        for (size_t i = 0; i < pext0; ++i)
        {
            norm += v(i) * v(i);
        }

        const T normc= sqrt(norm);

        //  const T normc=norm;
        #pragma omp parallel for simd shared(v,normc)
        for (size_t i = 0; i < pext0; ++i)
        {
            v(i)= v(i)/normc;
        }

        // Set column c of Q

        #pragma omp parallel for simd shared(Q,v,c)
        for (size_t i = 0; i < pext0; ++i)
        {
            Q(i,c) = v(i);
        }
    }

    // Compute R = Q^T * A
    size_t qtext[2];
    size_t qtstrides[2];


    datastruct<T> QT=Q.transpose(qtext,qtstrides);
    gpu_matrix_multiply_dot_w(QT,A,R);

    if(buffer==nullptr)
    {
        omp_free(tempM,omp_large_cap_mem_alloc);
        omp_free(tempS,omp_large_cap_mem_alloc);
        omp_free(tempC,omp_large_cap_mem_alloc);
    }


}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
inline void gpu_cross_product( const datastruct<T>& vec1, const  datastruct<T>& vec2, datastruct<T>& res)
{
    res(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    res(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    res(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);

}
#pragma omp end declare target








template <typename T>
inline void matrix_multiply_dot_g(  datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C,bool on_gpu=true, int devnum=0)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];
    if(on_gpu)
    {

        #pragma omp target teams distribute parallel for collapse(2) shared(A,B,C,rows,cols,inner_dim) device(devnum)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp simd
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j)= sum;
            }
        }
    }
    else
    {
        #pragma omp parallel for collapse(2) shared(A,B,C,rows,cols,inner_dim)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp simd
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j)= sum;
            }
        }
    }
}


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_dot_w( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];


    #pragma omp parallel for collapse(2) shared(A,B,C,rows,cols,inner_dim)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_dot_v( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];


    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_dot_s( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];


    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}
#pragma omp end declare target

template <typename T>
inline void gpu_matrix_add_g(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C,int devnum)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp target teams distribute parallel for simd collapse(2) shared(C,A,B,n,m)device(devnum)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }
}



#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_add_w(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp parallel for simd collapse(2) shared(C,A,B,n,m)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_add_v(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }


}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_add_s(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }


}
#pragma omp end declare target

template <typename T>
inline void gpu_matrix_subtract_g(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C,int devnum)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp target teams distribute parallel for simd collapse(2) shared(A,B,C,n,m)device(devnum)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }

}

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_subtract_w(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp parallel for simd collapse(2) shared(C,A,B,n,m)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_subtract_v(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_subtract_s(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }

}
#pragma omp end declare target



template <typename T>
inline void gpu_matrix_multiply_vector_g( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C, int devnum)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];
    #pragma omp target teams distribute parallel for simd collapse(2) shared(C,M,V,m,n)device(devnum)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i, j) * V(j);
        }
    }

}


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_w( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];
    #pragma omp parallel for shared(C,M,V,n,m)
    for (size_t i = 0; i <n; ++i)
    {
        #pragma omp simd
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i, j) * V(j);
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_v( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i, j) * V(j);
        }
    }

}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_s( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i, j) * V(j);
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_s( const datastruct<T>M, const T*V, datastruct<T> & C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V[i];
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_v( const datastruct<T>M, const T*V, datastruct<T> & C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V[i];
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_w( const datastruct<T>M, const T*V, datastruct<T> & C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp parallel for simd collapse(2) shared(C,M,V,n,m)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V[i];
        }
    }
}
#pragma omp end declare target


template <typename T>
inline void gpu_matrix_multiply_vector_g( const datastruct<T>M, const T*V, datastruct<T> & C, int devnum)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp target teams distribute parallel for simd collapse(2)shared(C,M,V,n,m)device(devnum)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V[i];
        }
    }
}




#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_scalar_s(  const datastruct<T>& M, const T V, datastruct<T>& C)
{


    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }

}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_scalar_v(  const datastruct<T>& M, const T V, datastruct<T>& C)
{


    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];

    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_scalar_w(  const datastruct<T>& M, const T V, datastruct<T>& C)
{

    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];
    #pragma omp parallel for simd collapse(2) shared(C,M,V,n,m)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }

}
#pragma omp end declare target

template <typename T>
inline void gpu_matrix_multiply_scalar_g(  const datastruct<T>& M, const T V, datastruct<T>& C, int devnum)
{

    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];
    #pragma omp target teams distribute parallel for simd collapse(2) shared(C,M,V,n,m) device(devnum)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }

}

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_scalar_multiply_s( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.dpextents[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}

#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_scalar_multiply_v( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.dpextents[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}

#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_scalar_multiply_w( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.dpextents[0];
    #pragma omp parallel for simd shared(res,vec,n,scalar)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}

#pragma omp end declare target

template <typename T>
inline void gpu_vector_scalar_multiply_g( const datastruct<T>& vec,const T scalar,datastruct<T>& res, int devnum)
{
    const size_t n=vec.dpextents[0];
    #pragma omp target teams distribute parallel for simd shared(res,vec,n,scalar)device(devnum)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}




#pragma omp begin declare target
template <typename T>
inline void gpu_vector_add_s( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.dpextents[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_add_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_vector_add_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp parallel for simd shared(res,vec1,n)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}
#pragma omp end declare target


template <typename T>
inline void gpu_vector_add_g( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res, int devnum)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp target teams distribute parallel for simd shared(res,vec1,vec2,n)device(devnum)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }
}


template <typename T>
inline void gpu_vector_subtract_g( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res, int devnum)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp target teams distribute parallel for simd device(devnum)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_subtract_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp parallel for simd shared(res,vec1,vec2,n)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_vector_subtract_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_subtract_s( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.dpextents[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline T gpu_dot_product_s(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result=0;

    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline T gpu_dot_product_v(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.dpextents[0];

    T result=0;
    #pragma omp simd reduction(+: result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline T gpu_dot_product_w(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result=0;
    #pragma omp parallel for simd shared(vec1,vec2,n) reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}
#pragma omp end declare target

template <typename T>
inline T gpu_dot_product_g(const  datastruct<T> &vec1, const datastruct<T> &vec2, int devnum)
{
    const size_t n=vec1.dpextents[0];
    T result=0;
    #pragma omp target parallel for simd shared(vec1,vec2,n) reduction(+:result)device(devnum)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}



template <typename T, typename CA, typename CB, typename CC>
bool strassen_multiply( mdspan<T, CA>& A,  mdspan<T, CB>& B, mdspan<T, CC>& C, matrix_multiplication_parameters & algorithm)
{
    // Dimensions of input matrices
    const  size_t n = A.extent(0); // Rows in A
    const  size_t m = A.extent(1); // Columns in A and rows in B
    const size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (n*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,algorithm.gpu_offload,algorithm.default_device,algorithm.devicenum);
        return true;
    }
    // Compute sizes for splitting
    const size_t half_n = n / 2;
    const size_t half_m = m / 2;
    const size_t half_p = p / 2;

    // Submatrices of A
    mdspan<T,CA> A11 = A.subspan({0, 0}, {half_n, half_m});
    mdspan<T,CA>  A12 = A.subspan({0, half_m}, {half_n, half_m});
    mdspan<T,CA>  A21 = A.subspan({half_n, 0}, {half_n, half_m});
    mdspan<T,CA>  A22 = A.subspan({half_n, half_m}, {half_n, half_m});



    // Submatrices of B
    mdspan<T,CB>  B11 = B.subspan({0, 0}, {half_m, half_p});
    mdspan<T,CB>  B12 = B.subspan({0, half_p}, {half_m, half_p});
    mdspan<T,CB>  B21 = B.subspan({half_m, 0}, {half_m, half_p});
    mdspan<T,CB>  B22 = B.subspan({half_m, half_p}, {half_m, half_p});


    // Temporary storage for intermediate results
    size_t s=half_n*half_p,
           s2=half_n*half_m,
           s3=half_m*half_p;

    T* __restrict M1_storage;
    T* __restrict M2_storage;
    T* __restrict M3_storage;
    T* __restrict M4_storage;
    T* __restrict M5_storage;
    T* __restrict M6_storage;
    T* __restrict M7_storage;
    T* __restrict A_result1_storage;
    T* __restrict B_result1_storage;
    T* __restrict A_result2_storage;
    T* __restrict B_result2_storage;
    T* __restrict A_result3_storage;
    T* __restrict B_result3_storage;
    T* __restrict A_result4_storage;
    T* __restrict B_result4_storage;
    T* __restrict A_result5_storage;
    T* __restrict B_result5_storage;


    if (algorithm.memmapped_files)
    {
        M1_storage=create_temp_mmap<T>(s);
        M2_storage=create_temp_mmap<T>(s);
        M3_storage=create_temp_mmap<T>(s);
        M4_storage=create_temp_mmap<T>(s);
        M5_storage=create_temp_mmap<T>(s);
        M6_storage=create_temp_mmap<T>(s);
        M7_storage=create_temp_mmap<T>(s);
        A_result1_storage=create_temp_mmap<T>(s2);
        A_result2_storage=create_temp_mmap<T>(s2);
        A_result2_storage=create_temp_mmap<T>(s2);
        A_result3_storage=create_temp_mmap<T>(s2);
        A_result4_storage=create_temp_mmap<T>(s2);
        A_result5_storage=create_temp_mmap<T>(s2);
        B_result1_storage=create_temp_mmap<T>(s3);
        B_result2_storage=create_temp_mmap<T>(s3);
        B_result3_storage=create_temp_mmap<T>(s3);
        B_result4_storage=create_temp_mmap<T>(s3);
        B_result5_storage=create_temp_mmap<T>(s3);

    }

    else
    {
        M1_storage =new T[s],
        M2_storage =new T[s],
        M3_storage =new T[s],
        M4_storage =new T[s],
        M5_storage =new T[s],
        M6_storage =new T[s],
        M7_storage =new T[s],
        A_result1_storage =new T[s2],
        A_result2_storage =new T[s2],
        A_result3_storage =new T[s2],
        A_result4_storage =new T[s2],
        A_result5_storage =new T[s2],
        B_result1_storage =new T[s3];
        B_result2_storage =new T[s3];
        B_result3_storage =new T[s3];
        B_result4_storage =new T[s3];
        B_result5_storage =new T[s3];

    }

    mdspan<T, CC>   M1(M1_storage, s, {half_n, half_p}, {half_p, 1}),
           M2(M2_storage, s, {half_n, half_p}, {half_p, 1}),
           M3(M3_storage, s, {half_n, half_p}, {half_p, 1}),
           M4(M4_storage, s, {half_n, half_p}, {half_p, 1}),
           M5(M5_storage, s, {half_n, half_p}, {half_p, 1}),
           M6(M6_storage, s, {half_n, half_p}, {half_p, 1}),
           M7(M7_storage, s, {half_n, half_p}, {half_p, 1});

    mdspan<T, CA> A_result1(A_result1_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result2(A_result2_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result3(A_result3_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result4(A_result4_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result5(A_result5_storage, s2, {half_n, half_m}, {half_m, 1});

    mdspan<T, CB> B_result1(B_result1_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result2(B_result2_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result3(B_result3_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result4(B_result4_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result5(B_result5_storage, s3, {half_m, half_p}, {half_p, 1});


    if (algorithm.omp)
    {
        #pragma omp parallel shared (A11,A22,A21,A12,B12,B21,B11,B22,\
        A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5)
        {
            matrix_add(A11, A22, A_result1);
            matrix_add(B11, B22, B_result1);
            matrix_add(A21, A22, A_result2);
            matrix_subtract(B12, B22, B_result2);
            matrix_subtract(B21, B11, B_result3);
            matrix_add(A11, A12, A_result3);
            matrix_subtract(A21, A11, A_result4);
            matrix_add(B11, B12, B_result4);
            matrix_subtract(A12, A22, A_result5);
            matrix_add(B21, B22, B_result5);
        }
    }
    else
    {
        matrix_add(A11, A22, A_result1);
        matrix_add(B11, B22, B_result1);
        matrix_add(A21, A22, A_result2);
        matrix_subtract(B12, B22, B_result2);
        matrix_subtract(B21, B11, B_result3);
        matrix_add(A11, A12, A_result3);
        matrix_subtract(A21, A11, A_result4);
        matrix_add(B11, B12, B_result4);
        matrix_subtract(A12, A22, A_result5);
        matrix_add(B21, B22, B_result5);
    }

    if (algorithm.mpi==true && n*p>=algorithm.size_for_mpi)
    {
        int myrank;
        MPI_Comm_rank(algorithm.comm, &myrank);

        int childdest=myrank*7;
        int commsize;
        MPI_Comm_size(algorithm.comm, &commsize);


        if (childdest+7<commsize)
        {
            int message=COMMAND_STRASSEN;
            MPI_Send(&message, 1, MPI_INT, childdest+1,0, algorithm.comm);
            A_result1.MPI_Send_mdspan(childdest+1,1,algorithm.comm);
            B_result1.MPI_Send_mdspan(childdest+1,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+2, 0, algorithm.comm);
            A_result2.MPI_Send_mdspan(childdest+2,1,algorithm.comm);
            B11.MPI_Send_mdspan(childdest+2,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+3, 0, algorithm.comm);
            A11.MPI_Send_mdspan(childdest+3,1,algorithm.comm);
            B_result2.MPI_Send_mdspan(childdest+3,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+4, 0, algorithm.comm);
            A22.MPI_Send_mdspan(childdest+4,1,algorithm.comm);
            B_result3.MPI_Send_mdspan(childdest+4,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+5, 0, algorithm.comm);
            A_result3.MPI_Send_mdspan(childdest+5,1,algorithm.comm);
            B22.MPI_Send_mdspan(childdest+5,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+6, 0, algorithm.comm);
            A_result4.MPI_Send_mdspan(childdest+6,1,algorithm.comm);
            B_result4.MPI_Send_mdspan(childdest+6,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+7, 0, algorithm.comm);
            A_result5.MPI_Send_mdspan(childdest+7,1,algorithm.comm);
            B_result5.MPI_Send_mdspan(childdest+7,2,algorithm.comm);

            M1.MPI_Recv_mdspan_pdata(childdest+1,3,algorithm.comm);
            M2.MPI_Recv_mdspan_pdata(childdest+2,3,algorithm.comm);
            M3.MPI_Recv_mdspan_pdata(childdest+3,3,algorithm.comm);
            M4.MPI_Recv_mdspan_pdata(childdest+4,3,algorithm.comm);
            M5.MPI_Recv_mdspan_pdata(childdest+5,3,algorithm.comm);
            M6.MPI_Recv_mdspan_pdata(childdest+6,3,algorithm.comm);
            M7.MPI_Recv_mdspan_pdata(childdest+7,3,algorithm.comm);


        }
        else
        {
            matrix_multiplication_parameters alg2=algorithm;
            alg2.mpi=false;
            if(algorithm.omp)
            {

                #pragma omp parallel shared(A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
                A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5,alg2)
                {
                    strassen_multiply(A_result1, B_result1, M1, alg2);
                    strassen_multiply(A_result2, B11, M2, alg2);
                    strassen_multiply(A11, B_result2, M3, alg2);
                    strassen_multiply(A22, B_result3, M4, alg2);
                    strassen_multiply(A_result3, B22, M5,alg2);
                    strassen_multiply(A_result4, B_result4, M6,alg2);
                    strassen_multiply(A_result5, B_result5, M7, alg2);
                }
            }
            else
            {
                strassen_multiply(A_result1, B_result1, M1, alg2);
                strassen_multiply(A_result2, B11, M2, alg2);
                strassen_multiply(A11, B_result2, M3, alg2);
                strassen_multiply(A22, B_result3, M4, alg2);
                strassen_multiply(A_result3, B22, M5,alg2);
                strassen_multiply(A_result4, B_result4, M6, alg2);
                strassen_multiply(A_result5, B_result5, M7, alg2);
            }
        }

    }
    else
    {
        if(algorithm.omp &&!algorithm.gpu_offload )
        {
            #pragma omp parallel shared(A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5,algorithm)
            {
                strassen_multiply(A_result1, B_result1, M1, algorithm);
                strassen_multiply(A_result2, B11, M2, algorithm);
                strassen_multiply(A11, B_result2, M3, algorithm);
                strassen_multiply(A22, B_result3, M4, algorithm);
                strassen_multiply(A_result3, B22, M5,algorithm);
                strassen_multiply(A_result4, B_result4, M6,algorithm);
                strassen_multiply(A_result5, B_result5, M7, algorithm);
            }
        }
        else
        {
            strassen_multiply(A_result1, B_result1, M1, algorithm);
            strassen_multiply(A_result2, B11, M2, algorithm);
            strassen_multiply(A11, B_result2, M3, algorithm);
            strassen_multiply(A22, B_result3, M4, algorithm);
            strassen_multiply(A_result3, B22, M5,algorithm);
            strassen_multiply(A_result4, B_result4, M6,algorithm);
            strassen_multiply(A_result5, B_result5, M7, algorithm);
        }
    }

    // Submatrices of C
    mdspan<T, CC> C11 = C.subspan({0, 0}, {half_n, half_p});
    mdspan<T, CC> C12 = C.subspan({0, half_p}, {half_n, half_p});
    mdspan<T, CC> C21 = C.subspan({half_n, 0}, {half_n, half_p});
    mdspan<T, CC> C22 = C.subspan({half_n, half_p}, {half_n, half_p});

    if (algorithm.omp)
    {
        #pragma omp parallel for collapse(2) shared(C11,C12,C21,C22,M1,M2,M3,M4,M5,M6,M7)
        for (size_t i = 0; i < half_n; i++)
            for (size_t j = 0; j < half_p; j++)
            {
                C11(i, j) =M1(i,j)+M4(i,j)-M5(i,j)+M7(i,j);
                C12(i, j) = M3(i, j) + M5(i, j);
                C21(i, j) = M2(i, j) + M4(i, j);
                C22(i, j) = M1(i, j) - M2(i, j) + M3(i, j) + M6(i, j);
            }
    }
    else
    {
        for (size_t i = 0; i < half_n; i++)
            for (size_t j = 0; j < half_p; j++)
            {
                C11(i, j) =M1(i,j)+M4(i,j)-M5(i,j)+M7(i,j);
                C12(i, j) = M3(i, j) + M5(i, j);
                C21(i, j) = M2(i, j) + M4(i, j);
                C22(i, j) = M1(i, j) - M2(i, j) + M3(i, j) + M6(i, j);
            }

    }



    if (algorithm.memmapped_files)
    {
        delete_temp_mmap<T>(M1_storage, s);
        delete_temp_mmap<T>(M2_storage, s);
        delete_temp_mmap<T>(M3_storage, s);
        delete_temp_mmap<T>(M4_storage, s);
        delete_temp_mmap<T>(M5_storage, s);
        delete_temp_mmap<T>(M6_storage, s);
        delete_temp_mmap<T>(M7_storage, s);
        delete_temp_mmap<T>(A_result1_storage, s2);
        delete_temp_mmap<T>(A_result2_storage, s2);
        delete_temp_mmap<T>(A_result3_storage, s2);
        delete_temp_mmap<T>(A_result4_storage, s2);
        delete_temp_mmap<T>(A_result5_storage, s2);
        delete_temp_mmap<T>(B_result1_storage, s3);
        delete_temp_mmap<T>(B_result2_storage, s3);
        delete_temp_mmap<T>(B_result3_storage, s3);
        delete_temp_mmap<T>(B_result4_storage, s3);
        delete_temp_mmap<T>(B_result5_storage, s3);
    }

    else
    {
        delete []M1_storage;
        delete []M2_storage;
        delete []M3_storage;
        delete []M4_storage;
        delete []M5_storage;
        delete []M6_storage;
        delete []M7_storage;
        delete[]A_result1_storage;
        delete[]A_result2_storage;
        delete[]A_result3_storage;
        delete[]A_result4_storage;
        delete[]A_result5_storage;
        delete[]B_result1_storage;
        delete[]B_result2_storage;
        delete[]B_result3_storage;
        delete[]B_result4_storage;
        delete[]B_result5_storage;
    }

    return true;
}

template <typename T, typename CA, typename CB, typename CC>
bool winograd_multiply(  mdspan<T, CA>& A,  mdspan<T, CB>& B, mdspan<T, CC>& C,matrix_multiplication_parameters& algorithm)
{
    // Dimensions of input matrices
    size_t n = A.extent(0); // Rows in A
    size_t m = A.extent(1); // Columns in A and rows in B
    size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (n*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,algorithm.gpu_offload,algorithm.default_device,algorithm.devicenum);
        return true;
    }
    // Compute sizes for splitting
    size_t half_n = n / 2;
    size_t half_m = m / 2;
    size_t half_p = p / 2;

    // Submatrices of A
    auto A11 = A.subspan({0, 0}, {half_n, half_m});
    auto A12 = A.subspan({0, half_m}, {half_n, half_m});
    auto A21 = A.subspan({half_n, 0}, {half_n, half_m});
    auto A22 = A.subspan({half_n, half_m}, {half_n, half_m});

    // Submatrices of B
    auto B11 = B.subspan({0, 0}, {half_m, half_p});
    auto B12 = B.subspan({0, half_p}, {half_m, half_p});
    auto B21 = B.subspan({half_m, 0}, {half_m, half_p});
    auto B22 = B.subspan({half_m, half_p}, {half_m, half_p});

    // Temporary storage for intermediate results
    size_t s=half_n*half_p;
    size_t s2=half_n*half_m;
    size_t s3=half_m*half_p;

    T* __restrict M1_storage;
    T* __restrict M2_storage;
    T* __restrict M3_storage;
    T* __restrict M4_storage;
    T* __restrict M5_storage;
    T* __restrict M6_storage;
    T* __restrict M7_storage;
    T* __restrict S1_result_storage;
    T* __restrict S2_result_storage;
    T* __restrict S3_result_storage;
    T* __restrict S4_result_storage;
    T* __restrict S5_result_storage;
    T* __restrict S6_result_storage;
    T* __restrict S7_result_storage;
    T* __restrict S8_result_storage;
    T* __restrict T1_result_storage;
    T* __restrict T2_result_storage;
    if (algorithm.memmapped_files)
    {
        M1_storage=create_temp_mmap<T>(s);
        M2_storage=create_temp_mmap<T>(s);
        M3_storage=create_temp_mmap<T>(s);
        M4_storage=create_temp_mmap<T>(s);
        M5_storage=create_temp_mmap<T>(s);
        M6_storage=create_temp_mmap<T>(s);
        M7_storage=create_temp_mmap<T>(s);
        S1_result_storage=create_temp_mmap<T>(s2);
        S2_result_storage=create_temp_mmap<T>(s2);
        S3_result_storage=create_temp_mmap<T>(s2);
        S4_result_storage=create_temp_mmap<T>(s2);
        S5_result_storage=create_temp_mmap<T>(s3);
        S6_result_storage=create_temp_mmap<T>(s3);
        S7_result_storage=create_temp_mmap<T>(s3);
        S8_result_storage=create_temp_mmap<T>(s3);
        T1_result_storage=create_temp_mmap<T>(s);
        T2_result_storage=create_temp_mmap<T>(s);
    }
    else
    {
        M1_storage=new T[s];
        M2_storage=new T[s];
        M3_storage=new T[s];
        M4_storage=new T[s];
        M5_storage=new T[s];
        M6_storage=new T[s];
        M7_storage=new T[s];
        S1_result_storage=new T[s2];
        S2_result_storage=new T[s2];
        S3_result_storage=new T[s2];
        S4_result_storage=new T[s2];
        S5_result_storage=new T[s3];
        S6_result_storage=new T[s3];
        S7_result_storage=new T[s3];
        S8_result_storage=new T[s3];
        T1_result_storage=new T[s];
        T2_result_storage=new T[s];
    }

    mdspan<T, CC> M1(M1_storage, s, {half_n, half_p}, {half_p, 1}),
           M2(M2_storage, s, {half_n, half_p}, {half_p, 1}),
           M3(M3_storage, s, {half_n, half_p}, {half_p, 1}),
           M4(M4_storage, s, {half_n, half_p}, {half_p, 1}),
           M5(M5_storage, s, {half_n, half_p}, {half_p, 1}),
           M6(M6_storage, s, {half_n, half_p}, {half_p, 1}),
           M7(M7_storage, s, {half_n, half_p}, {half_p, 1});

    mdspan<T, CA> S1(S1_result_storage, s2, {half_n, half_m}, {half_m, 1}),
           S2(S2_result_storage, s2, {half_n, half_m}, {half_m, 1}),
           S3(S3_result_storage,s2, {half_n, half_m}, {half_m, 1}),
           S4(S4_result_storage, s2, {half_n, half_m}, {half_m, 1});

    mdspan<T, CB> S5(S5_result_storage, s3, {half_m, half_p}, {half_p, 1}),
           S6(S6_result_storage,s3, {half_m, half_p}, {half_p, 1}),
           S7(S7_result_storage, s3, {half_m, half_p}, {half_p, 1}),
           S8(S8_result_storage, s3, {half_m, half_p}, {half_p, 1});

    if (algorithm.omp)
    {
        #pragma omp parallel shared(A11,A21,A12,A22,B11,B12,B22,B21,S1,S2,S3,S4,S5,S6,S7,S8)
        {
            #pragma omp single
            {
                matrix_add(A21, A22, S1);
                matrix_subtract(S1, A11, S2);
                matrix_subtract(A12, S2, S4);
            }
            #pragma omp single
            {
                matrix_subtract(A11, A21, S3);
                matrix_subtract(B22, B12, S7);
            }
            #pragma omp single
            {
                matrix_subtract(B12, B11, S5);
                matrix_subtract(B22, S5, S6);
                matrix_subtract(S6, B21, S8);
            }
        }
    }
    else
    {
        matrix_add(A21, A22, S1);
        matrix_subtract(S1, A11, S2);
        matrix_subtract(A12, S2, S4);
        matrix_subtract(A11, A21, S3);
        matrix_subtract(B22, B12, S7);
        matrix_subtract(B12, B11, S5);
        matrix_subtract(B22, S5, S6);
    }


    if (algorithm.mpi==true && n*p>=algorithm.size_for_mpi)
    {
        int myrank;
        MPI_Comm_rank(algorithm.comm, &myrank);

        int childdest=myrank*7;
        int commsize;
        MPI_Comm_size(algorithm.comm, &commsize);


        if (childdest+7<commsize )
        {

            int message=COMMAND_WINOGRAD;
            MPI_Send(&message, 1, MPI_INT, childdest+1, 0, algorithm.comm);
            S2.MPI_Send_mdspan(childdest+1,1,algorithm.comm);
            S6.MPI_Send_mdspan(childdest+1,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+2, 0, algorithm.comm);
            A11.MPI_Send_mdspan(childdest+2,1,algorithm.comm);
            B11.MPI_Send_mdspan(childdest+2,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+3, 0, algorithm.comm);
            A12.MPI_Send_mdspan(childdest+3,1,algorithm.comm);
            B21.MPI_Send_mdspan(childdest+3,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+4, 0, algorithm.comm);
            S3.MPI_Send_mdspan(childdest+4,1,algorithm.comm);
            S7.MPI_Send_mdspan(childdest+4,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+5, 0, algorithm.comm);
            S1.MPI_Send_mdspan(childdest+5,1,algorithm.comm);
            S5.MPI_Send_mdspan(childdest+5,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+6, 0, algorithm.comm);
            S4.MPI_Send_mdspan(childdest+6,1,algorithm.comm);
            B22.MPI_Send_mdspan(childdest+6,2,algorithm.comm);


            MPI_Send(&message, 1, MPI_INT, childdest+7, 0, algorithm.comm);
            A22.MPI_Send_mdspan(childdest+7,1,algorithm.comm);
            S8.MPI_Send_mdspan(childdest+7,2,algorithm.comm);

            M1.MPI_Recv_mdspan_pdata(childdest+1,3,algorithm.comm);
            M2.MPI_Recv_mdspan_pdata(childdest+2,3,algorithm.comm);
            M3.MPI_Recv_mdspan_pdata(childdest+3,3,algorithm.comm);
            M4.MPI_Recv_mdspan_pdata(childdest+4,3,algorithm.comm);
            M5.MPI_Recv_mdspan_pdata(childdest+5,3,algorithm.comm);
            M6.MPI_Recv_mdspan_pdata(childdest+6,3,algorithm.comm);
            M7.MPI_Recv_mdspan_pdata(childdest+7,3,algorithm.comm);
        }
        else
        {
            matrix_multiplication_parameters alg2=algorithm;
            alg2.mpi=false;
            if(algorithm.omp &&!algorithm.gpu_offload)
            {
                #pragma omp parallel shared(S1,S2,S3,S4,S5,S6,S7,S8,A11,A12,B11,B21,A22,B22,M1,M2,M3,M4,M5,M6,M7,alg2)
                {
                    winograd_multiply(S2,S6,M1,alg2);
                    winograd_multiply(A11,B11,M2,alg2);
                    winograd_multiply(A12,B21,M3,alg2);
                    winograd_multiply(S3,S7,M4,alg2);
                    winograd_multiply(S1,S5,M5,alg2);
                    winograd_multiply(S4,B22,M6,alg2);
                    winograd_multiply(A22,S8,M7,alg2);
                }
            }
            else
            {
                winograd_multiply(S2,S6,M1,alg2);
                winograd_multiply(A11,B11,M2,alg2);
                winograd_multiply(A12,B21,M3,alg2);
                winograd_multiply(S3,S7,M4,alg2);
                winograd_multiply(S1,S5,M5,alg2);
                winograd_multiply(S4,B22,M6,alg2);
                winograd_multiply(A22,S8,M7,alg2);
            }
        }
    }
    else
    {
        if(algorithm.omp)
        {
            #pragma omp parallel shared(S1,S2,S3,S4,S5,S6,S7,S8,A11,A12,B11,B21,A22,B22,M1,M2,M3,M4,M5,M6,M7,algorithm)
            {
                winograd_multiply(S2,S6,M1,algorithm);
                winograd_multiply(A11,B11,M2,algorithm);
                winograd_multiply(A12,B21,M3,algorithm);
                winograd_multiply(S3,S7,M4,algorithm);
                winograd_multiply(S1,S5,M5,algorithm);
                winograd_multiply(S4,B22,M6,algorithm);
                winograd_multiply(A22,S8,M7,algorithm);
            }
        }
        else
        {
            winograd_multiply(S2,S6,M1,algorithm);
            winograd_multiply(A11,B11,M2,algorithm);
            winograd_multiply(A12,B21,M3,algorithm);
            winograd_multiply(S3,S7,M4,algorithm);
            winograd_multiply(S1,S5,M5,algorithm);
            winograd_multiply(S4,B22,M6,algorithm);
            winograd_multiply(A22,S8,M7,algorithm);
        }
    }


    mdspan<T, CB> T1(T1_result_storage, s, {half_n, half_p}, {half_p, 1});
    mdspan<T, CB> T2(T2_result_storage, s, {half_n, half_p}, {half_p, 1});

    matrix_add(M1, M2, T1);
    matrix_add(T1, M4, T2);

    mdspan<T, CC> C11 = C.subspan({0, 0}, {half_n, half_p});
    mdspan<T, CC> C12 = C.subspan({0, half_p}, {half_n, half_p});
    mdspan<T, CC> C21 = C.subspan({half_n, 0}, {half_n, half_p});
    mdspan<T, CC> C22 = C.subspan({half_n, half_p}, {half_n, half_p});

    #pragma omp parallel for collapse(2) shared(M2,M3,M5,M6,M7,T1,T2,C11,C12,C21,C22)
    for (size_t i = 0; i < half_n; ++i)
    {
        for (size_t j = 0; j < half_p; ++j)
        {
            C11(i, j) = M2(i, j) + M3(i,j);
            C12(i, j) = T1(i, j) + M5(i,j)+M6(i,j);
            C21(i, j) = T2(i, j) - M7(i, j);
            C22(i, j) = T2(i, j) + M5(i, j);
        }
    }


    if (algorithm.memmapped_files)
    {
        delete_temp_mmap<T>(M1_storage, s);
        delete_temp_mmap<T>(M2_storage, s);
        delete_temp_mmap<T>(M3_storage, s);
        delete_temp_mmap<T>(M4_storage, s);
        delete_temp_mmap<T>(M5_storage, s);
        delete_temp_mmap<T>(M6_storage, s);
        delete_temp_mmap<T>(M7_storage, s);
        delete_temp_mmap<T>(S1_result_storage, s2);
        delete_temp_mmap<T>(S2_result_storage, s2);
        delete_temp_mmap<T>(S3_result_storage, s2);
        delete_temp_mmap<T>(S4_result_storage, s2);
        delete_temp_mmap<T>(S5_result_storage, s3);
        delete_temp_mmap<T>(S6_result_storage, s3);
        delete_temp_mmap<T>(S7_result_storage, s3);
        delete_temp_mmap<T>(S8_result_storage, s3);
        delete_temp_mmap<T>(T1_result_storage, s);
        delete_temp_mmap<T>(T2_result_storage, s);

    }
    else
    {
        delete []M1_storage;
        delete []M2_storage;
        delete []M3_storage;
        delete []M4_storage;
        delete []M5_storage;
        delete []M6_storage;
        delete []M7_storage;
        delete[]S3_result_storage;
        delete[]S7_result_storage;
        delete[]S2_result_storage;
        delete[]S6_result_storage;
        delete[]S1_result_storage;
        delete[]S5_result_storage;
        delete[]S4_result_storage;
        delete[]S8_result_storage;
        delete[]T1_result_storage;
        delete[]T2_result_storage;
    }
    return true;
}



template <typename T, typename CA>
void cholesky_decomposition(const mdspan<T, CA>& A, mdspan<T, CA>& L, matrix_multiplication_parameters algorithm, size_t step_size=0,  int gpu_offload=false,bool default_device=true,int devicenum=0)
{

    if(default_device)
        devicenum=omp_get_default_device();

    if (gpu_offload==true)
    {
        bool Loffloaded=L.is_offloaded(devicenum);

        datastruct<T> dA=A.get_datastruct();
        datastruct<T> dL=L.get_datastruct();

        const size_t n = dA.dpextents[0];

        const size_t nn=n*n;

        create_in_struct(dA,devicenum);
        create_in_struct(dL,devicenum);

        #pragma omp target teams distribute parallel for simd device(devicenum)
        for (size_t i=0; i<nn; i++)
        {
            dL.dpdata[i]=0;
        }



        for (size_t c = 0; c < n; ++c)
        {
            T tmp=0;
            #pragma omp target map(tofrom: tmp) device(devicenum)
            {
                tmp=dA(c, c);
            }

            #pragma omp target teams distribute parallel for simd reduction(-:tmp) shared(dL,c)device(devicenum)
            for (size_t k = 0; k < c; ++k)
            {
                const T tmp3=dL(c,k);
                tmp-= tmp3 * tmp3;
            }

            T tmp4=sqrt(tmp);
            #pragma omp target map (tofrom:tmp4)device(devicenum)
            {
                dL(c, c) =tmp4;
            }

            #pragma omp target teams distribute parallel for shared(dA,dL,c,tmp4)device(devicenum)
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = dA(i, c);
                #pragma omp simd reduction(-:tmp2)
                for (size_t k = 0; k < c; ++k)
                {
                    tmp2 -= dL(i, k) * dL(c, k);
                }
                dL(i, c)=tmp2/tmp4;
            }
        }

        if(!Loffloaded) update_host(dL,devicenum);

        release_struct(dA,devicenum);
        release_struct(dL,devicenum);

    }
    else
    {
        if(step_size==0)
            step_size=(size_t)pow(A.extent(0),0.8385);
        size_t n = A.extent(0);
        size_t nn=n*n;
        size_t tempsize=(n-step_size)*(n-step_size);

        T *__restrict sdata;
        T *__restrict adata;

        if(algorithm.memmapped_files)
        {
            sdata=create_temp_mmap<T>(tempsize);
            adata=create_temp_mmap<T>(nn);
        }
        else
        {
            sdata=(T*) omp_alloc(sizeof(T)*tempsize,omp_large_cap_mem_alloc);
            adata=(T*) omp_alloc(sizeof(T*)*nn,omp_large_cap_mem_alloc);
        }

        datastruct<T>dA=A.get_datastruct();
        datastruct<T>dL=L.get_datastruct();

#if (Unified_Shared_Memory==True)
        #pragma omp target teams distribute parallel for simd shared(dA,dL)device(devicenum)
#else
        #pragma omp parallel for simd shared(dA,dL)
#endif

        for (size_t i=0; i<nn; i++)
        {
            adata[i]=dA.dpdata[i];
            dL.dpdata[i]=0;
        }


        mdspan<T, CA> tempA(adata,dA.dprowmajor, n,n);


        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                // Extract submatrix R = L[c:n, z:c-1]

                auto R = L.subspanmatrix(c, z,u, c - z);

                // Compute S = RR^T using a fast matrix multiplication algorithm
                mdspan<T, CA> S(sdata,R.get_datastruct().dprowmajor, u,u);
                mdspan<T, CA> RT=R.transpose();

                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(R,RT,S,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(R,RT,S,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(R,RT,S,algorithm);
                }


#if (Unified_Shared_Memory==True)
                #pragma omp target teams distribute parallel for  collapse (2) shared(c,tempA,S)device(devicenum)
#else
                #pragma omp parallel for simd  collapse (2) shared(c,tempA,S)
#endif
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i, j) -= S(i - c, j - c);
                    }
                }
                // Update the block boundary
                z = c;
            }

            // Update the diagonal element L[c, c]
            T tmp=0;

            tmp=tempA(c, c);


#if (Unified_Shared_Memory==True)
            #pragma omp target teams distribute parallel for simd reduction(-: tmp) shared(c,L)device(devicenum)
#else
            #pragma omp parallel for simd reduction(-: tmp) shared(c,L)
#endif
            for (size_t k = z; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp-= tmp3 * tmp3;
            }
            L(c, c) =sqrt(tmp);

#if (Unified_Shared_Memory==True)
            #pragma omp target teams distribute shared(c, tempA,L)device(devicenum)
#else
            #pragma omp parallel for shared(c,tempA,L )
#endif
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = tempA(i, c);
                T tmp4=L(c, c);
#if (Unified_Shared_Memory==True)
                #pragma omp parallel for simd reduction(-:tmp2) shared(c,L)
#else
                #pragma omp simd reduction(-:tmp2)
#endif
                for (size_t k = z; k < c; ++k)
                {
                    tmp2 -= L(i, k) * L(c, k);
                }
                L(i, c)=tmp2/tmp4;
            }
        }
        if(algorithm.memmapped_files)
        {
            delete_temp_mmap<T>(sdata,tempsize);
            delete_temp_mmap<T>(adata,nn);
        }
        else
        {
            omp_free(sdata,omp_large_cap_mem_alloc);
            omp_free(adata,omp_large_cap_mem_alloc);
        }
    }
}

template <typename T, typename CA>
void lu_decomposition(mdspan<T, CA>& A, mdspan<T, CA>& L, mdspan<T, CA>& U, matrix_multiplication_parameters &algorithm,  size_t step_size=0,
                      bool gpu_offload=false,bool default_device=true,int devicenum=0)
{
    if(default_device)
        devicenum=omp_get_default_device();
    if (gpu_offload==true)
    {
        datastruct<T> dA=A.get_datastruct();
        datastruct<T> dL=L.get_datastruct();
        datastruct<T> dU=U.get_datastruct();
        bool Loffloaded=L.is_offloaded(devicenum);
        bool Uoffloaded=U.is_offloaded(devicenum);

        create_in_struct(dA,devicenum);
        create_out_struct(dL,devicenum);
        create_out_struct(dU,devicenum);
        size_t n = dA.dpextents[0];
        size_t nn=n*n;

        #pragma omp target teams distribute parallel for simd device(devicenum)
        for (size_t i=0; i<nn; i++)
        {
            dL.dpdata[i]=0;
            dU.dpdata[i]=0;
        }




        size_t z=0;
        for (size_t c = 0; c < n; ++c)
        {
            #pragma omp target teams distribute shared(dA,c,dL,dU)device(devicenum)
            for (size_t i = c; i < n; ++i)
            {
                T temp=dA(c,i);
                #pragma omp parallel for simd reduction(-:temp) shared(c,dL,dU)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= dU( k,i) * dL( c,k);
                }
                dU(c,i)=temp;
            }

            #pragma omp target teams distribute shared(dA,c,dL,dU)device(devicenum)
            for (size_t i = c; i < n; ++i)
            {
                const T temp4=dU(c,c);
                T temp = dA(i,c);
                #pragma omp parallel for simd reduction (-:temp)shared(dU,dL,c)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= dU(k,c) * dL( i,k);
                }
                dL(i,c)=temp/temp4;
            }
        }

        if (!Loffloaded)
            update_host(dL,devicenum);
        if(!Uoffloaded)
            update_host(dU,devicenum);

        release_struct(dL,devicenum);
        release_struct(dU,devicenum);
        release_struct(dA,devicenum);
    }
    else
    {
        if(step_size==0)
            step_size=(size_t)pow(A.extent(0),0.8385);
        size_t n = A.extent(0);
        size_t tempsize=(n-step_size)*(n-step_size);
        size_t nn=n*n;
        T * __restrict sdata;
        T * __restrict adata;
        if(algorithm.memmapped_files)
        {
            sdata=create_temp_mmap<T>(tempsize);
            adata=create_temp_mmap<T>(nn);
        }
        else
        {
            sdata=(T*)omp_alloc(sizeof(T)*tempsize,omp_large_cap_mem_alloc);
            adata=(T*)omp_alloc(sizeof(T)*nn,omp_large_cap_mem_alloc);
        }
        datastruct<T> dA=A.get_datastruct();
        datastruct<T> dL=L.get_datastruct();
        datastruct<T> dU=U.get_datastruct();
#if (Unified_Shared_Memory==True)
        #pragma omp target teams distribute parallel for simd shared(adata,A,L,U)device(devicenum)
#else
        #pragma omp parallel for simd  shared(adata,A,L,U)
#endif
        for (size_t i=0; i<nn; i++)
        {
            adata[i]=dA.dpdata[i];
            dL.dpdata[i]=0;
            dU.dpdata[i]=0;
        }
        mdspan<T, CA> tempA(adata,nn,dA.dprowmajor, {dA.dpextents[0],dA.dpextents[1]}, {dA.dpstrides[0], dA.dpstrides[1]});

        size_t z=0;
        #pragma omp ordered
        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;
                auto RL = L.subspanmatrix(c, z, u,v);
                auto RU = U.subspanmatrix(z, c, v, u);
                mdspan<T, CA> S(sdata,RU.get_datastruct().dprowmajor, u,u);
                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(RL,RU,S,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(RL,RU,S,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(RL,RU,S,algorithm);
                }


#if (Unified_Shared_Memory==True)
                #pragma omp target teams distribute parallel for collapse(2) shared(tempA,S )device(devicenum)
#else
                #pragma omp parallel for collapse(2) shared(tempA,S)
#endif
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i,j) -= S(i - c, j - c);
                    }
                }
                z = c;
            }

#if (Unified_Shared_Memory==True)
            #pragma omp target teams distribute shared(tempA,L,U,c)device(devicenum)
#else
            #pragma omp parallel for shared(tempA,L,U,c)
#endif
            for (size_t i = c; i < n; ++i)
            {
                T temp=tempA(c,i);
#if (Unified_Shared_Memory==True)
                #pragma omp parallel for simd reduction(-:temp) shared(L,U,c)
#else
                #pragma omp simd reduction(-:temp)
#endif
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U( k,i) * L( c,k);
                }
                U(c,i)=temp;
            }

            const T temp4=U(c,c);

#if (Unified_Shared_Memory==True)
            #pragma omp target teams distribute shared(tempA,L,U,c)device(devicenum)
#else
            #pragma omp parallel for shared(tempA,L,U,c)
#endif
            for (size_t i = c; i < n; ++i)
            {
                T temp = tempA(i,c);
#if (Unified_Shared_Memory==True)
                #pragma omp parallel for simd reduction(-:temp) shared(L,U,c)
#else
                #pragma omp simd reduction(-:temp)
#endif
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U(k,c) * L( i,k);
                }
                L(i,c)=temp/temp4;
            }
        }


        if(algorithm.memmapped_files)
        {
            delete_temp_mmap<T>(sdata,tempsize);
            delete_temp_mmap<T>(adata,nn);
        }
        else
        {
            omp_free(sdata,omp_large_cap_mem_alloc);
            omp_free(adata,omp_large_cap_mem_alloc);
        }
    }
}
// Fast QR Decomposition Algorithm for mdspan
template <typename T, typename CA>
void qr_decomposition(mdspan<T, CA>& A, mdspan<T, CA>& Q, mdspan<T, CA>& R,   matrix_multiplication_parameters algorithm,  size_t step_size=0,
                      bool gpu_offload=false, bool default_device=true,int devicenum=0)
{
    if(default_device)
        devicenum=omp_get_default_device();
    if (gpu_offload==true)
    {
        bool Qoffloaded=Q.is_offloaded(devicenum);
        bool Roffloaded=R.is_offloaded(devicenum);

        datastruct<T> dA=A.get_datastruct();
        datastruct<T> dQ=Q.get_datastruct();
        datastruct<T> dR=R.get_datastruct();

        const size_t m = dA.dpextents[1];


        size_t extm[2]= {dA.dpextents[0],dA.dpextents[1]};
        size_t strm[2]= {dA.dpstrides[0],dA.dpstrides[1]};

#if (Unified_Shared_Memory==True)
        T* Md=(T*) malloc(sizeof(T)*dA.dpdatalength);
        datastruct<T> dM(Md,dA.dpdatalength,dA.dprowmajor,2,extm,strm,false,false,false);
#else
        T* Md=(T*) omp_target_alloc(sizeof(T)*dA.dpdatalength,devicenum);
        datastruct<T> dM(Md,dA.spdatalength,dA.sprowmajor,2,extm,strm,true,false,false);
#endif




        create_in_struct(dA,devicenum);

        create_out_struct(dQ,devicenum);
        create_out_struct(dR,devicenum);
        create_out_struct(dM,devicenum);

        #pragma omp target teams distribute parallel for simd shared(dQ) device(devicenum)
        for (size_t i=0; i<dQ.dpdatalength; i++)
        {
            dQ.dpdata[i]=0;
        }
        #pragma omp target teams distribute parallel for simd shared(dR) device(devicenum)
        for (size_t i=0; i<dR.dpdatalength; i++)
        {
            dR.dpdata[i]=0;
        }
        #pragma omp target teams distribute parallel for simd shared(dM) device(devicenum)
        for (size_t i=0; i<dA.dpdatalength; i++)
        {
            dM.dpdata[i]=dA.dpdata[i];
        }

        size_t z = 0;


        for (size_t c = 0; c < m; ++c)
        {
            size_t pextv[1];
            size_t pstrv[1];
            size_t pext0=dM.dpextents[0];


            for (size_t j = z; j < c; ++j)
            {
                size_t pextu[1];
                size_t pstru[1];
                T dot_pr=0;
                #pragma omp target teams distribute parallel for simd shared(dQ,dM,pextu,pstru,pextv,pstrv,pext0) reduction(+:dot_pr)device(devicenum)
                for (size_t i = 0; i < pext0; ++i)
                {
                    datastruct<T> du = dQ.column(j,pextu,pstru);
                    datastruct<T> dv = dM.column(c,pextv,pstrv);
                    dot_pr += du(i) * dv(i);
                }

                const T cdot_pr=dot_pr;
                #pragma omp target teams distribute  parallel for simd shared(dQ,dM,pextu,pstru,pextv,pstrv, pext0,cdot_pr)device(devicenum)
                for (size_t i = 0; i < pext0; ++i)
                {
                    datastruct<T> du = dQ.column(j,pextu,pstru);
                    datastruct<T>  dv = dM.column(c,pextv,pstrv);
                    dv(i) -= cdot_pr * du(i);
                }
            }
            // Normalize v
            T norm=0;
            #pragma omp target teams distribute  parallel for simd shared(dM,pextv,pstrv,pext0)reduction(+: norm)device(devicenum)
            for (size_t i = 0; i < pext0; ++i)
            {
                datastruct<T>  dv = dM.column(c,pextv,pstrv);
                norm += dv(i) * dv(i);
            }

            const T normc= sqrt(norm);
            #pragma omp target teams distribute  parallel for simd shared(dM,pextv,pstrv,pext0,normc) device(devicenum)
            for (size_t i = 0; i < pext0; ++i)
            {
                datastruct<T>  dv = dM.column(c,pextv,pstrv);

                dv(i)= dv(i)/normc;
            }

            #pragma omp target teams distribute  parallel for simd shared(dM,pext0,pextv,pstrv) device(devicenum)
            for (size_t i = 0; i < pext0; ++i)
            {
                datastruct<T>  dv = dM.column(c,pextv,pstrv);
                dQ(i,c) = dv(i);
            }
        }



        const size_t rows = dQ.dpextents[0]; // Number of rows in A and C
        const size_t cols = dA.dpextents[1]; // Number of columns in B and C
        const  size_t inner_dim = dQ.dpextents[1]; // Number of columns in A and rows in B




        #pragma omp target teams distribute collapse(2) shared(dQ,dA,dR,rows,cols, inner_dim)device(devicenum)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += dQ(k,i) *dA(k,j);
                }
                dR(i,j)= sum;
            }
        }
        release_struct(dM,devicenum);

        if(!Qoffloaded)
            update_host(dQ,devicenum);
        if(!Roffloaded)
            update_host(dR,devicenum);

        release_struct(dR,devicenum);
        release_struct(dQ,devicenum);

        release_struct(dA,devicenum);

#if (Unified_Shared_Memory==True)
        free(Md);
#else
        omp_target_free(Md,devicenum);
#endif




    }
    else
    {

        if(step_size==0)
            step_size=(size_t)pow(A.extent(0),0.8385);
        size_t n = A.extent(0); // Number of rows (assuming 2D matrix)
        size_t m = A.extent(1); // Number of columns

        // Initialize Q and R matrices
        size_t nm=n*m, mm=m*m;

        T* __restrict tempC;
        T *__restrict tempS;
        T *__restrict tempM;

        if(algorithm.memmapped_files)
        {
            tempC=create_temp_mmap<T>(mm);
            tempS=create_temp_mmap<T>(nm);
            tempM=create_temp_mmap<T>(nm);
        }
        else
        {
            tempC=(T*)omp_alloc(sizeof(T)*m*m,omp_large_cap_mem_alloc);
            tempS=(T*)omp_alloc(sizeof(T)*nm,omp_large_cap_mem_alloc);
            tempM=(T*)omp_alloc(sizeof(T)*nm,omp_large_cap_mem_alloc);
        }

        datastruct<T> dA=A.get_datastruct();
#if (Unified_Shared_Memory==True)
        #pragma omp target teams distribute parallel for simd  shared(tempM,A)device(devicenum)
#else
        #pragma omp parallel for simd  shared(tempM,A)
#endif
        for (size_t i=0; i<nm; i++)
        {
            tempM[i]=dA.dpdata[i];
        }

        datastruct<T> dQ=Q.get_datastruct();
#if (Unified_Shared_Memory==True)
        #pragma omp target teams distribute parallel for simd  shared(Q)device(devicenum)
#else
        #pragma omp parallel for simd  shared(Q)
#endif
        for (size_t i=0; i<dQ.dpdatalength; i++)
        {
            dQ.dpdata[i]=0;
        }
        datastruct<T> dR=R.get_datastruct();

#if (Unified_Shared_Memory==True)
        #pragma omp target teams distribute parallel for simd  shared(R)device(devicenum)
#else
        #pragma omp parallel for simd  shared(R)
#endif
        for (size_t i=0; i<dR.dpdatalength; i++)
        {
            dR.dpdata[i]=0;
        }
        mdspan<T, CA> M(tempM,A.get_datastruct().dpdatalength,A.get_datastruct().dprowmajor, {A.get_datastruct().dpextents[0],A.get_datastruct().dpextents[1]}, {A.get_datastruct().dpstrides[0],A.get_datastruct().dpstrides[1]}); // Copy of A

        size_t z = 0;
        #pragma omp ordered
        for (size_t c = 0; c < m; ++c)
        {
            if (c == z +step_size)
            {
                size_t cz=c-z;
                size_t mc=m-c;
                // Extract submatrices

                auto BQ = Q.subspanmatrix(0, z, n, cz);
                auto BM = M.subspanmatrix(0, c, n,mc);

                // Compute C = BQ^T * BM
                auto C = mdspan<T, CA>(tempC, BM.get_datastruct().dprowmajor,cz, mc);

                auto BQT=BQ.transpose();
                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(BQT,BM,C,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(BQT,BM,C,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(BQT,BM,C,algorithm);
                }


                // Compute S = BQ * C
                auto S = mdspan<T, CA>(tempS, BQ.get_datastruct().dprowmajor, n, mc);

                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(BQ,C,S,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(BQ,C,S,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(BQ,C,S,algorithm);
                }

#if (Unified_Shared_Memory==True)
                #pragma omp target teams distribute parallel for simd shared(M,S)device(devicenum)
#else
                #pragma omp parallel for  simd collapse(2) shared(M,S)
#endif
                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        M(i, j) -= S(i, j-c);
                    }
                }
                z = c;
            }
            // Extract column c of M
            auto v = M.column(c);



            for (size_t j = z; j < c; ++j)
            {
                const auto u = Q.column(j);
                const T dot_pr =dot_product(u,v);
#if (Unified_Shared_Memory==True)
                #pragma omp target teams distribute parallel for simd  shared(u,v,dot_pr)device(devicenum)
#else
                #pragma omp parallel for   simd shared(u,v,dot_pr)
#endif
                for (size_t i = 0; i < n; ++i)
                {
                    v(i) -= dot_pr * u(i);
                }
            }

            // Normalize v
            const T norm = sqrt(dot_product(v,v));
#if (Unified_Shared_Memory==True)
            #pragma omp target teams distribute parallel for simd shared(v,norm)device(devicenum)
#else
            #pragma omp parallel for   simd shared(v,norm)
#endif
            for (size_t i = 0; i < n; ++i)
            {
                v(i) /= norm;
            }

            // Set column c of Q

#if (Unified_Shared_Memory==True)
            #pragma omp target teams distribute parallel for simd shared(Q,v,c)device(devicenum)
#else
            #pragma omp parallel for simd shared(Q,v,c)
#endif
            for (size_t i = 0; i < n; ++i)
            {
                Q(i,c) = v(i);
            }
        }

        // Compute R = Q^T * A
        auto QT=Q.transpose();
        switch (algorithm.algorithm_version)
        {
        case Matrix_Multiplication_Algorithm::Naive:
            matrix_multiply_dot(QT,A,R,algorithm.gpu_offload);
            break;
        case Matrix_Multiplication_Algorithm::Strassen:
            strassen_multiply(QT,A,R,algorithm);
            break;
        case Matrix_Multiplication_Algorithm::WinogradVariant:
            winograd_multiply(QT,A,R,algorithm);
        }

        if(algorithm.memmapped_files)
        {
            delete_temp_mmap<T>(tempC,mm);
            delete_temp_mmap<T>(tempS,nm);
            delete_temp_mmap<T>(tempM,nm);
        }
        else
        {
            omp_free(tempC,omp_large_cap_mem_alloc);
            omp_free(tempS,omp_large_cap_mem_alloc);
            omp_free(tempM,omp_large_cap_mem_alloc);
        }
    }

}


template <typename T, typename CA, typename CB, typename CC>
bool matrix_multiply_dot( mdspan<T, CA>& A,   mdspan<T, CB>& B, mdspan<T, CC>& C, bool on_gpu=false,bool default_device=true,int devicenum=0)
{

    if (on_gpu==true)
    {
        if(default_device)
            devicenum=omp_get_default_device();

        datastruct<T> dA=A.get_datastruct();
        datastruct<T> dB=B.get_datastruct();
        datastruct<T> dC=C.get_datastruct();

        bool offloadA=false,offloadB=false,offloadC=false;

        if(A.is_offloaded(devicenum)==false)
        {
            offloadA=true;
            create_in_struct(dA,devicenum);
        }
        if(B.is_offloaded(devicenum)==false)
        {
            offloadB=true;
            create_in_struct(dB,devicenum);
        }
        if(C.is_offloaded(devicenum)==false)
        {
            offloadC=true;
            create_out_struct(dC,devicenum);
        }

        const size_t rows = dA.dpextents[0];
        const size_t cols = dB.dpextents[1];
        const  size_t inner_dim = dA.dpextents[1];

        #pragma omp target teams distribute parallel for collapse(2) shared(dA,dB,dC,inner_dim, rows, cols)device(devicenum)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum=0;
                #pragma omp simd reduction (+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum+=dA(i,k)*dB(k,j);
                }
                dC(i,j)=sum;
            }
        }

        if (offloadC)
        {
            update_host(dC,devicenum);
            release_struct(dC,devicenum);
        }
        if (offloadA)
            release_struct(dA,devicenum);
        if (offloadB)
            release_struct(dB,devicenum);
    }
    else
    {
        datastruct<T> dA=A.get_datastruct();
        datastruct<T> dB=B.get_datastruct();
        datastruct<T> dC=C.get_datastruct();

        const size_t rows = dA.dpextents[0];
        const size_t cols = dB.dpextents[1];
        const  size_t inner_dim = dA.dpextents[1];


        #pragma omp parallel for collapse(2) shared(dC,dA,dB,rows,cols,inner_dim)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp simd reduction (+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += dA(i, k) * dB(k, j);
                }
                dC(i, j) = sum;
            }
        }
    }
    return true;
}

template <typename T, typename CA, typename CB, typename CC>
void matrix_multiply_dot_v(const mdspan<T, CA>& A, const  mdspan<T, CB>& B, mdspan<T, CC>& C)
{
    datastruct<T> dA=A.get_datastruct();
    datastruct<T> dB=B.get_datastruct();
    datastruct<T> dC=C.get_datastruct();

    const size_t rows = dA.pextents[0]; // Number of rows in A and C
    const size_t cols = dB.pextents[1]; // Number of columns in B and C
    const  size_t inner_dim = dA.pextents[1]; // Number of columns in A and rows in B

    const size_t strA0=dA.pstrides[0];
    const size_t strA1=dA.pstrides[1];

    const size_t strB0=dB.pstrides[0];
    const size_t strB1=dB.pstrides[1];

    const size_t strC0=dC.pstrides[0];
    const size_t strC1=dC.pstrides[1];

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd reduction (+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += dA(i, k,strA0,strA1) * dB(k, j,strB0,strB1);
            }
            dC(i, j,strC0,strC1) = sum;
        }
    }
}





template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_add( const mdspan<T, CA>& A,const   mdspan<T, CB>& B, mdspan<T, CC>& C, bool on_gpu=false,bool default_device=true,int devicenum=0)
{
    datastruct<T> dA=A.get_datastruct();
    datastruct<T> dB=B.get_datastruct();
    datastruct<T> dC=C.get_datastruct();

    const size_t rows = dC.dpextents[0];
    const size_t cols = dC.dpextents[1];



    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (A.is_offloaded(devicenum)==false) return false;
        if (B.is_offloaded(devicenum)==false) return false;
        if (C.is_offloaded(devicenum)==false) return false;

        #pragma omp target teams distribute parallel for simd collapse(2) shared(dA,dB,dC) device(devicenum)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)=dA(i,j)+dB(i,j);
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2) shared(dA,dB,dC)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)=dA(i,j)+dB(i,j);
            }
        }

    }

    return true;
}


template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_add_v( const mdspan<T, CA>& A,const   mdspan<T, CB>& B, mdspan<T, CC>& C)
{
    datastruct<T> dA=A.get_datastruct();
    datastruct<T> dB=B.get_datastruct();
    datastruct<T> dC=C.get_datastruct();

    const size_t rows = dC.pextents[0];
    const size_t cols = dC.pextents[1];



    #pragma omp simd collapse(2)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            dC(i,j)=dA(i,j)+dB(i,j);
        }
    }

    return true;
}



template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_subtract( const mdspan<T, CA>& A,  const mdspan<T, CB>& B, mdspan<T, CC>& C, bool on_gpu=false,bool default_device=true,int devicenum=0)
{
    datastruct<T> dA=A.get_datastruct();
    datastruct<T> dB=B.get_datastruct();
    datastruct<T> dC=C.get_datastruct();

    const size_t rows = dC.dpextents[0];
    const size_t cols = dC.dpextents[1];



    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (A.is_offloaded(devicenum)==false) return false;
        if (B.is_offloaded(devicenum)==false) return false;
        if (C.is_offloaded(devicenum)==false) return false;

        #pragma omp target teams distribute parallel for simd shared(dA,dB,dC)device(devicenum)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)=dA(i,j)-dB(i,j);
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2) shared(dA,dB,dC)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)=dA(i,j)-dB(i,j);
            }
        }
    }

    return true;
}


template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_subtract_v( const mdspan<T, CA>& A,  const mdspan<T, CB>& B, mdspan<T, CC>& C)
{
    datastruct<T> dA=A.get_datastruct();
    datastruct<T> dB=B.get_datastruct();
    datastruct<T> dC=C.get_datastruct();

    const size_t rows = dC.pextents[0];
    const size_t cols = dC.pextents[1];



    #pragma omp simd collapse(2)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            dC(i,j)=dA(i,j)-dB(i,j);
        }
    }


    return true;
}



template <typename T, typename CA, typename CB, typename CC>
inline bool matrix_multiply_vector(const mdspan<T, CA>& M, const mdspan<T, CB>& V, mdspan<T, CC>& C, bool on_gpu=false,bool default_device=true,int devicenum=0)
{
    datastruct<T> dM=M.get_datastruct();
    datastruct<T> dV=V.get_datastruct();
    datastruct<T> dC=C.get_datastruct();
    const size_t rows = dC.pextents[0];
    const size_t cols = dC.pextents[1];



    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (M.is_offloaded(devicenum)==false) return false;
        if (V.is_offloaded(devicenum)==false) return false;
        if (C.is_offloaded(devicenum)==false) return false;

        #pragma omp target teams distribute parallel for simd collapse(2) shared(dC,dM,dV) device(devicenum)

        for (size_t i = 0; i < rows; ++i)
        {

            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)= dM(i, j)* dV(j);  // This works because i, k, j are row/col indices
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2) shared(dC,dM,dV)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)= dM(i, j) * dV(j);  // This works because i, k, j are row/col indices
            }
        }

    }
    return true;
}

template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_multiply_vector(const mdspan<T, CA>& M,const  T*V, mdspan<T, CC>& C)
{
    datastruct<T> dM=M.get_datastruct();
    datastruct<T> dC=C.get_datastruct();
    const size_t rows = dC.pextents[0];
    const size_t cols = dC.pextents[1];
    #pragma omp parallel for simd collapse(2) shared(dC,dM,V)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            dC(i,j)= dM(i, j) * V[i];
        }
    }
    return true;
}

template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_multiply_vector_v(const mdspan<T, CA>& M,const  T*V, mdspan<T, CC>& C)
{
    datastruct<T> dM=M.get_datastruct();
    datastruct<T> dC=C.get_datastruct();
    const size_t rows = dC.pextents[0];
    const size_t cols = dC.pextents[1];



    #pragma omp simd collapse(2)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            dC(i,j)= dM(i, j) * V[i];
        }
    }

    return true;
}


template <typename T, typename CA, typename CC>
inline  bool matrix_multiply_scalar(const mdspan<T, CA>& M, const T& V, mdspan<T, CC>& C, bool on_gpu=false,bool default_device=true,int devicenum=0)
{

    datastruct<T> dM=M.get_datastruct();
    datastruct<T> dV=V.get_datastruct();
    datastruct<T> dC=C.get_datastruct();

    const size_t rows = dC.pextents[0];
    const size_t cols = dC.pextents[1];




    // Perform matrix multiplication: C = A * B
    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (M.is_offloaded(devicenum)==false) return false;
        if (V.is_offloaded(devicenum)==false) return false;
        if (C.is_offloaded(devicenum)==false) return false;


        #pragma omp target teams distribute parallel for simd collapse(2) shared(dC,dM,V) device(devicenum)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)= dM(i,j)*dV;
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2) shared(dC,dM,V)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dC(i,j)= dM(i,j)*V;
            }
        }
    }

    return true;
}

template <typename T, typename Container>
inline  T dot_product(const  mdspan<T, Container>& vec1,const   mdspan<T, Container>& vec2, bool on_gpu=false,bool default_device=true,int devicenum=0)
{
    datastruct<T> dvec1=vec1.get_datastruct();
    datastruct<T> dvec2=vec2.get_datastruct();
    const size_t n = dvec1.dpextents[0];
    T result = 0;
    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (vec1.is_offloaded(devicenum)==false) return false;
        if (vec2.is_offloaded(devicenum)==false) return false;


        #pragma omp target teams distribute parallel for simd reduction(+:result) shared(dvec1,dvec2) device(devicenum)
        for (size_t i = 0; i < n; ++i)
        {
            result += dvec1(i) * dvec2(i);
        }
    }
    else
    {
        #pragma omp parallel for simd reduction(+:result) shared(dvec1,dvec2)
        for (size_t i = 0; i < n; ++i)
        {
            result += dvec1(i) * dvec2(i);
        }
    }
    return result;
}

template <typename T, typename Container>
inline  T dot_product_v(const  mdspan<T, Container>& vec1,const   mdspan<T, Container>& vec2)
{
    datastruct<T> dvec1=vec1.get_datastruct();
    datastruct<T> dvec2=vec2.get_datastruct();
    const size_t n = dvec1.pextents[0];
    T result = 0;
    #pragma omp simd reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += dvec1(i) * dvec2(i);
    }
    return result;
}


template <typename T, typename Container>
inline  bool vector_scalar_multiply( const mdspan<T, Container>& vec, const T scalar,mdspan<T, Container>& res, bool on_gpu=false,bool default_device=true,int devicenum=0)
{
    datastruct<T> dvec=vec.get_datastruct();
    datastruct<T> dres=res.get_datastruct();

    const size_t n = dvec.pextents[0];

    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (vec.is_offloaded(devicenum)==false) return false;
        #pragma omp target teams distribute parallel for simd shared(dres,dvec, scalar) device(devicenum)
        for (size_t i = 0; i < n; ++i)
        {
            dres(i) = dvec(i)*scalar;
        }
    }
    else
    {
        #pragma omp parallel for simd shared(dres,dvec, scalar)
        for (size_t i = 0; i < n; ++i)
        {
            dres(i) = dvec(i)*scalar;
        }
    }
    return true;

}



template <typename T, typename Container>
inline  bool vector_scalar_multiply_v( const mdspan<T, Container>& vec, const T scalar,mdspan<T, Container>& res)
{
    datastruct<T> dvec=vec.get_datastruct();
    const size_t n = vec.pextents[0];

    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = dvec(i)*scalar;
    }
    return true;

}

template <typename T, typename Container>
inline  void cross_product(const mdspan<T, Container>& vec1,const  mdspan<T, Container>& vec2, mdspan<T, Container>& res)
{
    datastruct<T> dvec1=vec1.get_datastruct();
    datastruct<T> dvec2=vec2.get_datastruct();
    datastruct<T> dres=res.get_datastruct();

    dres(0) = dvec1(1) * dvec2(2) - dvec1(2) * dvec2(1);
    dres(1) = dvec1(2) * dvec2(0) - dvec1(0) * dvec2(2);
    dres(2) = dvec1(0) * dvec2(1) - dvec1(1) * dvec2(0);
}



template <typename T, typename Container>
inline  bool vector_add( const mdspan<T, Container>& vec1, const  mdspan<T, Container>& vec2, mdspan<T, Container>& vec3, bool on_gpu=false,bool default_device=true,int devicenum=0)
{
    datastruct<T> dvec1=vec1.get_datastruct();
    datastruct<T> dvec2=vec2.get_datastruct();
    datastruct<T> dvec3=vec3.get_datastruct();
    const size_t n = dvec1.pextents[0];
    const size_t strv1=dvec1.pstrides[0];
    const size_t strv2=dvec2.pstrides[0];
    const size_t strres=dvec3.pstrides[0];

    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (vec1.is_offloaded(devicenum)==false) return false;
        if (vec2.is_offloaded(devicenum)==false) return false;
        if (vec3.is_offloaded(devicenum)==false) return false;

        #pragma omp target teams distribute parallel for simd shared(dvec3,dvec2,dvec1,strres,strv1,strv2) device(devicenum)
        for(size_t i=0; i<n; i++)
        {
            dvec3(i,strres)=dvec1(i,strv1)+dvec2(i,strv2);
        }
    }
    else
    {
        #pragma omp parallel for simd shared(dvec3,dvec2,dvec1,strres,strv1,strv2)
        for(size_t i=0; i<n; i++)
        {
            dvec3(i,strres)=dvec1(i,strv1)+dvec2(i,strv2);
        }
    }
    return true;

}

template <typename T, typename Container>
inline  bool vector_add_v( const mdspan<T, Container>& vec1, const  mdspan<T, Container>& vec2, mdspan<T, Container>& vec3)
{
    datastruct<T> dvec1=vec1.get_datastruct();
    datastruct<T> dvec2=vec2.get_datastruct();
    datastruct<T> dvec3=vec3.get_datastruct();
    const size_t n = dvec1.pextents[0];
    const size_t strv1=dvec1.pstrides[0];
    const size_t strv2=dvec2.pstrides[0];
    const size_t strres=dvec3.pstrides[0];

    #pragma omp parallel for simd shared(dvec3,dvec2,dvec1,strres,strv1,strv2)
    for(size_t i=0; i<n; i++)
    {
        dvec3(i,strres)=dvec1(i,strv1)+dvec2(i,strv2);
    }

    return true;

}

template <typename T, typename Container>
inline  bool vector_subtract( const mdspan<T, Container>& vec1, const mdspan<T, Container>& vec2, mdspan<T, Container>& vec3, bool on_gpu=false,bool default_device=true,int devicenum=0)
{
    datastruct<T> dvec1=vec1.get_datastruct();
    datastruct<T> dvec2=vec2.get_datastruct();
    datastruct<T> dvec3=vec3.get_datastruct();
    const size_t n = dvec1.pextents[0];
    const size_t strv1=dvec1.pstrides[0];
    const size_t strv2=dvec2.pstrides[0];
    const size_t strres=dvec3.pstrides[0];

    if (on_gpu==true)
    {
        if(default_device) devicenum=omp_get_default_device();
        if (vec1.is_offloaded(devicenum)==false) return false;
        if (vec2.is_offloaded(devicenum)==false) return false;
        if (vec3.is_offloaded(devicenum)==false) return false;


        #pragma omp target teams distribute parallel for simd shared(dvec3,dvec2,dvec1,strres,strv1,strv2) device(devicenum)
        for(size_t i=0; i<n; i++)
        {
            dvec3(i,strres)=dvec1(i,strv1)-dvec2(i,strv2);
        }
    }
    else
    {
        #pragma omp parallel for simd shared(dvec3,dvec2,dvec1,strres,strv1,strv2)
        for(size_t i=0; i<n; i++)
        {
            dvec3(i,strres)=dvec1(i,strv1)-dvec2(i,strv2);
        }
    }
    return true;
}

template <typename T, typename Container>
inline  bool vector_subtract_v( const mdspan<T, Container>& vec1, const mdspan<T, Container>& vec2, mdspan<T, Container>& vec3)
{
    datastruct<T> dvec1=vec1.get_datastruct();
    datastruct<T> dvec2=vec2.get_datastruct();
    datastruct<T> dvec3=vec3.get_datastruct();
    const size_t n = dvec1.pextents[0];
    const size_t strv1=dvec1.pstrides[0];
    const size_t strv2=dvec2.pstrides[0];
    const size_t strres=dvec3.pstrides[0];

    #pragma omp simd
    for(size_t i=0; i<n; i++)
    {
        dvec3(i,strres)=dvec1(i,strv1)-dvec2(i,strv2);
    }
    return true;
}



template <typename T, typename Container>
void printmatrix(const mdspan<T, Container>& span)
{
    const size_t rows= span.get_datastruct().dpextents[0];
    const size_t cols= span.get_datastruct().dpextents[1];
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << span(i, j) << " ";
        }
        std::cout << "\n";
    }
    cout <<endl;
}





template <typename T, typename Container>
void printvector(const mdspan<T, Container>& span)
{
    const size_t rows= span.get_datastruct().pextent[0];
    const size_t str= span.get_datastruct().pstrides[0];
    for (size_t i = 0; i <rows; ++i)
    {
        std::cout << span(i,str) << " ";
    }
    cout <<endl;
}



#endif
