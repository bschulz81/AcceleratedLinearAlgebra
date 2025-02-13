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



#include <iostream>
#include <array>
#include <vector>

#include <numeric>
#include <cmath>
#include <numbers>

#include <omp.h>

#include <iostream>
#include <thread>
#include <mpi.h>


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
    COMMAND_SENDMATRIX
};

struct matrix_multiplication_parameters
{
    size_t algorithm_version{Matrix_Multiplication_Algorithm::Naive};
    size_t size_for_naive_algorithm=2;
    bool memmapped_files=true;
    bool gpu_offload=true;
    bool omp=true;
    bool mpi=false;
    MPI_Comm comm=MPI_COMM_NULL;
    MPI_Status status;
    bool size_for_mpi=2;
};

#pragma omp begin declare target
template <typename T>
struct datastruct
{
    T*  __restrict pdata = nullptr;
    size_t* __restrict pextents = nullptr;
    size_t*  __restrict pstrides = nullptr;
    size_t pdatalength = 0;
    size_t prank = 0;
    bool prowmajor = true;

    // Constructors
    datastruct(
        T* __restrict data,
        size_t pdatalength,
        bool rowm,
        size_t rank,
        size_t*  __restrict extents,
        size_t* __restrict  strides,
        bool pcompute_datalength,
        bool compute_strides_from_extents
    );

    datastruct(
        T* __restrict data,
        size_t pdatalength,
        bool rowm,
        size_t*  __restrict extents,
        size_t* __restrict  strides,
        bool pcompute_datalength,
        bool compute_strides_from_extents
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
        bool compute_strides_from_extents
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
        bool compute_strides_from_extents
    );

    datastruct(
        T* __restrict data,
        size_t datalength,
        bool rowm,  size_t rank,
        size_t* __restrict extents,
        size_t* __restrict strides
    );

    ~datastruct();

    // Operator overloads
    T& operator()(const size_t* __restrict indices) __attribute__((always_inline));
    T operator()(const size_t* __restrict indices) const __attribute__((always_inline));

    T& operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1) __attribute__((always_inline));
    T operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1) const __attribute__((always_inline));

    T& operator()(const size_t row, const size_t stride) __attribute__((always_inline));
    T operator()(const size_t row, const size_t stride) const __attribute__((always_inline));


    datastruct<T>substruct_t(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides);
    datastruct<T>substruct_t(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data );
    datastruct<T>substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides);
    datastruct<T>substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides);
    datastruct<T>substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides);
    datastruct<T>subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>subspanmatrix_t( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data);
    datastruct<T> subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data);
    datastruct<T>substruct_s(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T*__restrict sub_data);


    datastruct<T> transpose(size_t* __restrict newextents, size_t* __restrict newstrides);
    datastruct<T> row(const size_t row_index, size_t* __restrict newextents, size_t* __restrict newstrides);
    datastruct<T> column(const size_t col_index, size_t* __restrict newextents, size_t* __restrict newstrides);
};
#pragma omp end declare target



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
        #pragma omp parallel for  reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        // Column-major layout: iterate innermost to outermost
        #pragma omp parallel for  reduction(+ : offset)
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
    #pragma omp parallel for reduction(+:offset)
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
template<typename T>
inline T& datastruct<T>::operator()(const size_t row, const size_t stride)
{

    return pdata[row * stride];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T datastruct<T>::operator()(const size_t row, const size_t stride)const
{
    return pdata[row * stride];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T& datastruct<T>::operator()(const size_t*__restrict indices)
{
    return pdata[compute_offset_s(indices, pstrides, prank)];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T datastruct<T>::operator()(const size_t* __restrict indices)const
{
    return pdata[compute_offset_s(indices, pstrides, prank)];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T datastruct<T>::operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)const
{
    return pdata[row * strides0 + col *strides1];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T& datastruct<T>::operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)
{
    return pdata[row * strides0 + col * strides1];
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>inline datastruct<T> datastruct<T>::transpose(size_t* __restrict newextents, size_t *newstrides)
{
    newextents[0]=pextents[1];
    newextents[1]=pextents[0];

    newstrides[0]=pstrides[1];
    newstrides[1]=pstrides[0];
    return datastruct(pdata,pdatalength,prowmajor,2,newextents,newstrides);

}
#pragma omp end declare target

#pragma omp begin declare target
inline void fill_strides(const size_t* __restrict extents,size_t* __restrict strides, const size_t rank, const bool rowmajor)
{
    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[rank - 1] = 1;

        for (int i = rank - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
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
    bool compute_strides_from_extents
) : pdata(data),
    pextents(extents),
    pstrides(strides),
    pdatalength(datalength),
    prank(rank),
    prowmajor(rowm)

{
    if(compute_strides_from_extents==true && pextents!=nullptr && pstrides!=nullptr && rank !=0)
    {
        fill_strides(pextents,pstrides,rank,rowm);
    }
    if(compute_datalength==true && pextents!=nullptr && pstrides!=nullptr && rank !=0)
    {
        pdatalength=compute_data_length_s(pextents,pstrides,rank);
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
    size_t* __restrict strides
) : pdata(data),pextents(extents),
    pstrides(strides),
    pdatalength(datalength),
    prank(rank),
    prowmajor(rowm)
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
    bool compute_strides_from_extents
) : pdata(data),
    pextents(extents),
    pstrides(strides),
    pdatalength(datalength),
    prank(2),
    prowmajor(rowm)
{
    if(extents!=nullptr)
    {
        pextents[0]=(rowm==true)?rows:cols;
        pextents[1]=(rowm==true)?cols:rows;
    }
    if(pstrides!=nullptr && compute_strides_from_extents)
    {
        pstrides[0]=(rowm==true)? cols:1;
        pstrides[1]=(rowm==true)?1: rows;
    }
    if(compute_datalength==true && extents!=nullptr && strides!=nullptr)
    {
        pdatalength=(rows-1) * strides[0]+(cols-1)*strides[1]+1;
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
    bool compute_strides_from_extents
) : pdata(data),
    pextents(extents),
    pstrides(strides),
    pdatalength(datalength),
    prank(1),
    prowmajor(true)
{
    if(extents!=nullptr)
    {
        pextents[0]=noelements;
    }
    if(pstrides!=nullptr && compute_strides_from_extents)
    {
        if(rowvector)
            pstrides[0]=(rowm==true)? 1:noelements;
        else
            pstrides[0]=(rowm==true)? noelements:1;
    }
    if(compute_datalength==true && strides!=nullptr)
    {
        pdatalength=(noelements-1) * strides[0]+1;
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T> datastruct<T>::~datastruct()
{

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides)
{
    size_t offset_index = 0;
    const size_t r=this->prank;
    #pragma omp simd reduction( + : offset_index )
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += poffsets[i] * pstrides[i];
        psub_strides[i]=pstrides[i];
    }
    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
    return datastruct(pdata + offset_index,pl,this->prowmajor,r, psub_extents,psub_strides);

}
#pragma omp end  declare target

#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_v(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data)
{
    // Compute the new strides for the subspan
    size_t * __restrict indices;
    size_t *__restrict global_indices;
    const size_t r=this->prank;
    indices=new size_t[r];
    global_indices= new size_t[r];

    #pragma omp simd
    for (size_t i=0; i<r; i++)
    {
        indices[i]=0;
    }

    size_t largest_buffer_index=0;
    // Fill the supplied buffer with subspan data
    fill_strides(psub_extents,psub_strides,r,prowmajor);
    while (true)
    {
        // Compute the current global indices
        #pragma omp simd
        for (size_t i = 0; i < r; ++i)
        {
            global_indices[i] = poffsets[i] + indices[i];
        }

        // Compute the offsets for the original data and the new buffer
        size_t original_index = compute_offset_w(global_indices, pstrides, prowmajor);
        size_t buffer_index = compute_offset_w(indices,psub_strides, prowmajor);

        // Copy the data from the original tensor to the sub-buffer
        sub_data[buffer_index] = pdata[original_index];

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
    datastruct pd(sub_data,pl,prowmajor,psub_extents, psub_strides);

    delete[] global_indices;
    delete[] indices;

    return pd;
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t* __restrict psub_strides)
{
    size_t offset_index = 0;
    const size_t r=this->prank;
    #pragma omp parallel for reduction( + : offset_index )
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += poffsets[i] * pstrides[i];
        psub_strides[i]=pstrides[i];
    }
    size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
    return datastruct(pdata + offset_index,pl,this->prowmajor,r, psub_extents,psub_strides);

}
#pragma omp end  declare target

#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_w(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T* __restrict sub_data)
{
    // Compute the new strides for the subspan
    size_t * __restrict indices;
    size_t *__restrict global_indices;
    const size_t r=this->prank;
    indices=new size_t[r];
    global_indices= new size_t[r];

    #pragma omp parallel for
    for (size_t i=0; i<r; i++)
    {
        indices[i]=0;
    }

    size_t largest_buffer_index=0;
    // Fill the supplied buffer with subspan data
    fill_strides(psub_extents,psub_strides,r,prowmajor);
    while (true)
    {
        // Compute the current global indices
        #pragma omp parallel for
        for (size_t i = 0; i < r; ++i)
        {
            global_indices[i] = poffsets[i] + indices[i];
        }

        // Compute the offsets for the original data and the new buffer
        size_t original_index = compute_offset_w(global_indices, pstrides, prowmajor);
        size_t buffer_index = compute_offset_w(indices,psub_strides, prowmajor);

        // Copy the data from the original tensor to the sub-buffer
        sub_data[buffer_index] = pdata[original_index];

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
    datastruct pd(sub_data,pl,prowmajor,psub_extents, psub_strides);

    delete[] global_indices;
    delete[] indices;

    return pd;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides)
{
    psub_strides[0]=pstrides[0];
    psub_strides[1]=pstrides[1];
    psub_extents[0]=(prowmajor==true)?tile_rows:tile_cols;
    psub_extents[1]=(prowmajor==true)?tile_cols:tile_rows;
    size_t pl=(psub_extents[0]-1) * psub_strides[0]+ (psub_extents[1]-1) * psub_strides[1]+1;
    size_t offset=+row * pstrides[0]+col * pstrides[1];
    return datastruct(pdata+offset,pl,prowmajor,2,psub_extents,psub_strides);
}
#pragma omp end declare target


#pragma omp begin  declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data)
{
    if (prowmajor)
    {
        const size_t s0=pstrides[0];
        const size_t s1=pstrides[1];
        const T* __restrict pd=this->pdata;
        #pragma omp parallel for collapse(2)
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
        const size_t s0=pstrides[0];
        const size_t s1=pstrides[1];
        const T* __restrict pd=this->pdata;
        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < tile_cols; ++j)
        {
            for (size_t i = 0; i < tile_rows; ++i)
            {
                sub_data[j * tile_rows + i] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    fill_strides(psub_extents,psub_strides,2,this->prowmajor);
    size_t pl=(tile_rows-1) * psub_strides[0]+(tile_cols-1)*psub_strides[1]+1;
    return datastruct(sub_data,pl,prowmajor,tile_rows, tile_cols,psub_extents,psub_strides);
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_t( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data)
{
    if (prowmajor)
    {
        const size_t s0=pstrides[0];
        const size_t s1=pstrides[1];
        const T* __restrict pd=this->pdata;
        #pragma omp teams distribute parallel for collapse(2)
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
        const size_t s0=pstrides[0];
        const size_t s1=pstrides[1];
        const T* __restrict pd=this->pdata;
        #pragma omp teams distribute parallel for collapse(2)
        for (size_t j = 0; j < tile_cols; ++j)
        {
            for (size_t i = 0; i < tile_rows; ++i)
            {
                sub_data[j * tile_rows + i] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    fill_strides(psub_extents,psub_strides,2,this->prowmajor);
    size_t pl=compute_data_length_w(psub_extents,psub_strides,2);
    return datastruct(sub_data,pl,prowmajor,tile_rows, tile_cols,psub_extents,psub_strides);
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
datastruct<T>datastruct<T>::substruct_s(const size_t * __restrict poffsets,const size_t *__restrict psub_extents, size_t*__restrict psub_strides, T*__restrict sub_data)
{
    size_t offset_index = 0;
    const size_t r=this->prank;
    if(sub_data==nullptr)
    {
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += poffsets[i] * pstrides[i];
            psub_strides[i]=pstrides[i];
        }
        size_t pl=compute_data_length_w(psub_extents,psub_strides,r);
        return datastruct(pdata + offset_index,pl,this->prowmajor,r, psub_extents,psub_strides);
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
        fill_strides(psub_extents,psub_strides,r,prowmajor);
        while (true)
        {
            // Compute the current global indices
            for (size_t i = 0; i < r; ++i)
            {
                global_indices[i] = poffsets[i] + indices[i];
            }

            // Compute the offsets for the original data and the new buffer
            size_t original_index = compute_offset_s(global_indices, pstrides, prowmajor);
            size_t buffer_index = compute_offset_s(indices,psub_strides, prowmajor);

            // Copy the data from the original tensor to the sub-buffer
            sub_data[buffer_index] = pdata[original_index];

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
        datastruct pd(sub_data,pl,prowmajor,psub_extents, psub_strides);

        delete[] global_indices;
        delete[] indices;

        return pd;
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
datastruct<T>  datastruct<T>::subspanmatrix_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t * __restrict psub_extents,  size_t *__restrict psub_strides, T* __restrict sub_data)
{
    if (prowmajor)
    {
        const size_t s0=pstrides[0];
        const size_t s1=pstrides[1];
        const T* __restrict pd=this->pdata;
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
        const size_t s0=pstrides[0];
        const size_t s1=pstrides[1];
        const T* __restrict pd=this->pdata;
        // Column-major layout: fill column by column

        for (size_t j = 0; j < tile_cols; ++j)
        {
            for (size_t i = 0; i < tile_rows; ++i)
            {
                sub_data[j * tile_rows + i] = pd[compute_offset(row + i, col + j, s0, s1)];
            }
        }
    }
    fill_strides(psub_extents,psub_strides,2,this->prowmajor);
    size_t pl=(tile_rows-1) * psub_strides[0]+(tile_cols-1)*psub_strides[1]+1;
    return datastruct(sub_data,pl,prowmajor,tile_rows, tile_cols,psub_extents,psub_strides);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::row(const size_t row_index, size_t* __restrict extents,size_t *__restrict  new_strides)
{

    // Offset the data pointer to the start of the row
    T* __restrict row_data = pdata + row_index * pstrides[0];

    // Fill the extents array with the appropriate values for the row
    extents[0] = pextents[1]; // Extent for a row is the number of columns
    new_strides[0]=pstrides[0];
    return datastruct<T>(row_data,  pstrides[1] * extents[0],prowmajor,   1, &extents[0],    new_strides );
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
datastruct<T> datastruct<T>::column(const size_t col_index, size_t*__restrict  extents,size_t *__restrict new_strides)
{
    // Offset the data pointer to the start of the column
    T* __restrict col_data = pdata + col_index * pstrides[1];

    // Fill the extents array with the appropriate values for the column
    extents[0] = pextents[0]; // Extent for a column is the number of rows
    new_strides[0]=pstrides[0];
    return datastruct(col_data, pstrides[0] * extents[0],prowmajor,  1, &extents[0],   new_strides );
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void printmatrix(const datastruct<T>&span)
{
    const size_t rows= span.pextents[0];
    const size_t cols=span.pextents[1];
    const size_t str0= span.pstrides[0];
    const size_t str1= span.pstrides[1];
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            printf("%f ",span(i, j,str0,str1));
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
    const size_t str0= span.pstrides[0];
    for (size_t i = 0; i <rows; ++i)
    {
        printf("%f\n",span(i,str0));
    }

}
#pragma omp end declare target


template<typename T>
inline void update_device(datastruct<T>& dL)
{
      size_t l=dL.pdatalength;
    #pragma omp target update to(dL.pdata[0:l])
}

template<typename T>
inline void update_host(datastruct<T>& dL)
{
    size_t l=dL.pdatalength;
    #pragma omp target update from (dL.pdata[0:l])
}

template<typename T>
void inline create_in_struct(const datastruct<T>& dA)
{
      size_t l=dA.pdatalength;
      size_t r=dA.prank;
    #pragma omp target enter data map(to: dA,dA.pdata[0:l],dA.pextents[0:r],dA.pstrides[0:r])
}

template<typename T>
void inline create_out_struct(datastruct<T>& dA)
{
       size_t l=dA.pdatalength;
      size_t r=dA.prank;
   #pragma omp target enter data map(to: dA,dA.pdata[0:l],dA.pextents[0:r],dA.pstrides[0:r])
}
template<typename T>
inline void exit_struct(datastruct<T> &dA)
{
       size_t l=dA.pdatalength;
      size_t r=dA.prank;
    #pragma omp target exit data map(delete: dA.pstrides[0:r],dA.pextents[0:r],dA.pdata[0:l],dA)
}

template<typename T>
inline void exit_struct(const datastruct<T> &dA)
{
    #pragma omp target exit data map(delete: dA.pstrides[0:dA.prank],dA.pextents[0:dA.prank],dA.pdata[0:dA.pdatalength],dA)
}


template<typename T>
T* create_temp_mmap(const size_t array_size)
{
    size_t file_size = array_size * sizeof(double);

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

// Function to unmap the memory-mapped file
void delete_temp_mmap(double* mmap_ptr,const size_t array_size)
{
    size_t file_size = array_size * sizeof(double);
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
    mdspan(T* __restrict data,const bool rowm, const Container& extents);
    mdspan(T* __restrict data, const bool rowm,const size_t rows,const size_t cols);

    mdspan(const size_t datalength, const bool rowm, const bool memmap, const Container& extents, const Container& strides);
    mdspan(const bool rowm,const  bool memmap, const Container& extents, const Container& strides);
    mdspan(const bool rowm,const  bool memmap, const Container& extents);
    mdspan(const bool rowm, const bool memmap, const size_t rows, const size_t cols);

    mdspan(mdspan<T, Container>&& other) noexcept;
    mdspan& operator=(mdspan&& other) noexcept;
    ~mdspan();


    // Deleted copy constructor and copy assignment
    mdspan(const mdspan<T, Container>&) = delete;
    mdspan& operator=(const mdspan<T, Container>&) = delete;
    // Access operators
    inline T& operator()(const Container& extents);
    inline T operator()(const Container& extents)const;

    inline T& operator()(const size_t row, const size_t col, const size_t stride0, const size_t stride1);
    inline T operator()(const size_t row, const size_t col, const size_t stride0, const size_t stride1)const;

    inline T operator()(const size_t i, const size_t stride)const;
    inline T& operator()(const size_t i, const size_t stride);
    // Subspan methods
    mdspan<T, Container> subspan(const Container& offsets, const Container& sub_extents, T* __restrict sub_data=nullptr) const;
    mdspan<T, Container> subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*sub_data=nullptr)const;
    mdspan<T, Container> row(const size_t row_index);
    mdspan<T, Container> column(const size_t col_index);
    mdspan<T, Container> transpose() ;


    // Other utility methods
    size_t extent(const size_t dim) const;
    size_t rank() const;
    size_t stride(const size_t dim) const;

    // Member function declarations
    Container& extents()const;
    Container& strides()const;

    size_t datalength() const;
    // Data structure for parallel device allocation (assumed type)
    datastruct<T> pdatastruct;



private:
    bool pownsdata=false;
    bool pwith_memmap=false;
    void initialize_extents_and_strides(const Container&extents,const Container & strides);
    void initialize_extents(const Container&extents);
    void allocate_data(const bool memmap,const size_t datalength);

    // Private member variables
    Container pextents;  // Use the ExtentContainer type
    Container pstrides;  // Use the StrideContainer type
};




// Access operator for multidimensional indices
template <typename T, typename Container>
inline T& mdspan<T, Container>::operator()(const Container& indices)
{


    size_t offset = 0;
    #pragma omp simd  reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * pdatastruct.pstrides[i];
    }

    return pdatastruct.pdata[offset];
}

template <typename T, typename Container>
inline T& mdspan<T, Container>::operator()(const size_t row,const size_t col, const size_t stride0, const size_t stride1)
{


    return pdatastruct.pdata[row * stride0 + col * stride1];
}

template <typename T, typename Container>
inline T& mdspan<T, Container>::operator()(const size_t i, const size_t stride)
{

    return pdatastruct.pdata[i * stride];
}


// Access operator for multidimensional indices
template <typename T, typename Container>
T mdspan<T, Container>::operator()(const Container& indices)const
{

    size_t offset = 0;
    #pragma omp simd reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * pdatastruct.pstrides[i];
    }

    return pdatastruct.pdata[offset];
}

template <typename T, typename Container>
inline T mdspan<T, Container>::operator()(const size_t row,const size_t col, const size_t stride0, const size_t stride1)const
{
    return pdatastruct.pdata[row * stride0 + col * stride1];
}

template <typename T, typename Container>
T mdspan<T, Container>::operator()(const size_t i, const size_t stride)const
{

    return pdatastruct.pdata[i * stride];
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
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
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
    pdatastruct.pextents = pextents.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  size_t datalength,const  bool rowm, const Container& extents, const Container& strides)
    :pdatastruct(data,datalength,rowm,extents.size(),nullptr,nullptr,false,false),
     pownsdata(false),
     pwith_memmap(false)
{
    initialize_extents_and_strides(extents,strides);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm, const Container& extents, const Container& strides )
    : pdatastruct(data, 0,rowm,extents.size(),nullptr,nullptr,false,false),
      pownsdata(false),
      pwith_memmap(false)
      // Initialize pdatastruct with placeholders
{
    initialize_extents_and_strides(extents,strides);
    pdatastruct.pdatalength=compute_data_length_v(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const bool rowm,const  Container& extents)
    :  pdatastruct(data,0,rowm,extents.size(),nullptr,nullptr,false,false),
       pownsdata(false),
       pwith_memmap(false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length_w(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
}







template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm,  const size_t rows, const size_t cols)
    :  pdatastruct(data,0,rowm,2,nullptr,nullptr,false,false),
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

    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length_v(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
}





template <typename T, typename Container>
mdspan<T, Container>::mdspan( const size_t datalength,  const bool rowm,const bool memmap, const Container& extents, const Container& strides)
    :pdatastruct(nullptr, datalength,rowm,extents.size(),nullptr,nullptr,false,false)
{
    initialize_extents_and_strides(extents,strides,rowm);
    allocate_data(memmap,pdatastruct.pdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan( const bool rowm,const bool memmap, const Container& extents, const Container& strides )
    : pdatastruct(nullptr, 0,rowm,extents.size(),nullptr,nullptr,false,false)
{
    initialize_extents_and_strides(extents,strides,rowm);
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
    allocate_data(memmap,pdatastruct.pdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(const bool rowm,const bool memmap,const  Container& extents)
    :  pdatastruct(nullptr,0,rowm,extents.size(),nullptr,nullptr,false,false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    // Assign actual pointers to datastruct
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();

    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
    pownsdata=true;
    allocate_data(memmap,pdatastruct.pdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(const bool rowm,const bool memmap, const size_t rows, const size_t cols)
    :  pdatastruct(nullptr,0,rowm,2,nullptr,nullptr,false,false)

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

    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
    allocate_data(memmap,pdatastruct.pdatalength);

}

template <typename T, typename Container>
mdspan<T, Container>::~mdspan()
{
    if(pownsdata==true)
    {
        if (pwith_memmap==true)
            delete_temp_mmap(pdatastruct.pdata,sizeof(T)*pdatastruct.pdatalength);
        else
            delete[] pdatastruct.pdata;
    }
}

template <typename T, typename Container>
void mdspan<T, Container>::allocate_data(bool memmap, size_t datalength)
{
    pownsdata = true;
    if (memmap)
    {
        const size_t s=sizeof(T) * datalength;
        pdatastruct.pdata = create_temp_mmap<T>(s);
        pwith_memmap = true;
    }
    else
    {
        pdatastruct.pdata = new T[datalength];
        pwith_memmap = false;
    }
}

template <typename T, typename Container>
mdspan<T, Container>::mdspan(mdspan<T, Container>&& other) noexcept
    : pstrides(std::move(other.pstrides)),
      pextents(std::move(other.pextents)),
      pdatastruct(other.pdatastruct.pdata,other.pdatastruct.pdatalength,other.pdatastruct.rowmajor,nullptr,nullptr,false,false)
{
    pownsdata=other.pownsdata;
    pwith_memmap=other.pwith_memmap;
    // Update pointers in datastruct to the new strides and extents
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pextents = pextents.data();

    // Null out the other's pointers to avoid double delete
    other.pdatastruct.pdata = nullptr;
    other.pdatastruct.pstrides = nullptr;
    other.pdatastruct.pextents = nullptr;
    other.pdatastruct.pdata = nullptr;

}


// Move assignment operator

template <typename T, typename Container>
mdspan<T, Container> &  mdspan<T, Container>::operator=(mdspan<T, Container> && other) noexcept
{
    if (this != &other)
    {
        // Free existing resources
        if(pownsdata==true)
        {
            if (pwith_memmap==true)
                delete_temp_mmap(pdatastruct.pdata,sizeof(T)*pdatastruct.pdatalength);
            else
                delete[] pdatastruct.pdata;
        }
        pownsdata=other.pownsdata;
        pwith_memmap=other.pwith_memmap;
        // Move data members

        pdatastruct.pdata = other.pdatastruct.pdata;
        pdatastruct.pdatalength=other.pdatastruct.pdatalength;
        pdatastruct.prank=other.pdatastruct.prank;
        pdatastruct.prowmajor=other.pdatastruct.prowmajor;
        pstrides = std::move(other.pstrides);
        pextents = std::move(other.pextents);
        // Update pointers in datastruct to the new strides and extents
        pdatastruct.pstrides = pstrides.data();
        pdatastruct.pextents = pextents.data();
        // Null out the other's pointers to avoid double delete
        other.pdatastruct.pdata = nullptr;
        other.pdatastruct.pstrides = nullptr;
        other.pdatastruct.pextents = nullptr;
        other.pdatastruct.pdata = nullptr;
    }
    return *this;
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan(const Container&offsets, const Container &sub_extents, T* __restrict sub_data)const
{
    const size_t r=pdatastruct.prank;

    if (sub_data==nullptr)
    {
        // Compute the offset to the starting point
        size_t offset_index = 0;

        #pragma omp simd reduction( + : offset_index )
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += offsets[i] * pdatastruct.pstrides[i];
        }

        // Create a new mdspan_dynamic with the updated pointer, extents, and the same strides
        return mdspan(pdatastruct.pdata + offset_index,pdatastruct.prowmajor, sub_extents, pstrides);

    }
    else
    {
        // Compute the new strides for the subspan
        Container sub_strides;
        compute_strides(sub_extents, sub_strides, pdatastruct.prowmajor);
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
            size_t original_index = compute_offset_v(global_indices.data(), pdatastruct.pstrides,global_indices.size(), pdatastruct.prowmajor);
            size_t buffer_index = compute_offset_v(indices.data(),sub_strides.data(),indices.size(), pdatastruct.prowmajor);

            // Copy the data from the original tensor to the sub-buffer
            sub_data[buffer_index] = pdatastruct.pdata[original_index];

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
        return mdspan(sub_data, pdatastruct.prowmajor, sub_extents, sub_strides );
    }
}

template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*__restrict  sub_data)const
{

    if(sub_data==nullptr)
    {

        const size_t offset=row * pdatastruct.pstrides[0]+col * pdatastruct.pstrides[1];
        const Container ext= {tile_rows,tile_cols};
        return mdspan(pdatastruct.pdata +offset,pdatastruct.prowmajor,ext,pstrides);
    }
    else
    {
        if (pdatastruct.prowmajor)
        {
            // Row-major layout: fill row by row
            #pragma omp parallel for collapse (2)
            for (size_t i = 0; i < tile_rows; ++i)
            {
                for (size_t j = 0; j < tile_cols; ++j)
                {
                    sub_data[i * tile_cols + j] = pdatastruct.pdata[
                                                      compute_offset(row + i, col + j, pdatastruct.pstrides[0], pdatastruct.pstrides[1])
                                                  ];
                }
            }
        }
        else
        {
            #pragma omp parallel for collapse (2)
            for (size_t j = 0; j < tile_cols; ++j)
            {
                for (size_t i = 0; i < tile_rows; ++i)
                {
                    sub_data[j * tile_rows + i] = pdatastruct.pdata[
                                                      compute_offset(row + i, col + j, pdatastruct.pstrides[0], pdatastruct.pstrides[1])
                                                  ];
                }
            }
        }
        const Container sub_extents = {tile_rows, tile_cols};

        const Container sub_strides = (pdatastruct.prowmajor==true)? Container{tile_cols, 1} :
                                      Container{1,tile_rows};

        return mdspan(sub_data,pdatastruct.prowmajor, sub_extents, sub_strides );
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
        const size_t ext0=span.pdatastruct.pextents[0];
        const size_t ext1=span.pdatastruct.pextents[1];
        const size_t sstr0=span.pdatastruct.pstrides[0];
        const size_t sstr1=span.pdatastruct.pstrides[1];
        const size_t tstr0=target.pdatastruct.pstrides[0];
        const size_t tstr1=target.pdatastruct.pstrides[1];
        // Copy the current span into the target at the given offset
        #pragma omp parallel for collapse(2)
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

    const size_t num_rows = pdatastruct.pextents[0];
    const Container column_extents = {num_rows};
    const Container column_strides = {pdatastruct.pstrides[0]};

    return mdspan<T, Container>( pdatastruct.pdata + col_index * pdatastruct.pstrides[1], pdatastruct.prowmajor, column_extents, column_strides );
}

template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::row(const size_t row_index)
{

    const size_t num_cols = pdatastruct.pextents[1];
    const Container row_extents = {num_cols};
    const Container row_strides = {pdatastruct.pstrides[1]};

    return mdspan<T, Container>( pdatastruct.pdata + row_index * pdatastruct.pstrides[0], pdatastruct.prowmajor, row_extents, row_strides );
}



template <typename T>
mdspan<T, vector<size_t>> & MPI_recv_mdspan(const bool memmap, const int source,const int tag, const MPI_Comm pcomm)
{
    MPI_Status status;
    size_t pdatalength, prank;
    bool prowmajor;
    MPI_Recv(&pdatalength, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&prank, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&prowmajor, 1, MPI_C_BOOL, source, tag, pcomm, &status);

    vector<size_t> extents(prank,0);
    vector<size_t> strides(prank,0);

    MPI_Recv(extents.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(strides.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);

    mdspan<T, vector<size_t>> md(pdatalength,prowmajor,memmap,extents,strides);

    MPI_Recv(md.pdatastruct.pdata,sizeof(T)* md.pdatastruct.pdata, MPI_BYTE, source, tag, pcomm, &status);

    return md;
}

template <typename T>
mdspan<T, vector<size_t>> & MPI_Irecv_mdspan(const bool memmap, const int source, const int tag,const MPI_Comm pcomm)
{
    MPI_Request request;
    // Receive metadata
    size_t pdatalength, prank;
    bool prowmajor;
    MPI_Irecv(&pdatalength, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);
    MPI_Irecv(&prank, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);
    MPI_Irecv(&prowmajor, 1, MPI_C_BOOL, source, tag, pcomm, &request);

    vector<size_t> extents(prank,0);
    vector<size_t> strides(prank,0);
    // Allocate memory for extents and strides
    MPI_Irecv(extents.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);
    MPI_Irecv(strides.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);

    mdspan<T, vector<size_t>> md(pdatalength,prowmajor,memmap,extents,strides);

    MPI_Irecv(md.pdatastruct.pdata,sizeof(T)* md.pdatastruct.pdata, MPI_BYTE, source, tag, pcomm, &request);

    return md;
}


template <typename T>
void MPI_recv_mdspan_pdata(mdspan<T, vector<size_t>> & mds,const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(mds.pdatastruct.pdata,sizeof(T)* mds.pdatastruct.pdatalength, MPI_BYTE, source, tag, pcomm, &status);
}

template <typename T, typename Container>
void MPI_send_mdspan_pdata(mdspan<T, Container> & m, int dest, int tag,MPI_Comm pcomm)
{
    MPI_Send(m.pdatastruct.pdata,sizeof(T)* m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);
}

template <typename T>
void MPI_Irecv_mdspan_pdata(mdspan<T, vector<size_t>> & mds, const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Irecv(mds.pdatastruct.pdata,sizeof(T)* mds.pdatastruct.pdatalength, MPI_BYTE, source, tag, pcomm, &status);
}

template <typename T, typename Container>
void MPI_Isend_mdspan_pdata(mdspan<T, Container> & m, const int dest,const  int tag,const MPI_Comm pcomm)
{
    MPI_Isend(m.pdatastruct.pdata,sizeof(T)* m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);
}


template <typename T, typename Container>
void MPI_Isend_mdspan(mdspan<T, Container> & m, const int dest,const  int tag,const MPI_Comm pcomm)
{
    MPI_Isend(&m.pdatastruct.pdatalength, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Isend(&m.pdatastruct.prank, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Isend(&m.pdatastruct.prowmajor, 1, MPI_C_BOOL, dest, tag, pcomm);

    MPI_Isend(m.pdatastruct.pextents, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Isend(m.pdatastruct.pstrides, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);

    MPI_Isend(m.pdatastruct.pdata,sizeof(T)*m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);

}



template <typename T, typename Container>
void MPI_send_mdspan(mdspan<T, Container> & m, int dest, int tag, MPI_Comm pcomm)
{
    MPI_Send(&m.pdatastruct.pdatalength, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.pdatastruct.prank, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.pdatastruct.prowmajor, 1, MPI_C_BOOL, dest, tag, pcomm);

    MPI_Send(m.pdatastruct.pextents, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(m.pdatastruct.pstrides, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);

    MPI_Send(m.pdatastruct.pdata,sizeof(T)* m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);

}

template <typename T>
void MPI_listener(const MPI_Comm pcomm)
{
    while (true)
    {
        MPI_Status status;
        int message_type;

        MPI_Recv(&message_type, 1, MPI_INT, MPI_ANY_SOURCE, 0, pcomm, &status);

        switch (message_type)
        {
        case COMMAND_STRASSEN:
        {

            mdspan<T, vector<size_t>> A=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 1, pcomm);

            mdspan<T, vector<size_t>> B=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 2, pcomm);

            size_t rowsC=A.pdatastruct.pextents[0],
                   colsC=B.pdatastruct.pextents[1];

            mdspan<T, std::vector<size_t>> C(A.prowmajor,true, {rowsC, colsC});

            matrix_multiplication_parameters algorithm;

            algorithm.mpi=true;
            algorithm.memmapped_files=true;
            algorithm.gpu_offload=true;
            algorithm.comm=pcomm;
            algorithm.status=status;
            strassen_multiply(A,B,C,algorithm,true);
            MPI_send_mdspan_pdata(C,status.MPI_SOURCE,3,pcomm);
            break;
        }
        case COMMAND_WINOGRAD:
        {

            mdspan<T, vector<size_t>> A=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 1, pcomm);
            mdspan<T, vector<size_t>> B=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 2, pcomm);
            size_t rowsC=A.pdatastruct.pextents[0],colsC=B.pdatastruct.pextents[1];
            mdspan<T, std::vector<size_t>> C(A.prowmajor,true, {rowsC, colsC});
            matrix_multiplication_parameters algorithm;
            algorithm.mpi=true;
            algorithm.memmapped_files=true;
            algorithm.gpu_offload=true;
            algorithm.comm=pcomm;
            algorithm.status=status;
            winograd_multiply(A,B,C,algorithm,true);
            MPI_send_mdspan_pdata(C,status.MPI_SOURCE,3,pcomm);
            break;
        }
        case COMMAND_SENDMATRIX:
        {
            mdspan<T, vector<size_t>> A=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 1, pcomm);
            if(A.pdatastruct.prank==2)
                printmatrix(A);
            break;
        }

        }
    }
}

template <typename T, typename Container>
size_t mdspan<T, Container> ::extent(const size_t dim) const
{
    return pdatastruct.pextents[dim];
}



template <typename T, typename Container>
size_t mdspan<T, Container>  ::rank() const
{
    return pdatastruct.prank;
}


template <typename T, typename Container>
size_t mdspan<T, Container> ::stride(const size_t dim)const
{
    return pstrides[dim];
}


template <typename T, typename Container>
Container & mdspan<T, Container> ::extents()const
{
    return pextents;
}
template <typename T, typename Container>
Container & mdspan<T, Container> ::strides()const
{
    return pstrides;
}


template <typename T, typename Container>
size_t mdspan<T, Container> ::datalength()const
{
    return pdatastruct.pdatalength;
}





template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::transpose()
{
    Container transposed_extents = {pdatastruct.pextents[1], pdatastruct.pextents[0]};
    Container transposed_strides = {pdatastruct.pstrides[1], pdatastruct.pstrides[0]};
    return mdspan(pdatastruct.pdata,pdatastruct.pdatalength, pdatastruct.prowmajor,  transposed_extents,   transposed_strides);
}

#pragma omp begin declare target
template <typename T>
void gpu_cholesky_decomposition_w(const datastruct<T>& A, datastruct<T>& L, T* __restrict buffer=nullptr, size_t step_size=0)
{
    const size_t n = A.pextents[0];
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

    #pragma omp parallel for shared(A,L)
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=A.pdata[i];
        L.pdata[i]=0;
    }

    datastruct<T> tempA(adata, 0,A.prowmajor,n, n,pext3, pstrides3,true,true);
    const size_t strtA0=tempA.pstrides[0];
    const size_t strtA1=tempA.pstrides[1];
    const size_t strl0=L.pstrides[0];
    const size_t strl1=L.pstrides[1];

    for (size_t c = 0; c < n; ++c)
    {
        if (c == z + step_size)
        {
            size_t u=n-c;
            size_t v=c-z;
            size_t rtext[2];
            size_t rtstrides[2];

            size_t pSstrides[2];
            size_t pSext[2];
            size_t pRext[2];
            size_t pRstr[2];

            datastruct<T> R =L.subspanmatrix(c, z, u, v,pRext,pRstr);
            datastruct<T> RT=R.transpose(rtext,rtstrides);
            datastruct<T> S(sdata, 0, A.prowmajor,u, u, pSext, pSstrides,true,true);
            const size_t strs0=S.pstrides[0];
            const size_t strs1=S.pstrides[1];

            const size_t rows=R.pextents[0];
            const size_t cols=RT.pextents[1];
            const size_t inner_dim=R.pextents[1];

            const size_t strA0=R.pstrides[0];
            const size_t strA1=R.pstrides[1];

            const size_t strB0=RT.pstrides[0];
            const size_t strB1=RT.pstrides[1];

            const size_t strC0=S.pstrides[0];
            const size_t strC1=S.pstrides[1];

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += R(i,k,strA0,strA1) *RT(k,j,strB0,strB1);
                    }
                    S(i,j,strC0,strC1)= sum;
                }
            }


            const size_t h=c;
            #pragma omp parallel for collapse(2) shared(h,tempA,strtA0,strtA1,S,strs0,strs1)
            for (size_t i = h; i < n; ++i)
            {
                for (size_t j = h; j < n; ++j)
                {
                    tempA(i,j,strtA0,strtA1) -=S(i-h,j-h,strs0,strs1);
                }
            }

            z = c;
        }

        T temp = 0;
        #pragma omp parallel for reduction(+:temp) shared(L,strl0,strl1,c)
        for (size_t k = z; k < c; ++k)
        {
            T tmp3=L(c,k,strl0,strl1);
            temp +=  tmp3*tmp3;
        }

        temp=tempA(c,c,strtA0,strtA1)-temp;
        T temp4=sqrt(temp);
        L(c,c,strl0,strl1) = temp4;


        #pragma omp parallel for shared(c,L,strl0,strl1,tempA,strtA0,strtA1)
        for (size_t i = c + 1; i < n; ++i)
        {
            T temp2 =0;
            for (size_t k = z; k < c; ++k)
            {
                temp2 += L(i,k,strl0,strl1)*L(c,k,strl0,strl1);
            }
            temp2= tempA(i,c,strtA0,strtA1)-temp2;
            L(i,c,strl0,strl1) = temp2 / temp4;
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
void gpu_cholesky_decomposition_t(const datastruct<T>& A, datastruct<T>& L, T* __restrict buffer=nullptr, size_t step_size=0)
{
    const size_t n = A.pextents[0];
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
        adata=(T*) omp_alloc(sizeof(T*)*nn,omp_large_cap_mem_alloc,);
    }
    else
    {
        sdata=buffer;
        adata=buffer+tempsize;
    }

    #pragma omp teams distribute parallel for
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=A.pdata[i];
        L.pdata[i]=0;
    }

    datastruct<T> tempA(adata, 0,A.prowmajor,n, n,pext3, pstrides3,true,true);
    const size_t strtA0=tempA.pstrides[0];
    const size_t strtA1=tempA.pstrides[1];
    const size_t strl0=L.pstrides[0];
    const size_t strl1=L.pstrides[1];

    for (size_t c = 0; c < n; ++c)
    {
        if (c == z + step_size)
        {
            size_t u=n-c;
            size_t v=c-z;
            size_t rtext[2];
            size_t rtstrides[2];

            size_t pSstrides[2];
            size_t pSext[2];
            size_t pRext[2];
            size_t pRstr[2];
            datastruct<T> R = L.subspanmatrix(c, z, u, v,pRext,pRstr);
            datastruct<T> RT=R.transpose(rtext,rtstrides);
            datastruct<T> S(sdata, 0, A.prowmajor,u, u, pSext, pSstrides,true,true);


            const size_t strs0=S.pstrides[0];
            const size_t strs1=S.pstrides[1];

            const size_t rows=R.pextents[0];
            const size_t cols=RT.pextents[1];
            const size_t inner_dim=R.pextents[1];

            const size_t strA0=R.pstrides[0];
            const size_t strA1=R.pstrides[1];

            const size_t strB0=RT.pstrides[0];
            const size_t strB1=RT.pstrides[1];

            const size_t strC0=S.pstrides[0];
            const size_t strC1=S.pstrides[1];

            #pragma omp teams distribute collapse(2)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    #pragma omp parallel for reduction(+: sum)
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += R(i,k,strA0,strA1) *RT(k,j,strB0,strB1);
                    }
                    S(i,j,strC0,strC1)= sum;
                }
            }



            const size_t h=c;
            #pragma omp  teams distribute parallel for collapse(2) shared(h,tempA,strtA0,strtA1,S,strs0,strs1)
            for (size_t i = h; i < n; ++i)
            {
                for (size_t j = h; j < n; ++j)
                {
                    tempA(i,j,strtA0,strtA1) -=S(i-h,j-h,strs0,strs1);
                }
            }

            z = c;
        }

        T temp = 0;
        #pragma omp parallel for reduction(+:temp) shared(L,strl0,strl1,c)
        for (size_t k = z; k < c; ++k)
        {
            T tmp3=L(c,k,strl0,strl1);
            temp +=  tmp3*tmp3;
        }

        temp=tempA(c,c,strtA0,strtA1)-temp;
        T temp4=sqrt(temp);
        L(c,c,strl0,strl1) = temp4;


        #pragma omp  teams distribute shared(c,L,strl0,strl1,tempA,strtA0,strtA1)
        for (size_t i = c + 1; i < n; ++i)
        {
            T temp2 =0;
            #pragma omp parallel for reduction(+:temp2) shared(c,L,strl0,strl1,tempA,strtA0,strtA1)
            for (size_t k = z; k < c; ++k)
            {
                temp2 += L(i,k,strl0,strl1)*L(c,k,strl0,strl1);
            }
            temp2= tempA(i,c,strtA0,strtA1)-temp2;
            L(i,c,strl0,strl1) = temp2 / temp4;
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
inline  void gpu_lu_decomposition_t(const  datastruct<T>& dA, datastruct<T>& dL, datastruct<T>& dU, T* __restrict buffer=nullptr, size_t step_size=0)
{


    const size_t n = dA.pextents[0];
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

    #pragma omp teams distribute parallel for shared(dA,dL,dU)
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=dA.pdata[i];
        dL.pdata[i]=0;
        dU.pdata[i]=0;
    }
    datastruct<T> tempA(adata,  0, dA.prowmajor,n, n,pext3, pstrides3,true,true);
    const size_t strtA0=tempA.pstrides[0];
    const size_t strtA1=tempA.pstrides[1];

    const size_t strU0=dU.pstrides[0];
    const size_t strU1=dU.pstrides[1];
    const size_t strL0=dL.pstrides[0];
    const size_t strL1=dL.pstrides[1];

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

            datastruct<T> S(sdata,  0, dA.prowmajor,u, u,pSext, pSstrides,true,true);

            const size_t rows=RL.pextents[0];
            const size_t cols=RU.pextents[1];
            const size_t inner_dim=RL.pextents[1];

            const size_t strA0=RL.pstrides[0];
            const size_t strA1=RL.pstrides[1];

            const size_t strB0=RU.pstrides[0];
            const size_t strB1=RU.pstrides[1];

            const size_t strC0=S.pstrides[0];
            const size_t strC1=S.pstrides[1];


            #pragma omp teams distribute parallel for collapse(2)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;

                    #pragma omp parallel for reduction(+: sum)
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += RL(i,k,strA0,strA1) *RU(k,j,strB0,strB1);
                    }
                    S(i,j,strC0,strC1)= sum;
                }
            }


            const size_t strS0=S.pstrides[0];
            const size_t strS1=S.pstrides[1];
            #pragma omp  teams distribute  parallel for collapse(2) shared(S,tempA,strtA0,strtA1,strS0,strS1)
            for (size_t i = c; i < n; ++i)
            {
                for (size_t j = c; j < n; ++j)
                {
                    tempA(i,j,strtA0,strtA1) -= S(i - c, j - c,strS0,strS1);
                }
            }
            z = c;
        }


        #pragma omp teams distribute shared(dU,dL,strU0,strU1,strL0,strL1)
        for (size_t i = c; i < n; ++i)
        {
            T temp=0;
            #pragma omp parallel for reduction(+:temp)
            for (size_t k = z; k < c; ++k)
            {
                temp+= dU( k,i,strU0,strU1) * dL( c,k,strL0,strL1);
            }
            dU(c,i,strU0,strU1)=tempA(c,i,strtA0,strtA1)-temp;
        }
        const T temp4=dU(c,c,strU0,strU1);

        #pragma omp   teams distribute shared(dU,dL,strU0,strU1,strL0,strL1)
        for (size_t i = c; i < n; ++i)
        {
            T temp= 0;
            #pragma omp parallel for reduction(+:temp) shared(dU,dL,strU0,strU1,strL0,strL1)
            for (size_t k = z; k < c; ++k)
            {
                temp += dU(k,c,strU0,strU1) * dL( i,k,strL0,strL1);
            }
            temp=tempA(i,c,strtA0,strtA1)-temp;
            dL(i,c,strL0,strL1)=temp/temp4;
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

    const size_t n = dA.pextents[0];
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

    #pragma omp teams distribute parallel for shared(adata,dA,dL,dU)
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=dA.pdata[i];
        dL.pdata[i]=0;
        dU.pdata[i]=0;
    }
    datastruct<T> tempA(adata,  0, dA.prowmajor,n, n,pext3, pstrides3,true,true);
    const size_t strtA0=tempA.pstrides[0];
    const size_t strtA1=tempA.pstrides[1];

    const size_t strU0=dU.pstrides[0];
    const size_t strU1=dU.pstrides[1];
    const size_t strL0=dL.pstrides[0];
    const size_t strL1=dL.pstrides[1];

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

            datastruct<T> S(sdata,  0, dA.prowmajor,u, u,pSext, pSstrides,true,true);

            const size_t rows=RL.pextents[0];
            const size_t cols=RU.pextents[1];
            const size_t inner_dim=RL.pextents[1];

            const size_t strA0=RL.pstrides[0];
            const size_t strA1=RL.pstrides[1];

            const size_t strB0=RU.pstrides[0];
            const size_t strB1=RU.pstrides[1];

            const size_t strC0=S.pstrides[0];
            const size_t strC1=S.pstrides[1];

            #pragma omp parallel for  collapse(2)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += RL(i,k,strA0,strA1) *RU(k,j,strB0,strB1);
                    }
                    S(i,j,strC0,strC1)= sum;
                }
            }

            const size_t strS0=S.pstrides[0];
            const size_t strS1=S.pstrides[1];
            const size_t h=c;
            #pragma omp parallel for collapse(2) shared(h,S,tempA,strtA0,strtA1,strS0,strS1)
            for (size_t i = c; i < n; ++i)
            {
                for (size_t j = c; j < n; ++j)
                {
                    tempA(i,j,strtA0,strtA1) -= S(i - h, j - h,strS0,strS1);
                }
            }
            z = c;
        }

        #pragma omp parallel for shared(dU,dL,strU0,strU1,strL0,strL1)
        for (size_t i = c; i < n; ++i)
        {
            T temp=0;
            for (size_t k = z; k < c; ++k)
            {
                temp+= dU( k,i,strU0,strU1) * dL( c,k,strL0,strL1);
            }
            dU(c,i,strU0,strU1)=tempA(c,i,strtA0,strtA1)-temp;
        }
        const T temp4=dU(c,c,strU0,strU1);

        #pragma omp parallel for shared(dU,dL,strU0,strU1,strL0,strL1,temp4)
        for (size_t i = c; i < n; ++i)
        {
            T temp= 0;
            for (size_t k = z; k < c; ++k)
            {
                temp += dU(k,c,strU0,strU1) * dL( i,k,strL0,strL1);
            }
            temp=tempA(i,c,strtA0,strtA1)-temp;
            dL(i,c,strL0,strL1)=temp/temp4;
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
    const size_t n = A.pextents[0];
    const size_t m = A.pextents[1];

    if(step_size==0)
        step_size=(size_t)pow(A.pextents[0],0.8385);

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
    Mext[0]= A.pextents[1];
    Mext[1]=A.pextents[0];
    size_t Mstrides[2];
    Mstrides[0]= A.pstrides[0];
    Mstrides[1]=A.pstrides[1];



    #pragma omp parallel for shared(A,tempM)
    for (size_t i=0; i<nm; i++)
    {
        tempM[i]=A.pdata[i];
    }


    #pragma omp parallel for shared(Q)
    for (size_t i=0; i<Q.pdatalength; i++)
    {
        Q.pdata[i]=0;
    }

    #pragma omp parallel for shared(R)
    for (size_t i=0; i<R.pdatalength; i++)
    {
        R.pdata[i]=0;
    }


    datastruct<T> M(tempM,A.pdatalength,A.prowmajor,A.prank,Mext,Mstrides,false,false); //Copy of A
    size_t z = 0;


    const size_t mstr0=Mstrides[0];
    const size_t mstr1=Mstrides[1];
    const size_t qstr0=Q.pstrides[0];
    const size_t qstr1=Q.pstrides[1];

    for (size_t c = 0; c < m; ++c)
    {
        if (c == z +step_size)
        {
            // Extract submatrices
            size_t cz=c-z;
            size_t mc=m-c;

            size_t exts[2];
            size_t strs[2];
            size_t extbq[2];
            size_t extbm[2];
            size_t strbq[2];
            size_t strbm[2];
            size_t extc[2];
            size_t strc[2];
            size_t extbqt[2];
            size_t strbqt[2];

            datastruct<T> BQ = Q.subspanmatrix(0, z, n, cz,extbq,strbq);
            datastruct<T> BM = M.subspanmatrix(0, c, n, mc,extbm,strbm);

            // Compute C = BQ^T * BM
            datastruct<T> C(tempC,0, BM.prowmajor,cz, mc,extc,strc,true,true);

            datastruct<T> BQT=BQ.transpose(extbqt,strbqt);


            const size_t rows=BQT.pextents[0];
            const size_t cols=BM.pextents[1];
            const size_t inner_dim=BQT.pextents[1];

            const size_t strA0=BQT.pstrides[0];
            const size_t strA1=BQT.pstrides[1];

            const size_t strB0=BM.pstrides[0];
            const size_t strB1=BM.pstrides[1];

            const size_t strC0=C.pstrides[0];
            const size_t strC1=C.pstrides[1];

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += BQT(i,k,strA0,strA1) *BM(k,j,strB0,strB1);
                    }
                    C(i,j,strC0,strC1)= sum;
                }
            }
            // Compute S = BQ * C
            datastruct<T>S(tempS, 0,BQ.prowmajor,n, mc,exts,strs,true,true);

            const size_t rows2=BQ.pextents[0];
            const size_t cols2=C.pextents[1];
            const size_t inner_dim2=BQ.pextents[1];

            const size_t strA02=BQ.pstrides[0];
            const size_t strA12=BQ.pstrides[1];

            const size_t strB02=C.pstrides[0];
            const size_t strB12=C.pstrides[1];

            const size_t strC02=S.pstrides[0];
            const size_t strC12=S.pstrides[1];

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < rows2; ++i)
            {
                for (size_t j = 0; j < cols2; ++j)
                {
                    T sum = 0;
                    for (size_t k = 0; k < inner_dim2; ++k)
                    {
                        sum += BQ(i,k,strA02,strA12) *C(k,j,strB02,strB12);
                    }
                    S(i,j,strC02,strC12)= sum;
                }
            }


            const size_t strs0=strs[0];
            const size_t strs1=strs[1];
            // Update M: M[:, c:] -= S
            const size_t h=c;
            #pragma omp parallel for shared(mstr0,mstr1,M,S,strs0,strs1,h)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = h; j <n; ++j)
                {
                    M(i,  j, mstr0, mstr1) -= S(i, j-h,strs0,strs1);
                }
            }
            z = c;
        }
        // Extract column c of M
        size_t pextv[1];
        size_t pstrv[1];
        datastruct<T>  v = M.column(c,pextv,pstrv);
        const size_t pstrv0=pstrv[0];
        const size_t pext0=pextv[0];

        for (size_t j = z; j < c; ++j)
        {
            size_t pextu[1];
            size_t pstru[1];
            const datastruct<T> u = Q.column(j,pextu,pstru);
            const size_t pstru0=u.pstrides[0];
            T dot_pr=0;
            #pragma omp parallel for shared(u,v,pstru0,pstrv0)reduction(+:dot_pr)
            for (size_t i = 0; i < pext0; ++i)
            {
                dot_pr += u(i,pstru0) * v(i,pstrv0);
            }

            const T cdot_pr=dot_pr;
            #pragma omp parallel for shared(pstrv0,u,cdot_pr,pstru0)
            for (size_t i = 0; i < pext0; ++i)
            {
                v(i,pstrv0) -= cdot_pr * u(i,pstru0);
            }
        }
        // Normalize v
            T norm=0;
            #pragma omp parallel for shared(v,pstrv0) reduction(+: norm)
            for (size_t i = 0; i < pext0; ++i)
            {
                norm += v(i,pstrv0) * v(i,pstrv0);
            }

        const T normc= sqrt(norm);

        //  const T normc=norm;
        #pragma omp parallel for  shared(v,pstrv0,normc)
        for (size_t i = 0; i < pext0; ++i)
        {
            v(i,pstrv0)= v(i,pstrv0)/normc;
        }

        // Set column c of Q

        // removing the out commenting will lead clang to crash.
     //    #pragma omp parallel for shared(v,Q,c,pstrv0,qstr0,qstr1)
        for (size_t i = 0; i < pext0; ++i)
        {
            Q(i,c,qstr0,qstr1) = v(i,pstrv0);
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
template <typename T >
inline void gpu_qr_decomposition_t( const datastruct<T>&A, datastruct<T> Q, datastruct<T> &R, T* __restrict buffer=nullptr, size_t step_size=0)
{
    const size_t n = A.pextents[0];
    const size_t m = A.pextents[1];

    if(step_size==0)
        step_size=(size_t)pow(A.pextents[0],0.8385);

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
    Mext[0]= A.pextents[1];
    Mext[1]=A.pextents[0];
    size_t Mstrides[2];
    Mstrides[0]= A.pstrides[0];
    Mstrides[1]=A.pstrides[1];



    #pragma omp  teams distribute  parallel for shared(A,tempM)
    for (size_t i=0; i<nm; i++)
    {
        tempM[i]=A.pdata[i];
    }


    #pragma omp teams distribute  parallel for  shared(Q)
    for (size_t i=0; i<Q.pdatalength; i++)
    {
        Q.pdata[i]=0;
    }
//
//
    #pragma omp teams distribute  parallel for  shared(R)
    for (size_t i=0; i<R.pdatalength; i++)
    {
        R.pdata[i]=0;
    }


    datastruct<T> M(tempM,A.pdatalength,A.prowmajor,A.prank,Mext,Mstrides,false,false); //Copy of A
    size_t z = 0;


    const size_t mstr0=Mstrides[0];
    const size_t mstr1=Mstrides[1];
    const size_t qstr0=Q.pstrides[0];
    const size_t qstr1=Q.pstrides[1];

    for (size_t c = 0; c < m; ++c)
    {


        if (c == z +step_size)
        {
            // Extract submatrices
            size_t cz=c-z;
            size_t mc=m-c;

            size_t exts[2];
            size_t strs[2];
            size_t extbq[2];
            size_t extbm[2];
            size_t strbq[2];
            size_t strbm[2];
            size_t extc[2];
            size_t strc[2];
            size_t extbqt[2];
            size_t strbqt[2];

            datastruct<T> BQ = Q.subspanmatrix(0, z, n, cz,extbq,strbq);
            datastruct<T> BM = M.subspanmatrix(0, c, n, mc,extbm,strbm);

            // Compute C = BQ^T * BM
            datastruct<T> C(tempC,0, BM.prowmajor,cz, mc,extc,strc,true,true);

            datastruct<T> BQT=BQ.transpose(extbqt,strbqt);


            const size_t rows=BQT.pextents[0];
            const size_t cols=BM.pextents[1];
            const size_t inner_dim=BQT.pextents[1];

            const size_t strA0=BQT.pstrides[0];
            const size_t strA1=BQT.pstrides[1];

            const size_t strB0=BM.pstrides[0];
            const size_t strB1=BM.pstrides[1];

            const size_t strC0=C.pstrides[0];
            const size_t strC1=C.pstrides[1];

            #pragma omp teams distribute parallel for collapse(2)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum = 0;
                    #pragma omp parallel for reduction(+: sum)
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum += BQT(i,k,strA0,strA1) *BM(k,j,strB0,strB1);
                    }
                    C(i,j,strC0,strC1)= sum;
                }
            }
            // Compute S = BQ * C
            datastruct<T>S(tempS, 0,BQ.prowmajor,n, mc,exts,strs,true,true);

            const size_t rows2=BQ.pextents[0];
            const size_t cols2=C.pextents[1];
            const size_t inner_dim2=BQ.pextents[1];

            const size_t strA02=BQ.pstrides[0];
            const size_t strA12=BQ.pstrides[1];

            const size_t strB02=C.pstrides[0];
            const size_t strB12=C.pstrides[1];

            const size_t strC02=S.pstrides[0];
            const size_t strC12=S.pstrides[1];

            #pragma omp teams distribute parallel for collapse(2)
            for (size_t i = 0; i < rows2; ++i)
            {
                for (size_t j = 0; j < cols2; ++j)
                {
                    T sum = 0;
                    #pragma omp parallel for reduction(+: sum)
                    for (size_t k = 0; k < inner_dim2; ++k)
                    {
                        sum += BQ(i,k,strA02,strA12) *C(k,j,strB02,strB12);
                    }
                    S(i,j,strC02,strC12)= sum;
                }
            }

            const size_t strs0=strs[0];
            const size_t strs1=strs[1];
            // Update M: M[:, c:] -= S
            const size_t h=c;
            #pragma omp  teams distribute shared(M,S,h,strs0,strs1,mstr0,mstr1)
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp parallel for shared(M,S,h,strs0,strs1,mstr0,mstr1)
                for (size_t j = h; j <n; ++j)
                {
                    M(i,  j, mstr0, mstr1) -= S(i, j-h,strs0,strs1);
                }
            }
            z = c;
        }
        // Extract column c of M
        size_t pextv[1];
        size_t pstrv[1];
        datastruct<T>  v = M.column(c,pextv,pstrv);
        const size_t pstrv0=pstrv[0];
        const size_t pext0=pextv[0];

        for (size_t j = z; j < c; ++j)
        {
            size_t pextu[1];
            size_t pstru[1];
            const datastruct<T> u = Q.column(j,pextu,pstru);
            const size_t pstru0=u.pstrides[0];

            T dot_pr=0;
            #pragma omp parallel for shared(u,v,pstru0,pstrv0)reduction(+:dot_pr)
            for (size_t i = 0; i < pext0; ++i)
            {
                dot_pr += u(i,pstru0) * v(i,pstrv0);
            }

            const T cdot_pr=dot_pr;
            #pragma omp teams distribute  parallel for shared(u,cdot_pr,pstrv0,pstru0)
            for (size_t i = 0; i < pext0; ++i)
            {
                v(i,pstrv0) -= cdot_pr * u(i,pstru0);
            }
        }
        // Normalize v

 // Normalize v
            T norm=0;
            #pragma omp parallel for shared(v,pstrv0) reduction(+: norm)
            for (size_t i = 0; i < pext0; ++i)
            {
                norm += v(i,pstrv0) * v(i,pstrv0);
            }

        const T normc= sqrt(norm);

        //  const T normc=norm;
        #pragma omp teams distribute parallel for shared(v,pstrv0,normc)
        for (size_t i = 0; i < pext0; ++i)
        {
            v(i,pstrv0)= v(i,pstrv0)/normc;
        }

        // Set column c of Q

        // removing the out commenting will lead clang to crash.
       // #pragma omp   parallel for shared(v, Q,c,pstrv0,qstr0,qstr1)
        for (size_t i = 0; i < pext0; ++i)
        {

            Q.pdata[i * qstr0 + c *qstr1]= v.pdata[i*pstrv0];
        }
    }

    // Compute R = Q^T * A
    size_t qtext[2];
    size_t qtstrides[2];


    datastruct<T> QT=Q.transpose(qtext,qtstrides);
    gpu_matrix_multiply_dot_t(QT,A,R);

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
inline void gpu_matrix_multiply_dot_w( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
{

    const size_t rows=A.pextents[0];
    const size_t cols=B.pextents[1];
    const size_t inner_dim=A.pextents[1];

    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];

    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];

    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];

    #pragma omp parallel for collapse(2) shared(C,strC0,strC1,A,strA0,strA1,B,strB0,strB1,rows,cols, inner_dim)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k,strA0,strA1) *B(k,j,strB0,strB1);
            }
            C(i,j,strC0,strC1)= sum;
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_dot_t( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
{

    const size_t rows=A.pextents[0];
    const size_t cols=B.pextents[1];
    const size_t inner_dim=A.pextents[1];

    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];

    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];

    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];

    #pragma omp  teams distribute collapse(2) shared(C,strC0,strC1,A,strA0,strA1,B,strB0,strB1,rows,cols, inner_dim)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp parallel for reduction(+:sum) shared(C,strC0,strC1,A,strA0,strA1,B,strB0,strB1,rows,cols, inner_dim)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k,strA0,strA1) *B(k,j,strB0,strB1);
            }
            C(i,j,strC0,strC1)= sum;
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline bool gpu_matrix_add_t(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp teams distribute parallel for collapse(2) shared(C,strC0,strC1,A,strA0,strA1,B,strB0,strB1)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)+B(i,j,strB0,strB1);
        }
    }

    return true;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline bool gpu_matrix_subtract_t(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp teams distribute parallel for collapse(2)  shared(C,strC0,strC1,A,strA0,strA1,B,strB0,strB1)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)-B(i,j,strB0,strB1);
        }
    }
    return true;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline bool gpu_matrix_multiply_vector_t( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=V.pextents[0];
    const size_t strV0=V.pstrides[0];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp teams distribute parallel for collapse(2)  shared(C,strC0,strC1,M,strM0,strM1,V,strV0)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V(j,strV0);  // This works because i, k, j are row/col indices
        }
    }
    return true;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline bool gpu_matrix_multiply_vector_w( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=V.pextents[0];
    const size_t strV0=V.pstrides[0];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp parallel for collapse(2)  shared(C,strC0,strC1,M,strM0,strM1,V,strV0)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V(j,strV0);  // This works because i, k, j are row/col indices
        }
    }
    return true;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline bool gpu_matrix_multiply_vector_w( const datastruct<T>M, const T*V, datastruct<T> & C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=M.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp parallel for collapse(2)  shared(C,strC0,strC1,M,strM0,strM1,V)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V[i];  // This works because i, k, j are row/col indices
        }
    }

    return true;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline bool gpu_matrix_multiply_scalar_t(  const datastruct<T>& M, const T V, datastruct<T>& C)
{
    // Perform matrix multiplication: C = A * B

    const size_t n=C.pextents[0];
    const size_t m= C.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp teams distribute parallel for collapse(2) shared(C,strC0,strC1,M,strM0,strM1,V)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V;  // This works because i, k, j are row/col indices
        }
    }
    return true;
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline void gpu_vector_subtract_t( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    #pragma omp teams distribute parallel for  shared (res,strres,vec1,strv1,vec2,strv2)
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec1(i,strv1)-vec2(i,strv2);
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
inline bool gpu_matrix_multiply_vector_t( const datastruct<T>M, const T*V, datastruct<T> & C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=M.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp teams distribute parallel for collapse(2)  shared(C,strC0,strC1,M,strM0,strM1,V)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V[i];  // This works because i, k, j are row/col indices
        }
    }

    return true;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_vector_add_t( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    #pragma omp teams distribute parallel for shared (res,strres,vec1,strv1,vec2,strv2)
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec1(i,strv1)+vec2(i,strv2);
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_vector_subtract_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    #pragma omp parallel for  shared (res,strres,vec1,strv1,vec2,strv2)
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec1(i,strv1)-vec2(i,strv2);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_scalar_multiply_t( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.pextents[0];
    const size_t resstr=res.pstrides[0];
    const size_t vecstr=vec.pstrides[0];
    #pragma omp teams distribute parallel for  shared (res,resstr,vec,vecstr,scalar)
    for (size_t i = 0; i < n; ++i)
    {
        res(i,resstr) = vec(i,vecstr)*scalar;
    }
}
#pragma omp end  declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_vector_scalar_multiply_w( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.pextents[0];
    const size_t resstr=res.pstrides[0];
    const size_t vecstr=vec.pstrides[0];
    #pragma omp parallel for shared (res,resstr,vec,vecstr,scalar)
    for (size_t i = 0; i < n; ++i)
    {
        res(i,resstr) = vec(i,vecstr)*scalar;
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_cross_product( const datastruct<T>& vec1, const  datastruct<T>& vec2, datastruct<T>& res)
{
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    res(0,strres) = vec1(1,strv1) * vec2(2,strv2) - vec1(2,strv1) * vec2(1,strv2);
    res(1,strres) = vec1(2,strv1) * vec2(0,strv2) - vec1(0,strv1) * vec2(2,strv2);
    res(2,strres) = vec1(0,strv1) * vec2(1,strv2) - vec1(1,strv1) * vec2(0,strv2);

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_add_v(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)+B(i,j,strB0,strB1);
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_subtract_v(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)-B(i,j,strB0,strB1);
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_multiply_vector_v( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=V.pextents[0];
    const size_t strV0=V.pstrides[0];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V(j,strV0);  // This works because i, k, j are row/col indices
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_multiply_vector_v( const datastruct<T>M, const T*V, datastruct<T> & C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=M.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V[i];  // This works because i, k, j are row/col indices
        }
    }
}
#pragma omp end declare target
//
//
#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_multiply_scalar_v(  const datastruct<T>& M, const T V, datastruct<T>& C)
{
    // Perform matrix multiplication: C = A * B

    const size_t n=C.pextents[0];
    const size_t m= C.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V;  // This works because i, k, j are row/col indices
        }
    }
}
#pragma omp end declare target
//

//
#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_multiply_vector_w( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=V.pextents[0];
    const size_t strV0=V.pstrides[0];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp parallel for shared(C,strC0,strC1,strM0,strM1,V,strV0)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V(j,strV0);  // This works because i, k, j are row/col indices
        }
    }
}
#pragma omp end declare target
//
//

#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_add_w(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp parallel for collapse(2)shared(C,strC0,strC1,strA0,strA1,strB0,strB1)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)+B(i,j,strB0,strB1);
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_subtract_w(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp parallel for collapse(2) shared(C,strC0,strC1,strA0,strA1,strB0,strB1)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)-B(i,j,strB0,strB1);
        }
    }

}
#pragma omp end declare target




//

#pragma omp begin declare target
template <typename T>
inline  void gpu_matrix_multiply_scalar_w(  const datastruct<T>& M, const T V, datastruct<T>& C)
{
    // Perform matrix multiplication: C = A * B

    const size_t n=C.pextents[0];
    const size_t m= C.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    #pragma omp parallel for collapse(2) shared(C,strC0,strC1,strM0,strM1)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V;  // This works because i, k, j are row/col indices
        }
    }
}

#pragma omp end declare target





//
//
#pragma omp begin declare target
template <typename T>
inline  void gpu_vector_scalar_multiply_v( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.pextents[0];
    const size_t resstr=res.pstrides[0];
    const size_t vecstr=vec.pstrides[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i,resstr) = vec(i,vecstr)*scalar;
    }
}
#pragma omp end declare target
////
////
//
#pragma omp begin declare target
template <typename T>
inline void gpu_vector_add_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec1(i,strv1)+vec2(i,strv2);
    }

}
#pragma omp end declare target
//
#pragma omp begin declare target
template <typename T>
inline  void gpu_vector_subtract_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec1(i,strv1)-vec2(i,strv2);
    }

}
#pragma omp end declare target
//
//
#pragma omp begin declare target
template <typename T>
inline T gpu_dot_product_v(const  datastruct<T> vec1, const datastruct<T> vec2)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    T result=0;
    #pragma omp simd reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i,strv1) * vec2(i,strv2);
    }
    return result;
}

#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_dot_s( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t rows=A.pextents[0];
    const size_t cols=B.pextents[1];
    const size_t inner_dim=A.pextents[1];

    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k,strA0,strA1) *B(k,j,strB0,strB1);
            }
            C(i,j,strC0,strC1)= sum;
        }
    }
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_add_s(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)+B(i,j,strB0,strB1);
        }
    }


}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_subtract_s(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
    const size_t strA0=A.pstrides[0];
    const size_t strA1=A.pstrides[1];
    const size_t strB0=B.pstrides[0];
    const size_t strB1=B.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j,strC0,strC1) =A(i,j,strA0,strA1)-B(i,j,strB0,strB1);
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_s( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=V.pextents[0];
    const size_t strV0=V.pstrides[0];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V(j,strV0);  // This works because i, k, j are row/col indices
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_vector_s( const datastruct<T>M, const T*V, datastruct<T> & C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=M.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V[i];  // This works because i, k, j are row/col indices
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_matrix_multiply_scalar_s(  const datastruct<T>& M, const T V, datastruct<T>& C)
{
    // Perform matrix multiplication: C = A * B

    const size_t n=C.pextents[0];
    const size_t m= C.pextents[1];
    const size_t strM0=M.pstrides[0];
    const size_t strM1=M.pstrides[1];
    const size_t strC0=C.pstrides[0];
    const size_t strC1=C.pstrides[1];
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V;  // This works because i, k, j are row/col indices
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_scalar_multiply_s( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.pextents[0];
    const size_t resstr=res.pstrides[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i,resstr)*scalar;
    }
}

#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
inline void gpu_vector_add_s( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec1(i,strv1)+vec2(i,strv2);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline void gpu_vector_subtract_s( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    const size_t strres=res.pstrides[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec1(i,strv1)-vec2(i,strv2);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline T gpu_dot_product_s(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    T result=0;

    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i,strv1) * vec2(i,strv2);
    }
    return result;
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline T gpu_dot_product_w(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.pextents[0];
    const size_t strv1=vec1.pstrides[0];
    const size_t strv2=vec2.pstrides[0];
    T result=0;
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i,strv1) * vec2(i,strv2);
    }
    return result;
}
#pragma omp end declare target


template <typename T, typename CA, typename CB, typename CC>
bool strassen_multiply(const  mdspan<T, CA>& A,  const mdspan<T, CB>& B, mdspan<T, CC>& C, matrix_multiplication_parameters & algorithm)
{
    // Dimensions of input matrices
    const  size_t n = A.extent(0); // Rows in A
    const  size_t m = A.extent(1); // Columns in A and rows in B
    const size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (n*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,algorithm.gpu_offload);
        return true;
    }
    // Compute sizes for splitting
    const size_t half_n = n / 2;
    const size_t half_m = m / 2;
    const size_t half_p = p / 2;

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
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
        B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
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
    }

    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
        B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
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
    }

    mdspan<T, CC>   M1(M1_storage, s, {half_n, half_p}, {half_p, 1}),
           M2(M2_storage, s, {half_n, half_p}, {half_p, 1}),
           M3(M3_storage, s, {half_n, half_p}, {half_p, 1}),
           M4(M4_storage, s, {half_n, half_p}, {half_p, 1}),
           M5(M5_storage, s, {half_n, half_p}, {half_p, 1}),
           M6(M6_storage, s, {half_n, half_p}, {half_p, 1}),
           M7(M7_storage,s, {half_n, half_p}, {half_p, 1});

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

    #pragma omp parallel shared (A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
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

    if (algorithm.mpi==true && n*p>=algorithm.size_for_mpi)
{

    int childdest=algorithm.status.MPI_SOURCE*7;
    int commsize;
    MPI_Comm_size(algorithm.comm, &commsize);
        if (childdest+7<commsize)
        {

            MPI_Send(&m, 1, MPI_INT, childdest+1, 0, algorithm.comm);
            MPI_send_mdspan(A_result1,childdest+1,1,algorithm.comm);
            MPI_send_mdspan(B_result1,childdest+1,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+2, 0, algorithm.comm);
            MPI_send_mdspan(A_result2,childdest+2,1,algorithm.comm);
            MPI_send_mdspan(B11,childdest+2,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+3, 0, algorithm.comm);
            MPI_send_mdspan(A11,childdest+3,1,algorithm.comm);
            MPI_send_mdspan(B_result2,childdest+3,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+4, 0, algorithm.comm);
            MPI_send_mdspan(A22,childdest+4,1,algorithm.comm);
            MPI_send_mdspan(B_result3,childdest+4,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+5, 0, algorithm.comm);
            MPI_send_mdspan(A_result3,childdest+5,1,algorithm.comm);
            MPI_send_mdspan(B22,childdest+5,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+6, 0, algorithm.comm);
            MPI_send_mdspan(A_result4,childdest+6,1,algorithm.comm);
            MPI_send_mdspan(B_result4,childdest+6,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+7, 0, algorithm.comm);
            MPI_send_mdspan(A_result5,childdest+7,1,algorithm.comm);
            MPI_send_mdspan(B_result5,childdest+7,2,algorithm.comm);

            MPI_recv_mdspan_pdata(M1,childdest+1,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M2,childdest+2,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M3,childdest+3,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M4,childdest+4,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M5,childdest+5,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M6,childdest+6,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M7,childdest+7,3,algorithm.comm);
        }
        else
        {
            if(algorithm.omp)
            {
                #pragma omp parallel shared(A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
                A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5)
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

    }
    else
    {
        if(algorithm.omp)
        {
            #pragma omp parallel shared(A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
            A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5)
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
    auto C11 = C.subspan({0, 0}, {half_n, half_p});
    auto C12 = C.subspan({0, half_p}, {half_n, half_p});
    auto C21 = C.subspan({half_n, 0}, {half_n, half_p});
    auto C22 = C.subspan({half_n, half_p}, {half_n, half_p});

    #pragma omp parallel for collapse(2) shared(M2,M3,M5,M6,M7,C11,C12,C21,C22)
    for (size_t i = 0; i < half_n; ++i)
{
    for (size_t j = 0; j < half_p; ++j)
        {
            C11(i, j) = M1(i, j) + M4(i, j) - M5(i, j) + M7(i, j);
            C12(i, j) = M3(i, j) + M5(i, j);
            C21(i, j) = M2(i, j) + M4(i, j);
            C22(i, j) = M1(i, j) - M2(i, j) + M3(i, j) + M6(i, j);
        }
    }

    if (algorithm.memmapped_files)
{
    #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
    A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
    B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
    {
        delete_temp_mmap(M1_storage, s);
            delete_temp_mmap(M2_storage, s);
            delete_temp_mmap(M3_storage, s);
            delete_temp_mmap(M4_storage, s);
            delete_temp_mmap(M5_storage, s);
            delete_temp_mmap(M6_storage, s);
            delete_temp_mmap(M7_storage, s);

            delete_temp_mmap(A_result1_storage, s2);
            delete_temp_mmap(A_result2_storage, s2);
            delete_temp_mmap(A_result3_storage, s2);
            delete_temp_mmap(A_result4_storage, s2);
            delete_temp_mmap(A_result5_storage, s2);

            delete_temp_mmap(B_result1_storage, s3);
            delete_temp_mmap(B_result2_storage, s3);
            delete_temp_mmap(B_result3_storage, s3);
            delete_temp_mmap(B_result4_storage, s3);
            delete_temp_mmap(B_result5_storage, s3);
        }
    }
    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
        B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
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
    }

    return true;
}

template <typename T, typename CA, typename CB, typename CC>
bool winograd_multiply(const  mdspan<T, CA>& A, const mdspan<T, CB>& B, mdspan<T, CC>& C,matrix_multiplication_parameters& algorithm)
{
    // Dimensions of input matrices
    size_t n = A.extent(0); // Rows in A
    size_t m = A.extent(1); // Columns in A and rows in B
    size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (n*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,algorithm.gpu_offload);
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
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
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

    }
    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
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

    if (algorithm.mpi==true && n*p>=algorithm.size_for_mpi)
    {
        int source=algorithm.status.MPI_SOURCE;
        int childdest=source*7;
        int commsize;
        MPI_Comm_size(algorithm.comm, &commsize);

        if (childdest+7<commsize )
        {

            int m=COMMAND_WINOGRAD;
            MPI_Send(&m, 1, MPI_INT, childdest+1, 0, algorithm.comm);
            MPI_send_mdspan(S2,childdest+1,1,algorithm.comm);
            MPI_send_mdspan(S6,childdest+1,2,algorithm.comm);


            MPI_Send(&m, 1, MPI_INT, childdest+2, 0, algorithm.comm);
            MPI_send_mdspan(A11,childdest+2,1,algorithm.comm);
            MPI_send_mdspan(B11,childdest+2,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+3, 0, algorithm.comm);
            MPI_send_mdspan(A12,childdest+3,1,algorithm.comm);
            MPI_send_mdspan(B21,childdest+3,2,algorithm.comm);
            MPI_recv_mdspan_pdata(M3,childdest+3,3,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+4, 0, algorithm.comm);
            MPI_send_mdspan(S3,childdest+4,1,algorithm.comm);
            MPI_send_mdspan(S7,childdest+4,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+5, 0, algorithm.comm);
            MPI_send_mdspan(S1,childdest+5,1,algorithm.comm);
            MPI_send_mdspan(S5,childdest+5,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+6, 0, algorithm.comm);
            MPI_send_mdspan(S4,childdest+6,1,algorithm.comm);
            MPI_send_mdspan(B22,childdest+6,2,algorithm.comm);

            MPI_Send(&m, 1, MPI_INT, childdest+7, 0, algorithm.comm);
            MPI_send_mdspan(A22,childdest+7,1,algorithm.comm);
            MPI_send_mdspan(S8,childdest+7,2,algorithm.comm);

            MPI_recv_mdspan_pdata(M1,childdest+1,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M2,childdest+2,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M4,childdest+4,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M5,childdest+5,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M6,childdest+6,3,algorithm.comm);
            MPI_recv_mdspan_pdata(M7,childdest+7,3,algorithm.comm);

        }
        else
        {
            if(algorithm.omp)
            {
                #pragma omp parallel shared(S1,S2,S3,S4,S5,S6,S7,S8,A11,A12,B11,B21,A22,B22,M1,M2,M3,M4,M5,M6,algorithm)
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
    }
    else
    {
        if(algorithm.omp)
        {
            #pragma omp parallel shared(S1,S2,S3,S4,S5,S6,S7,S8,A11,A12,B11,B21,A22,B22,M1,M2,M3,M4,M5,M6,algorithm)
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

    auto C11 = C.subspan({0, 0}, {half_n, half_p});
    auto C12 = C.subspan({0, half_p}, {half_n, half_p});
    auto C21 = C.subspan({half_n, 0}, {half_n, half_p});
    auto C22 = C.subspan({half_n, half_p}, {half_n, half_p});

    #pragma omp parallel for collapse(2) shared(M2,M3,M5,M6,M7,T1,T2)
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
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
        {
            delete_temp_mmap(M1_storage, s);
            delete_temp_mmap(M2_storage, s);
            delete_temp_mmap(M3_storage, s);
            delete_temp_mmap(M4_storage, s);
            delete_temp_mmap(M5_storage, s);
            delete_temp_mmap(M6_storage, s);
            delete_temp_mmap(M7_storage, s);
            delete_temp_mmap(S1_result_storage, s2);
            delete_temp_mmap(S2_result_storage, s2);
            delete_temp_mmap(S3_result_storage, s2);
            delete_temp_mmap(S4_result_storage, s2);
            delete_temp_mmap(S5_result_storage, s3);
            delete_temp_mmap(S6_result_storage, s3);
            delete_temp_mmap(S7_result_storage, s3);
            delete_temp_mmap(S8_result_storage, s3);
            delete_temp_mmap(T1_result_storage, s);
            delete_temp_mmap(T2_result_storage, s);
        }
    }
    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
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
    }
    return true;
}



template <typename T, typename CA>
void cholesky_decomposition(const mdspan<T, CA>& A, mdspan<T, CA>& L, matrix_multiplication_parameters algorithm, size_t step_size=0,  bool gpu_offload=false)
{
    if (gpu_offload==true)
    {
        datastruct<T> dA=A.pdatastruct;
        datastruct<T> dL=L.pdatastruct;
        size_t bl=sizeof(T)*5*A.pdatastruct.pdatalength;
        T*buffer=(T*) omp_target_alloc(bl,omp_get_default_device());
        create_in_struct(dA);
        create_out_struct(dL);
        #pragma omp target
        {
            gpu_cholesky_decomposition_t(dA,dL, (T*) buffer,step_size);
        }
        update_host(dL);
        exit_struct(dA);
        exit_struct(dL);
        omp_target_free(buffer,omp_get_default_device());
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
            sdata=(T*) omp_alloc(sizeof(T)*tempsize,omp_null_allocator);
            adata=(T*) omp_alloc(sizeof(T*)*nn,omp_null_allocator);
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<nn; i++)
        {
            adata[i]=A.pdatastruct.pdata[i];
            L.pdatastruct.pdata[i]=0;
        }


        mdspan<T, CA> tempA(adata,A.pdatastruct.prowmajor, n,n);
        const size_t tempAstr0=tempA.pdatastruct.pstrides[0];
        const size_t tempAstr1=tempA.pdatastruct.pstrides[1];
        const size_t lstr0=L.pdatastruct.pstrides[0];
        const size_t lstr1=L.pdatastruct.pstrides[1];

        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                // Extract submatrix R = L[c:n, z:c-1]

                auto R = L.subspanmatrix(c, z,u, c - z);

                // Compute S = RR^T using a fast matrix multiplication algorithm
                mdspan<T, CA> S(sdata,R.pdatastruct.prowmajor, u,u);
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


                const size_t Sstr0=S.pdatastruct.pstrides[0];
                const size_t Sstr1=S.pdatastruct.pstrides[1];
                #pragma omp parallel for shared(tempA,tempAstr0,tempAstr1,S,Sstr0,Sstr1)
                for (size_t i = c; i < n; ++i)
                {
                    #pragma omp parallel for simd shared(c, tempA,tempAstr0,tempAstr1,S,Sstr0,Sstr1)
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i, j,tempAstr0,tempAstr1) -= S(i - c, j - c,Sstr0,Sstr1);
                    }
                }

                // Update the block boundary
                z = c;
            }

            // Update the diagonal element L[c, c]
            T tmp=tempA(c, c,tempAstr0,tempAstr1);
            #pragma omp parallel for simd reduction(-: tmp) shared(L,lstr0,lstr1)
            for (size_t k = z; k < c; ++k)
            {
                const T tmp3=L(c,k,lstr0,lstr1);
                tmp-= tmp3 * tmp3;
            }

            const T temp4= sqrt(tmp);
            L(c, c,lstr0,lstr1) =temp4;

            #pragma omp parallel for shared(tempAstr0,tempAstr1,tempA)
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = tempA(i, c,tempAstr0,tempAstr1);
                #pragma omp parallel for simd reduction(-:tmp2) shared(L,lstr0,lstr1,temp4)
                for (size_t k = z; k < c; ++k)
                {
                    tmp2 -= L(i, k,lstr0,lstr1) * L(c, k,lstr0,lstr1);
                }
                L(i, c,lstr0,lstr1)=tmp2/temp4;
            }
        }
        if(algorithm.memmapped_files)
        {
            delete_temp_mmap(sdata,tempsize);
            delete_temp_mmap(adata,nn);
        }
        else
        {
            omp_free(sdata,omp_null_allocator);
            omp_free(adata,omp_null_allocator);
        }
    }
}

template <typename T, typename CA>
void lu_decomposition(const mdspan<T, CA>& A, mdspan<T, CA>& L, mdspan<T, CA>& U, matrix_multiplication_parameters &algorithm,  size_t step_size=0,
                      bool gpu_offload=false)
{


    if (gpu_offload==true)
    {
        datastruct<T>dA=A.pdatastruct;

        datastruct<T> dL=L.pdatastruct, dU=U.pdatastruct;
        size_t bl=sizeof(T)*2*dA.pdatalength;
        T *__restrict buffer=(T*) omp_target_alloc(bl,omp_get_default_device());
        create_in_struct(dA);
        create_out_struct(dL);
        create_out_struct(dU);
        #pragma omp target
        {
            gpu_lu_decomposition_t( dA,  dL, dU, buffer,step_size);
        }
//
        update_host(dL);
        update_host(dU);
//
        exit_struct(dU);
        exit_struct(dL);
        exit_struct(dA);
       omp_target_free(buffer,omp_get_default_device());
//
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
            sdata=(T*)omp_alloc(sizeof(T)*tempsize,omp_null_allocator);
            adata=(T*)omp_alloc(sizeof(T)*nn,omp_null_allocator);
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<nn; i++)
        {
            adata[i]=A.pdatastruct.pdata[i];
            L.pdatastruct.pdata[i]=0;
            U.pdatastruct.pdata[i]=0;
        }
        mdspan<T, CA> tempA(adata,nn,A.pdatastruct.prowmajor, {A.pdatastruct.pextents[0],A.pdatastruct.pextents[1]}, {A.pdatastruct.pstrides[0], A.pdatastruct.pstrides[1]});
        const size_t tempAstr0=tempA.pdatastruct.pstrides[0];
        const size_t tempAstr1=tempA.pdatastruct.pstrides[1];
        const size_t lstr0=L.pdatastruct.pstrides[0];
        const size_t lstr1=L.pdatastruct.pstrides[1];
        const size_t ustr0=U.pdatastruct.pstrides[0];
        const size_t ustr1=U.pdatastruct.pstrides[1];
        size_t z=0;

        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;
                auto RL = L.subspanmatrix(c, z, u,v);
                auto RU = U.subspanmatrix(z, c, v, u);
                mdspan<T, CA> S(sdata,RU.pdatastruct.prowmajor, u,u);
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

                const size_t Sstr0=S.pdatastruct.pstrides[0];
                const size_t Sstr1=S.pdatastruct.pstrides[1];
                #pragma omp parallel for shared(tempA,S, tempAstr0,tempAstr1,Sstr0,Sstr1)
                for (size_t i = c; i < n; ++i)
                {
                    #pragma omp  parallel for simd shared(tempA,S, tempAstr0,tempAstr1,Sstr0,Sstr1)
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i,j,tempAstr0,tempAstr1) -= S(i - c, j - c,Sstr0,Sstr1);
                    }
                }
                z = c;
            }

            #pragma omp parallel for shared(tempA,tempAstr0,L,U,tempAstr1)
            for (size_t i = c; i < n; ++i)
            {
                T temp=tempA(c,i,tempAstr0,tempAstr1);
                #pragma omp parallel for simd reduction(-:temp) shared(L,U,ustr0,ustr1,lstr0,lstr1)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U( k,i,ustr0,ustr1) * L( c,k,lstr0,lstr1);
                }
                U(c,i,ustr0,ustr1)=temp;
            }

            const T temp4=U(c,c,ustr0,ustr1);
            #pragma omp parallel for shared(tempA,tempAstr0,L,U,tempAstr1)
            for (size_t i = c; i < n; ++i)
            {
                T temp = tempA(i,c,tempAstr0,tempAstr1);
                #pragma omp parallel for simd reduction(-:temp) shared(L,U,ustr0,ustr1,lstr0,lstr1)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U(k,c,ustr0,ustr1) * L( i,k,lstr0,lstr1);
                }
                L(i,c,lstr0,lstr1)=temp/temp4;
            }
        }


        if(algorithm.memmapped_files)
        {
            delete_temp_mmap(sdata,tempsize);
            delete_temp_mmap(adata,nn);
        }
        else
        {
            omp_free(sdata,omp_null_allocator);
            omp_free(adata,omp_null_allocator);
        }
    }
}
// Fast QR Decomposition Algorithm for mdspan
template <typename T, typename CA>
void qr_decomposition(const mdspan<T, CA>& A, mdspan<T, CA>& Q, mdspan<T, CA>& R,   matrix_multiplication_parameters algorithm,  size_t step_size=0,
                      bool gpu_offload=false)
{

    if (gpu_offload==true)
    {
        datastruct<T> dA=A.pdatastruct;
        datastruct<T> dQ=Q.pdatastruct;
        datastruct<T> dR=R.pdatastruct;
        size_t bl=sizeof(T)*(dA.pextents[1]*dA.pextents[1]+2*dA.pextents[0]* dA.pextents[1]);
        T* __restrict buffer=(T*) omp_target_alloc(bl,omp_get_default_device());
        create_in_struct(dA);
        create_out_struct(dQ);
        create_out_struct(dR);
        #pragma omp target
        {
            gpu_qr_decomposition_t(dA,dQ,dR, (T*) buffer,step_size);
        }
        update_host(dQ);
        update_host(dR);
        exit_struct(dR);
        exit_struct(dQ);
        exit_struct(dA);

       omp_target_free(buffer,omp_get_default_device());
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
            tempC=(T*)omp_alloc(sizeof(T)*m*m,omp_null_allocator);
            tempS=(T*)omp_alloc(sizeof(T)*nm,omp_null_allocator);
            tempM=(T*)omp_alloc(sizeof(T)*nm,omp_null_allocator);
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<nm; i++)
        {
            tempM[i]=A.pdatastruct.pdata[i];
        }
        #pragma omp parallel for simd
        for (size_t i=0; i<Q.pdatastruct.pdatalength; i++)
        {
            Q.pdatastruct.pdata[i]=0;
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<R.pdatastruct.pdatalength; i++)
        {
            R.pdatastruct.pdata[i]=0;
        }
        mdspan<T, CA> M(tempM,A.pdatastruct.pdatalength,A.pdatastruct.prowmajor, {A.pdatastruct.pextents[0],A.pdatastruct.pextents[1]}, {A.pdatastruct.pstrides[0],A.pdatastruct.pstrides[1]}); // Copy of A
        const size_t Mstr0=M.pdatastruct.pstrides[0];
        const size_t Mstr1=M.pdatastruct.pstrides[1];
        const size_t Qstr0=Q.pdatastruct.pstrides[0];
        const size_t Qstr1=Q.pdatastruct.pstrides[1];
        size_t z = 0;

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
                auto C = mdspan<T, CA>(tempC, BM.pdatastruct.prowmajor,cz, mc);

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
                auto S = mdspan<T, CA>(tempS, BQ.pdatastruct.prowmajor, n, mc);

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

                const size_t Sstr0=S.pdatastruct.pstrides[0];
                const size_t Sstr1=S.pdatastruct.pstrides[1];
                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i)
                {
                    #pragma omp parallel for simd shared(Sstr0,Sstr1,Mstr0,Mstr1)
                    for (size_t j = c; j < n; ++j)
                    {
                        M(i, j,Mstr0,Mstr1) -= S(i, j-c,Sstr0,Sstr1);
                    }
                }
                z = c;
            }
            // Extract column c of M
            auto v = M.column(c);
            const size_t vstr0=v.pdatastruct.pstrides[0];
            #pragma omp parallel for shared(vstr0,v)
            for (size_t j = z; j < c; ++j)
            {
                const auto u = Q.column(j);
                const size_t ustr0=u.pdatastruct.pstrides[0];
                const T dot_pr =dot_product(u,v);

                #pragma omp parallel for simd shared(ustr0,u,v,dot_pr)
                for (size_t i = 0; i < n; ++i)
                {
                    v(i,vstr0) -= dot_pr * u(i,ustr0);
                }
            }

            // Normalize v
            const T norm = sqrt(dot_product(v,v));
            #pragma omp parallel for simd shared(v,vstr0,norm)
            for (size_t i = 0; i < n; ++i)
            {
                v(i,vstr0) /= norm;
            }

            // Set column c of Q
            #pragma omp parallel for simd shared(Q,v,vstr0,Qstr0,Qstr1,c)
            for (size_t i = 0; i < n; ++i)
            {
                Q(i,c,Qstr0,Qstr1) = v(i,vstr0);
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
            delete_temp_mmap(tempC,mm);
            delete_temp_mmap(tempS,nm);
            delete_temp_mmap(tempM,nm);
        }
        else
        {
            omp_free(tempC,omp_null_allocator);
            omp_free(tempS,omp_null_allocator);
            omp_free(tempM,omp_null_allocator);
        }
    }

}

template <typename T, typename CA, typename CB, typename CC>
void matrix_multiply_dot(const mdspan<T, CA>& A, const  mdspan<T, CB>& B, mdspan<T, CC>& C, bool gpu_offload)
{


    const datastruct<T> dA=A.pdatastruct;
    const datastruct<T> dB=B.pdatastruct;
    datastruct<T> dC=C.pdatastruct;

    const size_t rows = dA.pextents[0]; // Number of rows in A and C
    const size_t cols = dB.pextents[1]; // Number of columns in B and C
    const  size_t inner_dim = dA.pextents[1]; // Number of columns in A and rows in B

    const size_t strA0=dA.pstrides[0];
    const size_t strA1=dA.pstrides[1];

    const size_t strB0=dB.pstrides[0];
    const size_t strB1=dB.pstrides[1];

    const size_t strC0=dC.pstrides[0];
    const size_t strC1=dC.pstrides[1];

    if (gpu_offload==true)
    {
        create_in_struct(dA);
        create_in_struct(dB);
        create_out_struct(dC);
        #pragma omp target enter data map(to: inner_dim, rows, cols,strA0,strA1,strB0,strB1,strC0,strC1)
        #pragma omp target teams distribute collapse(2) shared(dA, dB, dC,inner_dim, rows, cols,strA1,strA0,strB0,strB1,strC0,strC1)
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    T sum=0;
                    #pragma omp parallel for reduction(+:sum) shared(dA, dB, dC,inner_dim, rows, cols,strA1,strA0,strB0,strB1,strC0,strC1)
                    for (size_t k = 0; k < inner_dim; ++k)
                    {
                        sum+=dA(i,k,strA0,strA1)*dB(k,j,strB0,strB1);
                    }
                    dC(i,j,strC0,strC1)=sum;
                }
            }

        update_host(dC);

        #pragma omp target exit data map (delete: inner_dim, rows, cols,strA0,strA1,strB0,strB1,strC0,strC1)
        exit_struct(dA);
        exit_struct(dB);
        exit_struct(dC);
    }
    else
    {
        #pragma omp parallel for collapse(2) shared(dC,dA,dB,rows,cols,inner_dim, strC0,strC1,strA1,strA0,strB0,strB1)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp parallel for simd reduction (+:sum)  shared(inner_dim, strC0,strC1,strA1,strA0,strB0,strB1,dA,dB,dC)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += dA(i, k,strA0,strA1) * dB(k, j,strB0,strB1);
                }
                dC(i, j,strC0,strC1) = sum;
            }
        }
    }


}


template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_add( const mdspan<T, CA>& A,const   mdspan<T, CB>& B, mdspan<T, CC>& C)
{

    const size_t rows = C.pdatastruct.pextents[0];
    const size_t cols = C.pdatastruct.pextents[1];

    const size_t strA0=A.pdatastruct.pstrides[0];
    const size_t strA1=A.pdatastruct.pstrides[1];
    const size_t strB0=B.pdatastruct.pstrides[0];
    const size_t strB1=B.pdatastruct.pstrides[1];
    const size_t strC0=C.pdatastruct.pstrides[0];
    const size_t strC1=C.pdatastruct.pstrides[1];

    #pragma omp parallel for  shared(strC0,strC1,strA1,strA0,strB0,strB1,A,B,C)
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd  shared(strC0,strC1,strA1,strA0,strB0,strB1,A,B,C)
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j,strC0,strC1)=A(i,j,strA0,strA1)+B(i,j,strB0,strB1);
        }
    }

    return true;
}
template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_subtract( const mdspan<T, CA>& A,  const mdspan<T, CB>& B, mdspan<T, CC>& C)
{

    const size_t rows = C.pdatastruct.pextents[0];
    const size_t cols = C.pdatastruct.pextents[1];

    const size_t strA0=A.pdatastruct.pstrides[0];
    const size_t strA1=A.pdatastruct.pstrides[1];
    const size_t strB0=B.pdatastruct.pstrides[0];
    const size_t strB1=B.pdatastruct.pstrides[1];
    const size_t strC0=C.pdatastruct.pstrides[0];
    const size_t strC1=C.pdatastruct.pstrides[1];


    #pragma omp parallel for  shared(strC0,strC1,strA1,strA0,strB0,strB1,A,B,C)
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd  shared(strC0,strC1,strA1,strA0,strB0,strB1,A,B,C)
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j,strC0,strC1)=A(i,j,strA0,strA1)-B(i,j,strB0,strB1);
        }
    }

    return true;
}


template <typename T, typename CA, typename CB, typename CC>
inline bool matrix_multiply_vector(const mdspan<T, CA>& M, const mdspan<T, CB>& V, mdspan<T, CC>& C)
{


    const size_t rows = C.pdatastruct.pextents[0];
    const size_t cols = C.pdatastruct.pextents[1];

    const size_t strM0=M.pdatastruct.pstrides[0];
    const size_t strM1=M.pdatastruct.pstrides[1];
    const size_t strV0=V.pdatastruct.pstrides[0];
    const size_t strC0=C.pdatastruct.pstrides[0];
    const size_t strC1=C.pdatastruct.pstrides[1];


    // Perform matrix multiplication: C = A * B
    #pragma omp parallel for  shared(C,M,strM0,strM1,strC0,strC1,V,strV0)
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd shared(C,M,strM0,strM1,strC0,strC1,V,strV0)
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j),strM0,strM1 * V(j,strV0);  // This works because i, k, j are row/col indices
        }
    }

    return true;
}

template <typename T, typename CA, typename CB, typename CC>
inline  bool matrix_multiply_vector(const mdspan<T, CA>& M,const  T*V, mdspan<T, CC>& C)
{


    const size_t rows = C.pdatastruct.pextents[0];
    const size_t cols = C.pdatastruct.pextents[1];

    const size_t strM0=M.pdatastruct.pstrides[0];
    const size_t strM1=M.pdatastruct.pstrides[1];
    const size_t strC0=C.pdatastruct.pstrides[0];
    const size_t strC1=C.pdatastruct.pstrides[1];


    #pragma omp parallel for  shared(C,M,strM0,strM1,strC0,strC1,V)
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd  shared(C,M,strM0,strM1,strC0,strC1,V)
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j,strC0,strC1)= M(i, j,strM0,strM1) * V[i];
        }
    }

    return true;
}

template <typename T, typename CA, typename CC>
inline  bool matrix_multiply_scalar(const mdspan<T, CA>& M, const T& V, mdspan<T, CC>& C)
{



    const size_t rows = C.pdatastruct.pextents[0];
    const size_t cols = C.pdatastruct.pextents[1];

    const size_t strM0=M.pdatastruct.pstrides[0];
    const size_t strM1=M.pdatastruct.pstrides[1];
    const size_t strC0=C.pdatastruct.pstrides[0];
    const size_t strC1=C.pdatastruct.pstrides[1];


    // Perform matrix multiplication: C = A * B
    #pragma omp parallel for shared(C,M,strM0,strM1,strC0,strC1,V)
    for (size_t i = 0; i < rows; ++i)
    {
        T sum=0;
        #pragma omp parallel for simd shared(C,M,strM0,strM1,strC0,strC1,V)
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j,strC0,strC1)= M(i,j,strM0,strM1)*V;
        }

    }

    return true;
}

template <typename T, typename Container>
inline  T dot_product(const  mdspan<T, Container>& vec1,const   mdspan<T, Container>& vec2)
{

    const size_t n = vec1.pdatastruct.pextents[0];

    const size_t strv1=vec1.pdatastruct.pstrides[0];
    const size_t strv2=vec2.pdatastruct.pstrides[0];

    T result = 0;
    #pragma omp parallel for simd reduction(+:result) shared(strv1,strv2)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i,strv1) * vec2(i,strv2);
    }
    return result;
}

template <typename T, typename Container>
inline  void vector_scalar_multiply( const mdspan<T, Container>& vec, const T scalar,mdspan<T, Container>& res)
{
    const size_t n = vec.pdatastruct.pextents[0];

    const size_t strv=vec.pdatastruct.pstrides[0];
    const size_t strres=res.pdatastruct.pstrides[0];


    #pragma omp parallel for simd shared(res ,strres, strv, scalar)
    for (size_t i = 0; i < n; ++i)
    {
        res(i,strres) = vec(i,strv)*scalar;
    }
}


template <typename T, typename Container>
inline  void cross_product(const mdspan<T, Container>& vec1,const  mdspan<T, Container>& vec2, mdspan<T, Container>& res)
{
    const size_t strv1=vec1.pdatastruct.pstrides[0];
    const size_t strv2=vec2.pdatastruct.pstrides[0];
    const size_t strres=res.pdatastruct.pstrides[0];
    res(0,strres) = vec1(1,strv1) * vec2(2,strv2) - vec1(2,strv1) * vec2(1,strv2);
    res(1,strres) = vec1(2,strv1) * vec2(0,strv2) - vec1(0,strv1) * vec2(2,strv2);
    res(2,strres) = vec1(0,strv1) * vec2(1,strv2) - vec1(1,strv1) * vec2(0,strv2);


}
template <typename T, typename Container>
inline  void vector_add( const mdspan<T, Container>& vec1, const  mdspan<T, Container>& vec2, mdspan<T, Container>& vec3)
{
    const size_t n = vec1.pdatastruct.pextents[0];
    const size_t strv1=vec1.pdatastruct.pstrides[0];
    const size_t strv2=vec2.pdatastruct.pstrides[0];
    const size_t strres=vec3.pdatastruct.pstrides[0];
    #pragma omp parallel for simd shared(vec3,vec2,vec1,strres,strv1,strv2)
    for(size_t i=0; i<n; i++)
    {
        vec3(i,strres)=vec1(i,strv1)+vec2(i,strv2);
    }
}

template <typename T, typename Container>
inline  void vector_subtract( const mdspan<T, Container>& vec1, const mdspan<T, Container>& vec2, mdspan<T, Container>& vec3)
{
    const size_t n = vec1.pdatastruct.pextents[0];
    const size_t strv1=vec1.pdatastruct.pstrides[0];
    const size_t strv2=vec2.pdatastruct.pstrides[0];
    const size_t strres=vec3.pdatastruct.pstrides[0];
    #pragma omp parallel for simd shared(vec3,vec2,vec1,strres,strv1,strv2)
    for(size_t i=0; i<n; i++)
    {
        vec3(i,strres)=vec1(i,strv1)-vec2(i,strv2);
    }
}

template <typename T, typename Container>
void printmatrix(const mdspan<T, Container>& span)
{
    const size_t rows= span.pdatastruct.pextents[0];
    const size_t cols= span.pdatastruct.pextents[1];
    const size_t str0= span.pdatastruct.pstrides[0];
    const size_t str1= span.pdatastruct.pstrides[1];
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << span(i, j,str0,str1) << " ";
        }
        std::cout << "\n";
    }
    cout <<endl;
}

template <typename T, typename Container>
void printvector(const mdspan<T, Container>& span)
{
    const size_t rows= span.pdatastruct.pextent[0];
    const size_t str= span.pdatastruct.pstrides[0];
    for (size_t i = 0; i <rows; ++i)
    {
        std::cout << span(i,str) << " ";
    }
    cout <<endl;
}



#endif
