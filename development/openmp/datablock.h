#ifndef DATABLOCK
#define DATABLOCK

#include <omp.h>
#include <stdio.h>
#include "indiceshelperfunctions.h"


#if defined(Unified_Shared_Memory)
#pragma omp requires unified_shared_memory
#else
#pragma omp requires unified_address
#endif



#pragma omp begin declare target
inline void fill_strides(const size_t*    extents,size_t*    strides, const size_t rank, const bool rowmajor)
{
    if (rank==0)
        return;

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



template<typename T>
class DataBlock_GPU_Memory_Functions;

template<typename T>
class DataBlock_Host_Memory_Functions;

template<typename T>
class DataBlock_MPI_Functions;

template<typename T>
class BlockedDataView;

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

template<typename U, typename Container>
class mdspan_data;


class Math_Functions_Policy;

enum class DataBlockObject
{
    Scalar,
    Vector,
    Matrix,
    Tensor
};



#pragma omp begin declare target
template <typename T>
class DataBlock
{
public:
    friend class DataBlock_GPU_Memory_Functions<T>;
    friend class DataBlock_Host_Memory_Functions<T>;
    friend class DataBlock_MPI_Functions<T>;
    friend class In_Kernel_Mathfunctions<T>;
    friend class GPU_Math_Functions<T>;
    friend class Math_Functions<T>;
    friend class Math_Functions_MPI<T>;
    friend class BlockedDataView<T>;

    template<typename U, typename Container>
    friend class ::mdspan;

    template<typename U, typename Container>
    friend class ::mdspan_data;


    DataBlock() {};

    // Constructors
    DataBlock(T*  data, size_t datalength, bool rowm, size_t rank,size_t*   extents, size_t*   strides,
               bool compute_datalength,    bool compute_strides_from_extents,bool data_is_devptr,int devicenum=-1 );

    DataBlock(T*  data,size_t datalength,bool rowm,size_t rows, size_t cols,  size_t*  extents, size_t*  strides,
               bool compute_datalength, bool compute_strides_from_extents,  bool data_is_devptr,int devicenum=-1);


    DataBlock(T*  data, size_t datalength, bool rowm,  size_t rank, size_t*  extents, size_t*  strides, bool data_is_devptr,int devicenum=-1 );

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
    inline int devptr_num()const
    {
        return devptr_devicenum;
    }
    inline bool is_dev_ptr()const
    {
        return dpdata_is_devptr;
    }
    inline T* former_hostptr()const
    {
        return devptr_former_hostptr;
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

    inline DataBlock<T>subspan(const size_t *    poffsets,const size_t *   psub_extents, size_t* new_extents, size_t*    psub_strides)const;

    inline DataBlock<T>subspan_copy(const size_t *    poffsets,const size_t *   psub_extents, size_t* new_extents, size_t*   psub_strides, T*    sub_data)const;

    inline DataBlock<T>subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides)const;

    inline DataBlock<T>subspanmatrix_copy_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const;
    inline DataBlock<T>subspanmatrix_copy_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const ;
    inline DataBlock<T>subspanmatrix_copy_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const;

    inline DataBlock<T> transpose(size_t*    newextents, size_t*    newstrides)const;

    inline DataBlock<T> transpose_copy_v(size_t*    newextents, size_t*    newstrides,T* newdata)const;
    inline DataBlock<T> transpose_copy_w(size_t*    newextents, size_t*    newstrides,T* newdata)const;
    inline DataBlock<T> transpose_copy_s(size_t*    newextents, size_t*    newstrides,T* newdata)const;

    inline DataBlock<T> row(const size_t row_index, size_t*    newextents, size_t*    newstrides)const;
    inline DataBlock<T> column(const size_t col_index, size_t*    newextents, size_t*    newstrides)const;

    inline DataBlock<T> column_copy_w(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline DataBlock<T> column_copy_v(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline DataBlock<T> column_copy_s(const size_t col_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;

    inline DataBlock<T> row_copy_w(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline DataBlock<T> row_copy_v(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;
    inline DataBlock<T> row_copy_s(const size_t row_index, size_t*    newextents,size_t *    new_strides, T* newdata)const;

    size_t count_noncollapsed_dims() const;
    DataBlock<T> collapsed_view(size_t num_non_collapsed_dims,size_t* extents, size_t* strides) const;
    inline bool is_contiguous()const;
    inline void printtensor()const;


    template <typename Expr>
    requires requires(Expr e, DataBlock<T>& self, const Math_Functions_Policy* pol) {
        e.assign_to(self, pol);
    }
    DataBlock& operator=(const Expr& expr) {
        expr.assign_to(*this, nullptr);
        return *this;
    }

    template <typename Expr>
    requires requires(Expr e, DataBlock<T>& self, const Math_Functions_Policy* pol) {
        e.assign_to(self, pol);
    }
    DataBlock& assign(const Expr& expr, const Math_Functions_Policy* policy) {
        expr.assign_to(*this, policy);
        return *this;
    }

    enum Type
    {
        Scalar,
        Vector,
        Matrix,
        Tensor
    };

    inline Type ObjectType() const;

    inline bool is_scalar() const
    {
        return ObjectType() == Type::Scalar;
    }
    inline bool is_vector() const
    {
        return ObjectType() == Type::Vector;
    }
    inline bool is_matrix() const
    {
        return ObjectType() == Type::Matrix;
    }
    inline bool is_tensor() const
    {
        return ObjectType() == Type::Tensor;
    }

protected:
    void printtensor_recursive(size_t* indices, size_t depth,bool ondevice) const;

    T*          dpdata = nullptr;
    size_t*     dpextents = nullptr;
    size_t*     dpstrides = nullptr;
    size_t      dpdatalength = 0;
    size_t      dprank = 0;
    bool        dprowmajor = true;
    int         devptr_devicenum=-1;
    bool        dpdata_is_devptr=false;
    T*          devptr_former_hostptr=nullptr;
};
#pragma omp end declare target





#pragma omp begin declare target
template<typename T>
DataBlock<T>::DataBlock(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t*    extents,
    size_t*    strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr,
    int devicenum

) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
    devptr_devicenum( devicenum),
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
template<typename T> DataBlock<T>::DataBlock(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t*    extents,
    size_t*    strides,
    bool data_is_devptr,
    int devicenum
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
    devptr_devicenum( devicenum),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false)
#else
    dpdata_is_devptr(data_is_devptr)
#endif
{}
#pragma omp end declare target





#pragma omp begin declare target
template<typename T> DataBlock<T>::DataBlock(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rows,
    size_t cols,
    size_t*    extents,
    size_t*    strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr,
    int devicenum
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprowmajor(rowm),
    devptr_devicenum( devicenum),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false)
#else
    dpdata_is_devptr(data_is_devptr)
#endif
{
    if((rows>1) && (cols>1))
    {

        dprank=2;
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
    else
    {

        dprank=1;
        if (rows>1)
        {
            dpdatalength=rows;
            dpextents[0]=rows;
        }
        else
        {
            dpdatalength=cols;
            dpextents[0]=cols;
        }
        dpstrides[0]=1;
    }

}
#pragma omp end declare target






#pragma omp begin declare target
template<typename T>inline DataBlock<T> DataBlock<T>::transpose(size_t*    newextents, size_t *newstrides)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];
    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];

    return DataBlock(dpdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
DataBlock<T>::Type  DataBlock<T>::ObjectType() const
{
    if (dprank == 1)
    {
        if (dpextents[0] == 1) return Type::Scalar;
        return Type::Vector;
    }
    if (dprank == 2)
    {
        if (dpextents[0] == 1 && dpextents[1] == 1) return Type::Scalar;
        if (dpextents[0] == 1 || dpextents[1] == 1) return Type::Vector;
        return Type::Matrix;
    }
    if (dprank > 2) return Type::Tensor;

    // fallback
    return Type::Scalar;
}
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>inline DataBlock<T> DataBlock<T>::transpose_copy_w(size_t*    newextents, size_t *newstrides, T* newdata)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];
    T* pd=this->dpdata;

    const size_t rows=dpextents[0];
    const size_t cols=dpextents[1];
    const size_t old_s0=dpstrides[0];
    const size_t old_s1=dpstrides[1];
    const size_t new_s0=newstrides[0];
    const size_t new_s1=newstrides[1];
    if(omp_is_initial_device()&&dpdata_is_devptr)
    {

        #pragma omp target parallel for simd collapse(2) device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
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

    return DataBlock(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>inline DataBlock<T> DataBlock<T>::transpose_copy_v(size_t*    newextents, size_t *newstrides, T* newdata)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];

    T* pd=this->dpdata;

    const size_t rows=dpextents[0];
    const size_t cols=dpextents[1];
    const size_t old_s0=dpstrides[0];
    const size_t old_s1=dpstrides[1];
    const size_t new_s0=newstrides[0];
    const size_t new_s1=newstrides[1];


    if(omp_is_initial_device()&&dpdata_is_devptr)
    {
        #pragma omp target simd collapse(2) device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
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
    return DataBlock(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
inline DataBlock<T> DataBlock<T>::transpose_copy_s(size_t*    newextents, size_t *newstrides, T* newdata)const
{

    newextents[0]=dpextents[1];
    newextents[1]=dpextents[0];

    newstrides[0]=dpstrides[1];
    newstrides[1]=dpstrides[0];
    T* pd=this->dpdata;

    const size_t rows=dpextents[0];
    const size_t cols=dpextents[1];
    const size_t old_s0=dpstrides[0];
    const size_t old_s1=dpstrides[1];
    const size_t new_s0=newstrides[0];
    const size_t new_s1=newstrides[1];

    if(omp_is_initial_device()&&dpdata_is_devptr)
    {
        #pragma omp target  device(devptr_devicenum) is_device_ptr(newdata) is_device_ptr(pd)
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
        #pragma omp unroll
        for (size_t i=0; i<rows; i++)
            for (size_t j=0; j<cols; j++)
            {
                size_t dst_index = j*new_s0+i*new_s1;
                size_t src_index = i*old_s0+ j*old_s1;
                newdata[dst_index] = pd[src_index];
            }
    }



    return DataBlock(newdata,dpdatalength,dprowmajor,2,newextents,newstrides,dpdata_is_devptr);

}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
bool DataBlock<T>::is_contiguous() const
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
template <typename T>
void DataBlock<T>::printtensor() const
{
    size_t* indices= new size_t[dprank];
    #pragma omp simd
    for (size_t i = 0; i < dprank; ++i)
        indices[i] = 0;

    bool ondevice=omp_is_initial_device()&&dpdata_is_devptr;
    printtensor_recursive(indices, 0,ondevice);
    delete []indices;

    printf("\n");
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void DataBlock<T>::printtensor_recursive(size_t* indices, size_t depth,bool ondevice) const
{
    if (depth == dprank)
    {
        size_t offset=compute_offset_s(indices, dpstrides, dprank);
        T d;
        if(ondevice)
            omp_target_memcpy(&d,dpdata,sizeof(T),0,sizeof(T)*offset,omp_get_initial_device(),this->devptr_devicenum);
        else
            d= dpdata[offset];

        printf("%g",d); // element access via operator()(size_t*)
        return;
    }

    printf("[");

    for (size_t i = 0; i < dpextents[depth]; ++i)
    {
        indices[depth] = i;
        printtensor_recursive(indices, depth + 1,ondevice);

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
template<typename T>
size_t DataBlock<T>::count_noncollapsed_dims() const
{
    size_t count = 0;
    for (size_t i = 0; i < dprank; ++i)
        if (dpextents[i] > 1) ++count;
    return count == 0 ? 1 : count;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
DataBlock<T> DataBlock<T>::collapsed_view(size_t num_non_collapsed_dims,size_t *extents, size_t *strides) const
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

    // Create non-owning DataBlock
    DataBlock<T> view(
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
DataBlock<T>DataBlock<T>::subspan(const size_t * poffsets, const size_t * psub_extents,size_t* newextents, size_t*    new_strides)const
{
    const size_t r = dprank;
    size_t offset_index = 0;
    size_t length_index = 0; // for computing total length

    // Compute offset and copy strides
    #pragma omp unroll
    for (size_t i = 0; i < r; ++i)
    {
        offset_index  += poffsets[i] * dpstrides[i];
        length_index  += (psub_extents[i] - 1) * dpstrides[i];
    }

    // Count non-collapsed dimensions
    size_t rank_out = 0;
    #pragma omp unroll
    for (size_t i = 0; i < r; ++i)
        if (psub_extents[i] > 1) ++rank_out;
    if (rank_out == 0) rank_out = 1;

    // Allocate temporary arrays for collapsed extents/strides if needed
    if (rank_out != r)
    {
        size_t idx = 0;
        #pragma omp unroll
        for (size_t i = 0; i < r; ++i)
        {
            if (psub_extents[i] > 1)
            {
                newextents[idx] = psub_extents[i];
                new_strides[idx] = dpstrides[i] ;
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
               dpdata + offset_index,
               length_index + 1,
               dprowmajor,
               rank_out,
               newextents,
               new_strides,
               dpdata_is_devptr
           );
}
#pragma omp end  declare target


#pragma omp begin declare target
template<typename T>
DataBlock<T> DataBlock<T>::subspan_copy(
    const size_t* poffsets,
    const size_t* psub_extents,
    size_t* new_extents,
    size_t* new_strides,
    T* sub_data) const
{
    const size_t r = dprank;

    // Temporary strides for walking the sub-block during copy
    size_t* tempstr = new size_t[r];
    fill_strides(psub_extents, tempstr, r, dprowmajor);

    // Allocate index arrays
    size_t* indices        = new size_t[r]();
    size_t* global_indices = new size_t[r];

    bool tmcpy = omp_is_initial_device() && dpdata_is_devptr;

    // Copy loop
    while (true)
    {
        for (size_t i = 0; i < r; ++i)
            global_indices[i] = poffsets[i] + indices[i];

        size_t original_index = compute_offset_s(global_indices, dpstrides, r);
        size_t buffer_index   = compute_offset_s(indices, tempstr, r);

        if (tmcpy)
            omp_target_memcpy(sub_data,
                              dpdata,
                              sizeof(T),
                              sizeof(T) * buffer_index,
                              sizeof(T) * original_index,
                              devptr_devicenum,
                              devptr_devicenum);
        else
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

    // Collapse trivial dimensions
    size_t rank_out = 0;
    for (size_t i = 0; i < r; ++i)
        if (psub_extents[i] > 1) ++rank_out;
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

    // Compute pl AFTER collapsing
    size_t pl = compute_data_length_s(new_extents, new_strides, rank_out);

    return DataBlock(
               sub_data,
               pl,
               dprowmajor,
               rank_out,
               new_extents,
               new_strides,
               dpdata_is_devptr
           );
}
#pragma omp end declare target







#pragma omp begin declare target
template<typename T>
DataBlock<T>  DataBlock<T>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides)const
{
    psub_strides[0] = dpstrides[0];
    psub_strides[1] = dpstrides[1];
    psub_extents[0] = tile_rows;
    psub_extents[1] = tile_cols;

    size_t offset = row * dpstrides[0] + col * dpstrides[1];
    T* data_ptr = dpdata + offset;

    // Decide rank based on tile size
    if (tile_rows == 1 && tile_cols == 1)
    {
        psub_extents[0] = 1;
        psub_strides[0]=1;
        return DataBlock<T>(data_ptr, 1, dprowmajor, 1, psub_extents, psub_strides, dpdata_is_devptr);
    }
    else if (tile_rows == 1)
    {
        // Row vector
        psub_extents[0] = tile_cols;
        psub_strides[0] = dpstrides[1];
        return DataBlock<T>(data_ptr, tile_cols, dprowmajor, 1, psub_extents, psub_strides, dpdata_is_devptr);
    }
    else if (tile_cols == 1)
    {
        // Column vector
        psub_extents[0] = tile_rows;
        psub_strides[0] = dpstrides[0];
        return DataBlock<T>(data_ptr, tile_rows, dprowmajor, 1, psub_extents, psub_strides, dpdata_is_devptr);
    }
    else
    {
        // Full matrix
        size_t pl = (tile_rows-1) * dpstrides[0] + (tile_cols-1) * dpstrides[1] + 1;
        return DataBlock<T>(data_ptr, pl, dprowmajor, 2, psub_extents, psub_strides, dpdata_is_devptr);
    }
}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
DataBlock<T>  DataBlock<T>::subspanmatrix_copy_w( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const
{

    const size_t s0 = dpstrides[0];
    const size_t s1 = dpstrides[1];
    const T* pd = dpdata;

    // Set extents and strides
    psub_extents[0] = tile_rows;
    psub_extents[1] = tile_cols;
    psub_strides[0] = dprowmajor ? tile_cols : 1;
    psub_strides[1] = dprowmajor ? 1 : tile_rows;

    // Copy data
    if (omp_is_initial_device() && dpdata_is_devptr)
    {
        #pragma omp target parallel for simd collapse(2) device(devptr_devicenum) \
        is_device_ptr(sub_data) is_device_ptr(pd)
        for (size_t i = 0; i < tile_rows; ++i)
            for (size_t j = 0; j < tile_cols; ++j)
                sub_data[i * psub_strides[0] + j * psub_strides[1]] = pd[(row+i)*s0 + (col+j)*s1];
    }
    else
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i < tile_rows; ++i)
            for (size_t j = 0; j < tile_cols; ++j)
                sub_data[i * psub_strides[0] + j * psub_strides[1]] = pd[(row+i)*s0 + (col+j)*s1];
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
    return DataBlock<T>(sub_data, length, dprowmajor, rank_out, psub_extents, psub_strides, dpdata_is_devptr);

}
#pragma omp end declare target








#pragma omp begin declare target
template<typename T>
DataBlock<T>  DataBlock<T>::subspanmatrix_copy_v( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const
{


    const size_t s0 = dpstrides[0];
    const size_t s1 = dpstrides[1];
    const T* pd = dpdata;

    // Set extents and strides
    psub_extents[0] = tile_rows;
    psub_extents[1] = tile_cols;
    psub_strides[0] = dprowmajor ? tile_cols : 1;
    psub_strides[1] = dprowmajor ? 1 : tile_rows;

    // Copy data
    if (omp_is_initial_device() && dpdata_is_devptr)
    {
        #pragma omp target  simd collapse(2) device(devptr_devicenum) \
        is_device_ptr(sub_data) is_device_ptr(pd)
        for (size_t i = 0; i < tile_rows; ++i)
            for (size_t j = 0; j < tile_cols; ++j)
                sub_data[i * psub_strides[0] + j * psub_strides[1]] = pd[(row+i)*s0 + (col+j)*s1];
    }
    else
    {
        #pragma omp simd collapse(2)
        for (size_t i = 0; i < tile_rows; ++i)
            for (size_t j = 0; j < tile_cols; ++j)
                sub_data[i * psub_strides[0] + j * psub_strides[1]] = pd[(row+i)*s0 + (col+j)*s1];
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
    return DataBlock<T>(sub_data, length, dprowmajor, rank_out, psub_extents, psub_strides, dpdata_is_devptr);

}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
DataBlock<T>  DataBlock<T>::subspanmatrix_copy_s( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,  size_t *    psub_extents,  size_t *   psub_strides, T*    sub_data)const
{


    const size_t s0 = dpstrides[0];
    const size_t s1 = dpstrides[1];
    const T* pd = dpdata;

    // Set extents and strides
    psub_extents[0] = tile_rows;
    psub_extents[1] = tile_cols;
    psub_strides[0] = dprowmajor ? tile_cols : 1;
    psub_strides[1] = dprowmajor ? 1 : tile_rows;

    // Copy data
    if (omp_is_initial_device() && dpdata_is_devptr)
    {
        #pragma omp target device(devptr_devicenum) is_device_ptr(sub_data) is_device_ptr(pd)
        for (size_t i = 0; i < tile_rows; ++i)
            for (size_t j = 0; j < tile_cols; ++j)
                sub_data[i * psub_strides[0] + j * psub_strides[1]] = pd[(row+i)*s0 + (col+j)*s1];

    }
    else
    {
        #pragma omp unroll
        for (size_t i = 0; i < tile_rows; ++i)
            for (size_t j = 0; j < tile_cols; ++j)
                sub_data[i * psub_strides[0] + j * psub_strides[1]] = pd[(row+i)*s0 + (col+j)*s1];
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
    return DataBlock<T>(sub_data, length, dprowmajor, rank_out, psub_extents, psub_strides, dpdata_is_devptr);


}
#pragma omp end declare target






#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::row(const size_t row_index, size_t*    extents,size_t *    new_strides)const
{
    extents[0] = dpextents[1];
    new_strides[0]=dpstrides[1];

    return DataBlock<T>( dpdata + row_index * dpstrides[0],  dpstrides[1] * extents[0],dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::row_copy_w(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    const size_t pl=dpextents[1];
    extents[0] = pl;

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;

    if(omp_is_initial_device()&&dpdata_is_devptr)
    {

        #pragma omp target parallel for simd device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[row_index*s0+j*s1];
    }
    else
    {
        #pragma omp parallel for simd
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[row_index*s0+j*s1];
    }

    return DataBlock<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::row_copy_v(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{
    const size_t pl=dpextents[1];
    extents[0] = pl;
    new_strides[0] = 1;

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    if(omp_is_initial_device()&&dpdata_is_devptr)
    {
        #pragma omp target simd device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[row_index*s0+j*s1];
    }
    else
    {
        #pragma omp simd
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[row_index*s0+j*s1];
    }

    return DataBlock<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::row_copy_s(const size_t row_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{
    const size_t pl=dpextents[1];
    extents[0] = pl;
    new_strides[0] =1;

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    if(omp_is_initial_device()&&dpdata_is_devptr)
    {
        #pragma omp target  device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[row_index*s0+j*s1];
    }
    else
    {
        #pragma omp unroll
        for (size_t j = 0; j < pl; ++j)
            newdata[j] = pd[row_index*s0+j*s1];
    }

    return DataBlock<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::column_copy_v(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    const size_t pl=dpextents[0];
    extents[0] = pl;
    new_strides[0] = 1;

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];

    const T*    pd=dpdata;
    if(omp_is_initial_device()&&dpdata_is_devptr)
    {
        #pragma omp target simd device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[ i*s0+col_index*s1];
    }
    else
    {
        #pragma omp simd
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[ i*s0+col_index*s1];
    }


    return DataBlock<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::column_copy_s(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    const size_t pl=dpextents[0];
    extents[0] = pl;
    new_strides[0] = 1;

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    if(omp_is_initial_device()&&dpdata_is_devptr)
    {
        #pragma omp target device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[ i*s0+col_index*s1];
    }
    else
    {

        #pragma omp unroll
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[ i*s0+col_index*s1];
    }

    return DataBlock<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::column_copy_w(const size_t col_index, size_t*    extents,size_t *    new_strides, T* newdata)const
{

    const size_t pl=dpextents[0];
    extents[0] = pl;

    new_strides[0] = dprowmajor ? dpstrides[1] : dpstrides[0];

    const size_t s0=dpstrides[0];
    const size_t s1=dpstrides[1];
    const T*    pd=dpdata;
    if(omp_is_initial_device()&&dpdata_is_devptr)
    {
        #pragma omp target parallel for simd device(devptr_devicenum)is_device_ptr(newdata) is_device_ptr(pd)
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[i*s0+col_index*s1];
    }
    else
    {
        #pragma omp parallel for simd
        for (size_t i = 0; i < pl; ++i)
            newdata[i] = pd[i*s0+col_index*s1];
    }


    return DataBlock<T>(newdata,  pl,dprowmajor,   1, extents,    new_strides, dpdata_is_devptr);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
DataBlock<T> DataBlock<T>::column(const size_t col_index, size_t*    extents,size_t *   new_strides)const
{
    extents[0] = dpextents[0];
    new_strides[0]=dpstrides[0];
    return DataBlock(dpdata + col_index * dpstrides[1], dpstrides[0] * extents[0],dprowmajor,  1, extents,   new_strides,dpdata_is_devptr );
}
#pragma omp end declare target




#endif
