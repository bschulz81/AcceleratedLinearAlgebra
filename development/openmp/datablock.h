
#ifndef DATABLOCK
#define DATABLOCK
#include <climits>

#include <omp.h>

#include "datablockconfigstructs.h"
#include "datablockindiceshelperfunctions.hpp"

namespace expr
{
struct ExpressionValidationState;
}

class GPU_Memory_Functions;
class Host_Memory_Functions;
class DataBlock_MPI_Functions;
class In_Kernel_Mathfunctions;
class Math_Functions_MPI;
class GPU_Math_Functions;
class DataBlockUtilities;
class mdspan_utilities;
class Math_Functions_Policy;

template <typename T>
class BlockedDataView;

template <typename T, typename Container>
class mdspan;

template <typename T, typename Container>
class mdspan_data;

template <typename T>
class DistributedDataBlock;



template <typename T>
class DataBlockArray;


#pragma omp begin declare target
template <typename T>
class DataBlock
{
public:
    friend class GPU_Memory_Functions;
    friend class Host_Memory_Functions;
    friend class DataBlock_MPI_Functions;
    friend class In_Kernel_Mathfunctions;
    friend class GPU_Math_Functions;
    friend class Math_Functions;
    friend class Math_Functions_MPI;
    friend class DataBlockUtilities;
    friend class mdspan_utilities;
    friend struct expr::ExpressionValidationState;

    friend class BlockedDataView<T>;
    friend class DistributedDataBlock<T>;
    friend class DataBlockArray<T>;

    template <typename U, typename Container>
    friend class ::mdspan;

    template <typename U, typename Container>
    friend class ::mdspan_data;


    DataBlock() {};

    DataBlock(T*  data,    ptrdiff_t datalength,   ptrdiff_t   rank,
              ptrdiff_t*   extents,      ptrdiff_t*    strides,
              DataBlockConfig config, ComputeMetadata method);

    DataBlock(T*  data,    ptrdiff_t datalength,   ptrdiff_t   rank,
              ptrdiff_t*   extents,      ptrdiff_t*    strides,
              DataBlockConfig config);

    inline ptrdiff_t datalength() const;

    inline ptrdiff_t rank() const;

    inline bool rowmajor() const;


    inline int devptr_num()const;

    inline bool data_is_devptr()const;

    inline T* former_hostptr()const;

    inline T& data(ptrdiff_t i);
    inline T data(ptrdiff_t i)const;

    inline ptrdiff_t& extent(ptrdiff_t i);
    inline ptrdiff_t extent(ptrdiff_t i) const;

    inline ptrdiff_t& stride(ptrdiff_t i);
    inline ptrdiff_t stride(ptrdiff_t i) const;

    inline T* data();
    inline const T* data() const;

    inline ptrdiff_t* extents();

    inline const ptrdiff_t* extents() const;

    inline ptrdiff_t* strides();

    inline const ptrdiff_t* strides() const;

    inline T& operator()(const ptrdiff_t* indices);

    inline T& operator()(const ptrdiff_t row,  const ptrdiff_t col);

    inline T& operator()(const ptrdiff_t i);

    inline T operator()(const ptrdiff_t row, const ptrdiff_t col) const;

    inline T operator()(const ptrdiff_t i) const;

    inline T operator()(const ptrdiff_t* indices) const;
    inline bool is_conjugate() const;

    inline void print()const;
    ptrdiff_t print_to_buffer( char* buffer,   ptrdiff_t capacity) const;
    ptrdiff_t print_required_size() const;


    inline DataBlockObject DataShape() const;
    inline bool is_scalar() const;
    inline bool is_vector() const;
    inline bool is_matrix() const;
    inline bool is_tensor() const;

    inline bool is_contiguous()const;
protected:
    void printtensor_recursive(ptrdiff_t* indices, ptrdiff_t depth,bool ondevice) const;
    void printtensor_recursive_buffer( char*& cur, char* end,ptrdiff_t* indices,ptrdiff_t depth, bool ondevice) const;
    void printtensor_required_size_recursive(ptrdiff_t& count, ptrdiff_t* indices, ptrdiff_t depth,  bool ondevice) const;

    T*          dpdata = nullptr;
    ptrdiff_t      dpdatalength = 0;
    ptrdiff_t*     dpextents = nullptr;
    ptrdiff_t*     dpstrides = nullptr;

    ptrdiff_t      dprank = 0;

    DataBlockConfig dpconfig;
    T*          devptr_former_hostptr=nullptr;
    bool dpconjugate=false;

};
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
class DataBlockArray
{
public:


    inline T& operator()(const ptrdiff_t* indices,const ptrdiff_t blocknumber)
    {
        return pdata[compute_offset<OpenMPVariant::Sequential>(indices, pstridesbuffer, ptensor_rank,blocknumber)];
    };

    inline T operator()(const ptrdiff_t* indices,const ptrdiff_t blocknumber) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj( pdata[compute_offset<OpenMPVariant::Sequential>(indices, pstridesbuffer, ptensor_rank,blocknumber)]);
            }
        }
        return  pdata[compute_offset<OpenMPVariant::Sequential>(indices, pstridesbuffer, ptensor_rank,blocknumber)];
    }


    inline T& operator()(const ptrdiff_t row,  const ptrdiff_t col,const ptrdiff_t blocknumber)
    {
        T* const data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[2*blocknumber];
        const ptrdiff_t stride1=pstridesbuffer[2*blocknumber+1];

        return data_ptr[row*stride0+col*stride1];
    };

    inline T operator()(const ptrdiff_t row, const ptrdiff_t col, const ptrdiff_t blocknumber) const
    {
        const T* data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[2*blocknumber];
        const ptrdiff_t stride1=pstridesbuffer[2*blocknumber+1];

        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj(data_ptr[row*stride0+col*stride1]);
            }
        }

        return data_ptr[row*stride0+col*stride1];
    }

    inline T& operator()(const ptrdiff_t i,const ptrdiff_t blocknumber)
    {
        T* const data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[blocknumber];
        return data_ptr[i*stride0];
    };

    inline T operator()(const ptrdiff_t i,const ptrdiff_t blocknumber) const
    {
        const T* data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[blocknumber];
        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj(data_ptr[i*stride0]);
            }
        }

        return  data_ptr[i*stride0];
    }

    inline DataBlock<T> get_datablock_from_arrays(const ptrdiff_t i)const
    {

        ptrdiff_t len =(i + 1 <pnumblocks)? pblock_offsets[i+1] - pblock_offsets[i]: pdatalength - pblock_offsets[i];
        DataBlock<T>tempt(pdata + pblock_offsets[i],
                          len,  ptensor_rank,
                          pextentsbuffer + i*ptensor_rank,
                          pstridesbuffer + i*ptensor_rank,
                          ::DataBlockConfig{.dprowmajor=prowm,
                                            .data_is_devptr=pdata_is_devptr,
                                            .devicenum=pdata_is_devptr? pdevnum:-INT_MAX
                                           } );
        tempt.dpconjugate=pconjugate;
        return tempt;

    }

    T* pdata=nullptr;
    ptrdiff_t pdatalength=0;
    bool prowm=true;
    ptrdiff_t ptensor_rank=0;
    ptrdiff_t *pblock_offsets=nullptr;
    ptrdiff_t* pextentsbuffer=nullptr;
    ptrdiff_t* pstridesbuffer=nullptr;
    ptrdiff_t pnumblocks=0;
    bool pdata_is_devptr=false;
    int pdevnum=-INT_MAX;
    bool pconjugate=false;

};
#pragma omp end declare target

#include "datablock.hpp"

#endif
