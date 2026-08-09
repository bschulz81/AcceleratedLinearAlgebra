
#ifndef DATABLOCKSPARSEUTILS
#define DATABLOCKSPARSEUTILS

#include <climits>
#include<iostream>
#include <omp.h>

#include "datablock.h"


class GPU_Memory_Functions;

class Host_Memory_Functions;



class In_Kernel_Mathfunctions;
class Math_Functions;
class Math_Functions_MPI;

class GPU_Math_Functions;

template<typename U, typename Container>
class mdspan;

template<typename U, typename Container>
class mdspan_data;






#pragma omp begin declare target
template<typename T>
class BlockedDataView:
    protected DataBlock<T>
{


public:
    friend class GPU_Memory_Functions;
    friend class Host_Memory_Functions;
    friend class DataBlock_MPI_Functions;
    friend class In_Kernel_Mathfunctions;
    friend class GPU_Math_Functions;
    friend class Math_Functions;
    friend class Math_Functions_MPI;
    template<typename U, typename Container>
    friend class ::mdspan;

    template<typename U, typename Container>
    friend class ::mdspan_data;

    BlockedDataView(const DataBlock<T>& db, const ptrdiff_t* bshape, bool remove_zeroblocks)
        :DataBlock<T>(db)
    {
        block_shape=new ptrdiff_t[this->dprank];
        #pragma omp simd
        for (ptrdiff_t i=0; i<this->dprank; i++)
            block_shape[i]=bshape[i];

        offsets_starts_is_devptr=(this->dpconfig.data_is_devptr &&omp_is_initial_device());

        if(offsets_starts_is_devptr)
            devnum=this->dpconfig.devicenum;
        else
            devnum=-INT_MAX;

        switch(this->dprank)
        {
        case 1:
            build_blocks_rank1(bshape[0], remove_zeroblocks);
            break;
        case 2:
            build_blocks_rank2(bshape[0], bshape[1], remove_zeroblocks);
            break;
        default:
            build_blocks_arbitrary_rank(bshape, remove_zeroblocks);
            break;
        }
    }

    ~BlockedDataView()
    {
        delete[] block_shape;
        if (offsets_starts_is_devptr &&omp_is_initial_device())
        {
            omp_target_free(pooled_offsets_flat,devnum);
            omp_target_free(pooled_offsets_starts, devnum);
        }
        else
        {
            delete[] pooled_offsets_flat;
            delete[] pooled_offsets_starts;
        }
        devnum=-INT_MAX;
    }

    const DataBlock<T>& get_datablock()const
    {
        return *this;
    }

inline ptrdiff_t number_of_blocks() const
{
    return usedblocks;
}

inline ptrdiff_t block_volume() const
{
    ptrdiff_t volume = 1;

    for(ptrdiff_t i=0;i<this->dprank;i++)
        volume *= block_shape[i];

    return volume;
}

inline ptrdiff_t sparse_work_size() const
{
    return usedblocks * block_volume();
}

protected:
    ptrdiff_t* block_shape;
    ptrdiff_t* pooled_offsets_flat;
    ptrdiff_t* pooled_offsets_starts;
    ptrdiff_t usedblocks=0;
    bool offsets_starts_is_devptr=false;
    int  devnum=-INT_MAX;



    void build_blocks_rank1(ptrdiff_t block_size, bool remove_zeroblocks)
    {
        const ptrdiff_t nblocks = (this->dpextents[0] + block_size - 1) / block_size;

        pooled_offsets_flat = offsets_starts_is_devptr
                              ? (ptrdiff_t*)omp_target_alloc(sizeof(ptrdiff_t) * nblocks,devnum)
                              : new ptrdiff_t[nblocks];

        pooled_offsets_starts = offsets_starts_is_devptr
                                ? (ptrdiff_t*)omp_target_alloc(sizeof(ptrdiff_t) * (nblocks + 1),devnum)
                                : new ptrdiff_t[nblocks + 1];

        ptrdiff_t count = 0;
        ptrdiff_t ext0=this->dpextents[0] ;

        const T* pd=this->dpdata;
        if(offsets_starts_is_devptr)
        {
            #pragma omp target teams distribute parallel for map (tofrom: count)is_device_ptr(pd) device(devnum)
            for (ptrdiff_t bi = 0; bi < nblocks; ++bi)
            {
                const ptrdiff_t offset = bi * block_size;
                const ptrdiff_t diff   = ext0- offset;
                const ptrdiff_t len    = (block_size < diff) ? block_size : diff;

                bool keep = true;

                if (remove_zeroblocks)
                {
                    keep = false;
                    for (ptrdiff_t i = 0; i < len; ++i)
                    {
                        if (pd[offset + i] != T(0))
                        {
                            keep = true;
                            goto outofloop1;
                        }
                    }
                }
outofloop1:
                if (keep)
                {
                    ptrdiff_t slot;
                    #pragma omp atomic capture
                    slot = count++;
                    {
                        pooled_offsets_starts[slot] = slot;
                        pooled_offsets_flat[slot]   = offset;
                    }
                }
            }

            omp_target_memcpy(pooled_offsets_starts,&count,sizeof(ptrdiff_t),sizeof(ptrdiff_t)*count,0,devnum,omp_get_initial_device()); // sentinel
            usedblocks = count;
        }

        else
        {

            #pragma omp parallel for
            for (ptrdiff_t bi = 0; bi < nblocks; ++bi)
            {
                const ptrdiff_t offset = bi * block_size;
                const ptrdiff_t diff   = ext0- offset;
                const ptrdiff_t len    = (block_size < diff) ? block_size : diff;

                bool keep = true;

                if (remove_zeroblocks)
                {
                    keep = false;
                    for (ptrdiff_t i = 0; i < len; ++i)
                    {
                        if (pd[offset + i] != T(0))
                        {
                            keep = true;
                            goto outofloop2;

                        }
                    }
                }

outofloop2:
                if (keep)
                {
                    ptrdiff_t slot;
                    #pragma omp atomic capture
                    slot = count++;
                    {
                        pooled_offsets_starts[slot] = slot;
                        pooled_offsets_flat[slot]   = offset;
                    }
                }

            }
            pooled_offsets_starts[count] = count;
            usedblocks = count;
        }

    }


    void build_blocks_rank2(const ptrdiff_t block_rows,const ptrdiff_t block_cols,const bool remove_zeroblocks)
    {
        const ptrdiff_t nblocks_row = (this->dpextents[0] + block_rows - 1) / block_rows;
        const ptrdiff_t nblocks_col = (this->dpextents[1] + block_cols - 1) / block_cols;
        const ptrdiff_t maxblocks   = nblocks_row * nblocks_col;

        pooled_offsets_flat = offsets_starts_is_devptr
                              ? (ptrdiff_t*)omp_target_alloc(sizeof(ptrdiff_t) * 2 * maxblocks, devnum)
                              : new ptrdiff_t[2 * maxblocks];

        pooled_offsets_starts = offsets_starts_is_devptr
                                ? (ptrdiff_t*)omp_target_alloc(sizeof(ptrdiff_t) * (maxblocks + 1),devnum)
                                : new ptrdiff_t[maxblocks + 1];

        ptrdiff_t count  = 0; // block count
        const ptrdiff_t ext0=this->dpextents[0];
        const ptrdiff_t ext1=this->dpextents[1];
        const ptrdiff_t str0=this->dpstrides[0];
        const ptrdiff_t str1=this->dpstrides[1];
        const T* pd=this->dpdata;

        if(offsets_starts_is_devptr)
        {
            #pragma omp target teams distribute parallel for collapse(2) map(tofrom:count) is_device_ptr(pd) device(devnum)
            for (ptrdiff_t bi = 0; bi < nblocks_row; ++bi)
            {
                for (ptrdiff_t bj = 0; bj < nblocks_col; ++bj)
                {
                    const ptrdiff_t row_off = bi * block_rows;
                    const ptrdiff_t diff1   = ext0 - row_off;
                    const ptrdiff_t tile_rows = (block_rows < diff1) ? block_rows : diff1;



                    const ptrdiff_t col_off = bj * block_cols;
                    const ptrdiff_t diff2   = ext1 - col_off;
                    const ptrdiff_t tile_cols = (block_cols < diff2) ? block_cols : diff2;
                    bool keep = true;
                    if (remove_zeroblocks)
                    {
                        keep = false;
                        for (ptrdiff_t i = 0; i < tile_rows && !keep; ++i)
                        {
                            for (ptrdiff_t j = 0; j < tile_cols && !keep; ++j)
                            {
                                if (pd[(row_off + i) * str0 + (col_off + j) *str1] != T(0))
                                {
                                    keep = true;
                                    goto outofloop3;
                                }
                            }
                        }
                    }
outofloop3:
                    if (keep)
                    {
                        ptrdiff_t slot;
                        #pragma omp atomic capture
                        slot = count++;
                        {
                        ptrdiff_t pos = slot * 2;
                        pooled_offsets_starts[slot] = pos;
                        pooled_offsets_flat[pos]    = row_off;
                        pooled_offsets_flat[pos+1]  = col_off;
                        }

                    }

                }
            }

            ptrdiff_t count2=2*count;
            omp_target_memcpy(pooled_offsets_starts,&count2,sizeof(ptrdiff_t),sizeof(ptrdiff_t)*count,0,devnum,omp_get_initial_device()); // sentinel
            usedblocks = count;
        }
        else
        {
            #pragma omp parallel for collapse(2)
            for (ptrdiff_t bi = 0; bi < nblocks_row; ++bi)
            {
                for (ptrdiff_t bj = 0; bj < nblocks_col; ++bj)
                {
                    const ptrdiff_t row_off = bi * block_rows;
                    const ptrdiff_t diff1   = ext0 - row_off;
                    const ptrdiff_t tile_rows = (block_rows < diff1) ? block_rows : diff1;
                    bool keep = true;
                    const ptrdiff_t col_off = bj * block_cols;
                    const ptrdiff_t diff2   = ext1 - col_off;
                    const ptrdiff_t tile_cols = (block_cols < diff2) ? block_cols : diff2;

                    if (remove_zeroblocks)
                    {
                        keep = false;
                        for (ptrdiff_t i = 0; i < tile_rows && !keep; ++i)
                            for (ptrdiff_t j = 0; j < tile_cols && !keep; ++j)
                                if (pd[(row_off + i) * str0 + (col_off + j) *str1] != T(0))
                                {
                                    keep = true;
                                    goto outofloop4;
                                }
                    }
outofloop4:
                    if (keep)
                    {
                        ptrdiff_t slot;
                        #pragma omp atomic capture
                        slot = count++;
                        {
                        const ptrdiff_t pos = slot * 2;
                        pooled_offsets_starts[slot] = pos;
                        pooled_offsets_flat[pos]    = row_off;
                        pooled_offsets_flat[pos+1]  = col_off;
                        }

                    }
                }
            }
            pooled_offsets_starts[count] = count*2;
            usedblocks = count;
        }
    }

    bool is_nonzero_block(const ptrdiff_t* block_shape,
                          const ptrdiff_t* block_idx,
                          const ptrdiff_t* tile_extents,
                          const ptrdiff_t rank)
    {
        ptrdiff_t* idx=new ptrdiff_t[rank];
        #pragma omp simd
        for(ptrdiff_t i=0; i<rank; i++)
            idx[i]=0;


    bool b=check_nonzero_recursive(block_shape, block_idx, tile_extents, rank, 0, idx);
       delete []idx;
       return b;
    }

    bool check_nonzero_recursive(const  ptrdiff_t*  block_shape,
                                 const  ptrdiff_t*  block_idx,
                                 const  ptrdiff_t*  tile_extents,
                                 const ptrdiff_t rank,
                                 const ptrdiff_t dim,
                                  ptrdiff_t*  idx)
    {

        if (dim == rank)
        {

            ptrdiff_t linear = 0;
            #pragma omp simd reduction(+:linear)
            for (ptrdiff_t d = 0; d < rank; ++d)
            {
                const ptrdiff_t global_coord = block_idx[d] * block_shape[d] + idx[d];
                linear += global_coord * this->dpstrides[d];
            }

            T d;
            if(offsets_starts_is_devptr)
                omp_target_memcpy(&d,this->dpdata,sizeof(T),0,sizeof(T)*linear,omp_get_initial_device(),devnum);
            else
                d=this->dpdata[linear];

            return d != T(0);
        }
        for (ptrdiff_t i = 0; i < tile_extents[dim]; ++i)
        {
            idx[dim] = i;
            if (check_nonzero_recursive(block_shape, block_idx, tile_extents, rank, dim+1, idx))
                return true;
        }
        return false;
    }

    void build_blocks_arbitrary_rank(const  ptrdiff_t*  bshape,const bool remove_zeroblocks)
    {
        const ptrdiff_t r = this->dprank;

        ptrdiff_t* nblocks_dim = new ptrdiff_t[r];
        ptrdiff_t maxblocks = 1;

        #pragma omp unroll partial
        for (ptrdiff_t d = 0; d < r; ++d)
        {
            nblocks_dim[d] = (this->dpextents[d] + bshape[d] - 1) / bshape[d];
            maxblocks *= nblocks_dim[d];
        }

        pooled_offsets_flat = offsets_starts_is_devptr
                              ? (ptrdiff_t*)omp_target_alloc(sizeof(ptrdiff_t) * r * maxblocks, devnum)
                              : new ptrdiff_t[r * maxblocks];

        pooled_offsets_starts =offsets_starts_is_devptr
                               ? (ptrdiff_t*)omp_target_alloc(sizeof(ptrdiff_t) * (maxblocks + 1), devnum)
                               : new ptrdiff_t[maxblocks + 1];



        ptrdiff_t* idx = new ptrdiff_t[r];
        #pragma omp simd
        for (ptrdiff_t d = 0; d < r; ++d)
            idx[d] = 0;

        ptrdiff_t count  = 0;
        ptrdiff_t count2 = 0;

        while (true)
        {
            bool keep = true;
            if (remove_zeroblocks)
            {
                ptrdiff_t* tile_extents = new ptrdiff_t[r];
                #pragma omp simd
                for (ptrdiff_t d = 0; d < r; ++d)
                {
                    const ptrdiff_t offset = idx[d] * bshape[d];
                    const ptrdiff_t diff   = this->dpextents[d] - offset;
                    tile_extents[d]     = (bshape[d] < diff) ? bshape[d] : diff;
                }
                keep = is_nonzero_block(bshape, idx, tile_extents, r);
                delete[] tile_extents;
            }

            if (keep)
            {
                if(offsets_starts_is_devptr)
                {
                    omp_target_memcpy(pooled_offsets_starts,&count2,sizeof(ptrdiff_t),sizeof(ptrdiff_t)*count,0,devnum,omp_get_initial_device());
                    #pragma omp unroll partial
                    for (ptrdiff_t d = 0; d < r; ++d)
                    {
                        ptrdiff_t u= idx[d] * bshape[d];
                        omp_target_memcpy(pooled_offsets_flat,&u,sizeof(ptrdiff_t),sizeof(ptrdiff_t)*count2,0,devnum,omp_get_initial_device());
                    }
                    ++count2;
                    ++count;

                }
                else
                {
                    pooled_offsets_starts[count] = count2;
                    #pragma omp unroll partial
                    for (ptrdiff_t d = 0; d < r; ++d)
                        pooled_offsets_flat[count2++] = idx[d] * bshape[d];
                    ++count;
                }
            }


            ptrdiff_t dim = 0;
            for (; dim < r; ++dim)
            {
                idx[dim]++;
                if (idx[dim] < nblocks_dim[dim])
                    break;
                idx[dim] = 0;
            }
            if (dim == r) break;
        }
        if(offsets_starts_is_devptr)
            omp_target_memcpy(pooled_offsets_starts,&count2,sizeof(ptrdiff_t),sizeof(ptrdiff_t)*count,0,devnum,omp_get_initial_device());
        else
            pooled_offsets_starts[count] = count2;
        usedblocks = count;

        delete[] idx;
        delete[] nblocks_dim;
    }
};
#pragma omp end declare target

#endif
