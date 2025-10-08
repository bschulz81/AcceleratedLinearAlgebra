#ifndef DATABLOCKCONTAINER
#define DATABLOCKCONTAINER


#include "datablock.h"
#include<iostream>
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




template<typename T>
class BlockedDataView
{


public:
    friend class DataBlock_GPU_Memory_Functions<T>;
    friend class DataBlock_Host_Memory_Functions<T>;
    friend class DataBlock_MPI_Functions<T>;
    friend class In_Kernel_Mathfunctions<T>;
    friend class GPU_Math_Functions<T>;
    friend class Math_Functions<T>;
    friend class Math_Functions_MPI<T>;
    template<typename U, typename Container>
    friend class ::mdspan;

    template<typename U, typename Container>
    friend class ::mdspan_data;

    BlockedDataView(const DataBlock<T>& db, const size_t* block_shape, bool remove_zeroblocks)
        : dblock(db)
    {

        build_blocks(block_shape,remove_zeroblocks);
    }

    ~BlockedDataView()
    {
        delete[] block_shape;
        if (dblock.dpdata_is_devptr &&omp_is_initial_device())

        {
            omp_target_free(pooled_offsets_flat, dblock.devptr_devicenum);
            omp_target_free(pooled_offsets_starts, dblock.devptr_devicenum);
        }
        else
        {
            delete[] pooled_offsets_flat;
            delete[] pooled_offsets_starts;
        }
    }

    const DataBlock<T>& get_datablock()const
    {
        return dblock();
    }


protected:
    const DataBlock<T> & dblock;
    size_t* block_shape;
    size_t* pooled_offsets_flat;   // stores concatenated coordinates
    size_t* pooled_offsets_starts;
    size_t usedblocks=0;


    void build_blocks(const size_t* bshape, bool remove_zeroblocks = false)
    {

        block_shape=new size_t[dblock.dprank];
        #pragma omp simd
        for (size_t i=0; i<dblock.dprank; i++)
            block_shape[i]=bshape[i];

        bool devptr=(dblock.dpdata_is_devptr &&omp_is_initial_device());


        switch(dblock.dprank)
        {
        case 1:
            build_blocks_rank1(bshape[0], remove_zeroblocks,devptr);
            break;
        case 2:
            build_blocks_rank2(bshape[0], bshape[1], remove_zeroblocks,devptr);
            break;
        default:
            build_blocks_arbitrary_rank(bshape, remove_zeroblocks,devptr);
            break;
        }
    }

    // --- Rank-1 specialized ---
    void build_blocks_rank1(size_t block_size, bool remove_zeroblocks,bool devptr)
    {
        const size_t nblocks = (dblock.dpextents[0] + block_size - 1) / block_size;

        pooled_offsets_flat = devptr
                              ? (size_t*)omp_target_alloc(sizeof(size_t) * nblocks, dblock.devptr_devicenum)
                              : new size_t[nblocks];

        pooled_offsets_starts = devptr
                                ? (size_t*)omp_target_alloc(sizeof(size_t) * (nblocks + 1),dblock.devptr_devicenum)
                                : new size_t[nblocks + 1];

        size_t count = 0;
        size_t ext0=dblock.dpextents[0] ;

        const T* pd=dblock.dpdata;
        if(devptr)
        {
            #pragma omp target teams distribute parallel for map (tofrom: count) shared(count) is_device_ptr(pd,pooled_offsets_flat,pooled_offsets_starts)device(dblock.devptr_devicenum)
            for (size_t bi = 0; bi < nblocks; ++bi)
            {
                const size_t offset = bi * block_size;
                const size_t diff   = ext0- offset;
                const size_t len    = (block_size < diff) ? block_size : diff;

                bool keep = true;

                if (remove_zeroblocks)
                {
                    keep = false;

                    for (size_t i = 0; i < len; ++i)
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
                    size_t slot;
                    #pragma omp atomic capture
                    slot = count++;

                    {
                        pooled_offsets_starts[slot] = slot;
                        pooled_offsets_flat[slot]   = offset;
                    }
                }
            }

            omp_target_memcpy(pooled_offsets_starts,&count,sizeof(size_t),sizeof(size_t)*count,0,dblock.devptr_devicenum,omp_get_initial_device()); // sentinel
            usedblocks = count;
        }

        else
        {

            #pragma omp parallel for shared(count)
            for (size_t bi = 0; bi < nblocks; ++bi)
            {
                const size_t offset = bi * block_size;
                const size_t diff   = ext0- offset;
                const size_t len    = (block_size < diff) ? block_size : diff;

                bool keep = true;

                if (remove_zeroblocks)
                {
                    keep = false;

                    for (size_t i = 0; i < len; ++i)
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
                    size_t slot;
                    #pragma omp atomic capture
                    slot = count++;
                    {
                        pooled_offsets_starts[slot] = slot;
                        pooled_offsets_flat[slot]   = offset;
                    }
                }

            }
            pooled_offsets_starts[count] = count; // sentinel
            usedblocks = count;
        }

    }

// --- Rank-2 specialized ---
    void build_blocks_rank2(size_t block_rows, size_t block_cols, bool remove_zeroblocks,bool devptr)
    {
        const size_t nblocks_row = (dblock.dpextents[0] + block_rows - 1) / block_rows;
        const size_t nblocks_col = (dblock.dpextents[1] + block_cols - 1) / block_cols;
        const size_t maxblocks   = nblocks_row * nblocks_col;

        pooled_offsets_flat = devptr
                              ? (size_t*)omp_target_alloc(sizeof(size_t) * 2 * maxblocks, dblock.devptr_devicenum)
                              : new size_t[2 * maxblocks];

        pooled_offsets_starts = devptr
                                ? (size_t*)omp_target_alloc(sizeof(size_t) * (maxblocks + 1),dblock.devptr_devicenum)
                                : new size_t[maxblocks + 1];

        size_t count  = 0; // block count
        const size_t ext0=dblock.dpextents[0];
        const size_t ext1=dblock.dpextents[1];
        const size_t str0=dblock.dpstrides[0];
        const size_t str1=dblock.dpstrides[1];
        const T* pd=dblock.dpdata;

        if(devptr)
        {
            #pragma omp target teams distribute map(tofrom:count) shared(count) is_device_ptr(pd,pooled_offsets_flat,pooled_offsets_starts) device(dblock.devptr_devicenum)
            for (size_t bi = 0; bi < nblocks_row; ++bi)
            {
                #pragma omp parallel for shared(count)
                for (size_t bj = 0; bj < nblocks_col; ++bj)
                {
                    const size_t row_off = bi * block_rows;
                    const size_t diff1   = ext0 - row_off;
                    const size_t tile_rows = (block_rows < diff1) ? block_rows : diff1;

                    bool keep = true;

                    const size_t col_off = bj * block_cols;
                    const size_t diff2   = ext1 - col_off;
                    const size_t tile_cols = (block_cols < diff2) ? block_cols : diff2;

                    if (remove_zeroblocks)
                    {
                        keep = false;

                        for (size_t i = 0; i < tile_rows && !keep; ++i)
                            for (size_t j = 0; j < tile_cols && !keep; ++j)
                                if (pd[(row_off + i) * str0 + (col_off + j) *str1] != T(0))
                                {
                                    keep = true;
                                    goto outofloop3;
                                }
                    }
outofloop3:
                    if (keep)
                    {
                        size_t slot;
                        #pragma omp atomic capture
                        slot = count++;
                        const size_t pos = slot * 2;
                        pooled_offsets_starts[slot] = pos;
                        pooled_offsets_flat[pos]    = row_off;
                        pooled_offsets_flat[pos+1]  = col_off;

                    }

                }
            }

            size_t count2=2*count;
            omp_target_memcpy(pooled_offsets_starts,&count2,sizeof(size_t),sizeof(size_t)*count,0,dblock.devptr_devicenum,omp_get_initial_device()); // sentinel
            usedblocks = count;
        }
        else
        {
            #pragma omp parallel for collapse(2) shared(count)
            for (size_t bi = 0; bi < nblocks_row; ++bi)
            {
                for (size_t bj = 0; bj < nblocks_col; ++bj)
                {
                    const size_t row_off = bi * block_rows;
                    const size_t diff1   = ext0 - row_off;
                    const size_t tile_rows = (block_rows < diff1) ? block_rows : diff1;
                    bool keep = true;
                    const size_t col_off = bj * block_cols;
                    const size_t diff2   = ext1 - col_off;
                    const size_t tile_cols = (block_cols < diff2) ? block_cols : diff2;

                    if (remove_zeroblocks)
                    {
                        keep = false;
                        for (size_t i = 0; i < tile_rows && !keep; ++i)
                            for (size_t j = 0; j < tile_cols && !keep; ++j)
                                if (pd[(row_off + i) * str0 + (col_off + j) *str1] != T(0))
                                {
                                    keep = true;
                                    goto outofloop4;
                                }
                    }
outofloop4:
                    if (keep)
                    {
                        size_t slot;
                        #pragma omp atomic capture
                        slot = count++;    // atomically reserve a slot
                        const size_t pos = slot * 2;
                        pooled_offsets_starts[slot] = pos;
                        pooled_offsets_flat[pos]    = row_off;
                        pooled_offsets_flat[pos+1]  = col_off;

                    }
                }
            }
            pooled_offsets_starts[count] = count*2; // sentinel
            usedblocks = count;
        }
    }

    bool is_nonzero_block(const size_t* block_shape,
                          const size_t* block_idx,
                          const size_t* tile_extents,
                          size_t rank,bool devptr)
    {
        size_t* idx=new size_t[rank];

        for(size_t i=0; i<rank; i++)
            idx[i]=0;

        return check_nonzero_recursive(block_shape, block_idx, tile_extents, rank, 0, idx, devptr);
        delete []idx;
    }

    bool check_nonzero_recursive(const size_t* block_shape,
                                 const size_t* block_idx,
                                 const size_t* tile_extents,
                                 size_t rank,
                                 size_t dim,
                                 size_t* idx,bool devptr)
    {

        if (dim == rank)
        {
            // compute linear offset
            size_t linear = 0;
            for (size_t d = 0; d < rank; ++d)
            {
                const size_t global_coord = block_idx[d] * block_shape[d] + idx[d];
                linear += global_coord * dblock.dpstrides[d];
            }
            T d;

            if(devptr)
                omp_target_memcpy(&d,dblock.dpdata,sizeof(T),0,sizeof(T)*linear,omp_get_initial_device(),dblock.devptr_devicenum);
            else
                d=dblock.dpdata[linear];

            return d != T(0);
        }

        for (size_t i = 0; i < tile_extents[dim]; ++i)
        {
            idx[dim] = i;
            if (check_nonzero_recursive(block_shape, block_idx, tile_extents, rank, dim+1, idx,devptr))
                return true;
        }
        return false;
    }

    void build_blocks_arbitrary_rank(const size_t* bshape, bool remove_zeroblocks,bool devptr)
    {
        const size_t r = dblock.dprank;

        size_t* nblocks_dim = new size_t[r];
        size_t maxblocks = 1;


        for (size_t d = 0; d < r; ++d)
        {
            nblocks_dim[d] = (dblock.dpextents[d] + bshape[d] - 1) / bshape[d];
            maxblocks *= nblocks_dim[d];
        }

        pooled_offsets_flat = devptr
                              ? (size_t*)omp_target_alloc(sizeof(size_t) * r * maxblocks, dblock.devptr_devicenum)
                              : new size_t[r * maxblocks];

        pooled_offsets_starts =devptr
                               ? (size_t*)omp_target_alloc(sizeof(size_t) * (maxblocks + 1), dblock.devptr_devicenum)
                               : new size_t[maxblocks + 1];


        size_t* idx = new size_t[r];
        for (size_t d = 0; d < r; ++d)
            idx[d] = 0;

        size_t count  = 0;
        size_t count2 = 0;

        while (true)
        {
            bool keep = true;
            if (remove_zeroblocks)
            {
                size_t* tile_extents = new size_t[r];
                for (size_t d = 0; d < r; ++d)
                {
                    const size_t offset = idx[d] * bshape[d];
                    const size_t diff   = dblock.dpextents[d] - offset;
                    tile_extents[d]     = (bshape[d] < diff) ? bshape[d] : diff;
                }
                keep = is_nonzero_block(bshape, idx, tile_extents, r,devptr);
                delete[] tile_extents;
            }

            if (keep)
            {
                if(devptr)
                {
                    omp_target_memcpy(pooled_offsets_starts,&count2,sizeof(size_t),sizeof(size_t)*count,0,dblock.devptr_devicenum,omp_get_initial_device());

                    for (size_t d = 0; d < r; ++d)
                    {
                        size_t u= idx[d] * bshape[d];
                        omp_target_memcpy(pooled_offsets_flat,&u,sizeof(size_t),sizeof(size_t)*count2,0,dblock.devptr_devicenum,omp_get_initial_device());
                    }
                    ++count2;
                    ++count;

                }
                else
                {
                    pooled_offsets_starts[count] = count2;
                    for (size_t d = 0; d < r; ++d)
                        pooled_offsets_flat[count2++] = idx[d] * bshape[d];
                    ++count;
                }
            }

            // increment multidim index
            size_t dim = 0;
            for (; dim < r; ++dim)
            {
                idx[dim]++;
                if (idx[dim] < nblocks_dim[dim])
                    break;
                idx[dim] = 0;
            }
            if (dim == r) break;
        }
        if(devptr)
            omp_target_memcpy(pooled_offsets_starts,&count2,sizeof(size_t),sizeof(size_t)*count,0,dblock.devptr_devicenum,omp_get_initial_device());
        else
            pooled_offsets_starts[count] = count2; // sentinel
        usedblocks = count;

        delete[] idx;
        delete[] nblocks_dim;
    }
};

#endif
