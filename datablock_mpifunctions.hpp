#ifndef DATABLOCK_MPIFUNCTIONShpp
#define DATABLOCK_MPIFUNCTIONShpp

#include "datablock.h"
#include "host_memory_functions.h"
#include "gpu_memory_functions.h"


template <typename T>
void DataBlock_MPI_Functions::MPI_Free_DistributedDataBlock(
    DistributedDataBlock<T>& m)
{
    if(m.Dblockarray.pnumblocks > 0)
    {

        if(m.Dblockarray.pextentsbuffer!=nullptr)
        {
            free(m.Dblockarray.pextentsbuffer);
            m.Dblockarray.pextentsbuffer=nullptr;
        }
        if(m.Dblockarray.pstridesbuffer!=nullptr)
        {
            free(m.Dblockarray.pstridesbuffer);
            m.Dblockarray.pstridesbuffer=nullptr;
        }
        if(m.plocal_block_global_grid_indices!=nullptr)
        {
            free(m.plocal_block_global_grid_indices);
            m.plocal_block_global_grid_indices=nullptr;
        }

        if(m.Dblockarray.pdata != nullptr)
        {
#if defined(Unified_Shared_Memory)
            if (m.conf.pmemmap)
            {
                Host_Memory_Functions::delete_temp_mmap<T>(m.Dblockarray.pdata, m.Dblockarray.pdatalength);
            }
            else
            {
                free(m.Dblockarray.pdata);
            }
            m.Dblockarray.pdata=nullptr;

#else
            if(m.Dblockarray.conf.data_is_devptr)
            {
                omp_target_free(m.Dblockarray.pdata,m.Dblockarray.conf.devicenum);
            }
            else
            {
                if (m.Dblockarray.conf.pmemmap)
                {
                    Host_Memory_Functions::delete_temp_mmap<T>(m.Dblockarray.pdata,m.Dblockarray.pdatalength);
                }
                else
                {
                    free(m.Dblockarray.pdata);
                }
            }
            m.Dblockarray.pdata=nullptr;

#endif
        }

    }


    if(m.plocal_block_grid_coords)
    {
        free(m.plocal_block_grid_coords);
        m.plocal_block_grid_coords=nullptr;
    }
    if(m.pglobal_extents)
    {
        free(m.pglobal_extents);
        m.pglobal_extents=nullptr;
    }
    if(m.pdefault_block_shape)
    {
        free(m.pdefault_block_shape);
        m.pdefault_block_shape=nullptr;
    }

    if(m.pglobal_strides)
    {
        free(m.pglobal_strides);
        m.pglobal_strides=nullptr;
    }

    if(m.Dblockarray.pblock_offsets)
    {
        free(m.Dblockarray.pblock_offsets);
        m.Dblockarray.pblock_offsets=nullptr;
    }

    if(m.plocal_block_global_starts)
    {
        free(m.plocal_block_global_starts);
        m.plocal_block_global_starts=nullptr;
    }

    if(m.pblock_grid_extents)
    {
        free(m.pblock_grid_extents);
        m.pblock_grid_extents=nullptr;
    }

    if(m.pblock_grid_strides)
    {
        free(m.pblock_grid_strides);
        m.pblock_grid_strides=nullptr;
    }

    m.pglobal_grid_index_to_local_block.clear();


}



template <typename T>
void DataBlock_MPI_Functions::alloc_helper( DataBlockConfig loc, ptrdiff_t rank,ptrdiff_t datalength,ptrdiff_t*& pextents,ptrdiff_t *&pstrides,T *&pdata)
{
    pextents= (ptrdiff_t*)malloc(sizeof(ptrdiff_t)*rank);
    pstrides= (ptrdiff_t*)malloc(sizeof(ptrdiff_t)*rank);
    alloc_helper2(loc,datalength,pdata);

}


template <typename T>
void DataBlock_MPI_Functions::alloc_helper2( DataBlockConfig loc,ptrdiff_t datalength,T *&pdata)
{

#if defined(Unified_Shared_Memory)
    ondevice=false;
    devicenum=-INT_MAX;
    if(loc.pmemmap)
    {
        pdata=Host_Memory_Functions::create_temp_mmap<T>(pdatalength);
    }
    else
    {
        pdata=(T*)malloc(sizeof(T)*pdatalength);
    }
#else

    if(loc.data_is_devptr)
    {
        pdata=(T*)omp_target_alloc(sizeof(T)*datalength,loc.devicenum);
    }
    else
    {
        if(loc.pmemmap)
        {
            pdata=Host_Memory_Functions::create_temp_mmap<T>(datalength);
        }
        else
        {
            pdata=(T*)malloc(sizeof(T)*datalength);
        }
    }
#endif
}

template <typename T>
ptrdiff_t DistributedDataBlock<T>::tensor_rank() const
{
    return Dblockarray.ptensor_rank;
}

template <typename T>
ptrdiff_t* DistributedDataBlock<T>::global_extents() const
{
    return pglobal_extents;
}

template <typename T>
ptrdiff_t* DistributedDataBlock<T>::global_strides() const
{
    return pglobal_strides;
}

template <typename T>
ptrdiff_t DistributedDataBlock<T>::block_grid_rank() const
{
    return pblock_grid_rank;
}

template <typename T>
ptrdiff_t* DistributedDataBlock<T>::default_block_shape() const
{
    return pdefault_block_shape;
}

template <typename T>
const ptrdiff_t* DistributedDataBlock<T>::block_grid_extents() const
{
    return pblock_grid_extents;
}

template <typename T>
ptrdiff_t DistributedDataBlock<T>::block_grid_extent(ptrdiff_t dim) const
{
    return pblock_grid_extents[dim];
}


template <typename T>
ptrdiff_t DistributedDataBlock<T>::num_local_blocks() const
{
    return Dblockarray.pnumblocks;
}


template <typename T>
DataBlockArray<T>& DistributedDataBlock<T>::block_array()
{
    return Dblockarray;
}

template <typename T>
const DataBlockArray<T>& DistributedDataBlock<T>::block_array() const
{
    return Dblockarray;
}


template <typename T>
DataBlock<T> DistributedDataBlock<T>::local_block(
    ptrdiff_t local_block) const
{
    return Dblockarray.local_block(local_block);
}


template <typename T>
const ptrdiff_t*
DistributedDataBlock<T>::global_block_grid_coords(ptrdiff_t local_block) const
{
    return plocal_block_grid_coords +local_block * pblock_grid_rank;
}

template <typename T>
void DistributedDataBlock<T>::global_block_grid_coords(ptrdiff_t block_grid_index,ptrdiff_t* global_block_grid_coords) const
{
    // Convert a linear index in the GLOBAL BLOCK GRID
    // into multidimensional block-grid coordinates.

    ptrdiff_t remainder = block_grid_index;
    #pragma omp unroll partial
    for (ptrdiff_t d = pblock_grid_rank - 1; d >= 0; --d)
    {
        global_block_grid_coords[d] =remainder % pblock_grid_extents[d];

        remainder /=pblock_grid_extents[d];
    }
}


template <typename T>
ptrdiff_t DistributedDataBlock<T>::global_block_grid_index(ptrdiff_t local_block) const
{
    return plocal_block_global_grid_indices[local_block];
}

template <typename T>
ptrdiff_t DistributedDataBlock<T>::global_block_grid_index(const ptrdiff_t* global_block_grid_coords) const
{
    ptrdiff_t index = 0;
    #pragma omp unroll partial
    for (ptrdiff_t d = 0; d < pblock_grid_rank; ++d)
    {
        index = index * pblock_grid_extents[d]+ global_block_grid_coords[d];
    }

    return index;
}

template <typename T>
const ptrdiff_t* DistributedDataBlock<T>::block_grid_starts(ptrdiff_t local_block) const
{
    return plocal_block_global_starts +local_block * Dblockarray.ptensor_rank;
}

template <typename T>
const ptrdiff_t*DistributedDataBlock<T>::local_block_extents(ptrdiff_t local_block) const
{
    return Dblockarray.pextentsbuffer +local_block * Dblockarray.ptensor_rank;
}

template <typename T>
void DistributedDataBlock<T>::global_block_extents(
    const ptrdiff_t* global_block_grid_coords,
    ptrdiff_t* extents) const
{
    #pragma omp parallel for simd if(parallel:pblock_grid_rank > 30)
    for (ptrdiff_t d = 0; d < pblock_grid_rank; ++d)
    {
        const ptrdiff_t start =global_block_grid_coords[d]* pdefault_block_shape[d];

        const ptrdiff_t remaining =pglobal_extents[d] - start;

        extents[d] =std::min(pdefault_block_shape[d],remaining);
    }

    #pragma omp parallel for simd if(parallel:Dblockarray.ptensor_rank - pblock_grid_rank > 30)
    for (ptrdiff_t d = pblock_grid_rank; d < Dblockarray.ptensor_rank; ++d)
    {
        extents[d] = pglobal_extents[d];
    }
}

template <typename T>const ptrdiff_t*DistributedDataBlock<T>::local_block_strides(
    ptrdiff_t local_block) const
{
    return Dblockarray.pstridesbuffer +local_block * Dblockarray.ptensor_rank;
}

template <typename T>
ptrdiff_t DistributedDataBlock<T>::total_block_num() const
{
    ptrdiff_t total = 1;

    #pragma omp parallel for simd reduction (*:total) if(parallel:pblock_grid_rank>30)
    for (ptrdiff_t d = 0; d < pblock_grid_rank; ++d)
    {
        total *= pblock_grid_extents[d];
    }

    return total;
}

template <typename T>
int DistributedDataBlock<T>::owner_rank(
    const ptrdiff_t* block_coords) const
{
    int rank=ppolicy->owner_mpi_rank(block_coords,pblock_grid_rank,*pctx);

    return rank;

}





template <typename T>
void DistributedDataBlock<T>::block_grid_starts(
    const ptrdiff_t* global_block_grid_coords,
    ptrdiff_t* starts) const
{
    #pragma omp parallel for simd if(parallel:pblock_grid_rank > 30)
    for (ptrdiff_t d = 0; d < pblock_grid_rank; ++d)
    {
        starts[d] =global_block_grid_coords[d]* pdefault_block_shape[d];
    }

    #pragma omp parallel for simd if(parallel:Dblockarray.ptensor_rank - pblock_grid_rank > 30)
    for (ptrdiff_t d = pblock_grid_rank; d < Dblockarray.ptensor_rank; ++d)
    {
        starts[d] = 0;
    }
}






template <typename T>
void DataBlock_MPI_Functions::free_helper( DataBlockConfig loc, ptrdiff_t datalength,ptrdiff_t*& pextents,ptrdiff_t *&pstrides,T *&pdata)
{
    free_helper2(loc,datalength,pdata);

    free(pextents);
    free(pstrides);
    pextents=nullptr;
    pstrides=nullptr;
}



template <typename T>
void DataBlock_MPI_Functions::free_helper2( DataBlockConfig loc, ptrdiff_t datalength,T *&pdata)
{

#if defined(Unified_Shared_Memory)
    if(loc.pmemmap)
    {
        Host_Memory_Functions::delete_temp_mmap<T>(pdata,pdatalength)
    }
    else
    {
        free(pdata);
    }
#else
    if(loc.data_is_devptr)
    {
        omp_target_free(pdata,loc.devicenum);
    }
    else
    {
        if(loc.pmemmap)
        {
            Host_Memory_Functions::delete_temp_mmap<T>(pdata,datalength);
        }
        else
        {
            free(pdata);
        }
    }
#endif
    pdata=nullptr;

}



template<typename T>
void DistributedDataBlock<T>::print(int rootrank) const
{
    int rank, size;
    MPI_Comm_rank(pctx->pcomm,&rank);
    MPI_Comm_size(pctx->pcomm,&size);



    char* buffer=nullptr;
    int len=0;


    if(Dblockarray.pnumblocks == 0)
    {
        len += snprintf(nullptr, 0,"\n=== MPI Rank %d ===\n",rank);
        len += snprintf(nullptr, 0,"[]\n");
    }
    else
    {
        for(ptrdiff_t i=0; i<Dblockarray.pnumblocks; ++i)
        {
            len += snprintf(nullptr, 0,"\n=== MPI Rank %d ===\n",rank);
            len += snprintf(nullptr, 0,"Block %zu coords=(",i);

            ptrdiff_t* coords = plocal_block_grid_coords + i * Dblockarray.ptensor_rank;

            for(ptrdiff_t d=0; d<Dblockarray.ptensor_rank; ++d)
            {
                len += snprintf(nullptr,0,"%zu%s",coords[d],  (d + 1 < Dblockarray.ptensor_rank) ? "," : "");
            }

            len += snprintf(nullptr, 0,")\n");

            if(Dblockarray.pdata != nullptr)
            {
                DataBlock<T> block =Dblockarray.local_block(i);
                len += block.print_required_size();
            }
            else
            {
                len += snprintf(nullptr,0,"[]");
            }
            len += 1; // trailing '\n'
        }
    }

    buffer = (char*)malloc(len + 1);

    char* cur = buffer;
    ptrdiff_t remaining = len + 1;

    if(Dblockarray.pnumblocks == 0)
    {
        int n;

        n = snprintf(cur,  remaining,  "\n=== MPI Rank %d ===\n",rank);
        cur += n;
        remaining -= n;
        n = snprintf(cur,remaining,"[]\n");
        cur += n;
        remaining -= n;
    }
    else
    {
        for(ptrdiff_t i=0; i<Dblockarray.pnumblocks; ++i)
        {
            int n;
            n = snprintf(cur,remaining,"\n=== MPI Rank %d ===\n",rank);
            cur += n;
            remaining -= n;
            n = snprintf(cur,remaining,"Block %zu coords=(",i);

            cur += n;
            remaining -= n;

            ptrdiff_t* coords =plocal_block_grid_coords +i * Dblockarray.ptensor_rank;

            for(ptrdiff_t d=0; d<Dblockarray.ptensor_rank; ++d)
            {
                n = snprintf( cur, remaining, "%zu%s",coords[d],(d + 1 < Dblockarray.ptensor_rank) ? "," : "");
                cur += n;
                remaining -= n;
            }

            n = snprintf(cur, remaining, ")\n");

            cur += n;
            remaining -= n;
            if(Dblockarray.pdata != nullptr)
            {
                DataBlock<T> block =
                    Dblockarray.local_block(i);

                ptrdiff_t tensor_len =
                    block.print_required_size();

                block.print_to_buffer(cur, tensor_len + 1);

                cur += tensor_len;
                remaining -= tensor_len;
            }
            else
            {
                n = snprintf(cur, remaining,"[]");

                cur += n;
                remaining -= n;
            }

            *cur++ = '\n';
            --remaining;
        }
    }

    *cur = '\0';


    //
    ptrdiff_t entiresize=0;
    int *sizes= rank==rootrank? (int*)malloc(size*sizeof(int)):nullptr;
    int *displs=rank==rootrank? (int*)malloc(size*sizeof(int)):nullptr;
    char* msg=nullptr;



    MPI_Gather(&len, 1, mpi_get_type<int>(), rank==rootrank ? sizes : nullptr,1, mpi_get_type<int>(), 0, pctx->pcomm);

    if( rank==0)
    {
        for(int i=0; i<size; i++)
        {
            displs[i]=entiresize;
            entiresize+= sizes[i];
        }
        msg=(char*) malloc(entiresize+1);
    }
    MPI_Gatherv(buffer,len, mpi_get_type<char>(),rank==rootrank? msg:nullptr, rank==rootrank? sizes:nullptr,rank==rootrank? displs: nullptr,mpi_get_type<char>(),0,pctx->pcomm);

    if( rank==rootrank)
    {

        fwrite(msg,1,entiresize,stdout);
        free(msg);
        free(displs);
        free(sizes);

    }
    free(buffer);
}



template <typename T>
int DistributedDataBlock<T>::owner_rank_from_global_flat_index(const ptrdiff_t global_flat_index) const
{
    const ptrdiff_t rank = Dblockarray.ptensor_rank;

    if (rank <= 0 || global_flat_index < 0 ||
            pglobal_extents == nullptr ||
            pdefault_block_shape == nullptr ||
            ppolicy == nullptr || pctx == nullptr)
    {
        throw std::invalid_argument("Invalid distributed tensor metadata or flat index");
    }

    std::vector<ptrdiff_t> global_coords(rank);

    ptrdiff_t tmp = global_flat_index;

    #pragma omp unroll partial
    for (ptrdiff_t d = rank-1; d>=0; d--)
    {
        const ptrdiff_t extent = pglobal_extents[d];

        global_coords[d] = tmp % extent;
        tmp /= extent;
    }

    if (tmp != 0)
        throw std::out_of_range("Global flat index is outside the tensor");

    std::vector<ptrdiff_t> block_coords(rank, 0);

    #pragma omp parallel for simd if(parallel:pblock_grid_rank>30)
    for (ptrdiff_t d = 0; d < pblock_grid_rank; ++d)
    {
        block_coords[d] = global_coords[d] / pdefault_block_shape[d];
    }

    return ppolicy->owner_mpi_rank(block_coords.data(),rank,*pctx);
}



template<typename T>
inline TensorBlockLocation
DataBlock_MPI_Functions::get_tensor_block_location(
    const DistributedDataBlock<T>& source,
    ptrdiff_t global_flat_index)
{
    const ptrdiff_t rank = source.Dblockarray.ptensor_rank;
    const ptrdiff_t blockrank = source.pblock_grid_rank;

    ptrdiff_t global_coords[rank];
    ptrdiff_t block_coords[rank];

    ptrdiff_t remaining = global_flat_index;


    #pragma omp unroll partial
    for (ptrdiff_t d = rank - 1; d >= 0; --d)
    {
        global_coords[d] = remaining % source.pglobal_extents[d];
        remaining /= source.pglobal_extents[d];
    }


    #pragma omp unroll partial
    for (ptrdiff_t d = 0; d < rank; ++d)
    {
        block_coords[d] =(d < blockrank)? global_coords[d] / source.pdefault_block_shape[d]: 0;
    }


    ptrdiff_t global_block_index = 0;
    ptrdiff_t local_flat_index = 0;

    for (ptrdiff_t d = 0; d < rank; ++d)
    {
        ptrdiff_t local_coord = global_coords[d];
        ptrdiff_t actual_extent = source.pglobal_extents[d];

        if (d < blockrank)
        {
            const ptrdiff_t block_start =block_coords[d] * source.pdefault_block_shape[d];

            local_coord -= block_start;

            const ptrdiff_t diff =source.pglobal_extents[d] - block_start;

            const ptrdiff_t block_extent =source.pdefault_block_shape[d];

            actual_extent = block_extent < diff? block_extent: diff;

            global_block_index +=block_coords[d] * source.pblock_grid_strides[d];
        }

        local_flat_index =local_flat_index * actual_extent + local_coord;
    }

    return{source.owner_rank_from_global_flat_index(global_flat_index),global_block_index,local_flat_index};
}

template <typename T>
inline void DataBlock_MPI_Functions::MPI_Send_DataBlockArray(const DataBlockArray<T>& array,int dest,int tag,MPI_Comm pcomm)
{
    if (pcomm == MPI_COMM_NULL)
    {
        return;
    }

    MPI_Send(&array.pnumblocks,1,mpi_get_type<ptrdiff_t>(),dest, tag, pcomm);

    MPI_Send(&array.ptensor_rank,1,mpi_get_type<ptrdiff_t>(),dest, tag, pcomm);

    MPI_Send(&array.pdatalength,1,mpi_get_type<ptrdiff_t>(),dest, tag, pcomm);

    MPI_Send(&array.pconjugate,1,mpi_get_type<bool>(),dest, tag, pcomm);

    if (array.pnumblocks > 0)
    {
        MPI_Send(array.pblock_offsets,array.pnumblocks,mpi_get_type<ptrdiff_t>(),dest, tag, pcomm);

        MPI_Send(array.pextentsbuffer,array.pnumblocks * array.ptensor_rank,mpi_get_type<ptrdiff_t>(),dest, tag, pcomm);

        MPI_Send(array.pstridesbuffer,array.pnumblocks * array.ptensor_rank,mpi_get_type<ptrdiff_t>(),dest, tag, pcomm);
    }

    if (array.pdatalength > 0)
    {
        MPI_Send(array.pdata,sizeof(T) * array.pdatalength,MPI_BYTE,dest, tag, pcomm);
    }
}

template <typename T>
inline DataBlockArray<T>
DataBlock_MPI_Functions::MPI_Recv_alloc_DataBlockArray(
    DataBlockConfig loc,
    int source,
    int tag,
    MPI_Comm pcomm)
{
    DataBlockArray<T> array{};

    if (pcomm == MPI_COMM_NULL)
        return array;

    MPI_Status status;

    MPI_Recv(&array.pnumblocks, 1,mpi_get_type<ptrdiff_t>(),source, tag, pcomm, &status);

    MPI_Recv(&array.ptensor_rank, 1,mpi_get_type<ptrdiff_t>(),source, tag, pcomm, &status);

    MPI_Recv(&array.pdatalength, 1,mpi_get_type<ptrdiff_t>(),source, tag, pcomm, &status);

    MPI_Recv(&array.pconjugate, 1,mpi_get_type<bool>(),source, tag, pcomm, &status);

    array.conf = loc;

    if (array.pnumblocks > 0)
    {
        array.pblock_offsets =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * array.pnumblocks);

        array.pextentsbuffer =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) *array.pnumblocks * array.ptensor_rank);

        array.pstridesbuffer =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) *array.pnumblocks * array.ptensor_rank);

        MPI_Recv(array.pblock_offsets, array.pnumblocks,mpi_get_type<ptrdiff_t>(),source, tag, pcomm, &status);

        MPI_Recv(array.pextentsbuffer,array.pnumblocks * array.ptensor_rank,mpi_get_type<ptrdiff_t>(),source, tag, pcomm, &status);

        MPI_Recv(array.pstridesbuffer,array.pnumblocks * array.ptensor_rank,mpi_get_type<ptrdiff_t>(),source, tag, pcomm, &status);
    }

    if (array.pdatalength > 0)
    {
        alloc_helper2(loc, array.pdatalength, array.pdata);

        MPI_Recv(array.pdata,sizeof(T) * array.pdatalength,MPI_BYTE, source, tag, pcomm, &status);
    }

    return array;
}


template <typename T>
inline void
DataBlock_MPI_Functions::MPI_Free_DataBlockArray(DataBlockArray<T>&array)
{

    if (array.pnumblocks > 0)
    {
        free(array.pblock_offsets,sizeof(ptrdiff_t));
        free(array.pextentsbuffer,sizeof(ptrdiff_t));
        free(array.pstridesbuffer,sizeof(ptrdiff_t));
    }
    array.pblock_offsets = nullptr;
    array.pextentsbuffer = nullptr;
    array.pstridesbuffer = nullptr;

    if (array.pdatalength > 0)
    {
        free_helper2(array.conf,array.pdatalength,array.pdata);
    }
    array.pdata = nullptr;
    array.pdatalength = 0;
    array.ptensor_rank = 0;
    array.pnumblocks = 0;
    array.pconjugate = false;
}

template <typename T>
inline void DataBlock_MPI_Functions::MPI_Bcast_alloc_DataBlockArray(DataBlockArray<T>&array,DataBlockConfig loc,int rootrank,MPI_Comm pcomm)
{
    if (pcomm == MPI_COMM_NULL)
    {
        return;
    }


    MPI_Bcast(&array.pnumblocks,1,mpi_get_type<ptrdiff_t>(),rootrank, pcomm );

    MPI_Bcast(&array.ptensor_rank,1,mpi_get_type<ptrdiff_t>(),rootrank, pcomm);

    MPI_Bcast(&array.pdatalength,1,mpi_get_type<ptrdiff_t>(),rootrank, pcomm);

    MPI_Bcast(&array.pconjugate,1,mpi_get_type<bool>(),rootrank, pcomm);

    int rank;
    MPI_Comm_rank(pcomm, &rank);

    if(rank!=rootrank && array.pnumblocks > 0)
    {
        array.pblock_offsets =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) *array.pnumblocks);

        array.pextentsbuffer =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) *array.pnumblocks *array.ptensor_rank);

        array.pstridesbuffer =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) *array.pnumblocks *array.ptensor_rank);

    }

    MPI_Bcast(array.pblock_offsets,array.pnumblocks,mpi_get_type<ptrdiff_t>(),rootrank, pcomm );

    MPI_Bcast(array.pextentsbuffer,array.pnumblocks * array.ptensor_rank,mpi_get_type<ptrdiff_t>(),rootrank, pcomm);

    MPI_Bcast(array.pstridesbuffer,array.pnumblocks * array.ptensor_rank,mpi_get_type<ptrdiff_t>(),rootrank, pcomm);


    if(rank!=rootrank)
        array.conf=loc;


    if (array.pnumblocks > 0)
    {
        alloc_helper2(loc,array.pdatalength,array.pdata);
    }

    MPI_Bcast(array.pdata,sizeof(T) * array.pdatalength,MPI_BYTE,rootrank, pcomm );

    return;

}




template<typename T>
inline void DataBlock_MPI_Functions::MPI_Bcast_DataBlock (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (&db.dpdatalength, 1,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (&db.dprank,1,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, com);
    MPI_Bcast (&db.dpconjugate, 1,  mpi_get_type<bool>(), rootrank, com);

}



template<typename T>
inline void DataBlock_MPI_Functions::MPI_Bcast_DataBlock_pdata (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, com);
}




template<typename T>
inline void DataBlock_MPI_Functions::MPI_Bcast_DataBlock_meta (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (&db.dpdatalength, 1,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (&db.dprank,1,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (&db.dpconjugate, 1,  mpi_get_type<bool>(), rootrank, com);


}



template<typename T>
inline void DataBlock_MPI_Functions::MPI_Bcast_DataBlock_extents_strides (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<ptrdiff_t>(), rootrank, com);
}



template<typename T>
inline void DataBlock_MPI_Functions::MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,DataBlockConfig loc,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }

    int rank;
    MPI_Comm_rank(com, &rank);
    MPI_Bcast (&db.dpdatalength,1,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (&db.dprank,    1,    mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (&db.dpconjugate, 1,  mpi_get_type<bool>(), rootrank, com);

    if (rank != rootrank)
    {
        alloc_helper(loc,db.dprank,db.dpdatalength,db.dpextents,db.dpstrides,db.dpdata);
        db.devptr_former_hostptr=nullptr;
        db.dpconfig=loc;
    }
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<ptrdiff_t>(), rootrank, com);
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, com);

}


template<typename T>
inline void DataBlock_MPI_Functions::MPI_Scatter_matrix_to_rows_alloc(
    DistributedDataBlock<T>& recv_db,
    DataBlockConfig loc,
    MPI_CartesianContext* ctx,
    BlockMappingPolicy* policy,
    int rootrank,
    const DataBlock<T>* send_db)
{
    int rank;
    ptrdiff_t cols=0;
    MPI_Comm_rank(ctx->pcomm, &rank);
    if (rank==rootrank)cols=send_db->dpextents[1];

    MPI_Bcast(&cols,1,mpi_get_type<ptrdiff_t>(),rootrank,ctx->pcomm);
    MPI_Scatter_matrix_to_submatrices_alloc(1,cols,recv_db,loc,ctx,policy, rootrank,send_db);
}




template<typename T>
inline void DataBlock_MPI_Functions::MPI_Scatter_matrix_to_columns_alloc(
    DistributedDataBlock<T>& recv_db,DataBlockConfig loc,
    MPI_CartesianContext* ctx,
    BlockMappingPolicy* policy,
    int rootrank,
    const DataBlock<T>* send_db)
{
    int rank;
    ptrdiff_t rows=0;
    MPI_Comm_rank(ctx->pcomm, &rank);
    if (rank==rootrank)rows=send_db->dpextents[0];

    MPI_Bcast(&rows,1,mpi_get_type<ptrdiff_t>(),rootrank,ctx->pcomm);

    MPI_Scatter_matrix_to_submatrices_alloc(rows,1,recv_db,loc,ctx,policy, rootrank,send_db);
}


template<typename T>
inline void DataBlock_MPI_Functions::MPI_Gather_matrix_from_rows_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank, DataBlockConfig loc,
    DataBlock<T>* recv_db
)
{
    MPI_Gather_matrix_from_submatrices_alloc(send_db,rootrank, recv_db,loc);
}


template<typename T>
inline void DataBlock_MPI_Functions::MPI_Gather_matrix_from_columns_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,DataBlockConfig loc,
    DataBlock<T>* recv_db
)
{
    MPI_Gather_matrix_from_submatrices_alloc(send_db,rootrank, recv_db,loc);
}

template<typename T>
inline void DataBlock_MPI_Functions::
MPI_Scatter_matrix_to_submatrices_alloc(ptrdiff_t br,ptrdiff_t bc,DistributedDataBlock<T>& recv_db,DataBlockConfig loc,
                                        MPI_CartesianContext* ctx,BlockMappingPolicy* policy,
                                        int rootrank,const DataBlock<T>* send_db
)
{
    recv_db.pctx = ctx;
    recv_db.ppolicy = policy;

    if (ctx->pcomm == MPI_COMM_NULL)
        return;

    constexpr ptrdiff_t tensor_rank = 2;

    const ptrdiff_t block_rows = std::abs(br);
    const ptrdiff_t block_cols = std::abs(bc);

    recv_db.Dblockarray.ptensor_rank = tensor_rank;
    recv_db.pblock_grid_rank = tensor_rank;

    recv_db.pglobal_extents = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * tensor_rank);
    recv_db.pglobal_strides = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * tensor_rank);
    recv_db.pdefault_block_shape = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * tensor_rank);
    recv_db.pblock_grid_extents = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * tensor_rank);
    recv_db.pblock_grid_strides = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * tensor_rank);

    int rank;
    MPI_Comm_rank(ctx->pcomm, &rank);

    // Root supplies the global matrix metadata.
    if (rank == rootrank)
    {
        recv_db.pglobal_extents[0] = send_db->dpextents[0];
        recv_db.pglobal_extents[1] = send_db->dpextents[1];

        recv_db.pglobal_strides[0] = send_db->dpstrides[0];
        recv_db.pglobal_strides[1] = send_db->dpstrides[1];

        recv_db.Dblockarray.pconjugate = send_db->dpconjugate;

        recv_db.pdefault_block_shape[0] = block_rows;
        recv_db.pdefault_block_shape[1] = block_cols;
    }

    MPI_Bcast(recv_db.pglobal_extents,2,mpi_get_type<ptrdiff_t>(),rootrank,ctx->pcomm);

    MPI_Bcast(recv_db.pglobal_strides,2,mpi_get_type<ptrdiff_t>(),rootrank,ctx->pcomm);

    MPI_Bcast(&recv_db.Dblockarray.pconjugate,1,mpi_get_type<bool>(),rootrank,ctx->pcomm);

    MPI_Bcast(recv_db.pdefault_block_shape,2,mpi_get_type<ptrdiff_t>(),rootrank,ctx->pcomm);

    const ptrdiff_t M = std::abs(recv_db.pglobal_extents[0]);
    const ptrdiff_t N = std::abs(recv_db.pglobal_extents[1]);

    recv_db.pblock_grid_extents[0] =(M + block_rows - 1) / block_rows;

    recv_db.pblock_grid_extents[1] =(N + block_cols - 1) / block_cols;

    const ptrdiff_t grid_r = recv_db.pblock_grid_extents[0];
    const ptrdiff_t grid_c = recv_db.pblock_grid_extents[1];

    recv_db.pblock_grid_strides[1] = 1;
    recv_db.pblock_grid_strides[0] = grid_c;

    const ptrdiff_t total_blocks =recv_db.pblock_grid_extents[0] *recv_db.pblock_grid_extents[1];


    ptrdiff_t local_blocks = 0;

    ptrdiff_t* local_block_indices =total_blocks > 0 ? new ptrdiff_t[total_blocks] : nullptr;

    ptrdiff_t bcoords[2];

    for (ptrdiff_t b = 0; b < total_blocks; ++b)
    {
        bcoords[0] = b / recv_db.pblock_grid_strides[0];
        bcoords[1] = (b / recv_db.pblock_grid_strides[1])% recv_db.pblock_grid_extents[1];

        const int owner = policy->owner_mpi_rank(bcoords,recv_db.pblock_grid_rank,*ctx);

        if (owner == rank)
            local_block_indices[local_blocks++] = b;
    }

    recv_db.Dblockarray.pnumblocks = local_blocks;

    recv_db.plocal_block_global_starts =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * 2 * local_blocks): nullptr;

    recv_db.plocal_block_global_grid_indices =local_blocks > 0?(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.plocal_block_grid_coords =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * 2 * local_blocks): nullptr;

    recv_db.Dblockarray.pblock_offsets =local_blocks > 0?(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.Dblockarray.pextentsbuffer =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * 2 * local_blocks): nullptr;

    recv_db.Dblockarray.pstridesbuffer =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * 2 * local_blocks): nullptr;

    recv_db.pglobal_grid_index_to_local_block.reserve(local_blocks);

    struct BlockInfo
    {
        ptrdiff_t global_index;
        ptrdiff_t bi;
        ptrdiff_t bj;
        ptrdiff_t rows;
        ptrdiff_t cols;
        ptrdiff_t blocksize;
    };

    BlockInfo* blocks =local_blocks > 0 ? new BlockInfo[local_blocks] : nullptr;

    ptrdiff_t total_recv_elems = 0;

    #pragma omp parallel for simd reduction(+:total_recv_elems)
    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        const ptrdiff_t b = local_block_indices[i];

        const ptrdiff_t bi =b / recv_db.pblock_grid_strides[0];

        const ptrdiff_t bj =(b / recv_db.pblock_grid_strides[1])% recv_db.pblock_grid_extents[1];

        const ptrdiff_t r0 = bi * block_rows;
        const ptrdiff_t c0 = bj * block_cols;

        const ptrdiff_t rows = (block_rows < M - r0) ? block_rows : M - r0;
        const ptrdiff_t cols = (block_cols < N - c0) ? block_cols : N - c0;

        const ptrdiff_t blocksize = rows * cols;

        blocks[i] = {b, bi, bj, rows, cols, blocksize};

        recv_db.plocal_block_global_starts[2 * i] = r0;
        recv_db.plocal_block_global_starts[2 * i + 1] = c0;

        recv_db.plocal_block_grid_coords[2 * i] = bi;
        recv_db.plocal_block_grid_coords[2 * i + 1] = bj;

        recv_db.plocal_block_global_grid_indices[i] = b;

        total_recv_elems += blocksize;
    }

    delete[] local_block_indices;

    recv_db.Dblockarray.pdatalength = total_recv_elems;
    recv_db.Dblockarray.pdata = nullptr;
    recv_db.Dblockarray.conf = loc;

    if (total_recv_elems > 0)
        alloc_helper2(loc, total_recv_elems, recv_db.Dblockarray.pdata);

    MPI_Request* reqs =local_blocks > 0 ? new MPI_Request[local_blocks] : nullptr;

    ptrdiff_t offset = 0;

    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        T* ptr = recv_db.Dblockarray.pdata + offset;

        const ptrdiff_t b = blocks[i].global_index;

        MPI_Irecv(ptr,(int)blocks[i].blocksize,mpi_get_type<T>(),rootrank,(int)b,ctx->pcomm,&reqs[i]);

        recv_db.pglobal_grid_index_to_local_block[b] = i;

        recv_db.Dblockarray.pblock_offsets[i] = offset;

        offset += blocks[i].blocksize;
    }


    if (rank == rootrank)
    {
        MPI_Datatype blocktype0 =make_strided_2d_rowmajor_type<T>(block_rows,block_cols,send_db->dpstrides[0],send_db->dpstrides[1]);

        MPI_Type_commit(&blocktype0);

        MPI_Request* sendreqs =total_blocks > 0 ? new MPI_Request[total_blocks] : nullptr;

        for (ptrdiff_t bi = 0; bi < grid_r; ++bi)
        {
            for (ptrdiff_t bj = 0; bj < grid_c; ++bj)
            {
                const ptrdiff_t b =bi * recv_db.pblock_grid_strides[0] +bj * recv_db.pblock_grid_strides[1];

                bcoords[0] = bi;
                bcoords[1] = bj;

                const int owner = policy->owner_mpi_rank(bcoords,recv_db.pblock_grid_rank,*ctx);

                const ptrdiff_t r0 = bi * block_rows;
                const ptrdiff_t c0 = bj * block_cols;

 const ptrdiff_t rows = (block_rows < M - r0) ? block_rows : M - r0;
const ptrdiff_t cols = (block_cols < N - c0) ? block_cols : N - c0;

                const bool edgecase =rows < block_rows || cols < block_cols;

                MPI_Datatype sendtype = blocktype0;

                if (edgecase)
                {
                    sendtype = make_strided_2d_rowmajor_type<T>(rows,cols,send_db->dpstrides[0],send_db->dpstrides[1]);
                    MPI_Type_commit(&sendtype);
                }

                T* start =send_db->dpdata+ r0 * send_db->dpstrides[0]+ c0 * send_db->dpstrides[1];

                MPI_Isend(start,1,sendtype,owner,(int)b,ctx->pcomm,&sendreqs[b]);

                if (edgecase)
                    MPI_Type_free(&sendtype);
            }
        }

        MPI_Waitall((int)total_blocks,sendreqs,MPI_STATUSES_IGNORE);

        MPI_Type_free(&blocktype0);

        delete[] sendreqs;
    }

    MPI_Waitall((int)local_blocks,reqs,MPI_STATUSES_IGNORE
    );

    delete[] reqs;


    #pragma omp parallel for simd
    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        ptrdiff_t* bext_i =recv_db.Dblockarray.pextentsbuffer + 2 * i;

        ptrdiff_t* bstr_i =recv_db.Dblockarray.pstridesbuffer + 2 * i;

        bext_i[0] = blocks[i].rows;
        bext_i[1] = blocks[i].cols;

        bstr_i[0] = blocks[i].cols;
        bstr_i[1] = 1;
    }

    delete[] blocks;
}


template<typename T>
inline void DataBlock_MPI_Functions::MPI_Gather_matrix_from_submatrices_alloc(const DistributedDataBlock<T>& send_db,int rootrank,DataBlockConfig loc,DataBlock<T>* recv_db)
{

    if (send_db.pctx->pcomm == MPI_COMM_NULL)
        return;
    int rank;

    MPI_Comm_rank(send_db.pctx->pcomm,&rank);

    ptrdiff_t M = send_db.pglobal_extents[0];
    ptrdiff_t N = send_db.pglobal_extents[1];


    ptrdiff_t br=0, bc=0;

    br = send_db.pdefault_block_shape[0];
    bc = send_db.pdefault_block_shape[1];

    ptrdiff_t grid_r = (M + br - 1) / br;
    ptrdiff_t grid_c = (N + bc - 1) / bc;
    ptrdiff_t total_blocks = grid_r * grid_c;

    if(rank==rootrank)
    {
        ptrdiff_t *ext=nullptr;
        ptrdiff_t *str=nullptr;
        T *pdata=nullptr;


        ptrdiff_t datalen=compute_storage_span(send_db.pglobal_extents,send_db.pglobal_strides,2);

        alloc_helper(loc,2,datalen,ext,str,pdata);

        ext[0]=M;
        ext[1]=N;

        str[1]=send_db.pglobal_strides[1];
        str[0]=send_db.pglobal_strides[0];


        *recv_db = DataBlock<T>(pdata,datalen,2,ext,str,loc);
        recv_db->dpconjugate=send_db.Dblockarray.pconjugate;
    }


    MPI_Request *reqs=nullptr;
    ptrdiff_t recv_idx=0;
    if(rank==rootrank)
    {
        reqs= new MPI_Request[total_blocks];
        MPI_Datatype type;

        type=make_strided_2d_rowmajor_type<T>(br,bc,recv_db->dpstrides[0],recv_db->dpstrides[1]);
        MPI_Type_commit(&type);
        ptrdiff_t *grid_coords=new ptrdiff_t [send_db.pctx->pgridrank];

        for(ptrdiff_t bi=0; bi<grid_r; bi++)
        {
            for(ptrdiff_t bj=0; bj<grid_c; bj++)
            {
                MPI_Datatype type1;
                ptrdiff_t b = bi*grid_c + bj;

                ptrdiff_t bcoords[2] = {bi, bj};

                int owner = send_db.ppolicy->owner_mpi_rank(bcoords,2,*send_db.pctx);

                ptrdiff_t r0 = bi*br;
                ptrdiff_t c0 = bj*bc;
                ptrdiff_t diff1=M-r0;
                ptrdiff_t diff2=N-c0;

                bool edgecase=false;
                if(diff1<br|| diff2<bc)
                {
                    ptrdiff_t rows = br<=diff1? br:diff1;
                    ptrdiff_t cols = bc<=diff2? bc:diff2;

                    type1=make_strided_2d_rowmajor_type<T>(rows,cols,recv_db->dpstrides[0],recv_db->dpstrides[1]);
                    MPI_Type_commit(&type1);
                    edgecase=true;
                }

                T* start =recv_db->dpdata +r0*recv_db->dpstrides[0] +c0*recv_db->dpstrides[1];

                MPI_Irecv(start,1,edgecase? type1:type,owner,b,send_db.pctx->pcomm,&reqs[recv_idx]);

                recv_idx++;

                if(edgecase)
                    MPI_Type_free(&type1);
            }
        }
        delete[] grid_coords;
        MPI_Type_free(&type);
    }



    MPI_Request* sendreqs =(send_db.Dblockarray.pnumblocks>0)? new MPI_Request[send_db.Dblockarray.pnumblocks]: nullptr;

    ptrdiff_t send_idx=0;

    for(ptrdiff_t i=0; i<send_db.Dblockarray.pnumblocks; i++)
    {
        ptrdiff_t b = send_db.plocal_block_global_grid_indices[i];

        const ptrdiff_t* ext =send_db.Dblockarray.pextentsbuffer + 2*i;

        ptrdiff_t rows = ext[0];
        ptrdiff_t cols = ext[1];
        T* buffer= send_db.Dblockarray.pdata + send_db.Dblockarray.pblock_offsets[i];
        MPI_Isend(buffer,rows * cols,mpi_get_type<T>(),rootrank,b,send_db.pctx->pcomm,&sendreqs[send_idx++]);
    }

    if(send_idx>0)
        MPI_Waitall(send_idx,sendreqs,MPI_STATUSES_IGNORE);

    if(sendreqs)
        delete[] sendreqs;

    if(rank==rootrank)
    {
        MPI_Waitall(total_blocks,reqs,MPI_STATUSES_IGNORE);
        delete[] reqs;
    }
}



template<typename T>
inline void DataBlock_MPI_Functions::MPI_Scatter_tensor_to_subtensors_alloc(ptrdiff_t blockrank,const ptrdiff_t* block_extents,DistributedDataBlock<T>& recv_db,
                                                                            DataBlockConfig loc,MPI_CartesianContext* ctx,BlockMappingPolicy* policy,
                                                                            int rootrank,const DataBlock<T>* send_db)
{
    recv_db.pctx = ctx;
    recv_db.ppolicy = policy;

    if (ctx->pcomm == MPI_COMM_NULL)
        return;

    int mpi_rank;
    MPI_Comm_rank(ctx->pcomm, &mpi_rank);

    if (mpi_rank == rootrank)
    {
        recv_db.Dblockarray.ptensor_rank = send_db->dprank;

        recv_db.pblock_grid_rank =std::min(blockrank, send_db->dprank);

        recv_db.Dblockarray.pconjugate = send_db->dpconjugate;

        const ptrdiff_t rank = recv_db.Dblockarray.ptensor_rank;
        const ptrdiff_t brank = recv_db.pblock_grid_rank;

        recv_db.pglobal_extents =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * rank);

        recv_db.pglobal_strides =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * rank);

        recv_db.pdefault_block_shape =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * brank);
        #pragma omp parallel for simd if(parallel:rank>50)
        for (ptrdiff_t d = 0; d < rank; ++d)
        {
            recv_db.pglobal_extents[d] = send_db->dpextents[d];
            recv_db.pglobal_strides[d] = send_db->dpstrides[d];
        }
        #pragma omp parallel for simd if(parallel:brank>50)
        for (ptrdiff_t d = 0; d < brank; ++d)
            recv_db.pdefault_block_shape[d] = block_extents[d];
    }

    MPI_Bcast(&recv_db.Dblockarray.ptensor_rank,1, mpi_get_type<ptrdiff_t>(),rootrank, ctx->pcomm);

    MPI_Bcast(&recv_db.pblock_grid_rank,1, mpi_get_type<ptrdiff_t>(),rootrank, ctx->pcomm);

    MPI_Bcast(&recv_db.Dblockarray.pconjugate,1, mpi_get_type<bool>(),rootrank, ctx->pcomm);

    const ptrdiff_t rank = recv_db.Dblockarray.ptensor_rank;
    const ptrdiff_t brank = recv_db.pblock_grid_rank;

    if (mpi_rank != rootrank)
    {
        recv_db.pglobal_extents =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * rank);

        recv_db.pglobal_strides =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * rank);

        recv_db.pdefault_block_shape =(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * brank);
    }

    MPI_Bcast(recv_db.pglobal_extents,static_cast<int>(rank),mpi_get_type<ptrdiff_t>(),rootrank, ctx->pcomm);

    MPI_Bcast(recv_db.pglobal_strides,static_cast<int>(rank),mpi_get_type<ptrdiff_t>(),rootrank, ctx->pcomm);

    MPI_Bcast(recv_db.pdefault_block_shape,static_cast<int>(brank),mpi_get_type<ptrdiff_t>(),rootrank, ctx->pcomm);

    recv_db.pblock_grid_extents =rank > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * rank): nullptr;

    #pragma omp parallel for simd if(parallel:brank>50)
    for (ptrdiff_t d = 0; d < brank; ++d)
    {

        recv_db.pblock_grid_extents[d] =(recv_db.pglobal_extents[d] +recv_db.pdefault_block_shape[d] - 1) /recv_db.pdefault_block_shape[d];
    }
    #pragma omp parallel for simd if(parallel:rank-brank>50)
    for (ptrdiff_t d = brank; d < rank; ++d)
    {
        recv_db.pblock_grid_extents[d] = 1;
    }


    recv_db.pblock_grid_strides = nullptr;

    if (brank > 0)
    {
        recv_db.pblock_grid_strides = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * brank);
        ptrdiff_t stride = 1;
        #pragma omp unroll partial
        for (ptrdiff_t d = brank - 1; d >= 0; --d)
        {
            recv_db.pblock_grid_strides[d] = stride;

            stride *= recv_db.pblock_grid_extents[d];
        }
    }


    ptrdiff_t total_blocks = 1;
    #pragma omp parallel for simd reduction(*:total_blocks) if(parallel:rank>50)
    for (ptrdiff_t d = 0; d < rank; ++d)
    {
        total_blocks *= recv_db.pblock_grid_extents[d];
    }

    struct BlockInfo
    {
        ptrdiff_t linear_idx;
        ptrdiff_t blocksize;

        ptrdiff_t* coords;
        ptrdiff_t* starts;
        ptrdiff_t* extents;
    };

    std::vector<BlockInfo> blocks;

    if (total_blocks > 0)
        blocks.reserve(total_blocks);

    ptrdiff_t* bcoords = new ptrdiff_t[rank];

    for (ptrdiff_t b = 0; b < total_blocks; ++b)
    {
        ptrdiff_t tmp = b;
        #pragma omp unroll partial
        for (ptrdiff_t d = rank - 1; d >= 0; --d)
        {
            bcoords[d] =tmp % recv_db.pblock_grid_extents[d];

            tmp /= recv_db.pblock_grid_extents[d];
        }

        const int owner = policy->owner_mpi_rank(bcoords, brank, *ctx);

        if (owner != mpi_rank)
            continue;

        BlockInfo block;

        block.linear_idx = b;

        block.coords = new ptrdiff_t[rank];

        block.starts =brank > 0 ? new ptrdiff_t[brank] : nullptr;

        block.extents =brank > 0 ? new ptrdiff_t[brank] : nullptr;

        #pragma omp simd
        for (ptrdiff_t d = 0; d < rank; ++d)
            block.coords[d] = bcoords[d];

        ptrdiff_t blocksize = 1;
        #pragma omp simd reduction(*:blocksize)
        for (ptrdiff_t d = 0; d < brank; ++d)
        {
            const ptrdiff_t start =bcoords[d] * recv_db.pdefault_block_shape[d];

            const ptrdiff_t diff =recv_db.pglobal_extents[d] - start;

           const ptrdiff_t len = (recv_db.pdefault_block_shape[d] < diff) ? recv_db.pdefault_block_shape[d] : diff;

            block.starts[d] = start;
            block.extents[d] = len;

            blocksize *= len;
        }

        // Unpartitioned dimensions belong entirely to each block.
        #pragma omp simd reduction(*:blocksize)
        for (ptrdiff_t d = brank; d < rank; ++d)
        {
            blocksize *= recv_db.pglobal_extents[d];
        }

        block.blocksize = blocksize;

        blocks.push_back(block);
    }

    delete[] bcoords;



    const ptrdiff_t local_blocks = blocks.size();

    recv_db.Dblockarray.pnumblocks = local_blocks;

    recv_db.Dblockarray.pblock_offsets =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.plocal_block_global_grid_indices =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.plocal_block_grid_coords =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks * rank): nullptr;

    recv_db.plocal_block_global_starts =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks * brank): nullptr;


    ptrdiff_t total_recv_elems = 0;

    #pragma omp unroll partial
    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        recv_db.Dblockarray.pblock_offsets[i] =total_recv_elems;

        total_recv_elems += blocks[i].blocksize;
    }

    // Copy block metadata into contiguous arrays.
    #pragma omp parallel for if(parallel:local_blocks>50)
    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        #pragma omp simd
        for (ptrdiff_t d = 0; d < rank; ++d)
        {
            recv_db.plocal_block_grid_coords[i * rank + d] = blocks[i].coords[d];
        }
        #pragma omp simd
        for (ptrdiff_t d = 0; d < brank; ++d)
        {
            recv_db.plocal_block_global_starts[i * brank + d] = blocks[i].starts[d];
        }

        recv_db.plocal_block_global_grid_indices[i] =blocks[i].linear_idx;
    }



    recv_db.Dblockarray.pdatalength = total_recv_elems;
    recv_db.Dblockarray.pdata = nullptr;

    if (total_recv_elems > 0)
    {
        alloc_helper2(loc,total_recv_elems,recv_db.Dblockarray.pdata);
    }

    recv_db.Dblockarray.conf = loc;

    recv_db.pglobal_grid_index_to_local_block.reserve(local_blocks);

    MPI_Request* reqs =new MPI_Request[local_blocks];


    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        T* buffer =recv_db.Dblockarray.pdata +recv_db.Dblockarray.pblock_offsets[i];

        MPI_Irecv(buffer,(int)blocks[i].blocksize,mpi_get_type<T>(),rootrank,(int)blocks[i].linear_idx,ctx->pcomm,&reqs[i]);

        recv_db.pglobal_grid_index_to_local_block[blocks[i].linear_idx] = i;
    }


    if (mpi_rank == rootrank)
    {
        MPI_Request* sendreqs =new MPI_Request[total_blocks];

        ptrdiff_t* send_bcoords = new ptrdiff_t[rank];

        for (ptrdiff_t b = 0; b < total_blocks; ++b)
        {
            ptrdiff_t tmp = b;
            #pragma omp unroll partial
            for (ptrdiff_t d = rank - 1; d >= 0; --d)
            {
                send_bcoords[d] =tmp % recv_db.pblock_grid_extents[d];

                tmp /= recv_db.pblock_grid_extents[d];
            }

            const int owner = policy->owner_mpi_rank(send_bcoords, brank, *ctx);

            ptrdiff_t* block_ext = new ptrdiff_t[rank];
            ptrdiff_t* block_start = new ptrdiff_t[rank];


            #pragma omp simd
            for (ptrdiff_t d = 0; d < brank; ++d)
            {
                block_start[d] =send_bcoords[d] * block_extents[d];

                const ptrdiff_t diff =recv_db.pglobal_extents[d] -block_start[d];

                block_ext[d] =block_extents[d]<diff? block_extents[d]:diff;
            }


            #pragma omp simd
            for (ptrdiff_t d = brank; d < rank; ++d)
            {
                block_start[d] = 0;
                block_ext[d] = recv_db.pglobal_extents[d];
            }


            T* start = send_db->dpdata;
            ptrdiff_t total_offset = 0;

            #pragma omp simd reduction(+:total_offset)
            for (ptrdiff_t d = 0; d < rank; ++d)
            {
                total_offset += block_start[d] * send_db->dpstrides[d];
            }

            start += total_offset;

            MPI_Datatype blocktype =make_strided_nd_rowmajor_type<T>(rank,block_ext,send_db->dpstrides);

            MPI_Type_commit(&blocktype);

            MPI_Isend(start,1,blocktype,owner,(int)b,ctx->pcomm,&sendreqs[b]);

            MPI_Type_free(&blocktype);

            delete[] block_ext;
            delete[] block_start;
        }

        MPI_Waitall((int)total_blocks,sendreqs,MPI_STATUSES_IGNORE);

        delete[] send_bcoords;
        delete[] sendreqs;
    }

    MPI_Waitall((int)local_blocks,reqs,MPI_STATUSES_IGNORE);

    delete[] reqs;



    recv_db.Dblockarray.pextentsbuffer =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * rank * local_blocks): nullptr;

    recv_db.Dblockarray.pstridesbuffer =local_blocks > 0? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * rank * local_blocks): nullptr;

    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        ptrdiff_t* bext =recv_db.Dblockarray.pextentsbuffer + i * rank;

        ptrdiff_t* bstr =recv_db.Dblockarray.pstridesbuffer + i * rank;

        #pragma omp simd
        for (ptrdiff_t d = 0; d < brank; ++d)
        {
            bext[d] = blocks[i].extents[d];
        }


        #pragma omp simd
        for (ptrdiff_t d = brank; d < rank; ++d)
        {
            bext[d] = recv_db.pglobal_extents[d];
        }


        bstr[rank - 1] = 1;

        #pragma omp unroll partial
        for (ptrdiff_t d = rank - 2; d >= 0; --d)
        {
            bstr[d] = bstr[d + 1] * bext[d + 1];
        }

        delete[] blocks[i].coords;
        delete[] blocks[i].starts;
        delete[] blocks[i].extents;
    }

    blocks.clear();
}





template<typename T>
inline void DataBlock_MPI_Functions::MPI_Gather_tensor_from_subtensors_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,DataBlockConfig loc,
    DataBlock<T>* recv_db
)
{
    if (send_db.pctx->pcomm == MPI_COMM_NULL)
        return;

    int rank,size;
    MPI_Comm_rank(send_db.pctx->pcomm,&rank);
    MPI_Comm_size(send_db.pctx->pcomm,&size);

    ptrdiff_t rank_t    = send_db.Dblockarray.ptensor_rank;
    ptrdiff_t blockrank = send_db.pblock_grid_rank;


    ptrdiff_t* global_ext = send_db.pglobal_extents;
    ptrdiff_t* global_str = send_db.pglobal_strides;
    ptrdiff_t* block_ext  = send_db.pdefault_block_shape;



    ptrdiff_t* grid = new ptrdiff_t[rank_t];
    ptrdiff_t total_blocks = 1;

    #pragma omp parallel for simd if(parallel: blockrank>50)
    for(ptrdiff_t d=0; d<blockrank; d++)
        grid[d] = (global_ext[d] + block_ext[d] - 1) / block_ext[d];

    #pragma omp parallel for simd if(parallel: rank_t-blockrank>50)
    for(ptrdiff_t d=blockrank; d<rank_t; d++)
        grid[d] = 1;

    #pragma omp parallel for simd reduction(*:total_blocks)if(parallel:rank_t>50)
    for(ptrdiff_t d=0; d<rank_t; d++)
        total_blocks *= grid[d];



    if(rank==rootrank)
    {
        ptrdiff_t *ext=nullptr;
        ptrdiff_t *str=nullptr;
        T *pdata=nullptr;

        ptrdiff_t datalen=compute_storage_span(global_ext,global_str,rank_t);

        alloc_helper(loc,rank_t,datalen,ext,str,pdata);

        #pragma omp parallel for simd if(parallel:rank_t>50)
        for(ptrdiff_t d=0; d<rank_t; d++)
            ext[d]=global_ext[d];

        str[rank_t-1]=1;
        #pragma omp unroll partial
        for(int d=rank_t-2; d>=0; d--)
            str[d]=str[d+1]*ext[d+1];


        *recv_db = DataBlock<T>(pdata,datalen,rank_t,ext,str,loc);
        recv_db->dpconjugate=send_db.Dblockarray.pconjugate;
    }



    MPI_Request* reqs = nullptr;

    if(rank==rootrank)
        reqs = new MPI_Request[total_blocks];

    ptrdiff_t recv_idx = 0;

    if(rank==rootrank)
    {

        ptrdiff_t* bcoords=new ptrdiff_t[rank_t];
        for(ptrdiff_t b=0; b<total_blocks; b++)
        {
            ptrdiff_t tmp=b;
            #pragma omp unroll partial
            for(int d=rank_t-1; d>=0; d--)
            {
                bcoords[d] = tmp % grid[d];
                tmp /= grid[d];
            }

            int owner = send_db.ppolicy->owner_mpi_rank(bcoords,send_db.Dblockarray.ptensor_rank,*send_db.pctx);

            ptrdiff_t* block_ext =new ptrdiff_t[rank_t];

            ptrdiff_t* block_start =new ptrdiff_t[rank_t];

            #pragma omp parallel for simd if(parallel: blockrank>50)
            for(ptrdiff_t d = 0; d < blockrank; ++d)
            {
                block_start[d] =bcoords[d] * send_db.pdefault_block_shape[d];

                ptrdiff_t diff =global_ext[d] - block_start[d];

                block_ext[d] =(send_db.pdefault_block_shape[d] < diff)? send_db.pdefault_block_shape[d]: diff;
            }
            #pragma omp parallel for simd if(parallel:rank_t-blockrank>50)
            for(ptrdiff_t d = blockrank; d < rank_t; ++d)
            {
                block_start[d] = 0;
                block_ext[d] = global_ext[d];
            }


            ptrdiff_t offset=0;
            #pragma omp parallel for simd reduction(+:offset) if(parallel:rank_t>50)
            for(ptrdiff_t d = 0; d < rank_t; ++d)
            {
                offset +=block_start[d] *recv_db->dpstrides[d];
            }
            T* start = recv_db->dpdata;
            start+=offset;



            MPI_Datatype  blocktype=make_strided_nd_rowmajor_type<T>(rank_t,block_ext,recv_db->dpstrides);
            MPI_Type_commit(&blocktype);

            MPI_Irecv(start,1,blocktype,owner,b,send_db.pctx->pcomm,&reqs[recv_idx++]);

            MPI_Type_free(&blocktype);
            delete[] block_ext;
            delete[] block_start;

        }

        delete[]bcoords;
    }



    MPI_Request* sendreqs =send_db.Dblockarray.pnumblocks>0 ?new MPI_Request[send_db.Dblockarray.pnumblocks] :nullptr;

    ptrdiff_t send_idx=0;

    for(ptrdiff_t i=0; i<send_db.Dblockarray.pnumblocks; i++)
    {
        ptrdiff_t b = send_db.plocal_block_global_grid_indices[i];

        const ptrdiff_t* ext =send_db.Dblockarray.pextentsbuffer + i*rank_t;

        ptrdiff_t elems=1;

        #pragma omp parallel for simd reduction(*:elems) if(parallel:rank_t>30)
        for(ptrdiff_t d=0; d<rank_t; d++)
            elems *= ext[d];

        T* buffer=send_db.Dblockarray.pdata + send_db.Dblockarray.pblock_offsets[i];

        MPI_Isend(buffer,elems,mpi_get_type<T>(),rootrank,b,send_db.pctx->pcomm,&sendreqs[send_idx++]);
    }

    if(send_idx)
        MPI_Waitall(send_idx,sendreqs,MPI_STATUSES_IGNORE);

    if(sendreqs) delete[] sendreqs;

    if(rank==rootrank)
    {
        MPI_Waitall(total_blocks,reqs,MPI_STATUSES_IGNORE);
        delete[] reqs;
    }

    delete[] grid;
}

template<typename T>
inline void DataBlock_MPI_Functions::MPI_All_Gather_tensor_from_subtensors_alloc(
    const DistributedDataBlock<T>& send_db,DataBlockConfig loc,DataBlock<T>& recv_db)
{
    int rootrank=0;
    MPI_Gather_tensor_from_subtensors_alloc(send_db,0,loc,&recv_db);
    DataBlock_MPI_Functions::MPI_Bcast_alloc_DataBlock (recv_db,loc,send_db.pctx->pcomm,0 );
}

template<typename T>
inline void DataBlock_MPI_Functions::MPI_All_Gather_matrix_from_submatrices_alloc(
    const DistributedDataBlock<T>& send_db,DataBlockConfig loc,DataBlock<T>& recv_db)
{
    int rootrank=0;
    MPI_Gather_matrix_from_submatrices_alloc(send_db,0,loc,&recv_db);
    DataBlock_MPI_Functions::MPI_Bcast_alloc_DataBlock (recv_db,loc,send_db.pctx->pcomm,0 );
}

template<typename T>
inline void DataBlock_MPI_Functions::MPI_All_Gather_vector_from_subvectors_alloc(
    const DistributedDataBlock<T>& send_db,DataBlockConfig loc,DataBlock<T>& recv_db)
{
    int rootrank=0;
    MPI_Gather_vector_from_subvectors_alloc(send_db,0,loc,&recv_db);
    DataBlock_MPI_Functions::MPI_Bcast_alloc_DataBlock (recv_db,loc,send_db->pcom,0 );
}


template<typename T>
inline void DataBlock_MPI_Functions::MPI_Scatter_vector_to_subvectors_alloc(ptrdiff_t blocksize,
    DistributedDataBlock<T>& recv_db,DataBlockConfig loc,MPI_CartesianContext* ctx,
    BlockMappingPolicy* policy,int rootrank,const DataBlock<T>* send_db)
{
    recv_db.pctx = ctx;
    recv_db.ppolicy = policy;

    if (ctx->pcomm == MPI_COMM_NULL)
        return;

    constexpr ptrdiff_t tensor_rank = 1;

    const ptrdiff_t bs = std::abs(blocksize);

    int rank;
    MPI_Comm_rank(ctx->pcomm, &rank);

    recv_db.Dblockarray.ptensor_rank = tensor_rank;
    recv_db.pblock_grid_rank = tensor_rank;

    recv_db.pglobal_extents =(ptrdiff_t*)malloc(sizeof(ptrdiff_t));

    recv_db.pglobal_strides =(ptrdiff_t*)malloc(sizeof(ptrdiff_t));

    recv_db.pdefault_block_shape =(ptrdiff_t*)malloc(sizeof(ptrdiff_t));

    recv_db.pblock_grid_extents =(ptrdiff_t*)malloc(sizeof(ptrdiff_t));

    recv_db.pblock_grid_strides =(ptrdiff_t*)malloc(sizeof(ptrdiff_t));

    if (rank == rootrank)
    {
        recv_db.pglobal_extents[0] = send_db->dpextents[0];
        recv_db.pglobal_strides[0] = send_db->dpstrides[0];
        recv_db.pdefault_block_shape[0] = bs;
        recv_db.Dblockarray.pconjugate = send_db->dpconjugate;
    }

    MPI_Bcast(recv_db.pglobal_extents, 1,mpi_get_type<ptrdiff_t>(), rootrank, ctx->pcomm);

    MPI_Bcast(recv_db.pglobal_strides, 1,mpi_get_type<ptrdiff_t>(), rootrank, ctx->pcomm);

    MPI_Bcast(recv_db.pdefault_block_shape, 1,mpi_get_type<ptrdiff_t>(), rootrank, ctx->pcomm);

    MPI_Bcast(&recv_db.Dblockarray.pconjugate, 1,mpi_get_type<bool>(), rootrank, ctx->pcomm);

    const ptrdiff_t N = recv_db.pglobal_extents[0];

    recv_db.pblock_grid_extents[0] =(N + bs - 1) / bs;

    recv_db.pblock_grid_strides[0] = 1;

    const ptrdiff_t total_blocks =recv_db.pblock_grid_extents[0];



    ptrdiff_t local_blocks = 0;

    ptrdiff_t* local_block_indices =total_blocks > 0 ? new ptrdiff_t[total_blocks] : nullptr;

    ptrdiff_t bcoords[1];

    for (ptrdiff_t b = 0; b < total_blocks; ++b)
    {
        bcoords[0] = b;

        const int owner = policy->owner_mpi_rank(bcoords,recv_db.pblock_grid_rank,*ctx);

        if (owner == rank)
            local_block_indices[local_blocks++] = b;
    }

    recv_db.Dblockarray.pnumblocks = local_blocks;

    recv_db.plocal_block_grid_coords =local_blocks? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.plocal_block_global_starts =local_blocks?(ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.plocal_block_global_grid_indices =local_blocks? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.Dblockarray.pblock_offsets =local_blocks? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.Dblockarray.pextentsbuffer =local_blocks? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.Dblockarray.pstridesbuffer =local_blocks? (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * local_blocks): nullptr;

    recv_db.pglobal_grid_index_to_local_block.reserve(local_blocks);


   ptrdiff_t total_recv_elems = 0;
    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        const ptrdiff_t b = local_block_indices[i];

        const ptrdiff_t global_block_index =b * recv_db.pblock_grid_strides[0];

        const ptrdiff_t start =global_block_index * bs;

        const ptrdiff_t len =std::min(bs, N - start);

        recv_db.plocal_block_grid_coords[i] = b;
        recv_db.plocal_block_global_starts[i] = start;

        recv_db.plocal_block_global_grid_indices[i] =global_block_index;

        recv_db.Dblockarray.pblock_offsets[i] = total_recv_elems;

        total_recv_elems += len;
    }



    delete[] local_block_indices;

    recv_db.Dblockarray.pdatalength = total_recv_elems;
    recv_db.Dblockarray.pdata = nullptr;
    recv_db.Dblockarray.conf = loc;

    if (total_recv_elems > 0)
        alloc_helper2(loc, total_recv_elems, recv_db.Dblockarray.pdata);


    MPI_Request* reqs =local_blocks > 0 ? new MPI_Request[local_blocks] : nullptr;

    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        const ptrdiff_t len =(i + 1 < local_blocks)? recv_db.Dblockarray.pblock_offsets[i + 1]- recv_db.Dblockarray.pblock_offsets[i]
            : total_recv_elems- recv_db.Dblockarray.pblock_offsets[i];

        const ptrdiff_t b =recv_db.plocal_block_global_grid_indices[i];

        MPI_Irecv(recv_db.Dblockarray.pdata+ recv_db.Dblockarray.pblock_offsets[i],(int)len,mpi_get_type<T>(),rootrank,(int)b,ctx->pcomm,&reqs[i]);

        recv_db.pglobal_grid_index_to_local_block[b] = i;
    }


    if (rank == rootrank)
    {
        MPI_Datatype block_type;

        MPI_Type_create_hvector((int)bs,1,(MPI_Aint)send_db->dpstrides[0] * sizeof(T),mpi_get_type<T>(),&block_type);

        MPI_Type_commit(&block_type);

        MPI_Request* sendreqs =total_blocks > 0 ? new MPI_Request[total_blocks] : nullptr;

        for (ptrdiff_t b = 0; b < total_blocks; ++b)
        {
            bcoords[0] = b;

            const int owner = policy->owner_mpi_rank(bcoords,recv_db.pblock_grid_rank,*ctx);

            const ptrdiff_t global_block_index =bcoords[0] * recv_db.pblock_grid_strides[0];

            const ptrdiff_t start =global_block_index * bs;

            const ptrdiff_t len =std::min(bs, N - start);

            const bool edgecase = (len < bs);

            MPI_Datatype send_type = block_type;

            if (edgecase)
            {
                MPI_Type_create_hvector((int)len,1,(MPI_Aint)send_db->dpstrides[0] * sizeof(T),mpi_get_type<T>(),&send_type);

                MPI_Type_commit(&send_type);
            }

            T* ptr =send_db->dpdata + start * send_db->dpstrides[0];

            MPI_Isend(ptr,1,send_type,owner,(int)global_block_index,ctx->pcomm,&sendreqs[b]);

            if (edgecase)
                MPI_Type_free(&send_type);
        }

        MPI_Waitall((int)total_blocks,sendreqs,MPI_STATUSES_IGNORE);

        MPI_Type_free(&block_type);

        delete[] sendreqs;
    }

    MPI_Waitall((int)local_blocks,reqs,MPI_STATUSES_IGNORE);

    delete[] reqs;


    #pragma omp parallel for simd
    for (ptrdiff_t i = 0; i < local_blocks; ++i)
    {
        const ptrdiff_t len =(i + 1 < local_blocks)?
                                        recv_db.Dblockarray.pblock_offsets[i + 1]- recv_db.Dblockarray.pblock_offsets[i]
                                    : total_recv_elems- recv_db.Dblockarray.pblock_offsets[i];

        recv_db.Dblockarray.pextentsbuffer[i] = len;
        recv_db.Dblockarray.pstridesbuffer[i] = 1;
    }
}



template<typename T>
inline void DataBlock_MPI_Functions::MPI_Gather_vector_from_subvectors_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,DataBlockConfig loc,
    DataBlock<T>* recv_db
)
{
    if(send_db.pctx==nullptr)
        return;
    if (send_db.pctx->pcomm == MPI_COMM_NULL)
        return;

    int rank, size;
    MPI_Comm_rank(send_db.pctx->pcomm, &rank);
    MPI_Comm_size(send_db.pctx->pcomm, &size);

    ptrdiff_t N      = send_db.pglobal_extents[0];
    ptrdiff_t bs     = send_db.pdefault_block_shape[0];

    ptrdiff_t grid = (N + bs - 1) / bs;
    ptrdiff_t total_blocks = grid;

    if (rank == rootrank)
    {
        ptrdiff_t *ext = nullptr;
        ptrdiff_t *str = nullptr;
        T *pdata = nullptr;

        ptrdiff_t datalen =compute_storage_span(send_db.pglobal_extents,send_db.pglobal_strides,1);

        alloc_helper(loc,1,datalen,ext,str,pdata);

        ext[0] = N;
        str[0] =  send_db.pglobal_strides[0];;

        *recv_db = DataBlock<T>(pdata,datalen,1,ext,str,loc);
        recv_db->dpconjugate=send_db.Dblockarray.pconjugate;

    }


    MPI_Request* reqs = nullptr;
    ptrdiff_t recv_idx = 0;

    if (rank == rootrank)
        reqs = new MPI_Request[total_blocks];

    ptrdiff_t gridrank=send_db.pctx->pgridrank;
    if (rank == rootrank)
    {
        int* temp_coords=new int [gridrank];
        for (ptrdiff_t b = 0; b < total_blocks; b++)
        {
            ptrdiff_t bcoords[1] = { b };

            int owner = send_db.ppolicy->owner_mpi_rank(bcoords,1,*send_db.pctx);
            ptrdiff_t start = b * bs;
            ptrdiff_t diff=N - start;
            ptrdiff_t len   = bs<diff?bs:diff;

            T* ptr = recv_db->dpdata + start;

            MPI_Irecv(ptr,len,mpi_get_type<T>(),owner,b,send_db.pctx->pcomm,&reqs[recv_idx++]);
        }
        delete[]temp_coords;

    }



    MPI_Request* sendreqs =(send_db.Dblockarray.pnumblocks > 0)? new MPI_Request[send_db.Dblockarray.pnumblocks]: nullptr;

    ptrdiff_t send_idx = 0;

    for (ptrdiff_t i = 0; i < send_db.Dblockarray.pnumblocks; i++)
    {
        ptrdiff_t b = send_db.plocal_block_global_grid_indices[i];

        ptrdiff_t len =send_db.Dblockarray.pextentsbuffer[i];
        T* buffer=send_db.Dblockarray.pdata + send_db.Dblockarray.pblock_offsets[i];
        MPI_Isend(buffer,len,mpi_get_type<T>(),rootrank,b,send_db.pctx->pcomm,&sendreqs[send_idx++]);
    }

    if (send_idx > 0)
        MPI_Waitall(send_idx, sendreqs, MPI_STATUSES_IGNORE);

    if (sendreqs)
        delete[] sendreqs;


    if (rank == rootrank)
    {
        MPI_Waitall(total_blocks, reqs, MPI_STATUSES_IGNORE);
        delete[] reqs;
    }
}


template<typename T>
inline  void DataBlock_MPI_Functions::MPI_Send_DataBlock(DataBlock<T> &m, int dest, int tag, MPI_Comm pcomm)
{

    MPI_Send(&m.dpdatalength, 1, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprank, 1, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);
    MPI_Send(m.dpextents, m.dprank, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);
    MPI_Send(m.dpstrides, m.dprank, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);
    MPI_Send(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm);
    MPI_Send(&m.dpconjugate,sizeof(bool), mpi_get_type<bool>(), dest, tag, pcomm);

}


template<typename T>
inline  void DataBlock_MPI_Functions::MPI_Send_DataBlock_meta(DataBlock<T> &m, int dest, int tag, MPI_Comm pcomm)
{
    MPI_Send(&m.dpdatalength, 1, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprank, 1, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);

    MPI_Send(m.dpextents, m.dprank, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);
    MPI_Send(m.dpstrides, m.dprank, mpi_get_type<ptrdiff_t>(), dest, tag, pcomm);
    MPI_Send(&m.dpconjugate,sizeof(bool), mpi_get_type<bool>(), dest, tag, pcomm);

}



template<typename T>
inline  void DataBlock_MPI_Functions::MPI_Isend_DataBlock_pdata(DataBlock<T> &m,const int dest,const  int tag,const MPI_Comm pcomm,MPI_Request *request)
{
    MPI_Isend(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm,request);
}

template<typename T>
inline  void DataBlock_MPI_Functions::MPI_Send_DataBlock_pdata(DataBlock<T> &m,const int dest,const int tag,const MPI_Comm pcomm)
{
    MPI_Send(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}

template<typename T>
inline  DataBlock<T> DataBlock_MPI_Functions::MPI_Recv_alloc_DataBlock(DataBlockConfig loc, const int source,const  int tag, MPI_Comm pcomm)
{

    MPI_Status status;
    ptrdiff_t pdatalength, prank;
    MPI_Recv(&pdatalength, 1, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(&prank, 1, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);

    ptrdiff_t *pextents=nullptr,
               *pstrides=nullptr;
    T* pdata=nullptr;

    alloc_helper(loc,prank,pdatalength,pextents,pstrides,pdata);

    MPI_Recv(pextents,prank, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);

    MPI_Recv(pstrides,prank, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);

    MPI_Recv(pdata,sizeof(T)*pdatalength, MPI_BYTE, source, tag, pcomm, &status);
    bool conjugate;
    MPI_Recv(&conjugate,sizeof(bool), mpi_get_type<bool>(), source, tag, pcomm,&status);


    DataBlock<T> tempt(pdata,pdatalength,prank,pextents,pstrides,loc);
    tempt.dpconjugate=conjugate;
    return tempt;

}



template <typename T>
void DataBlock_MPI_Functions::MPI_Free_DataBlock(DataBlock<T>&m)
{

    if(m.dpdata!=nullptr)
    {
#if defined(Unified_Shared_Memory)
        if(dpconfig.pmemmap)
            Host_Memory_Functions<T>::delete_temp_mmap<T>(m.dpdata,m.dpdatalength);
        else;
        free(m.dpdata);
#else
        if(m.dpconfig.data_is_devptr)
            omp_target_free(m.dpdata,m.dpconfig.devicenum);
        else
        {
            if(m.dpconfig.pmemmap)
                Host_Memory_Functions::delete_temp_mmap(m.dpdata,m.dpdatalength);
            else
                free(m.dpdata);
        }
#endif
    }
    if(m.dpextents!=nullptr) free(m.dpextents);
    if(m.dpstrides!=nullptr) free(m.dpstrides);
}


template<typename T>
void DataBlock_MPI_Functions::MPI_Recv_DataBlock(DataBlock<T>& m,const int source,const  int tag, MPI_Comm pcomm)
{
    MPI_Status status;

    MPI_Recv(&m.dpdatalength, 1, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprank, 1, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(m.dpextents,m.dprank, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(m.dpstrides,m.dprank, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(m.dpdata,sizeof(T)*m.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);
    MPI_Recv(&m.dpconjugate,sizeof(bool), mpi_get_type<bool>(), source, tag, pcomm,&status);


}


template<typename T>
void DataBlock_MPI_Functions::MPI_Recv_DataBlock_meta(DataBlock<T>& m,const int source,const  int tag, MPI_Comm pcomm)
{
    MPI_Status status;

    MPI_Recv(&m.dpdatalength, 1, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprank, 1, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(m.dpextents,m.dprank, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(m.dpstrides,m.dprank, mpi_get_type<ptrdiff_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dpconjugate,sizeof(bool), mpi_get_type<bool>(), source, tag, pcomm,&status);


}




template<typename T>
inline  void DataBlock_MPI_Functions::MPI_Irecv_DataBlock_pdata(DataBlock<T> &mds, const int source, const int tag,const  MPI_Comm pcomm,  MPI_Request *request)
{
    MPI_Irecv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, request);
}

template<typename T>
inline  void DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(DataBlock<T>& mds,const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);
}


inline MPI_CartesianContext::MPI_CartesianContext(MPI_Comm comm): pcomm(comm)
{
    MPI_Comm_size(pcomm, &psize);

    int ndims;
    MPI_Cartdim_get(comm, &ndims);
    pgridrank= (ptrdiff_t)ndims;

    pdims    = new int[pgridrank];
    pperiods = new int[pgridrank];

    int* temp_cartesian_coords = new int[pgridrank];

    MPI_Cart_get(pcomm,(int)pgridrank,pdims,pperiods,temp_cartesian_coords);

    delete[] temp_cartesian_coords;
}

inline MPI_CartesianContext::~MPI_CartesianContext()
{
    delete[] pdims;
    delete[] pperiods;
}

inline int MPI_CartesianContext::rank_from_coords(int *coords) const
{
    int rank;
    MPI_Cart_rank(pcomm, coords, &rank);
    return rank;
}

inline BlockMappingPolicy::BlockMappingPolicy(
    ptrdiff_t gridrank,
    const int* index_map,
    const ptrdiff_t* cyclic_block)
    : pgridrank(gridrank),
      pindex_map(new int[gridrank]),
      pcyclic_block(new ptrdiff_t[gridrank])
{
    #pragma omp unroll partial
    for (ptrdiff_t d = 0; d < pgridrank; ++d)
    {
        pindex_map[d] = index_map ? index_map[d] : static_cast<int>(d);
        pcyclic_block[d] = cyclic_block ? cyclic_block[d] : 1;
    }
}



inline void BlockMappingPolicy::owner_mpi_cartesian_coords(const ptrdiff_t* block_grid_coords,ptrdiff_t block_grid_rank,const MPI_CartesianContext& ctx,int* cartesian_coords) const
{
    #pragma omp unroll partial
    for (ptrdiff_t d = 0; d < pgridrank; ++d)
    {
        const int idx = pindex_map[d];

        ptrdiff_t x =(idx >= 0 && (ptrdiff_t)idx < block_grid_rank)? block_grid_coords[idx]: 0;

        const ptrdiff_t grouped =x / pcyclic_block[d];

        cartesian_coords[d] =(int)(grouped % ctx.pdims[d]);
    }
}

inline int BlockMappingPolicy::owner_mpi_rank(const ptrdiff_t* block_grid_coords,ptrdiff_t block_grid_rank,
        const MPI_CartesianContext& ctx) const
{
    int* temp_cartesian_coords = new int[ctx.pgridrank];

    owner_mpi_cartesian_coords(block_grid_coords,block_grid_rank,ctx,temp_cartesian_coords);

    int rank = ctx.rank_from_coords(temp_cartesian_coords);

    delete[] temp_cartesian_coords;

    return rank;
}

#endif
