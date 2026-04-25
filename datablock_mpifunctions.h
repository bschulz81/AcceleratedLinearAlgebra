#ifndef DATABLOCK_MPIFUNCTIONS
#define DATABLOCK_MPIFUNCTIONS

#include <mpi.h>

#include <memory>
#include <complex>
#include <type_traits>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "datablock.h"
#include "datablock_host_memory_functions.h"
#include "datablock_gpu_memory_functions.h"




template <typename T>
struct mpi_type_map
{
    static inline const MPI_Datatype value = MPI_DATATYPE_NULL;
};

// --- Fundamental types ---
template <> struct mpi_type_map<char>
{
    static inline const MPI_Datatype value = MPI_CHAR;
};
template <> struct mpi_type_map<signed char>
{
    static inline const MPI_Datatype value = MPI_SIGNED_CHAR;
};
template <> struct mpi_type_map<unsigned char>
{
    static inline const MPI_Datatype value = MPI_UNSIGNED_CHAR;
};
template <> struct mpi_type_map<wchar_t>
{
    static inline const MPI_Datatype value = MPI_WCHAR;
};
template <> struct mpi_type_map<short>
{
    static inline const MPI_Datatype value = MPI_SHORT;
};
template <> struct mpi_type_map<unsigned short>
{
    static inline const MPI_Datatype value = MPI_UNSIGNED_SHORT;
};
template <> struct mpi_type_map<int>
{
    static inline const MPI_Datatype value = MPI_INT;
};
template <> struct mpi_type_map<unsigned int>
{
    static inline const MPI_Datatype value = MPI_UNSIGNED;
};
template <> struct mpi_type_map<long>
{
    static inline const MPI_Datatype value = MPI_LONG;
};
template <> struct mpi_type_map<unsigned long>
{
    static inline const MPI_Datatype value = MPI_UNSIGNED_LONG;
};
template <> struct mpi_type_map<long long>
{
    static inline const MPI_Datatype value = MPI_LONG_LONG;
};
template <> struct mpi_type_map<unsigned long long>
{
    static inline const MPI_Datatype value = MPI_UNSIGNED_LONG_LONG;
};
template <> struct mpi_type_map<float>
{
    static inline const MPI_Datatype value = MPI_FLOAT;
};
template <> struct mpi_type_map<double>
{
    static inline const MPI_Datatype value = MPI_DOUBLE;
};
template <> struct mpi_type_map<long double>
{
    static inline const MPI_Datatype value = MPI_LONG_DOUBLE;
};
template <> struct mpi_type_map<bool>
{
    static inline const MPI_Datatype value = MPI_C_BOOL;
};



template <> struct mpi_type_map<std::complex<float>>
{
    static inline const MPI_Datatype value = MPI_C_COMPLEX;
};
template <> struct mpi_type_map<std::complex<double>>
{
    static inline const MPI_Datatype value = MPI_C_DOUBLE_COMPLEX;
};
template <> struct mpi_type_map<std::complex<long double>>
{
    static inline const MPI_Datatype value = MPI_C_LONG_DOUBLE_COMPLEX;
};

template <typename T>
MPI_Datatype mpi_get_type() noexcept
{
    return mpi_type_map<T>::value;
}


class MPI_CartesianContext
{
public:
    MPI_Comm comm;

    size_t gridrank;
    int size;

    int* dims;
    int* periods;

public:

    MPI_CartesianContext(MPI_Comm comm_)
        : comm(comm_)
    {
        MPI_Comm_size(comm, &size);

        int ndims;
        MPI_Cartdim_get(comm, &ndims);
        gridrank= (size_t)ndims;

        dims    = new int[gridrank];
        periods = new int[gridrank];

        int* tmp_coords = new int[gridrank];

        MPI_Cart_get(comm,
                     (int)gridrank,
                     dims,
                     periods,
                     tmp_coords);

        delete[] tmp_coords;
    }

    ~MPI_CartesianContext()
    {
        delete[] dims;
        delete[] periods;
    }

    inline int rank_from_coords(int* coords) const
    {
        int rank;
        MPI_Cart_rank(comm, coords, &rank);
        return rank;
    }
};

class BlockMappingPolicy
{
public:
    size_t gridrank;

    int* index_map;
    size_t* cyclic_block;

public:

    BlockMappingPolicy(
        size_t gridrank_,
        const int* index_map_ = nullptr,
        const size_t* cyclic_block_ = nullptr)
        : gridrank(gridrank_)
    {
        index_map = new int[gridrank];
        cyclic_block = new size_t[gridrank];

        for (size_t d = 0; d < gridrank; d++)
        {
            cyclic_block[d] = cyclic_block_ ? cyclic_block_[d] : 1;
        }

        if (index_map_ != nullptr)
        {
            for (size_t d = 0; d < gridrank; d++)
                index_map[d] = index_map_[d];
        }
        else
        {
            for (size_t d = 0; d < gridrank; d++)
                index_map[d] = (int)d;
        }
    }

    ~BlockMappingPolicy()
    {
        delete[] index_map;
        delete[] cyclic_block;
    }

    inline void create_coords(
        const size_t* in,
        size_t* out,
        size_t in_rank) const
    {
        for (size_t g = 0; g < gridrank; g++)
        {
            int idx = index_map[g];

            if (idx >= 0 && (size_t)idx < in_rank)
                out[g] = in[idx];
            else
                out[g] = 0;
        }
    }

    inline int owner(
        const size_t* grid_coords,
        const MPI_CartesianContext& ctx,
        int* temp_coords) const
    {
        for (size_t d = 0; d < gridrank; d++)
        {
            size_t grouped = grid_coords[d] / cyclic_block[d];
            temp_coords[d] = (int)(grouped % ctx.dims[d]);
        }

        return ctx.rank_from_coords(temp_coords);
    }
};



class MPI_Policy
{
public:
    MPI_Comm comm = MPI_COMM_WORLD;
    bool mpi_enabled = true;

    int mpi_rank = 0;
    int mpi_size = 1;


    MPI_Policy(bool mpi=true, MPI_Comm com = MPI_COMM_WORLD)
    {
        if (mpi)
        {
            int init;
            MPI_Initialized(&init);
            if(init)
            {
                comm=com;
                MPI_Comm_rank(comm, &mpi_rank);
                MPI_Comm_size(comm, &mpi_size);
            }
            else
            {
                mpi_enabled=false;
                mpi_size=0;
                mpi_rank=0;
            }
        }
        else
        {
            mpi_enabled=false;
            mpi_size=0;
            mpi_rank=0;
        }
    }


};



template<typename T>
class DataBlock_MPI_Functions;

template<typename T>
class Math_Functions_MPI;

class Math_MPI_Functions_Policy;


template<typename T>
class DistributedDataBlock
{
    friend class DataBlock_MPI_Functions<T>;
    friend class Math_Functions_MPI<T>;
    friend class Math_MPI_Functions_Policy;
public:

    size_t globalrank()const
    {
        return pglobalrank;
    }
    bool global_rowmajor()const
    {
        return pglobal_rowmajor;
    }
    size_t* global_extents()const
    {
        return pglobal_extents;
    }
    size_t* global_strides()const
    {
        return pglobal_strides;
    }



    size_t local_blocknumber()const
    {
        return plocal_blocknumber;
    }



    DataBlock<T> *blocks()const
    {
        return pblocks;
    }




    void printblockcoordinates( )const
    {
        int rank, size;
        MPI_Comm_rank(pctx->comm, &rank);
        MPI_Comm_size(pctx->comm, &size);
        for(int r=0; r<size; r++)
        {
            if(rank==r)
            {
                printf("\n=== MPI Rank %d ===\n",rank);
                printf("blocks: %zu\n",plocal_blocknumber);
                for(size_t i=0; i< plocal_blocknumber * pglobalrank; i++)
                {
                    printf("block coords %zu",pcoordsbuffer[i]);
                    printf(" ");
                    fflush(stdout);
                }

            }
        }
    }
    void printtensors()const
    {
        int rank, size;
        MPI_Comm_rank(pctx->comm,&rank);
        MPI_Comm_size(pctx->comm,&size);

        for(int r=0; r<size; r++)
        {
            if(rank==r)
            {
                printf("\n=== MPI Rank %d ===\n",rank);
                printf("blocks: %zu\n",plocal_blocknumber);

                for (size_t i=0; i<plocal_blocknumber; i++)
                {
                    if(pblocks[i].dpdata!=nullptr)
                    {
                        printf("\n");
                        printf("Block %zu\n",i);
                        pblocks[i].printtensor();
                        printf("\n");
                        fflush(stdout);
                    }
                    else
                    {
                        printf("[]");
                        fflush(stdout);
                    }
                }
            }

        }
    }
protected:


    T* pdata = nullptr;
    size_t pdatalength=0;

    size_t* pextentsbuffer=nullptr;
    size_t* pstridesbuffer=nullptr;

    size_t* pcoordsbuffer=nullptr;
    size_t* startsbuffer=nullptr;
    size_t* pblock_indices=nullptr;
    size_t* pblock_offsets=nullptr;
    size_t* pblock_linear_idx = nullptr;
    size_t  pglobalrank=0;
    size_t* pglobal_extents=nullptr;
    size_t* pglobal_strides=nullptr;

    size_t  pblock_rank;
    size_t* pblock_extents=nullptr;
    size_t* pblock_coords=nullptr;
    bool pdpdata_is_devptr=false;
    bool pmemmap=false;
    int pdevptr_devicenum=-1;

    bool pglobal_rowmajor=true;
    size_t plocal_blocknumber=0;
    DataBlock<T>* pblocks=nullptr;

    MPI_CartesianContext* pctx;
    BlockMappingPolicy* ppolicy;

    std::unordered_map<size_t, size_t> pglobal_to_local_index;
};

template<typename T>
class DataBlock_MPI_Functions
{
public:

    inline static void MPI_Bcast_DataBlock (DataBlock<T> &db,MPI_Comm com, int rootrank);
    inline static void MPI_Bcast_DataBlock_meta (DataBlock<T> &db,MPI_Comm com, int rootrank);
    inline static void MPI_Bcast_DataBlock_extents_strides (DataBlock<T> &db,MPI_Comm com, int rootrank);
    inline static void MPI_Bcast_DataBlock_pdata (DataBlock<T> &db,MPI_Comm com, int rootrank);
    inline static void MPI_IBcast_DataBlock_pdata (DataBlock<T> &db,MPI_Comm com,MPI_Request*req, int rootrank);

    inline static void MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,bool with_memmap,bool ondevice, int devicenum,MPI_Comm com, int rootrank);

    inline static void MPI_Scatter_vector_to_subvectors_alloc(  size_t blocksize,    DistributedDataBlock<T>& recv_db,
            bool memmap,      bool ondevice,    int devicenum,    MPI_CartesianContext *ctx,    BlockMappingPolicy* policy,   int rootrank,    const DataBlock<T>* send_db);

    inline static void MPI_Gather_vector_from_subvectors_alloc(    const DistributedDataBlock<T>& send_db,    int rootrank,
            DataBlock<T>* recv_db=nullptr,    bool memmap=false,    bool ongpu=false,
            int devicenum=-1);

    inline static void MPI_Scatter_matrix_to_rows_alloc( DistributedDataBlock<T>& recv_db, bool memmap, bool ondevice, int devicenum,
            MPI_CartesianContext* ctx,  BlockMappingPolicy* policy,  int rootrank,   const DataBlock<T>* send_db);

    inline static void MPI_Gather_matrix_from_rows_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank,    DataBlock<T>* recv_db=nullptr,    bool memmap=false,    bool ondevice=false,    int devicenum=-1);

    inline static void MPI_Scatter_matrix_to_columns_alloc( DistributedDataBlock<T>& recv_db, bool memmap, bool ondevice, int devicenum,
            MPI_CartesianContext* ctx,  BlockMappingPolicy* policy,  int rootrank,   const DataBlock<T>* send_db=nullptr);

    inline static void MPI_Gather_matrix_from_columns_alloc(   const DistributedDataBlock<T>& send_db,      int rootrank,DataBlock<T>* recv_db = nullptr,
            bool memmap=false, bool ongpu=false, int devicenum=-1);


    inline static void MPI_Scatter_matrix_to_submatrices_alloc(    size_t br,    size_t bc,    DistributedDataBlock<T>& recv_db,    bool memmap,
            bool ondevice,    int devicenum,     MPI_CartesianContext *ctx,    BlockMappingPolicy* policy, int rootrank,     const DataBlock<T>* send_db=nullptr  );



    inline static void MPI_Gather_matrix_from_submatrices_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank,DataBlock<T>* recv_db = nullptr,bool memmap=false, bool ongpu=false, int devicenum=-1 );

    inline static void MPI_Scatter_tensor_to_subtensors_alloc(    size_t blockrank,    const size_t* block_extents,
            DistributedDataBlock<T>& recv_db,    bool memmap,    bool ondevice,    int devicenum,    MPI_CartesianContext *ctx,    BlockMappingPolicy* policy, int rootrank,
            const DataBlock<T>* send_db=nullptr);


    inline static void MPI_Gather_tensor_from_subtensors_alloc(  const DistributedDataBlock<T>& send_db,  int rootrank,
            DataBlock<T>* recv_db = nullptr, bool memmap=false, bool ongpu=false, int devicenum=-1);

    inline static DataBlock<T> MPI_Recv_alloc_DataBlock(bool with_memmap,bool ondevice, int devicenum, const int source,const  int tag, MPI_Comm pcomm);

    inline static void MPI_Free_DataBlock(DataBlock<T>&m, bool with_memmap=false);
    inline static void MPI_Free_DistributedDataBlock(DistributedDataBlock<T>&m);

    inline static void MPI_Send_DataBlock(DataBlock<T> &m,const int dest, const int tag, MPI_Comm pcomm);
    inline static void MPI_Recv_DataBlock(DataBlock<T>& m, const int source,const  int tag, MPI_Comm pcomm);

    inline static void MPI_Send_DataBlock_meta(DataBlock<T> &m,const int dest, const int tag, MPI_Comm pcomm);
    inline static void MPI_Recv_DataBlock_meta(DataBlock<T>& m, const int source,const  int tag, MPI_Comm pcomm);


    inline static void MPI_Recv_DataBlock_pdata(DataBlock<T>& mds,const int source, const int tag,const  MPI_Comm pcomm);
    inline static void MPI_Send_DataBlock_pdata(DataBlock<T> &m,const int dest,const int tag,const MPI_Comm pcomm);


    inline static void MPI_Isend_DataBlock_pdata(DataBlock<T> &m,const int dest,const  int tag,const MPI_Comm pcomm,MPI_Request *request);
    inline static void MPI_Irecv_DataBlock_pdata(DataBlock<T> &mds, const int source, const int tag,const  MPI_Comm pcomm,MPI_Request *request);



    inline static std::optional<MPI_Policy> default_policy;


    static const MPI_Policy& get_default_policy(MPI_Comm com,size_t blockrank)
    {
        if (!default_policy.has_value())
        {

            default_policy.emplace(true,com);
        }
        return *default_policy;
    }

    inline static void alloc_helper(bool &memmap,bool& ondevice, int& devnum, size_t rank,size_t datalength,size_t* &pextents,size_t *&pstrides,T *&pdata);
    inline static void alloc_helper2(bool &memmap,bool &ondevice, int& devicenum, size_t datalength,T *&pdata);

    inline static void free_helper(bool memmap,bool ondevice, int devnum,size_t datalength,size_t* &pextents,size_t *&pstrides,T *&pdata);

    inline static void free_helper2(bool memmap,bool ondevice, int devicenum, size_t datalength,T *&pdata);

    inline static int compute_owner(const size_t* bcoords,const size_t* proc_grid, const size_t* cyclic_block,size_t gridrank);
};



template <typename T>
void DataBlock_MPI_Functions<T>::MPI_Free_DistributedDataBlock(
    DistributedDataBlock<T>& m)
{
    if(m.plocal_blocknumber > 0 && m.pblocks != nullptr)
    {

        if(m.pextentsbuffer!=nullptr)
        {
            free(m.pextentsbuffer);
            m.pextentsbuffer=nullptr;
        }
        if(m.pstridesbuffer!=nullptr)
        {
            free(m.pstridesbuffer);
            m.pstridesbuffer=nullptr;
        }
        if(m.pblock_linear_idx!=nullptr)
        {
            free(m.pstridesbuffer);
            m.pblock_linear_idx=nullptr;
        }

        if(m.pdata != nullptr)
        {
#if defined(Unified_Shared_Memory)
            if (m.pmemmap)
            {
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(m.pdata, m.pdatalength);
            }
            else
            {
                free(m.pdata);
            }
            m.pdata=nullptr;

#else
            if(m.pdpdata_is_devptr)
            {
                omp_target_free(m.pdata,m.pdevptr_devicenum);
            }
            else
            {
                if (m.pmemmap)
                {
                    DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(m.pdata,m.pdatalength);
                }
                else
                {
                    free(m.pdata);
                }
            }
            m.pdata=nullptr;

#endif
        }
        free(m.pblocks);
        m.pblocks=nullptr;
    }


    if(m.pcoordsbuffer)
    {
        free(m.pcoordsbuffer);
        m.pcoordsbuffer=nullptr;
    }
    if(m.pglobal_extents)
    {
        free(m.pglobal_extents);
        m.pglobal_extents=nullptr;
    }
    if(m.pblock_extents)
    {
        free(m.pglobal_extents);
        m.pglobal_extents=nullptr;
    }

    if(m.pglobal_strides)
    {
        free(m.pglobal_strides);
        m.pglobal_strides=nullptr;
    }


    m.pglobal_to_local_index.clear();


}



template <typename T>
void DataBlock_MPI_Functions<T>::alloc_helper(bool &memmap,bool &ondevice, int& devicenum, size_t rank,size_t datalength,size_t*& pextents,size_t *&pstrides,T *&pdata)
{
    pextents= (size_t*)malloc(sizeof(size_t)*rank);
    pstrides= (size_t*)malloc(sizeof(size_t)*rank);
    alloc_helper2(memmap,ondevice,devicenum,datalength,pdata);

}


template <typename T>
void DataBlock_MPI_Functions<T>::alloc_helper2(bool &memmap,bool &ondevice, int& devicenum,size_t datalength,T *&pdata)
{

#if defined(Unified_Shared_Memory)
    ondevice=false;
    devicenum=-1;
    if(with_memmap)
    {
        pdata=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(pdatalength);
    }
    else
    {
        pdata=(T*)malloc(sizeof(T)*pdatalength);
    }
#else

    if(ondevice)
    {
        memmap=false;
        pdata=(T*)omp_target_alloc(sizeof(T)*datalength,devicenum);
    }
    else
    {
        devicenum=-1;
        if(memmap)
        {
            pdata=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(datalength);
        }
        else
        {
            pdata=(T*)malloc(sizeof(T)*datalength);
        }
    }
#endif
}




template <typename T>
void DataBlock_MPI_Functions<T>::free_helper(bool memmap,bool ondevice, int devicenum, size_t datalength,size_t*& pextents,size_t *&pstrides,T *&pdata)
{
    free_helper2(memmap,ondevice,devicenum,datalength,pdata);

    free(pextents);
    free(pstrides);
    pextents=nullptr;
    pstrides=nullptr;
}



template <typename T>
void DataBlock_MPI_Functions<T>::free_helper2(bool memmap,bool ondevice, int devicenum, size_t datalength,T *&pdata)
{

#if defined(Unified_Shared_Memory)
    if(with_memmap)
    {
        DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(pdata,pdatalength)
    }
    else
    {
        free(pdata);
    }
#else
    if(ondevice)
    {
        omp_target_free(pdata,devicenum);
    }
    else
    {
        if(memmap)
        {
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(pdata,datalength);
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
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_DataBlock (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (&db.dpdatalength, 1,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprank,1,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprowmajor,1,  mpi_get_type<bool>(), rootrank, com);
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, com);
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_DataBlock_pdata (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, com);
}




template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_DataBlock_meta (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (&db.dpdatalength, 1,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprank,1,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprowmajor,1,  mpi_get_type<bool>(), rootrank, com);
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_DataBlock_extents_strides (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,bool memmap,bool ondevice, int devicenum,MPI_Comm com, int rootrank)
{
    if (com == MPI_COMM_NULL)
    {
        return;
    }

    int rank;
    MPI_Comm_rank(com, &rank);
    MPI_Bcast (&db.dpdatalength,1,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprank,    1,    mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprowmajor,1,  mpi_get_type<bool>(), rootrank, com);

    if (rank != rootrank)
    {
        alloc_helper(memmap,ondevice,devicenum,db.dprank,db.dpdatalength,db.dpextents,db.dpstrides,db.dpdata);
        db.devptr_devicenum=devicenum;
        db.dpdata_is_devptr=ondevice;
        db.devptr_former_hostptr=nullptr;
    }
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, com);
}


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_matrix_to_rows_alloc(
    DistributedDataBlock<T>& recv_db,
    bool memmap, bool ondevice, int devicenum,
    MPI_CartesianContext* ctx,
    BlockMappingPolicy* policy,
    int rootrank,
    const DataBlock<T>* send_db)
{
    int rank;
    size_t cols=0;
    MPI_Comm_rank(ctx->comm, &rank);
    if (rank==rootrank)cols=send_db->dpextents[1];

    MPI_Bcast(&cols,1,mpi_get_type<size_t>(),rootrank,ctx->comm);
    MPI_Scatter_matrix_to_submatrices_alloc(1,cols,recv_db,memmap,ondevice, devicenum,ctx,policy, rootrank,send_db);
}




template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_matrix_to_columns_alloc(
    DistributedDataBlock<T>& recv_db,
    bool memmap,
    bool ondevice,
    int devicenum,
    MPI_CartesianContext* ctx,
    BlockMappingPolicy* policy,
    int rootrank,
    const DataBlock<T>* send_db)
{
    int rank;
    size_t rows=0;
    MPI_Comm_rank(ctx->comm, &rank);
    if (rank==rootrank)rows=send_db->dpextents[0];

    MPI_Bcast(&rows,1,mpi_get_type<size_t>(),rootrank,ctx->comm);

    MPI_Scatter_matrix_to_submatrices_alloc(rows,1,recv_db,memmap,ondevice, devicenum,ctx,policy, rootrank,send_db);
}


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_matrix_from_rows_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,
    DataBlock<T>* recv_db,
    bool memmap,
    bool ondevice,
    int devicenum)
{
    MPI_Gather_matrix_from_submatrices_alloc(send_db,rootrank, recv_db,memmap,ondevice,devicenum);
}


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_matrix_from_columns_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,
    DataBlock<T>* recv_db,
    bool memmap,
    bool ondevice,
    int devicenum)
{
    MPI_Gather_matrix_from_submatrices_alloc(send_db,rootrank, recv_db,memmap,ondevice,devicenum);
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::
MPI_Scatter_matrix_to_submatrices_alloc(
    size_t br,
    size_t bc,
    DistributedDataBlock<T>& recv_db,
    bool memmap,
    bool ondevice,
    int devicenum,
    MPI_CartesianContext* ctx,
    BlockMappingPolicy* policy,
    int rootrank,
    const DataBlock<T>* send_db
)
{
    recv_db.pctx =ctx;
    recv_db.ppolicy = policy;

    if (ctx->comm == MPI_COMM_NULL)
    {
        return;
    }




    recv_db.pglobalrank = 2;

    recv_db.pglobal_extents=(size_t*)malloc(sizeof(size_t)*2);
    recv_db.pglobal_strides=(size_t*)malloc(sizeof(size_t)*2);


    recv_db.pblock_extents=(size_t*)malloc(sizeof(size_t)*2);

    recv_db.pblock_rank=2;

    size_t gridrank = ctx->gridrank;


    int rank;
    MPI_Comm_rank(ctx->comm, &rank);


    if(rank==rootrank)
    {

        recv_db.pglobal_extents[0]=send_db->dpextents[0];
        recv_db.pglobal_extents[1]=send_db->dpextents[1];
        recv_db.pglobal_strides[0]=send_db->dpstrides[0];
        recv_db.pglobal_strides[1]=send_db->dpstrides[1];
        recv_db.pglobal_rowmajor=send_db->dprowmajor;
        recv_db.pblock_extents[0]=br;
        recv_db.pblock_extents[1]=bc;
    }

    MPI_Bcast(recv_db.pglobal_extents,2,mpi_get_type<size_t>(),rootrank,ctx->comm);
    MPI_Bcast(recv_db.pglobal_strides,2,mpi_get_type<size_t>(),rootrank,ctx->comm);
    MPI_Bcast(&recv_db.pglobal_rowmajor,1,mpi_get_type<bool>(),rootrank,ctx->comm);
    MPI_Bcast(recv_db.pblock_extents,2,mpi_get_type<size_t>(),rootrank,ctx->comm);



    size_t M = recv_db.pglobal_extents[0];
    size_t N = recv_db.pglobal_extents[1];

    size_t grid_r = (M + br - 1) / br;
    size_t grid_c = (N + bc - 1) / bc;

    size_t total_blocks = grid_r * grid_c;

    size_t local_blocks=0;

    size_t* local_block_indices = new size_t[total_blocks];

    size_t bi, bj;
    size_t bcoords[2];
    size_t *grid_coords=new size_t[gridrank];
    int *temp_coords=new int[gridrank];
    for(size_t b = 0; b < total_blocks; b++)
    {
        bi = b / grid_c;
        bj = b % grid_c;

        bcoords[0]=bi;
        bcoords[1]=bj;
        policy->create_coords(bcoords, grid_coords, recv_db.pglobalrank);
        int owner = policy->owner(grid_coords, *ctx, temp_coords);

        if(owner == rank)
        {
            local_block_indices[local_blocks] = b;
            local_blocks++;
        }
    }
    delete[] grid_coords;
    delete[] temp_coords;

    recv_db.plocal_blocknumber = local_blocks;

    recv_db.plocal_blocknumber=local_blocks;

    recv_db.pblocks =
        (local_blocks>0)?(DataBlock<T>*)malloc(sizeof(DataBlock<T>)*local_blocks):nullptr;

    recv_db.pblock_coords =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*2*local_blocks):nullptr;

    recv_db.pblock_indices =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*2*local_blocks):nullptr;

    recv_db.pblock_linear_idx =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*local_blocks):nullptr;

    recv_db.pblock_offsets =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*local_blocks):nullptr;

    recv_db.pextentsbuffer = (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*2*local_blocks):nullptr;

    recv_db.pstridesbuffer= (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*2*local_blocks):nullptr;

    recv_db.pglobal_to_local_index.reserve(local_blocks);



    struct BlockInfo
    {
        size_t my_idx;
        size_t bi, bj;
        size_t rows, cols;
        size_t blocksize;
    };

    BlockInfo* blocks= new BlockInfo[local_blocks];
    size_t total_recv_elems=0;

    #pragma omp parallel for reduction(+:total_recv_elems)
    for(size_t i = 0; i < local_blocks; i++)
    {
        size_t b = local_block_indices[i];
        size_t bi = b / grid_c;
        size_t bj = b % grid_c;

        size_t r0 = bi * br;
        size_t c0 = bj * bc;

        size_t rows = (br < (M-r0)) ? br : (M-r0);
        size_t cols = (bc < (N-c0)) ? bc : (N-c0);
        size_t blocksize=rows*cols;
        blocks[i] = {b, bi, bj, rows, cols,blocksize};

        recv_db.pblock_indices[2*i] = bi;
        recv_db.pblock_indices[2*i+1] = bj;
        recv_db.pblock_coords[2*i] = r0;
        recv_db.pblock_coords[2*i+1] = c0;

        recv_db.pblock_linear_idx[i] = b;


        total_recv_elems += blocksize;
    }

    delete []local_block_indices;


    recv_db.pdatalength=total_recv_elems;
    recv_db.pdata=nullptr;

    if(total_recv_elems>0)
        alloc_helper2(memmap,ondevice,devicenum,total_recv_elems,recv_db.pdata);

    recv_db.pdpdata_is_devptr=ondevice;
    recv_db.pdevptr_devicenum=devicenum;
    recv_db.pmemmap=memmap;

    MPI_Request* reqs=new MPI_Request[local_blocks];

    size_t offset=0;
    for(size_t i = 0; i < local_blocks; i++)
    {
        T* ptr = recv_db.pdata + offset;
        size_t b = blocks[i].bi * grid_c + blocks[i].bj;
        MPI_Irecv(
            ptr,
            blocks[i].rows * blocks[i].cols,
            mpi_get_type<T>(),
            rootrank,
            b,
            ctx->comm,
            &reqs[i]);
        recv_db.pglobal_to_local_index[blocks[i].my_idx] = i;
        recv_db.pblock_offsets[i]=offset;
        offset+=blocks[i].blocksize;

    }





    if(rank==rootrank)
    {
        MPI_Datatype tmp0,blocktype0;

        MPI_Type_vector(
            send_db->dprowmajor? br:bc,
            send_db->dprowmajor? bc:br,
            send_db->dprowmajor? send_db->dpstrides[0]:send_db->dpstrides[1],
            mpi_get_type<T>(),
            &tmp0);

        MPI_Type_create_resized(tmp0,0,sizeof(T),&blocktype0);
        MPI_Type_commit(&blocktype0);
        MPI_Type_free(&tmp0);


        MPI_Request* sendreqs=new MPI_Request[total_blocks];
        size_t *grid_coords=new size_t [gridrank];
        int *temp_coords=new int [gridrank];
        size_t bcoords[2];

        for(size_t bi=0; bi<grid_r; bi++)
        {
            for(size_t bj=0; bj<grid_c; bj++)
            {
                size_t b = bi * grid_c + bj;

                bcoords[0] = bi;
                bcoords[1] = bj;
                policy->create_coords(bcoords, grid_coords, recv_db.pglobalrank);
                int owner = policy->owner(grid_coords, *ctx, temp_coords);

                size_t r0 = bi * br;
                size_t c0 = bj * bc;

                size_t diff1=M-r0,
                       diff2=N-c0;

                bool edgecase=false;

                MPI_Datatype tmp1,blocktype1;
                if(diff1 < br || diff2 < bc)
                {
                    size_t rows=br<diff1? br:diff1;
                    size_t cols=bc<diff2? bc:diff2;

                    MPI_Type_vector(
                        send_db->dprowmajor? rows:cols,
                        send_db->dprowmajor? cols:rows,
                        send_db->dprowmajor? send_db->dpstrides[0]:send_db->dpstrides[1],
                        mpi_get_type<T>(),
                        &tmp1);
                    MPI_Type_create_resized(tmp1,0,sizeof(T),&blocktype1);
                    MPI_Type_commit(&blocktype1);
                    MPI_Type_free(&tmp1);
                    edgecase=true;
                }


                T* start =
                    send_db->dpdata +
                    r0*send_db->dpstrides[0] +
                    c0*send_db->dpstrides[1];

                MPI_Isend(
                    start,
                    1,
                    edgecase? blocktype1: blocktype0,
                    owner,
                    b,
                    ctx->comm,
                    &sendreqs[b]);

                if(edgecase)
                    MPI_Type_free(&blocktype1);
            }
        }



        MPI_Waitall(total_blocks,sendreqs,MPI_STATUSES_IGNORE);
        MPI_Type_free(&blocktype0);
        delete[] grid_coords;
        delete[] temp_coords;
        delete[] sendreqs;
    }

    MPI_Waitall(local_blocks,reqs,MPI_STATUSES_IGNORE);

    delete[] reqs;

    #pragma omp parallel for
    for (size_t i=0; i<local_blocks; i++)
    {


        size_t* bext_i = recv_db.pextentsbuffer + i*2;
        size_t* bstr_i = recv_db.pstridesbuffer + i*2;

        bext_i[0] = blocks[i].rows;
        bext_i[1] = blocks[i].cols;

        bstr_i[0] = recv_db.pglobal_rowmajor ? blocks[i].cols : 1;
        bstr_i[1] = recv_db.pglobal_rowmajor ? 1 : blocks[i].rows;

        recv_db.pblocks[i] =
            DataBlock<T>(
                recv_db.pdata+recv_db.pblock_offsets[i],
                blocks[i].rows * blocks[i].cols,
                recv_db.pglobal_rowmajor,
                2,
                bext_i,
                bstr_i,
                ondevice,
                devicenum);

    }
    delete[] blocks;

}


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_matrix_from_submatrices_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,
    DataBlock<T>* recv_db,
    bool memmap,
    bool ongpu,
    int devicenum)
{

    if (send_db.pctx->comm == MPI_COMM_NULL)
        return;
    int rank;

    MPI_Comm_rank(send_db.pctx->comm,&rank);


    size_t M = send_db.pglobal_extents[0];
    size_t N = send_db.pglobal_extents[1];
    bool rowmajor = send_db.pglobal_rowmajor;



    size_t br=0, bc=0;


    br = send_db.pblock_extents[0];
    bc = send_db.pblock_extents[1];



    size_t grid_r = (M + br - 1) / br;
    size_t grid_c = (N + bc - 1) / bc;
    size_t total_blocks = grid_r * grid_c;



    if(rank==rootrank)
    {
        size_t *ext=nullptr;
        size_t *str=nullptr;
        T *pdata=nullptr;

        size_t datalen=M*N;

        alloc_helper(
            memmap,
            ongpu,
            devicenum,
            2,
            datalen,
            ext,
            str,
            pdata);

        ext[0]=M;
        ext[1]=N;

        str[1]=rowmajor?1:M;
        str[0]=rowmajor?N:1;

        *recv_db = DataBlock<T>(
                       pdata,
                       datalen,
                       rowmajor,
                       2,
                       ext,
                       str,
                       ongpu,
                       devicenum);
    }



    size_t recv_idx=0;
    MPI_Request *reqs  = new MPI_Request[total_blocks];
    if(rank==rootrank)
    {

        MPI_Datatype tmp,type;
        MPI_Type_vector(
            rowmajor? br:bc,
            rowmajor? bc:br,
            rowmajor? recv_db->dpstrides[0]:recv_db->dpstrides[1],
            mpi_get_type<T>(),&tmp);
        MPI_Type_create_resized(tmp,0,sizeof(T),&type);
        MPI_Type_commit(&type);
        MPI_Type_free(&tmp);

        size_t *grid_coords=new size_t [send_db.pctx->gridrank];
        int *tempcoords=new int[send_db.pctx->gridrank];

        for(size_t bi=0; bi<grid_r; bi++)
        {
            for(size_t bj=0; bj<grid_c; bj++)
            {
                MPI_Datatype type1;
                size_t b = bi*grid_c + bj;

                size_t bcoords[2] = {bi, bj};

                send_db.ppolicy->create_coords(bcoords, grid_coords, send_db.pglobalrank);
                int owner = send_db.ppolicy->owner(grid_coords,*send_db.pctx, tempcoords);

                size_t r0 = bi*br;
                size_t c0 = bj*bc;
                size_t diff1=M-r0;
                size_t diff2=N-c0;

                bool edgecase=false;
                if(diff1<br|| diff2<bc)
                {
                    size_t rows = br<=diff1? br:diff1;
                    size_t cols = bc<=diff2? bc:diff2;

                    MPI_Datatype tmp1;

                    MPI_Type_vector(
                        rowmajor? rows:cols,
                        rowmajor? cols:rows,
                        rowmajor? recv_db->dpstrides[0]:recv_db->dpstrides[1],
                        mpi_get_type<T>(),
                        &tmp1);

                    MPI_Type_create_resized(tmp1,0,sizeof(T),&type1);
                    MPI_Type_commit(&type1);
                    MPI_Type_free(&tmp1);
                    edgecase=true;
                }

                T* start =
                    recv_db->dpdata +
                    r0*recv_db->dpstrides[0] +
                    c0*recv_db->dpstrides[1];

                MPI_Irecv(
                    start,
                    1,
                    edgecase? type1:type,
                    owner,
                    b,
                    send_db.pctx->comm,
                    &reqs[recv_idx]);

                recv_idx++;

                if(edgecase)
                    MPI_Type_free(&type1);
            }
        }
        delete []tempcoords;
        delete[] grid_coords;
        MPI_Type_free(&type);
    }



    MPI_Request* sendreqs =(send_db.plocal_blocknumber>0)? new MPI_Request[send_db.plocal_blocknumber]: nullptr;

    size_t send_idx=0;

    for(size_t i=0; i<send_db.plocal_blocknumber; i++)
    {
        size_t b = send_db.pblock_linear_idx[i];

        size_t rows = send_db.pblocks[i].dpextents[0];
        size_t cols = send_db.pblocks[i].dpextents[1];

        MPI_Isend(
            send_db.pblocks[i].dpdata,
            rows*cols,
            mpi_get_type<T>(),
            rootrank,
            b,
            send_db.pctx->comm,
            &sendreqs[send_idx++]);
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
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_tensor_to_subtensors_alloc(
    size_t blockrank,
    const size_t* block_extents,
    DistributedDataBlock<T>& recv_db,
    bool memmap,
    bool ondevice,
    int devicenum,
    MPI_CartesianContext *ctx,
    BlockMappingPolicy* policy, int rootrank,
    const DataBlock<T>* send_db)

{

    recv_db.pctx =ctx;
    recv_db.ppolicy = policy;

    if (ctx->comm == MPI_COMM_NULL)
    {
        return;
    }

    int rank;
    MPI_Comm_rank(ctx->comm,&rank);

    if(rank == rootrank)
    {

        recv_db.pblock_rank = blockrank< send_db->dprank?blockrank:send_db->dprank;

        recv_db.pglobalrank = send_db->dprank;
        recv_db.pglobal_rowmajor = send_db->dprowmajor;
        recv_db.pglobal_extents = (size_t*)malloc(sizeof(size_t)*recv_db.pglobalrank);
        recv_db.pglobal_strides = (size_t*)malloc(sizeof(size_t)*recv_db.pglobalrank);
        recv_db.pblock_extents = (size_t*)malloc(sizeof(size_t)*recv_db.pblock_rank);
        #pragma omp parallel for simd if(parallel:recv_db.pglobalrank>30)
        for(size_t d=0; d<recv_db.pglobalrank; d++)
        {
            recv_db.pglobal_extents[d] = send_db->dpextents[d];
            recv_db.pglobal_strides[d] = send_db->dpstrides[d];
        }
        #pragma omp parallel for simd if(parallel:recv_db.pblock_rank>30)
        for(size_t d=0; d<recv_db.pblock_rank; d++)
            recv_db.pblock_extents[d] = block_extents[d];
    }
    MPI_Bcast(&recv_db.pglobalrank, 1, mpi_get_type<size_t>(), rootrank, ctx->comm );
    MPI_Bcast(&recv_db.pblock_rank, 1, mpi_get_type<size_t>(), rootrank, ctx->comm );
    MPI_Bcast(&recv_db.pglobal_rowmajor, 1, mpi_get_type<bool>(), rootrank, ctx->comm );


    if(rank != rootrank)
    {
        recv_db.pglobal_extents = (size_t*)malloc(sizeof(size_t) * recv_db.pglobalrank);
        recv_db.pglobal_strides = (size_t*)malloc(sizeof(size_t) * recv_db.pglobalrank);
        recv_db.pblock_extents = (size_t*)malloc(sizeof(size_t) * recv_db.pblock_rank);
    }

    MPI_Bcast(recv_db.pglobal_extents, recv_db.pglobalrank, mpi_get_type<size_t>(), rootrank, ctx->comm );
    MPI_Bcast(recv_db.pglobal_strides, recv_db.pglobalrank, mpi_get_type<size_t>(), rootrank, ctx->comm );
    MPI_Bcast(recv_db.pblock_extents, recv_db.pblock_rank, mpi_get_type<size_t>(), rootrank, ctx->comm );

    size_t* grid = new size_t[recv_db.pglobalrank];


    #pragma omp parallel for simd if(parallel: blockrank > 30)
    for(size_t d = 0; d < blockrank; d++)
    {
        grid[d] = (recv_db.pglobal_extents[d] + recv_db.pblock_extents[d] - 1) / recv_db.pblock_extents[d];
    }

    #pragma omp parallel for if(parallel: recv_db.pglobalrank-blockrank > 30)
    for(size_t d = blockrank; d < recv_db.pglobalrank; d++)
        grid[d] = 1;

    size_t total_blocks = 1;
    #pragma omp parallel for simd reduction(*:total_blocks) if(parallel: recv_db.pglobalrank > 30)
    for(size_t d = 0; d < recv_db.pglobalrank; d++)
        total_blocks *= grid[d];


    size_t local_blocks = 0;

    struct BlockInfo
    {
        size_t linear_idx;
        size_t blocksize;
        size_t* coords;
        size_t* starts;
        size_t* extents;
    };

    std::vector<BlockInfo> blocks;
    if (total_blocks > 0)
        blocks.reserve(total_blocks);

    size_t* bcoords = new size_t[recv_db.pglobalrank];
    size_t *grid_coords= new size_t[ctx->gridrank];
    int *tmpcoords=new int [ctx->gridrank];
    for (size_t b = 0; b < total_blocks; b++)
    {
        size_t tmp = b;

        #pragma omp unroll partial
        for (int d = recv_db.pglobalrank - 1; d >= 0; d--)
        {
            bcoords[d] = tmp % grid[d];
            tmp /= grid[d];
        }

        policy->create_coords(bcoords, grid_coords, recv_db.pglobalrank);
        int owner = policy->owner(grid_coords, *ctx, tmpcoords);

        if (owner != rank)
        {
            continue;
        }
        BlockInfo block;
        block.linear_idx = b;
        block.coords  = new size_t[recv_db.pglobalrank];
        block.starts  = new size_t[recv_db.pblock_rank];
        block.extents = new size_t[recv_db.pblock_rank];

        #pragma omp parallel for simd  if(parallel:recv_db.pglobalrank>30)
        for (size_t d = 0; d < recv_db.pglobalrank; d++)
            block.coords[d] = bcoords[d];

        size_t blocksize = 1;
        #pragma omp parallel for simd reduction(*:blocksize) if(parallel:recv_db.pblock_rank>30)
        for (size_t d = 0; d < recv_db.pblock_rank; d++)
        {
            size_t start = bcoords[d] * recv_db.pblock_extents[d];
            size_t diff  = recv_db.pglobal_extents[d] - start;
            size_t len   = (recv_db.pblock_extents[d] <= diff) ? recv_db.pblock_extents[d] : diff;

            block.starts[d]  = start;
            block.extents[d] = len;
            blocksize *= len;
        }

        #pragma omp parallel for simd reduction(*:blocksize) if(parallel:recv_db.pglobalrank>30)
        for (size_t d = recv_db.pblock_rank; d < recv_db.pglobalrank; d++)
            blocksize *= recv_db.pglobal_extents[d];

        block.blocksize = blocksize;
        blocks.push_back(block);

    }
    delete[] bcoords;
    delete[] grid_coords;
    delete[] tmpcoords;


    local_blocks=blocks.size();
    recv_db.plocal_blocknumber = local_blocks;

    recv_db.pblock_offsets = (size_t*)malloc(sizeof(size_t)*local_blocks);

    recv_db.pblock_linear_idx =
        (local_blocks > 0) ? (size_t*)malloc(sizeof(size_t)*local_blocks) : nullptr;

    recv_db.pcoordsbuffer =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*local_blocks*recv_db.pglobalrank):nullptr;

    recv_db.pblock_indices =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*local_blocks*recv_db.pblock_rank):nullptr;

    size_t total_recv_elems = 0;



    for(size_t i = 0; i < local_blocks; i++)
    {
        recv_db.pblock_offsets[i] = total_recv_elems;
        total_recv_elems += blocks[i].blocksize;
    }


    #pragma omp parallel for
    for(size_t i = 0; i < local_blocks; i++)
    {
        #pragma omp simd
        for(size_t d = 0; d < recv_db.pglobalrank; d++)
        {
            recv_db.pcoordsbuffer[i*recv_db.pglobalrank + d] = blocks[i].coords[d];
            recv_db.pblock_indices[i*recv_db.pblock_rank + d] = blocks[i].starts[d];
        }
        recv_db.pblock_linear_idx[i] = blocks[i].linear_idx;
    }


    recv_db.pdatalength = total_recv_elems;
    recv_db.pdata = nullptr;

    if(total_recv_elems > 0)
        alloc_helper2(memmap,ondevice,devicenum,total_recv_elems,recv_db.pdata);

    recv_db.pglobal_to_local_index.reserve(local_blocks);

    MPI_Request* reqs = new MPI_Request[local_blocks];



    for(size_t i=0; i<local_blocks; i++)
    {
        MPI_Irecv(
            recv_db.pdata + recv_db.pblock_offsets[i],
            blocks[i].blocksize,
            mpi_get_type<T>(),
            rootrank,
            blocks[i].linear_idx,
            ctx->comm,
            &reqs[i]);

        recv_db.pglobal_to_local_index[blocks[i].linear_idx] = i;
    }
    if(rank == rootrank)
    {
        MPI_Request* sendreqs = new MPI_Request[total_blocks];

        size_t *bcoords=new size_t [recv_db.pglobalrank];
        size_t *grid_coords= new size_t[ctx->gridrank];
        int* tmpcoords=new int[ctx->gridrank];
        for(size_t b = 0; b < total_blocks; b++)
        {
            size_t tmp = b;
            #pragma omp unroll
            for(int d = recv_db.pglobalrank-1; d >= 0; d--)
            {
                bcoords[d] = tmp % grid[d];
                tmp /= grid[d];
            }

            policy->create_coords(bcoords, grid_coords, recv_db.pglobalrank);
            int owner = policy->owner(grid_coords, *ctx, tmpcoords);

            MPI_Datatype tmp_type, blocktype;
            int* sizes  = new int[blockrank];
            int* subs   = new int[blockrank];
            int* starts = new int[blockrank];
            #pragma omp parallel for simd if(parallel: blockrank>30)
            for(size_t d = 0; d < blockrank; d++)
            {
                sizes[d]   = (int)send_db->dpextents[d];
                starts[d] = (int)(bcoords[d] * block_extents[d]);

                int diff = recv_db.pglobal_extents[d] - starts[d];
                subs[d]    = (int)block_extents[d]<diff? block_extents[d]:diff;
            }

            MPI_Type_create_subarray(
                blockrank,
                sizes,
                subs,
                starts,
                send_db->dprowmajor ? MPI_ORDER_C : MPI_ORDER_FORTRAN,
                mpi_get_type<T>(),
                &tmp_type);

            MPI_Type_create_resized(tmp_type, 0, sizeof(T), &blocktype);
            MPI_Type_commit(&blocktype);
            MPI_Type_free(&tmp_type);

            MPI_Isend(
                send_db->dpdata,
                1,
                blocktype,
                owner,
                b,
                ctx->comm,
                &sendreqs[b]);

            MPI_Type_free(&blocktype);

            delete[] sizes;
            delete[] subs;
            delete[] starts;
        }

        MPI_Waitall(total_blocks, sendreqs, MPI_STATUSES_IGNORE);
        delete[]bcoords;
        delete[]grid_coords;
        delete[]tmpcoords;
        delete[] sendreqs;
    }



    MPI_Waitall(local_blocks, reqs, MPI_STATUSES_IGNORE);
    delete[] reqs;
    delete[] grid;

    recv_db.pblocks =
        (local_blocks>0)?(DataBlock<T>*)malloc(sizeof(DataBlock<T>)*local_blocks):nullptr;

    recv_db.pextentsbuffer =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*blockrank*local_blocks):nullptr;

    recv_db.pstridesbuffer =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*blockrank*local_blocks):nullptr;

    #pragma omp parallel for
    for(size_t i=0; i<local_blocks; i++)
    {
        size_t* bext = recv_db.pextentsbuffer + i*blockrank;
        size_t* bstr = recv_db.pstridesbuffer + i*blockrank;
        #pragma omp simd
        for(size_t d=0; d<blockrank; d++)
            bext[d] = blocks[i].extents[d];

        if(recv_db.pglobal_rowmajor)
        {
            bstr[blockrank-1] = 1;
            #pragma omp unroll partial
            for(int d=blockrank-2; d>=0; d--)
                bstr[d] = bstr[d+1] * bext[d+1];
        }
        else
        {
            bstr[0] = 1;
            #pragma omp unroll partial
            for(size_t d=1; d<blockrank; d++)
                bstr[d] = bstr[d-1] * bext[d-1];
        }

        recv_db.pblocks[i] =
            DataBlock<T>(
                recv_db.pdata + recv_db.pblock_offsets[i],
                blocks[i].blocksize,
                recv_db.pglobal_rowmajor,
                blockrank,
                bext,
                bstr,
                ondevice,
                devicenum);
        delete[] blocks[i].coords;
        delete[] blocks[i].starts;
        delete[] blocks[i].extents;
    }
    blocks.clear();
}


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_tensor_from_subtensors_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,
    DataBlock<T>* recv_db,
    bool memmap,
    bool ongpu,
    int devicenum)
{
    if (send_db.pctx->comm == MPI_COMM_NULL)
        return;

    int rank,size;
    MPI_Comm_rank(send_db.pctx->comm,&rank);
    MPI_Comm_size(send_db.pctx->comm,&size);

    size_t rank_t    = send_db.pglobalrank;
    size_t blockrank = send_db.pblock_rank;


    size_t* global_ext = send_db.pglobal_extents;
    size_t* block_ext  = send_db.pblock_extents;

    bool rowmajor = send_db.pglobal_rowmajor;



    size_t* grid = new size_t[rank_t];
    size_t total_blocks = 1;

    #pragma omp parallel for simd if(parallel: blockrank>30)
    for(size_t d=0; d<blockrank; d++)
        grid[d] = (global_ext[d] + block_ext[d] - 1) / block_ext[d];

    #pragma omp parallel for simd if(parallel: rank_t-blockrank>30)
    for(size_t d=blockrank; d<rank_t; d++)
        grid[d] = 1;

    #pragma omp parallel for simd reduction(*:total_blocks)if(parallel:rank_t>30)
    for(size_t d=0; d<rank_t; d++)
        total_blocks *= grid[d];



    if(rank==rootrank)
    {
        size_t *ext=nullptr;
        size_t *str=nullptr;
        T *pdata=nullptr;

        size_t datalen=1;

        #pragma omp parallel for simd reduction(*:datalen)if(parallel:rank_t>30)
        for(size_t d=0; d<rank_t; d++)
            datalen*=global_ext[d];

        alloc_helper(
            memmap,
            ongpu,
            devicenum,
            rank_t,
            datalen,
            ext,
            str,
            pdata);

        #pragma omp parallel for simd if(parallel:rank_t>30)
        for(size_t d=0; d<rank_t; d++)
            ext[d]=global_ext[d];

        if(rowmajor)
        {
            str[rank_t-1]=1;
            #pragma omp unroll partial
            for(int d=rank_t-2; d>=0; d--)
                str[d]=str[d+1]*ext[d+1];
        }
        else
        {
            str[0]=1;
            #pragma omp unroll partial
            for(size_t d=1; d<rank_t; d++)
                str[d]=str[d-1]*ext[d-1];
        }

        *recv_db = DataBlock<T>(
                       pdata,
                       datalen,
                       rowmajor,
                       rank_t,
                       ext,
                       str,
                       ongpu,
                       devicenum);
    }



    MPI_Request* reqs = nullptr;

    if(rank==rootrank)
        reqs = new MPI_Request[total_blocks];

    size_t recv_idx = 0;

    if(rank==rootrank)
    {
        int* sizes  = new int[rank_t];
        int* subs   = new int[rank_t];
        int* starts = new int[rank_t];

        #pragma omp parallel for simd if(parallel:rank_t>30)
        for(size_t d=0; d<rank_t; d++)
            sizes[d]=(int)global_ext[d];


        size_t* bcoords=new size_t[rank_t];
        size_t *grid_coords=new size_t [send_db.pctx->gridrank];
        int *tempcoords=new int[send_db.pctx->gridrank];
        for(size_t b=0; b<total_blocks; b++)
        {
            size_t tmp=b;


            #pragma omp unroll partial
            for(int d=rank_t-1; d>=0; d--)
            {
                bcoords[d] = tmp % grid[d];
                tmp /= grid[d];
            }

            send_db.ppolicy->create_coords(bcoords, grid_coords, send_db.pglobalrank);
            int owner = send_db.ppolicy->owner(grid_coords,*send_db.pctx, tempcoords);

            #pragma omp parallel for simd if(parallel:rank_t>30)
            for(size_t d=0; d<rank_t; d++)
            {
                starts[d] = (int)(bcoords[d]*block_ext[d]);
                size_t diff = global_ext[d]-starts[d];
                subs[d] =(int)(block_ext[d] <= diff ?  block_ext[d] : diff);
            }

            MPI_Datatype tmp_type,blocktype;

            MPI_Type_create_subarray(
                rank_t,
                sizes,
                subs,
                starts,
                rowmajor?MPI_ORDER_C:MPI_ORDER_FORTRAN,
                mpi_get_type<T>(),
                &tmp_type);

            MPI_Type_create_resized(tmp_type,0,sizeof(T),&blocktype);
            MPI_Type_commit(&blocktype);
            MPI_Type_free(&tmp_type);

            MPI_Irecv(
                recv_db->dpdata,
                1,
                blocktype,
                owner,
                b,
                send_db.pctx->comm,
                &reqs[recv_idx++]);

            MPI_Type_free(&blocktype);

        }
        delete[] sizes;
        delete[] subs;
        delete[] starts;
        delete[]bcoords;
        delete[]grid_coords;
        delete[] tempcoords;
    }



    MPI_Request* sendreqs =
        send_db.plocal_blocknumber ?
        new MPI_Request[send_db.plocal_blocknumber] :
        nullptr;

    size_t send_idx=0;

    for(size_t i=0; i<send_db.plocal_blocknumber; i++)
    {
        size_t b = send_db.pblock_linear_idx[i];


        size_t elems=1;

        #pragma omp parallel for simd reduction(*:elems) if(parallel:blockrank>30)
        for(size_t d=0; d<blockrank; d++)
            elems*=send_db.pblocks[i].dpextents[d];

        #pragma omp parallel for simd reduction(*:elems) if(parallel:rank_t-blockrank>30)
        for(size_t d=blockrank; d<rank_t; d++)
            elems*=global_ext[d];

        MPI_Isend(
            send_db.pblocks[i].dpdata,
            elems,
            mpi_get_type<T>(),
            rootrank,
            b,
            send_db.pctx->comm,
            &sendreqs[send_idx++]);
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
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_vector_to_subvectors_alloc(
    size_t blocksize,
    DistributedDataBlock<T>& recv_db,
    bool memmap,
    bool ondevice,
    int devicenum,
    MPI_CartesianContext *ctx,
    BlockMappingPolicy* policy,
    int rootrank,
    const DataBlock<T>* send_db)
{
    recv_db.pctx =ctx;
    recv_db.ppolicy = policy;

    if (ctx->comm == MPI_COMM_NULL)
    {
        return;
    }

    int rank, size;
    MPI_Comm_rank(ctx->comm, &rank);

    recv_db.pglobalrank = 1;

    recv_db.pglobal_extents  = (size_t*)malloc(sizeof(size_t));
    recv_db.pglobal_strides  = (size_t*)malloc(sizeof(size_t));
    recv_db.pblock_extents   = (size_t*)malloc(sizeof(size_t));

    recv_db.pblock_rank = 1;
    recv_db.pglobal_rowmajor=true;


    if (rank == rootrank)
    {
        recv_db.pglobal_extents[0] = send_db->dpextents[0];
        recv_db.pglobal_strides[0] = send_db->dpstrides[0];
        recv_db.pblock_extents[0] = blocksize;
    }


    MPI_Bcast(recv_db.pglobal_extents, 1, mpi_get_type<size_t>(), rootrank, ctx->comm);
    MPI_Bcast(recv_db.pglobal_strides, 1, mpi_get_type<size_t>(), rootrank, ctx->comm);
    MPI_Bcast(recv_db.pblock_extents, 1, mpi_get_type<size_t>(), rootrank, ctx->comm);

    size_t N  = recv_db.pglobal_extents[0];
    size_t bs = recv_db.pblock_extents[0];

    size_t grid = (N + bs - 1) / bs;
    size_t total_blocks = grid;


    size_t local_blocks = 0;
    size_t* local_block_indices = new size_t[total_blocks];

    size_t* grid_coords=new size_t[ctx->gridrank];
    int *temp_coords=new int[ctx->gridrank];

    for (size_t b = 0; b < total_blocks; b++)
    {
        size_t bcoords[1] = {b};
        policy->create_coords(bcoords, grid_coords, recv_db.pglobalrank);
        int owner = policy->owner(grid_coords, *ctx, temp_coords);

        if (owner == rank)
            local_block_indices[local_blocks++] = b;
    }

    delete []grid_coords;
    delete []temp_coords;

    recv_db.plocal_blocknumber = local_blocks;


    recv_db.pblocks =
        local_blocks ? (DataBlock<T>*)malloc(sizeof(DataBlock<T>) * local_blocks) : nullptr;

    recv_db.pcoordsbuffer =
        local_blocks ? (size_t*)malloc(sizeof(size_t) * local_blocks) : nullptr;

    recv_db.pblock_indices =
        local_blocks ? (size_t*)malloc(sizeof(size_t) * local_blocks) : nullptr;

    recv_db.pblock_linear_idx =
        local_blocks ? (size_t*)malloc(sizeof(size_t) * local_blocks) : nullptr;

    recv_db.pblock_offsets =
        local_blocks ? (size_t*)malloc(sizeof(size_t) * local_blocks) : nullptr;

    recv_db.pextentsbuffer =
        local_blocks ? (size_t*)malloc(sizeof(size_t) * local_blocks) : nullptr;

    recv_db.pstridesbuffer =
        local_blocks ? (size_t*)malloc(sizeof(size_t) * local_blocks) : nullptr;

    recv_db.pglobal_to_local_index.reserve(local_blocks);


    size_t total_recv_elems = 0;

    for (size_t i = 0; i < local_blocks; i++)
    {
        size_t b = local_block_indices[i];

        size_t start = b * bs;
        size_t diff=N-start;
        size_t len   = bs<diff? bs:diff;

        recv_db.pcoordsbuffer[i]     = start;
        recv_db.pblock_indices[i]    = start;
        recv_db.pblock_linear_idx[i] = b;
        recv_db.pblock_offsets[i]    = total_recv_elems;

        total_recv_elems += len;
    }

    delete[] local_block_indices;



    recv_db.pdatalength = total_recv_elems;
    recv_db.pdata = nullptr;

    if (total_recv_elems > 0)
        alloc_helper2(memmap, ondevice, devicenum, total_recv_elems, recv_db.pdata);

    recv_db.pdpdata_is_devptr = ondevice;
    recv_db.pdevptr_devicenum = devicenum;
    recv_db.pmemmap = memmap;



    MPI_Request* reqs = new MPI_Request[local_blocks];

    for (size_t i = 0; i < local_blocks; i++)
    {
        size_t len =
            (i + 1 < local_blocks)
            ? recv_db.pblock_offsets[i+1] - recv_db.pblock_offsets[i]
            : (total_recv_elems - recv_db.pblock_offsets[i]);

        MPI_Irecv(
            recv_db.pdata + recv_db.pblock_offsets[i],
            len,
            mpi_get_type<T>(),
            rootrank,
            recv_db.pblock_linear_idx[i],
            ctx->comm,
            &reqs[i]);

        recv_db.pglobal_to_local_index[recv_db.pblock_linear_idx[i]] = i;
    }



    if (rank == rootrank)
    {
        MPI_Datatype base_type, block_type;


        MPI_Type_vector(
            (int)bs,
            1,
            (int)send_db->dpstrides[0],
            mpi_get_type<T>(),
            &base_type);

        MPI_Type_create_resized(base_type, 0, sizeof(T), &block_type);
        MPI_Type_commit(&block_type);
        MPI_Type_free(&base_type);

        MPI_Request* sendreqs = new MPI_Request[total_blocks];
        size_t* grid_coords=new size_t[ctx->gridrank];
        int *temp_coords=new int[ctx->gridrank];

        for (size_t b = 0; b < total_blocks; b++)
        {
            size_t bcoords[1] = { b };

            policy->create_coords(bcoords, grid_coords, recv_db.pglobalrank);
            int owner = policy->owner(grid_coords, *ctx, temp_coords);


            size_t start = b * bs;
            size_t diff=N-start;
            size_t len   = bs<diff? bs:diff;

            bool edgecase = (len != bs);

            MPI_Datatype send_type = block_type;
            MPI_Datatype tmp_edge;

            if (edgecase)
            {
                MPI_Type_vector(
                    (int)len,
                    1,
                    (int)send_db->dpstrides[0],
                    mpi_get_type<T>(),
                    &tmp_edge);

                MPI_Type_create_resized(tmp_edge, 0, sizeof(T), &send_type);
                MPI_Type_commit(&send_type);
                MPI_Type_free(&tmp_edge);
            }

            T* ptr = send_db->dpdata + start * send_db->dpstrides[0];

            MPI_Isend(
                ptr,
                1,
                send_type,
                owner,
                b,
                ctx->comm,
                &sendreqs[b]);

            if (edgecase)
                MPI_Type_free(&send_type);
        }

        MPI_Waitall(total_blocks, sendreqs, MPI_STATUSES_IGNORE);
        MPI_Type_free(&block_type);
        delete[] sendreqs;
        delete []grid_coords;
        delete []temp_coords;
    }

    MPI_Waitall(local_blocks, reqs, MPI_STATUSES_IGNORE);
    delete[] reqs;



    for (size_t i = 0; i < local_blocks; i++)
    {
        size_t len =
            (i + 1 < local_blocks)
            ? recv_db.pblock_offsets[i+1] - recv_db.pblock_offsets[i]
            : (total_recv_elems - recv_db.pblock_offsets[i]);

        size_t* ext = recv_db.pextentsbuffer + i;
        size_t* str = recv_db.pstridesbuffer + i;

        ext[0] = len;
        str[0] = 1;

        recv_db.pblocks[i] =
            DataBlock<T>(
                recv_db.pdata + recv_db.pblock_offsets[i],
                len,
                recv_db.pglobal_rowmajor,
                1,
                ext,
                str,
                ondevice,
                devicenum);
    }
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_vector_from_subvectors_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,
    DataBlock<T>* recv_db,
    bool memmap,
    bool ongpu,
    int devicenum)
{
    if(send_db.pctx==nullptr)
        return;
    if (send_db.pctx->comm == MPI_COMM_NULL)
        return;

    int rank, size;
    MPI_Comm_rank(send_db.pctx->comm, &rank);
    MPI_Comm_size(send_db.pctx->comm, &size);

    size_t N      = send_db.pglobal_extents[0];
    size_t bs     = send_db.pblock_extents[0];
    bool rowmajor =true;

    size_t grid = (N + bs - 1) / bs;
    size_t total_blocks = grid;

    if (rank == rootrank)
    {
        size_t *ext = nullptr;
        size_t *str = nullptr;
        T *pdata = nullptr;

        size_t datalen = N;

        alloc_helper(
            memmap,
            ongpu,
            devicenum,
            1,
            datalen,
            ext,
            str,
            pdata);

        ext[0] = N;
        str[0] = 1;

        *recv_db = DataBlock<T>(
                       pdata,
                       datalen,
                       rowmajor,
                       1,
                       ext,
                       str,
                       ongpu,
                       devicenum);
    }


    MPI_Request* reqs = nullptr;
    size_t recv_idx = 0;

    if (rank == rootrank)
        reqs = new MPI_Request[total_blocks];

    size_t gridrank=send_db.pctx->gridrank;
    if (rank == rootrank)
    {
        size_t *grid_coords= new size_t[gridrank];
        int* temp_coords=new int [gridrank];
        for (size_t b = 0; b < total_blocks; b++)
        {
            size_t bcoords[1] = { b };

            send_db.ppolicy->create_coords(bcoords,grid_coords,send_db.pglobalrank);

            int owner = send_db.ppolicy->owner(grid_coords,*send_db.pctx,temp_coords);
            size_t start = b * bs;
            size_t diff=N - start;
            size_t len   = bs<diff?bs:diff;

            T* ptr = recv_db->dpdata + start;

            MPI_Irecv(
                ptr,
                len,
                mpi_get_type<T>(),
                owner,
                b,
                send_db.pctx->comm,
                &reqs[recv_idx++]);
        }
    }



    MPI_Request* sendreqs =
        (send_db.plocal_blocknumber > 0)
        ? new MPI_Request[send_db.plocal_blocknumber]
        : nullptr;

    size_t send_idx = 0;

    for (size_t i = 0; i < send_db.plocal_blocknumber; i++)
    {
        size_t b = send_db.pblock_linear_idx[i];

        size_t len = send_db.pblocks[i].dpextents[0];

        MPI_Isend(
            send_db.pblocks[i].dpdata,
            len,
            mpi_get_type<T>(),
            rootrank,
            b,
            send_db.pctx->comm,
            &sendreqs[send_idx++]);
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
inline  void DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(DataBlock<T> &m, int dest, int tag, MPI_Comm pcomm)
{

    MPI_Send(&m.dpdatalength, 1, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprank, 1, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprowmajor, 1, mpi_get_type<bool>(), dest, tag, pcomm);
    MPI_Send(m.dpextents, m.dprank, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(m.dpstrides, m.dprank, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}


template<typename T>
inline  void DataBlock_MPI_Functions<T>::MPI_Send_DataBlock_meta(DataBlock<T> &m, int dest, int tag, MPI_Comm pcomm)
{
    MPI_Send(&m.dpdatalength, 1, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprank, 1, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprowmajor, 1, mpi_get_type<bool>(), dest, tag, pcomm);
    MPI_Send(m.dpextents, m.dprank, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(m.dpstrides, m.dprank, mpi_get_type<size_t>(), dest, tag, pcomm);
}



template<typename T>
inline  void DataBlock_MPI_Functions<T>::MPI_Isend_DataBlock_pdata(DataBlock<T> &m,const int dest,const  int tag,const MPI_Comm pcomm,MPI_Request *request)
{
    MPI_Isend(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm,request);
}

template<typename T>
inline  void DataBlock_MPI_Functions<T>::MPI_Send_DataBlock_pdata(DataBlock<T> &m,const int dest,const int tag,const MPI_Comm pcomm)
{
    MPI_Send(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}

template<typename T>
inline  DataBlock<T> DataBlock_MPI_Functions<T>::MPI_Recv_alloc_DataBlock(bool with_memmap,bool ondevice, int devicenum, const int source,const  int tag, MPI_Comm pcomm)
{
    MPI_Status status;
    size_t pdatalength, prank;
    bool prowmajor;

    MPI_Recv(&pdatalength, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&prank, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&prowmajor, 1, mpi_get_type<bool>(), source, tag, pcomm, &status);

    size_t *pextents=nullptr,
            *pstrides=nullptr;
    T* pdata=nullptr;

    alloc_helper(with_memmap,ondevice,devicenum,prank,pdatalength,pextents,pstrides,pdata);

    MPI_Recv(pextents,prank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    MPI_Recv(pstrides,prank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    MPI_Recv(pdata,sizeof(T)*pdatalength, MPI_BYTE, source, tag, pcomm, &status);

    return DataBlock<T>(pdata,pdatalength,prowmajor,prank,pextents,pstrides,ondevice, devicenum);

}



template <typename T>
void DataBlock_MPI_Functions<T>::MPI_Free_DataBlock(DataBlock<T>&m, bool with_memmap)
{

    if(m.dpdata!=nullptr)
    {
#if defined(Unified_Shared_Memory)
        if(with_memmap)
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(m.dpdata,size);
        else;
        free(m.dpdata);
#else
        if(m.dpdata_is_devptr)
            omp_target_free(m.dpdata,m.devptr_devicenum);
        else
            free(m.dpdata);
#endif
    }
    if(m.dpextents!=nullptr) free(m.dpextents);
    if(m.dpstrides!=nullptr) free(m.dpstrides);
}


template<typename T>
void DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock(DataBlock<T>& m,const int source,const  int tag, MPI_Comm pcomm)
{
    MPI_Status status;

    MPI_Recv(&m.dpdatalength, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprank, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprowmajor, 1, mpi_get_type<bool>(), source, tag, pcomm, &status);

    MPI_Recv(m.dpextents,m.dprank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    MPI_Recv(m.dpstrides,m.dprank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    MPI_Recv(m.dpdata,sizeof(T)*m.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);

}


template<typename T>
void DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_meta(DataBlock<T>& m,const int source,const  int tag, MPI_Comm pcomm)
{
    MPI_Status status;

    MPI_Recv(&m.dpdatalength, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprank, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprowmajor, 1, mpi_get_type<bool>(), source, tag, pcomm, &status);

    MPI_Recv(m.dpextents,m.dprank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    MPI_Recv(m.dpstrides,m.dprank, mpi_get_type<size_t>(), source, tag, pcomm, &status);
}



template<typename T>
inline  void DataBlock_MPI_Functions<T>::MPI_Irecv_DataBlock_pdata(DataBlock<T> &mds, const int source, const int tag,const  MPI_Comm pcomm,  MPI_Request *request)
{
    MPI_Irecv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, request);
}

template<typename T>
inline  void DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(DataBlock<T>& mds,const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);
}


#endif



