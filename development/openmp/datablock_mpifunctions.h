#ifndef DATABLOCK_MPIFUNCTIONS
#define DATABLOCK_MPIFUNCTIONS

#include <mpi.h>


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


class MPI_GridPol
{
public:
    bool defaultgrid=true;
    size_t* pproc_grid=nullptr;
    size_t* pcyclic_block=nullptr;
    size_t pgridrank=0;

    void balanced_process_grid(int processsize, size_t gridrank)
    {
        if(pproc_grid)
        {
            delete[]pproc_grid;
            pproc_grid=nullptr;
        }
        pproc_grid=new size_t[gridrank];
        pgridrank=gridrank;

        int remaining = processsize;

        #pragma omp unroll partial
        for(size_t d = 0; d < gridrank; d++)
        {
            double root = std::pow((double)remaining, 1.0/(gridrank-d));
            int guess = (int)root;

            while(remaining % guess != 0)
                guess--;

            pproc_grid[d] = guess;
            remaining /= guess;
        }
    }

    void balanced_bloc_grid(int processsize, size_t gridrank)
    {
        if(pcyclic_block)
        {
            delete[] pcyclic_block;
            pcyclic_block=nullptr;
        }

        pcyclic_block=new size_t[gridrank];
        pgridrank=gridrank;

        #pragma omp parallel for simd if(parallel: gridrank>30)
        for(size_t d=0; d<gridrank; d++)
            pcyclic_block[d] = 1;
    }


    MPI_GridPol(MPI_Comm com, size_t blockrank,bool defaultgrid=true, size_t gridrank=0,size_t *proc_grid=nullptr,size_t *cyclic_block=nullptr)
    {
        if(defaultgrid)
        {
            int size;
            MPI_Comm_size(com, &size);
            pgridrank = blockrank<3?blockrank:3;
            balanced_process_grid(size, pgridrank);
            balanced_bloc_grid(size,pgridrank);
        }
        else
        {
            set_grid( gridrank, proc_grid, cyclic_block);
        }
    }

    MPI_GridPol(  size_t gridrank=0,size_t *proc_grid=nullptr,size_t *cyclic_block=nullptr)
    {
        set_grid( gridrank, proc_grid, cyclic_block);
    }

    void set_grid( size_t gridrank,size_t *proc_grid=nullptr,size_t *cyclic_block=nullptr)
    {
        if(pproc_grid)
        {
            delete[]pproc_grid;
            pproc_grid=nullptr;
        }

        if(pcyclic_block)
        {
            delete[] pcyclic_block;
            pcyclic_block=nullptr;
        }
        pproc_grid=new size_t[gridrank];
        pcyclic_block=new size_t[gridrank];
        memcpy(pproc_grid,proc_grid, sizeof(size_t)*gridrank);
        memcpy(pcyclic_block,cyclic_block,sizeof(size_t)*gridrank);
        pgridrank=gridrank;
    }
    // Copy constructor (deep copy)
    MPI_GridPol(const MPI_GridPol& other)
        : defaultgrid(other.defaultgrid), pgridrank(other.pgridrank)
    {
        if (other.pproc_grid)
        {
            pproc_grid = new size_t[pgridrank];
            std::memcpy(pproc_grid, other.pproc_grid, sizeof(size_t) * pgridrank);
        }
        if (other.pcyclic_block)
        {
            pcyclic_block = new size_t[pgridrank];
            std::memcpy(pcyclic_block, other.pcyclic_block, sizeof(size_t) * pgridrank);
        }
    }

    // Copy assignment operator (deep copy)
    MPI_GridPol& operator=(const MPI_GridPol& other)
    {
        if (this == &other) return *this; // self-assignment check

        // Delete old arrays
        delete[] pproc_grid;
        delete[] pcyclic_block;

        defaultgrid = other.defaultgrid;
        pgridrank = other.pgridrank;

        // Allocate new arrays and copy
        if (other.pproc_grid)
        {
            pproc_grid = new size_t[pgridrank];
            std::memcpy(pproc_grid, other.pproc_grid, sizeof(size_t) * pgridrank);
        }
        else
            pproc_grid = nullptr;

        if (other.pcyclic_block)
        {
            pcyclic_block = new size_t[pgridrank];
            std::memcpy(pcyclic_block, other.pcyclic_block, sizeof(size_t) * pgridrank);
        }
        else
            pcyclic_block = nullptr;

        return *this;
    }


    virtual ~MPI_GridPol()
    {
        if(pproc_grid)
        {
            delete[]pproc_grid;
            pproc_grid=nullptr;
        }

        if(pcyclic_block)
        {
            delete[] pcyclic_block;
            pcyclic_block=nullptr;
        }
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

    MPI_Comm mpi_comm()const
    {
        return pcomm;
    }

    size_t local_blocknumber()const
    {
        return plocal_blocknumber;
    }

    size_t *block_coords()const
    {
        return pblock_coords;
    }

    DataBlock<T> *blocks()const
    {
        return pblocks;
    }

    size_t *process_grid()const
    {
        return pproc_grid;
    }
    size_t *cyclic_block()const
    {
        return pcyclic_block;
    }
    size_t *block_extents()const
    {
        return pblock_extents;
    }
    size_t blockrank()const
    {
        return pblock_rank;
    }
    size_t gridrank()const
    {
        return pgridrank;
    }

    void printblockcoordinates( )const
    {
        int rank, size;
        MPI_Comm_rank(pcomm,&rank);
        MPI_Comm_size(pcomm,&size);
        for(int r=0; r<size; r++)
        {
            if(rank==r)
            {
                printf("\n=== MPI Rank %d ===\n",rank);
                printf("blocks: %zu\n",plocal_blocknumber);
                for(size_t i=0; i< plocal_blocknumber * pglobalrank; i++)
                {
                    printf("block coords %zu",pblock_coords[i]);
                    printf(" ");
                    fflush(stdout);
                }

            }
        }
    }
    void printtensors()const
    {
        int rank, size;
        MPI_Comm_rank(pcomm,&rank);
        MPI_Comm_size(pcomm,&size);

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
    size_t* pproc_grid=nullptr;
    size_t* pcyclic_block=nullptr;
    size_t *pblock_extents=nullptr;
    size_t pblock_rank=0;
    size_t pgridrank=0;
    T* pdata = nullptr;
    size_t pdatalength=0;
    size_t* pextentsbuffer;
    size_t* pstridesbuffer;
    bool pdpdata_is_devptr=false;
    bool pmemmap=false;
    int pdevptr_devicenum=-1;
    size_t pglobalrank=0;
    size_t* pglobal_extents=nullptr;
    size_t* pglobal_strides=nullptr;

    bool pglobal_rowmajor=true;
    MPI_Comm pcomm;
    size_t plocal_blocknumber=0;
    size_t* pblock_coords=nullptr;

    size_t* pblock_indices=nullptr;
    size_t* pblock_linear_idx=nullptr;

    DataBlock<T>* pblocks=nullptr;
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


    inline static void MPI_Scatter_matrix_to_rows_alloc   (  DistributedDataBlock<T>& recv_db, bool memmap, bool ondevice, int devicenum,
            MPI_Comm comm,   int rootrank,    const DataBlock<T>* send_db=nullptr);
    inline static void MPI_Gather_matrix_from_rows_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank,    DataBlock<T>* recv_db=nullptr,    bool memmap=false,    bool ondevice=false,    int devicenum=-1);

    inline static void MPI_Scatter_matrix_to_columns_alloc(  DistributedDataBlock<T>& recv_db, bool memmap, bool ondevice, int devicenum,
            MPI_Comm comm,   int rootrank,    const DataBlock<T>* send_db=nullptr);
    inline static void MPI_Gather_matrix_from_columns_alloc(   const DistributedDataBlock<T>& send_db,      int rootrank,DataBlock<T>* recv_db = nullptr,
            bool memmap=false, bool ongpu=false, int devicenum=-1);

    inline static void MPI_Scatter_matrix_to_submatrices_alloc(  size_t br,    size_t bc,    DistributedDataBlock<T>& recv_db,
            bool memmap, bool ondevice, int devicenum,    MPI_Comm comm,    int rootrank,    const DataBlock<T>* send_db=nullptr, MPI_GridPol *ppol=nullptr);

    inline static void MPI_Gather_matrix_from_submatrices_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank,DataBlock<T>* recv_db = nullptr,bool memmap=false, bool ongpu=false, int devicenum=-1 );

    inline static void MPI_Scatter_tensor_to_subtensors_alloc(    size_t blockrank,    const size_t* block_extents,
            DistributedDataBlock<T>& recv_db,    bool memmap,    bool ondevice,    int devicenum,    MPI_Comm comm,
            int rootrank,    const DataBlock<T>* send_db=nullptr, MPI_GridPol *ppol=nullptr);


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



    inline static std::optional<MPI_GridPol> default_grid_policy;


    static const MPI_GridPol& get_default_grid_policy(MPI_Comm com,size_t blockrank)
    {
        if (!default_grid_policy.has_value())
        {

            default_grid_policy.emplace(com,blockrank,true);
        }
        return *default_grid_policy;
    }

    inline static void alloc_helper(bool &memmap,bool& ondevice, int& devnum, size_t rank,size_t datalength,size_t* &pextents,size_t *&pstrides,T *&pdata);
    inline static void alloc_helper2(bool &memmap,bool &ondevice, int& devicenum, size_t datalength,T *&pdata);

    inline static void free_helper(bool memmap,bool ondevice, int devnum,size_t datalength,size_t* &pextents,size_t *&pstrides,T *&pdata);

    inline static void free_helper2(bool memmap,bool ondevice, int devicenum, size_t datalength,T *&pdata);

    inline static int compute_owner(const size_t* bcoords,const size_t* proc_grid, const size_t* cyclic_block,size_t gridrank);
};

template <typename T>
inline int DataBlock_MPI_Functions<T>::compute_owner(
    const size_t* bcoords,
    const size_t* proc_grid,
    const size_t* cyclic_block,
    size_t gridrank)
{
    int owner=0;
    #pragma omp unroll partial
    for(size_t d=0; d<gridrank; d++)
    {
        int group = bcoords[d] / cyclic_block[d];
        int proc_coord = group % proc_grid[d];
        owner = owner * proc_grid[d] + proc_coord;
    }

    return owner;

}





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
    if(m.pcyclic_block)
    {
        free(m.pcyclic_block);
        m.pcyclic_block=nullptr;
    }
    if(m.pblock_extents)
    {
        free(m.pblock_extents);
        m.pblock_extents=nullptr;
    }
    if(m.pproc_grid)
    {
        free(m.pproc_grid);
        m.pproc_grid=nullptr;
    }
    if(m.pblock_coords)
    {
        free(m.pblock_coords);
        m.pblock_coords=nullptr;
    }
    if(m.pglobal_extents)
    {
        free(m.pglobal_extents);
        m.pglobal_extents=nullptr;
    }
    if(m.pglobal_strides)
    {
        free(m.pglobal_strides);
        m.pglobal_strides=nullptr;
    }

    if(m.pblock_indices)
    {
        free(m.pblock_indices);
        m.pblock_indices=nullptr;
    }
    if(m.pblock_linear_idx)
    {
        free(m.pblock_linear_idx);
        m.pblock_linear_idx=nullptr;
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
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, com);
}




template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_DataBlock_meta (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    MPI_Bcast (&db.dpdatalength, 1,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprank,1,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (&db.dprowmajor,1,  mpi_get_type<bool>(), rootrank, com);
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_DataBlock_extents_strides (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<size_t>(), rootrank, com);
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,bool memmap,bool ondevice, int devicenum,MPI_Comm com, int rootrank)
{
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
    MPI_Comm comm,
    int rootrank,
    const DataBlock<T>* send_db)
{
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);


    recv_db.pglobal_extents=(size_t*)malloc(sizeof(size_t)*2);
    recv_db.pglobal_strides=(size_t*)malloc(sizeof(size_t)*2);

    recv_db.pproc_grid=(size_t*)malloc(sizeof(size_t));
    recv_db.pcyclic_block=(size_t*)malloc(sizeof(size_t));
    recv_db.pblock_extents=(size_t*)malloc(sizeof(size_t));

    recv_db.pglobalrank=2;
    recv_db.pcomm=comm;

    recv_db.pblock_rank=1;
    recv_db.pproc_grid[0]=size;
    recv_db.pcyclic_block[0]=1;

    if(rank==rootrank)
    {
        recv_db.pglobal_extents[0]=send_db->dpextents[0];
        recv_db.pglobal_extents[1]=send_db->dpextents[1];
        recv_db.pglobal_strides[0]=send_db->dpstrides[0];
        recv_db.pglobal_strides[1]=send_db->dpstrides[1];
        recv_db.pglobal_rowmajor=send_db->dprowmajor;
        recv_db.pblock_extents[0]=send_db->dpextents[1];
    }


    MPI_Bcast(recv_db.pglobal_extents,2,mpi_get_type<size_t>(),rootrank,comm);
    MPI_Bcast(recv_db.pglobal_strides,2,mpi_get_type<size_t>(),rootrank,comm);
    MPI_Bcast(&recv_db.pglobal_rowmajor,1,mpi_get_type<bool>(),rootrank,comm);
    MPI_Bcast(recv_db.pblock_extents,1,mpi_get_type<size_t>(),rootrank,comm);

    size_t M = recv_db.pglobal_extents[0];
    size_t N = recv_db.pglobal_extents[1];



    size_t local_rows=0;

    #pragma omp parallel for simd reduction(+:local_rows) if(parallel: N > 30)
    for(size_t r=0; r<M; r++)
        if(r%size==(size_t)rank)
            local_rows++;

    recv_db.plocal_blocknumber=local_rows;

    recv_db.pblocks =
        (local_rows>0)?(DataBlock<T>*)malloc(sizeof(DataBlock<T>)*local_rows):nullptr;

    recv_db.pblock_coords =
        (local_rows>0)?(size_t*)malloc(sizeof(size_t)*2*local_rows):nullptr;

    recv_db.pblock_indices =(local_rows>0)?(size_t*)malloc(sizeof(size_t)*recv_db.pblock_rank*local_rows):nullptr;

    recv_db.pblock_linear_idx =(local_rows>0)?(size_t*)malloc(sizeof(size_t)*local_rows):nullptr;

    recv_db.pglobal_to_local_index.reserve(local_rows);

    size_t idx=0;



    for(size_t r = 0; r < M; r++)
    {
        if(r % size != (size_t)rank)
            continue;

        recv_db.pblock_coords[2*idx]     = r;
        recv_db.pblock_coords[2*idx + 1] = 0;
        recv_db.pblock_indices[idx] = r;
        recv_db.pblock_linear_idx[idx] = r;
        recv_db.pglobal_to_local_index[r] = idx;

        idx++;
    }


    recv_db.pdatalength=N*local_rows;

    recv_db.pdata=nullptr;

    if(local_rows>0)
        alloc_helper2(memmap,ondevice,devicenum,recv_db.pdatalength,recv_db.pdata);

    recv_db.pdpdata_is_devptr=ondevice;
    recv_db.pdevptr_devicenum=devicenum;
    recv_db.pmemmap=memmap;

    MPI_Request* reqs=new MPI_Request[local_rows];

    size_t offset=0;
    size_t req_idx=0;

    for(size_t r=0; r<M; r++)
    {
        if(r%size!=(size_t)rank)
            continue;

        MPI_Irecv(
            recv_db.pdata + offset,
            N,
            mpi_get_type<T>(),
            rootrank,
            r,
            comm,
            &reqs[req_idx]);

        offset += N;
        req_idx++;
    }



    if(rank==rootrank)
    {
        MPI_Request* sendreqs=new MPI_Request[M];
        MPI_Datatype rowtype;
        if(!send_db->dprowmajor)
        {
            MPI_Datatype tmp;

            MPI_Type_vector(
                N,
                1,
                send_db->dpstrides[1],
                mpi_get_type<T>(),
                &tmp);

            MPI_Type_create_resized(tmp,0,sizeof(T),&rowtype);
            MPI_Type_commit(&rowtype);
            MPI_Type_free(&tmp);
        }

        for(size_t r=0; r<M; r++)
        {
            int owner = r % size;

            T* start =
                send_db->dpdata +
                r*send_db->dpstrides[0];

            if(send_db->dprowmajor)
            {
                MPI_Isend(
                    start,
                    N,
                    mpi_get_type<T>(),
                    owner,
                    r,
                    comm,
                    &sendreqs[r]);
            }
            else
            {
                MPI_Isend(
                    start,
                    1,
                    rowtype,
                    owner,
                    r,
                    comm,
                    &sendreqs[r]);
            }
        }


        MPI_Waitall(M,sendreqs,MPI_STATUSES_IGNORE);
        delete[] sendreqs;
    }

    MPI_Waitall(local_rows,reqs,MPI_STATUSES_IGNORE);
    delete[] reqs;



    offset=0;

    recv_db.pextentsbuffer=(size_t*)malloc(sizeof(size_t)*local_rows);
    recv_db.pstridesbuffer=(size_t*)malloc(sizeof(size_t)*local_rows);

    for(size_t i=0; i<local_rows; i++)
    {

        size_t* bext_i =  recv_db.pextentsbuffer + i;
        size_t* bstr_i = recv_db.pstridesbuffer + i;
        bext_i[0]=N;
        bstr_i[0]=1;

        recv_db.pblocks[i] =
            DataBlock<T>(
                recv_db.pdata+offset,
                N,
                true,
                1,
                bext_i,
                bstr_i,
                ondevice,
                devicenum);

        offset += N;
    }
}


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_matrix_to_columns_alloc(
    DistributedDataBlock<T>& recv_db,
    bool memmap,
    bool ondevice,
    int devicenum,
    MPI_Comm comm,
    int rootrank,
    const DataBlock<T>* send_db)
{
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    recv_db.pcomm = comm;
    recv_db.pglobalrank = 2;

    recv_db.pglobal_extents = (size_t*)malloc(sizeof(size_t)*2);
    recv_db.pglobal_strides = (size_t*)malloc(sizeof(size_t)*2);

    recv_db.pcyclic_block= (size_t*)malloc(sizeof(size_t)*1);
    recv_db.pblock_extents= (size_t*)malloc(sizeof(size_t)*1);
    recv_db.pproc_grid= (size_t*)malloc(sizeof(size_t)*1);


    recv_db.pproc_grid[0]=size;
    recv_db.pblock_rank= 1;
    recv_db.pcyclic_block[0]=1;

    if(rank==rootrank)
    {
        recv_db.pglobal_extents[0] = send_db->dpextents[0];
        recv_db.pglobal_extents[1] = send_db->dpextents[1];
        recv_db.pglobal_strides[0] = send_db->dpstrides[0];
        recv_db.pglobal_strides[1] = send_db->dpstrides[1];
        recv_db.pglobal_rowmajor  = send_db->dprowmajor;
        recv_db.pblock_extents[0]=send_db->dpextents[0];
    }

    MPI_Bcast(recv_db.pglobal_extents,2,mpi_get_type<size_t>(),rootrank,comm);
    MPI_Bcast(recv_db.pglobal_strides,2,mpi_get_type<size_t>(),rootrank,comm);
    MPI_Bcast(&recv_db.pglobal_rowmajor,1,mpi_get_type<bool>(),rootrank,comm);
    MPI_Bcast(recv_db.pblock_extents,1,mpi_get_type<size_t>(),rootrank,comm);

    size_t M = recv_db.pglobal_extents[0];
    size_t N = recv_db.pglobal_extents[1];



    size_t local_cols=0;
    #pragma omp parallel for simd reduction(+:local_cols) if(parallel: N > 30)
    for(size_t c=0; c<N; c++)
        if(c%size==(size_t)rank)
            local_cols++;

    recv_db.plocal_blocknumber = local_cols;

    recv_db.pblocks =
        (local_cols>0)?(DataBlock<T>*)malloc(sizeof(DataBlock<T>)*local_cols):nullptr;

    recv_db.pblock_coords =
        (local_cols>0)?(size_t*)malloc(sizeof(size_t)*2*local_cols):nullptr;

    recv_db.pblock_indices =(local_cols>0)?(size_t*)malloc(sizeof(size_t)*recv_db.pblock_rank*local_cols):nullptr;

    recv_db.pblock_linear_idx =(local_cols>0)?(size_t*)malloc(sizeof(size_t)*local_cols):nullptr;

    recv_db.pglobal_to_local_index.reserve(local_cols);

    size_t idx=0;

    for(size_t c=0; c<N; c++)
    {
        if(c%size!=(size_t)rank)
            continue;

        recv_db.pblock_coords[2*idx]   = 0;
        recv_db.pblock_coords[2*idx+1] = c;

        recv_db.pblock_indices[idx] = c;
        recv_db.pblock_linear_idx[idx] = c;
        recv_db.pglobal_to_local_index[c] = idx;

        idx++;
    }


    recv_db.pdatalength=M*local_cols;

    recv_db.pdata=nullptr;

    if(local_cols>0)
        alloc_helper2(memmap,ondevice,devicenum,recv_db.pdatalength,recv_db.pdata);

    recv_db.pdpdata_is_devptr=ondevice;
    recv_db.pdevptr_devicenum=devicenum;
    recv_db.pmemmap=memmap;


    MPI_Request* reqs = new MPI_Request[local_cols];

    size_t offset=0;
    size_t req_idx=0;

    for(size_t c=0; c<N; c++)
    {
        if(c%size!=(size_t)rank)
            continue;

        MPI_Irecv(
            recv_db.pdata + offset,
            M,
            mpi_get_type<T>(),
            rootrank,
            c,
            comm,
            &reqs[req_idx]);

        offset += M;
        req_idx++;
    }



    if(rank==rootrank)
    {
        MPI_Request* sendreqs = new MPI_Request[N];

        for(size_t c=0; c<N; c++)
        {
            int owner = c % size;

            MPI_Datatype tmp,coltype;

            MPI_Type_vector(
                M,
                1,
                send_db->dpstrides[0],
                mpi_get_type<T>(),
                &tmp);

            MPI_Type_create_resized(tmp,0,sizeof(T),&coltype);
            MPI_Type_commit(&coltype);
            MPI_Type_free(&tmp);

            T* start =
                send_db->dpdata +
                c*send_db->dpstrides[1];

            MPI_Isend(
                start,
                1,
                coltype,
                owner,
                c,
                comm,
                &sendreqs[c]);

            MPI_Type_free(&coltype);
        }

        MPI_Waitall(N,sendreqs,MPI_STATUSES_IGNORE);
        delete[] sendreqs;
    }

    MPI_Waitall(local_cols,reqs,MPI_STATUSES_IGNORE);
    delete[] reqs;



    offset=0;

    recv_db.pstridesbuffer=(size_t*)malloc(sizeof(size_t)*local_cols);
    recv_db.pextentsbuffer=(size_t*)malloc(sizeof(size_t)*local_cols);


    for(size_t i=0; i<local_cols; i++)
    {

        size_t* bext_i = recv_db.pextentsbuffer + i;
        size_t* bstr_i = recv_db.pstridesbuffer + i;
        bext_i[0]=M;
        bstr_i[0]=1;

        recv_db.pblocks[i] =
            DataBlock<T>(
                recv_db.pdata+offset,
                M,
                true,
                1,
                bext_i,
                bstr_i,
                ondevice,
                devicenum);

        offset += M;
    }
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_matrix_to_submatrices_alloc(
    size_t br,
    size_t bc,
    DistributedDataBlock<T>& recv_db,
    bool memmap,
    bool ondevice,
    int devicenum,
    MPI_Comm active_comm,
    int rootrank,
    const DataBlock<T>* send_db, MPI_GridPol *pol)
{

    recv_db.pcomm = active_comm;
    if (active_comm == MPI_COMM_NULL)
    {
        return;
    }

    int rank,size;
    MPI_Comm_rank(active_comm,&rank);
    MPI_Comm_size(active_comm,&size);

    recv_db.pglobalrank = 2;

    recv_db.pglobal_extents=(size_t*)malloc(sizeof(size_t)*2);
    recv_db.pglobal_strides=(size_t*)malloc(sizeof(size_t)*2);

    recv_db.pcyclic_block=(size_t*)malloc(sizeof(size_t)*2);
    recv_db.pblock_extents=(size_t*)malloc(sizeof(size_t)*2);

    recv_db.pproc_grid=(size_t*)malloc(sizeof(size_t)*2);

    recv_db.pblock_rank=2;

    recv_db.pcyclic_block[0]=1;
    recv_db.pcyclic_block[1]=1;

    int Pr=1;
    int Pc=1;
    if(rank==rootrank)
    {
        recv_db.pglobal_extents[0]=send_db->dpextents[0];
        recv_db.pglobal_extents[1]=send_db->dpextents[1];
        recv_db.pglobal_strides[0]=send_db->dpstrides[0];
        recv_db.pglobal_strides[1]=send_db->dpstrides[1];
        recv_db.pglobal_rowmajor=send_db->dprowmajor;
        recv_db.pblock_extents[0]=br;
        recv_db.pblock_extents[1]=bc;

        const MPI_GridPol& policy = (pol != nullptr) ? *pol : get_default_grid_policy(active_comm,2);
        Pr=policy.pproc_grid[0];
        Pc=policy.pproc_grid[1];
        recv_db.pproc_grid[0]=Pr;
        recv_db.pproc_grid[1]=Pc;
    }


    MPI_Bcast(recv_db.pglobal_extents,2,mpi_get_type<size_t>(),rootrank,active_comm);
    MPI_Bcast(recv_db.pglobal_strides,2,mpi_get_type<size_t>(),rootrank,active_comm);
    MPI_Bcast(&recv_db.pglobal_rowmajor,1,mpi_get_type<bool>(),rootrank,active_comm);

    MPI_Bcast(recv_db.pblock_extents,2,mpi_get_type<size_t>(),rootrank,active_comm);
    MPI_Bcast(recv_db.pproc_grid,2,mpi_get_type<size_t>(),rootrank,active_comm);

    Pr=recv_db.pproc_grid[0];
    Pc=recv_db.pproc_grid[1];

    size_t M = recv_db.pglobal_extents[0];
    size_t N = recv_db.pglobal_extents[1];

    size_t grid_r = (M + br - 1) / br;
    size_t grid_c = (N + bc - 1) / bc;

    size_t total_blocks = grid_r * grid_c;



    size_t local_blocks=0;

    #pragma omp parallel for simd collapse(2)
    for(size_t bi=0; bi<grid_r; bi++)
    {
        for(size_t bj=0; bj<grid_c; bj++)
        {
            size_t bcoords[2] = {bi, bj};
            int owner = compute_owner(
                            bcoords,
                            recv_db.pproc_grid,
                            recv_db.pcyclic_block,
                            2);


            if(owner == rank)
            {
                #pragma omp atomic
                local_blocks++;
            }
        }
    }

    recv_db.plocal_blocknumber=local_blocks;

    recv_db.pblocks =
        (local_blocks>0)?(DataBlock<T>*)malloc(sizeof(DataBlock<T>)*local_blocks):nullptr;

    recv_db.pblock_coords =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*2*local_blocks):nullptr;

    recv_db.pblock_indices =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*2*local_blocks):nullptr;

    recv_db.pblock_linear_idx =
        (local_blocks>0)?(size_t*)malloc(sizeof(size_t)*local_blocks):nullptr;





    #pragma omp parallel for collapse(2)
    for(size_t bi=0; bi<grid_r; bi++)
    {
        for(size_t bj=0; bj<grid_c; bj++)
        {
            size_t bcoords[2] = {bi, bj};
            int owner = compute_owner(
                            bcoords,
                            recv_db.pproc_grid,
                            recv_db.pcyclic_block,
                            2);

            if(owner != rank)
                continue;


            size_t proc_row = (bi / recv_db.pcyclic_block[0]) % Pr;
            size_t proc_col = (bj / recv_db.pcyclic_block[1]) % Pc;

            size_t local_cols = (grid_c + Pc - proc_col - 1) / Pc;
            size_t local_i = bi / Pr;
            size_t local_j = bj / Pc;

            size_t my_idx = local_i * local_cols + local_j;
            recv_db.pblock_indices[2*my_idx]   = bi;
            recv_db.pblock_indices[2*my_idx+1] = bj;

            size_t r0 = bi * br;
            size_t c0 = bj * bc;

            recv_db.pblock_coords[2*my_idx]   = r0;
            recv_db.pblock_coords[2*my_idx+1] = c0;

            recv_db.pblock_linear_idx[my_idx] = bi * grid_c + bj;
        }
    }



    size_t total_recv_elems=0;

    #pragma omp parallel for simd reduction(+:total_recv_elems)
    for(size_t i=0; i<local_blocks; i++)
    {
        size_t r0 = recv_db.pblock_coords[2*i];
        size_t c0 = recv_db.pblock_coords[2*i+1];
        size_t diff=M-r0;
        size_t rows = br<=diff? br:diff;
        diff=N-c0;
        size_t cols =bc<=diff?bc:diff;
        total_recv_elems += rows*cols;
    }

    recv_db.pdatalength=total_recv_elems;
    recv_db.pdata=nullptr;

    if(total_recv_elems>0)
        alloc_helper2(memmap,ondevice,devicenum,total_recv_elems,recv_db.pdata);

    recv_db.pdpdata_is_devptr=ondevice;
    recv_db.pdevptr_devicenum=devicenum;
    recv_db.pmemmap=memmap;

    MPI_Request* reqs=new MPI_Request[local_blocks];

    size_t offset=0;


    size_t req_idx = 0;

    for(size_t bi=0; bi<grid_r; bi++)
    {
        for(size_t bj=0; bj<grid_c; bj++)
        {
            size_t bcoords[2] = {bi, bj};
            int owner = compute_owner(
                            bcoords,
                            recv_db.pproc_grid,
                            recv_db.pcyclic_block,
                            2);

            if(owner != rank)
                continue;

            size_t b = bi * grid_c + bj;
            size_t r0 = bi * br;
            size_t c0 = bj * bc;
            size_t diff= M - r0;
            size_t rows = br<=diff? br:diff;
            diff= N - c0;
            size_t cols =bc<=diff?bc:diff;

            MPI_Irecv(
                recv_db.pdata + offset,
                rows * cols,
                mpi_get_type<T>(),
                rootrank,
                b,
                active_comm,
                &reqs[req_idx]);

            offset += rows * cols;
            req_idx++;
        }
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

        for(size_t bi=0; bi<grid_r; bi++)
            for(size_t bj=0; bj<grid_c; bj++)
            {
                size_t b = bi * grid_c + bj;

                size_t bcoords[2] = {bi, bj};
                int owner = compute_owner(
                                bcoords,
                                recv_db.pproc_grid,
                                recv_db.pcyclic_block,
                                2);
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
                    active_comm,
                    &sendreqs[b]);

                if(edgecase)
                    MPI_Type_free(&blocktype1);
            }


        MPI_Waitall(total_blocks,sendreqs,MPI_STATUSES_IGNORE);
        MPI_Type_free(&blocktype0);

        delete[] sendreqs;
    }

    MPI_Waitall(local_blocks,reqs,MPI_STATUSES_IGNORE);

    delete[] reqs;

    offset=0;

    recv_db.pextentsbuffer =(size_t*)malloc(sizeof(size_t)*2*local_blocks);
    recv_db.pstridesbuffer=(size_t*)malloc(sizeof(size_t)*2*local_blocks);

    recv_db.pglobal_to_local_index.reserve(local_blocks);
    #pragma omp parallel for ordered schedule(static)
    for(size_t i=0; i<local_blocks; i++)
    {
        size_t* bext_i = recv_db.pextentsbuffer + i*2;
        size_t* bstr_i = recv_db.pstridesbuffer + i*2;

        size_t r0 = recv_db.pblock_coords[2*i];
        size_t c0 = recv_db.pblock_coords[2*i+1];
        size_t diff=M-r0;
        size_t rows =br<diff? br:diff;
        diff=N-c0;
        size_t cols = bc<=diff? bc:diff;

        bext_i[0]=rows;
        bext_i[1]=cols;
        bstr_i[0]=recv_db.pglobal_rowmajor? cols:1;
        bstr_i[1]=recv_db.pglobal_rowmajor? 1:rows;

        #pragma omp ordered
        {
            recv_db.pblocks[i] =
                DataBlock<T>(
                    recv_db.pdata+offset,
                    rows*cols,
                    recv_db.pglobal_rowmajor,
                    2,
                    bext_i,
                    bstr_i,
                    ondevice,
                    devicenum);
            size_t lin = recv_db.pblock_linear_idx[i];
            recv_db.pglobal_to_local_index[lin]=i;

            offset += rows*cols;
        }

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
    MPI_Comm active_comm,
    int rootrank,
    const DataBlock<T>* send_db,
    MPI_GridPol* ppol)
{

    recv_db.pcomm = active_comm;

    if (active_comm == MPI_COMM_NULL)
        return;

    int rank,size;
    MPI_Comm_rank(active_comm,&rank);
    MPI_Comm_size(active_comm,&size);




    recv_db.pglobalrank = blockrank;
    size_t pgridrank;


    if(rank == rootrank)
    {
        const MPI_GridPol& policy = (ppol != nullptr) ? *ppol : get_default_grid_policy(active_comm,blockrank);
        pgridrank = std::min(policy.pgridrank, send_db->dprank);
        recv_db.pgridrank = pgridrank;

        recv_db.pproc_grid = (size_t*)malloc(sizeof(size_t) * pgridrank);
        recv_db.pcyclic_block = (size_t*)malloc(sizeof(size_t) * pgridrank);

        #pragma omp parallel for simd if(parallel: pgridrank > 30)
        for(size_t d = 0; d < pgridrank; d++)
        {
            recv_db.pproc_grid[d] = policy.pproc_grid[d];
            recv_db.pcyclic_block[d] = policy.pcyclic_block[d];
        }


        recv_db.pglobalrank = send_db->dprank;
        recv_db.pglobal_extents = (size_t*)malloc(sizeof(size_t) * send_db->dprank);
        recv_db.pglobal_strides = (size_t*)malloc(sizeof(size_t) * send_db->dprank);
        #pragma omp parallel for simd if(parallel: send_db->dprank > 30)
        for(size_t d = 0; d < send_db->dprank; d++)
        {
            recv_db.pglobal_extents[d] = send_db->dpextents[d];
            recv_db.pglobal_strides[d] = send_db->dpstrides[d];
        }


        if(send_db->dprank < blockrank)
            blockrank = send_db->dprank;

        recv_db.pblock_extents = (size_t*)malloc(sizeof(size_t) * blockrank);
        recv_db.pblock_rank = blockrank;

        #pragma omp parallel for simd if(parallel: blockrank > 30)
        for(size_t d = 0; d < blockrank; d++)
            recv_db.pblock_extents[d] = block_extents[d];

        recv_db.pglobal_rowmajor = send_db->dprowmajor;
    }


    MPI_Bcast(&recv_db.pgridrank, 1, mpi_get_type<size_t>(), rootrank, active_comm);
    MPI_Bcast(&recv_db.pglobalrank, 1, mpi_get_type<size_t>(), rootrank, active_comm);
    MPI_Bcast(&recv_db.pblock_rank, 1, mpi_get_type<size_t>(), rootrank, active_comm);
    MPI_Bcast(&recv_db.pglobal_rowmajor, 1, mpi_get_type<bool>(), rootrank, active_comm);

    pgridrank = recv_db.pgridrank;

    if(rank != rootrank)
    {
        recv_db.pproc_grid = (size_t*)malloc(sizeof(size_t) * pgridrank);
        recv_db.pcyclic_block = (size_t*)malloc(sizeof(size_t) * pgridrank);
        recv_db.pglobal_extents = (size_t*)malloc(sizeof(size_t) * recv_db.pglobalrank);
        recv_db.pglobal_strides = (size_t*)malloc(sizeof(size_t) * recv_db.pglobalrank);
        recv_db.pblock_extents = (size_t*)malloc(sizeof(size_t) * recv_db.pblock_rank);
    }

    MPI_Bcast(recv_db.pproc_grid, recv_db.pgridrank, mpi_get_type<size_t>(), rootrank, active_comm);
    MPI_Bcast(recv_db.pcyclic_block, recv_db.pgridrank, mpi_get_type<size_t>(), rootrank, active_comm);
    MPI_Bcast(recv_db.pglobal_extents, recv_db.pglobalrank, mpi_get_type<size_t>(), rootrank, active_comm);
    MPI_Bcast(recv_db.pglobal_strides, recv_db.pglobalrank, mpi_get_type<size_t>(), rootrank, active_comm);
    MPI_Bcast(recv_db.pblock_extents, recv_db.pblock_rank, mpi_get_type<size_t>(), rootrank, active_comm);


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

    #pragma omp parallel
    {
        size_t* bcoords = new size_t[recv_db.pglobalrank];
        #pragma omp for nowait
        for(size_t b = 0; b < total_blocks; b++)
        {
            size_t tmp = b;


            for(int d = recv_db.pglobalrank-1; d >= 0; d--)
            {
                bcoords[d] = tmp % grid[d];
                tmp /= grid[d];
            }

            int owner = compute_owner(bcoords, recv_db.pproc_grid, recv_db.pcyclic_block, recv_db.pgridrank);
            if(owner == rank)
            {
                #pragma omp atomic
                local_blocks++;
            }
        }
        delete[] bcoords;
    }

    recv_db.plocal_blocknumber = local_blocks;


    recv_db.pblocks = (local_blocks > 0) ? (DataBlock<T>*)malloc(sizeof(DataBlock<T>) * local_blocks) : nullptr;
    recv_db.pblock_coords = (local_blocks > 0) ? (size_t*)malloc(sizeof(size_t) * blockrank * local_blocks) : nullptr;

    recv_db.pblock_indices =(local_blocks > 0) ? (size_t*)malloc(sizeof(size_t) * blockrank * local_blocks) : nullptr;
    recv_db.pblock_linear_idx =(local_blocks>0)?(size_t*)malloc(sizeof(size_t)*local_blocks):nullptr;

    recv_db.pglobal_to_local_index.reserve(local_blocks);

    size_t idx = 0;
    #pragma omp parallel
    {
        size_t* bcoords = new size_t[recv_db.pglobalrank];
        #pragma omp for ordered
        for(size_t b = 0; b < total_blocks; b++)
        {

            size_t tmp = b;

            #pragma omp unroll partial
            for(int d = recv_db.pglobalrank-1; d >= 0; d--)
            {
                bcoords[d] = tmp % grid[d];
                tmp /= grid[d];
            }

            int owner = compute_owner(bcoords, recv_db.pproc_grid, recv_db.pcyclic_block, recv_db.pgridrank);
            if(owner != rank)
            {
                continue;
            }

            #pragma omp ordered
            {
                for(size_t d = 0; d < blockrank; d++)
                {
                    recv_db.pblock_indices[idx* blockrank+ d] = bcoords[d];
                    recv_db.pblock_coords[idx*blockrank + d] = bcoords[d] * recv_db.pblock_extents[d];
                }
                recv_db.pblock_linear_idx[idx] = b;
                recv_db.pglobal_to_local_index[b] = idx;

                idx++;
            }
        }

        delete[] bcoords;
    }

    size_t* block_sizes = new size_t[local_blocks];

    size_t total_recv_elems = 0;
    #pragma omp parallel for reduction(+:total_recv_elems)
    for(size_t i = 0; i < local_blocks; i++)
    {
        size_t tmp = 1;

        #pragma omp simd reduction (*:tmp )

        for(size_t d = 0; d < blockrank; d++)
        {
            size_t start = recv_db.pblock_coords[i*blockrank + d];
            size_t diff = recv_db.pglobal_extents[d] - start;
            size_t len = (recv_db.pblock_extents[d] <= diff) ? recv_db.pblock_extents[d] : diff;
            tmp *= len;
        }
        block_sizes[i]=tmp;

        size_t elems=block_sizes[i] ;

        #pragma omp simd reduction(*:elems)
        for(size_t d = blockrank; d < recv_db.pglobalrank; d++)
            elems *= recv_db.pglobal_extents[d];

        total_recv_elems += elems;
    }

    recv_db.pmemmap = memmap;
    recv_db.pdatalength = total_recv_elems;
    recv_db.pdpdata_is_devptr = ondevice;
    recv_db.pdevptr_devicenum = devicenum;
    recv_db.pdata = nullptr;

    if(total_recv_elems > 0)
        alloc_helper2(memmap, ondevice, devicenum, total_recv_elems, recv_db.pdata);




    MPI_Request* reqs = new MPI_Request[local_blocks];
    size_t offset = 0;
    size_t req_idx = 0;
    for(size_t i = 0; i < local_blocks; i++)
    {
        size_t elems = block_sizes[i];

        #pragma omp parallel for simd reduction(*:elems) if(parallel:blockrank>30)
        for(size_t d = blockrank; d < recv_db.pglobalrank; d++)
            elems *= recv_db.pglobal_extents[d];

        size_t b = recv_db.pblock_linear_idx[i];

        MPI_Irecv(recv_db.pdata + offset, elems,
                  mpi_get_type<T>(), rootrank, b, active_comm, &reqs[req_idx]);

        offset += elems;
        req_idx++;
    }


    if(rank == rootrank)
    {
        MPI_Request* sendreqs = new MPI_Request[total_blocks];

        int* sizes  = new int[blockrank];
        int* subs   = new int[blockrank];
        int* starts = new int[blockrank];

        #pragma omp parallel for simd if(parallel:blockrank>30)
        for(size_t d = 0; d < blockrank; d++)
            sizes[d] = (int)send_db->dpextents[d];

        size_t* bcoords = new size_t[recv_db.pglobalrank];

        for(size_t b = 0; b < total_blocks; b++)
        {

            size_t tmp = b;

            #pragma omp unroll partial
            for(int d = blockrank-1; d >= 0; d--)
            {
                bcoords[d] = tmp % grid[d];
                tmp /= grid[d];
            }

            #pragma omp parallel for simd if(parallel:blockrank>30)
            for(int d = 0; d < (int)blockrank; d++)
            {
                starts[d] = (int)(bcoords[d] * block_extents[d]);
                size_t diff = recv_db.pglobal_extents[d] - starts[d];
                subs[d] = (int)((block_extents[d] <= diff) ? block_extents[d] : diff);
            }

            int owner = compute_owner(bcoords,
                                      recv_db.pproc_grid,
                                      recv_db.pcyclic_block,
                                      recv_db.pgridrank);

            MPI_Datatype tmp_type, blocktype;

            MPI_Type_create_subarray(blockrank, sizes, subs, starts,
                                     send_db->dprowmajor ? MPI_ORDER_C : MPI_ORDER_FORTRAN,
                                     mpi_get_type<T>(), &tmp_type);

            MPI_Type_create_resized(tmp_type, 0, sizeof(T), &blocktype);
            MPI_Type_commit(&blocktype);
            MPI_Type_free(&tmp_type);

            MPI_Isend(send_db->dpdata, 1, blocktype,
                      owner, b, active_comm, &sendreqs[b]);

            MPI_Type_free(&blocktype);
        }

        MPI_Waitall(total_blocks, sendreqs, MPI_STATUSES_IGNORE);


        delete[] bcoords;
        delete[] sendreqs;
        delete[] sizes;
        delete[] subs;
        delete[] starts;
    }

    MPI_Waitall(local_blocks, reqs, MPI_STATUSES_IGNORE);

    delete[] reqs;
    delete[] grid;

    offset = 0;


    recv_db.pextentsbuffer= (size_t*)malloc(sizeof(size_t) * blockrank*local_blocks);
    recv_db.pstridesbuffer = (size_t*)malloc(sizeof(size_t) * blockrank*local_blocks);




    size_t* offsets = new size_t[local_blocks];


    offsets[0] = 0;

    #pragma omp unroll partial
    for(size_t i = 1; i < local_blocks; i++)
        offsets[i] = offsets[i-1] + block_sizes[i-1];


    #pragma omp parallel for
    for(size_t i = 0; i < local_blocks; i++)
    {

        size_t* bext_i = recv_db.pextentsbuffer + i*blockrank;
        size_t* bstr_i = recv_db.pstridesbuffer + i*blockrank;

        #pragma omp simd
        for(size_t d = 0; d < blockrank; d++)
        {
            size_t start = recv_db.pblock_coords[i*blockrank + d];
            size_t diff  = recv_db.pglobal_extents[d] - start;
            size_t len   = (block_extents[d] <= diff) ? block_extents[d] : diff;
            bext_i[d] = len;
        }

        if(recv_db.pglobal_rowmajor)
        {
            bstr_i[blockrank-1] = 1;
            #pragma omp unroll partial
            for(int d = blockrank-2; d >= 0; d--)
                bstr_i[d] = bstr_i[d+1] * bext_i[d+1];
        }
        else
        {
            bstr_i[0] = 1;
            #pragma omp unroll partial
            for(size_t d = 1; d < blockrank; d++)
                bstr_i[d] = bstr_i[d-1] * bext_i[d-1];
        }

        recv_db.pblocks[i] =
            DataBlock<T>(recv_db.pdata + offsets[i],
                         block_sizes[i],
                         recv_db.pglobal_rowmajor,
                         blockrank,
                         bext_i,
                         bstr_i,
                         ondevice,
                         devicenum);
    }
    delete[]offsets;
    delete[]block_sizes;
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

    if (send_db.pcomm == MPI_COMM_NULL)
        return;

    int rank,size;

    MPI_Comm_rank(send_db.pcomm,&rank);
    MPI_Comm_size(send_db.pcomm,&size);

    size_t M = send_db.pglobal_extents[0];
    size_t N = send_db.pglobal_extents[1];

    size_t local_rows = send_db.plocal_blocknumber;

    T* send_buffer=nullptr;
    if(local_rows>0)
        send_buffer = send_db.pblocks[0].dpdata;



    if(rank==rootrank && recv_db!=nullptr)
    {
        size_t rank2=2;
        size_t datalen=M*N;

        size_t* ext=nullptr;
        size_t* str=nullptr;
        T* data=nullptr;

        alloc_helper(memmap,ondevice,devicenum,
                     rank2,datalen,
                     ext,str,data);

        ext[0]=M;
        ext[1]=N;

        str[1]=send_db.pglobal_rowmajor?1:M;
        str[0]=send_db.pglobal_rowmajor?N:1;

        *recv_db = DataBlock<T>(
                       data,datalen,send_db.pglobal_rowmajor,rank2,ext,str,ondevice,devicenum);
    }



    MPI_Request sendreq;

    if(local_rows>0)
    {
        MPI_Isend(
            send_buffer,
            local_rows*N,
            mpi_get_type<T>(),
            rootrank,
            rank,
            send_db.pcomm,
            &sendreq);
    }



    if(rank==rootrank)
    {
        MPI_Request* reqs=new MPI_Request[size];

        for(int p=0; p<size; p++)
        {

            size_t rows=0;

            #pragma omp parallel for simd reduction(+:rows)
            for(size_t r=p; r<M; r+=size)
                rows++;

            if(rows==0)
            {
                reqs[p]=MPI_REQUEST_NULL;
                continue;
            }

            MPI_Datatype type=MPI_DATATYPE_NULL;

            MPI_Datatype rowtype, tmp;


            MPI_Type_vector(
                N,
                1,
                recv_db->dpstrides[1],
                mpi_get_type<T>(),
                &rowtype);

            MPI_Type_commit(&rowtype);

            MPI_Type_create_hvector(
                rows,
                1,
                sizeof(T) * size * recv_db->dpstrides[0],
                rowtype,
                &tmp);


            MPI_Type_create_resized(
                tmp,
                0,
                sizeof(T) * recv_db->dpstrides[0],
                &type);

            MPI_Type_commit(&type);

            MPI_Type_free(&rowtype);
            MPI_Type_free(&tmp);

            T* start = recv_db->dpdata + p * recv_db->dpstrides[0];

            MPI_Irecv(start, 1, type, p, p, send_db.pcomm, &reqs[p]);
            MPI_Type_free(&type);
        }

        MPI_Waitall(size,reqs,MPI_STATUSES_IGNORE);
        delete[] reqs;

    }

    if(rank!=rootrank && local_rows>0)
        MPI_Wait(&sendreq,MPI_STATUS_IGNORE);
}




template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_matrix_from_columns_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,
    DataBlock<T>* recv_db,
    bool memmap,
    bool ongpu,
    int devicenum)
{
    if (send_db.pcomm == MPI_COMM_NULL)
        return;

    int rank, size;
    MPI_Comm_rank(send_db.pcomm, &rank);
    MPI_Comm_size(send_db.pcomm, &size);

    size_t M = send_db.pglobal_extents[0];
    size_t N = send_db.pglobal_extents[1];

    size_t local_cols = send_db.local_blocknumber();


    T* send_buffer=nullptr;
    if(local_cols>0)
        send_buffer = send_db.pblocks[0].dpdata;

    if(rank == rootrank && recv_db != nullptr)
    {
        size_t rank2 = 2;
        size_t datalen = M * N;

        size_t* ext = nullptr;
        size_t* str = nullptr;
        T* data = nullptr;

        alloc_helper(memmap, ongpu, devicenum,
                     rank2, datalen,
                     ext, str, data);
        ext[0] = M;
        ext[1] = N;

        str[1] =send_db.pglobal_rowmajor? 1:M;
        str[0] =send_db.pglobal_rowmajor? N:1;

        *recv_db = DataBlock<T>(
                       data, datalen, send_db.pglobal_rowmajor, rank2, ext, str, ongpu, devicenum);
    }


    MPI_Request sendreq;

    if(local_cols > 0)
    {
        MPI_Isend(
            send_buffer,
            M * local_cols,
            mpi_get_type<T>(),
            rootrank,
            rank,
            send_db.pcomm,
            &sendreq);
    }

    if(rank == rootrank)
    {
        MPI_Request *reqs= new MPI_Request[size];

        for(int p = 0; p < size; p++)
        {


            size_t cols = 0;

            #pragma omp parallel for simd reduction(+:cols)
            for(size_t c = p; c < N; c += size)
                cols++;

            if(cols == 0)
            {
                reqs[p] = MPI_REQUEST_NULL;
                continue;
            }
            MPI_Datatype type=MPI_DATATYPE_NULL;

            MPI_Datatype coltype;
            MPI_Datatype tmp;


            MPI_Type_vector(
                M,
                1,
                recv_db->dpstrides[0],
                mpi_get_type<T>(),
                &coltype);

            MPI_Type_commit(&coltype);


            MPI_Type_create_hvector(
                cols,
                1,
                sizeof(T) * size * recv_db->dpstrides[1],
                coltype,
                &tmp);

            MPI_Type_create_resized(tmp,0,sizeof(T)*M,&type);
            MPI_Type_commit(&type);

            MPI_Type_free(&coltype);
            MPI_Type_free(&tmp);

            T* start = recv_db->dpdata + p * recv_db->dpstrides[1];

            MPI_Irecv(start, 1, type,  p, p, send_db.pcomm,  &reqs[p]);

            MPI_Type_free(&type);
        }

        MPI_Waitall(size, reqs, MPI_STATUSES_IGNORE);
        delete[] reqs;

    }

    if(rank != rootrank && local_cols > 0)
        MPI_Wait(&sendreq, MPI_STATUS_IGNORE);

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
    if (send_db.pcomm == MPI_COMM_NULL)
        return;
    int rank,size;
    MPI_Comm_rank(send_db.pcomm,&rank);
    MPI_Comm_size(send_db.pcomm,&size);

    size_t Pr= send_db.pproc_grid[0];
    size_t Pc=send_db.pproc_grid[1];

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

        for(size_t bi=0; bi<grid_r; bi++)
        {
            for(size_t bj=0; bj<grid_c; bj++)
            {
                MPI_Datatype type1;
                size_t b = bi*grid_c + bj;


                size_t bcoords[2] = {bi, bj};
                int owner = compute_owner(
                                bcoords,
                                send_db.pproc_grid,
                                send_db.pcyclic_block,
                                2);

                size_t r0 = bi*br;
                size_t c0 = bj*bc;
                size_t diff1=M-r0;
                size_t diff2=N-c0;

                bool edgecase=false;
                if(diff1<br|| diff2<bc)
                {
                    size_t rows =br<=diff1? br:diff1;
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
                    send_db.pcomm,
                    &reqs[recv_idx]);

                recv_idx++;

                if(edgecase)
                    MPI_Type_free(&type1);
            }
        }

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
            send_db.pcomm,
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
inline void DataBlock_MPI_Functions<T>::MPI_Gather_tensor_from_subtensors_alloc(
    const DistributedDataBlock<T>& send_db,
    int rootrank,
    DataBlock<T>* recv_db,
    bool memmap,
    bool ongpu,
    int devicenum)
{
    if (send_db.pcomm == MPI_COMM_NULL)
        return;

    int rank,size;
    MPI_Comm_rank(send_db.pcomm,&rank);
    MPI_Comm_size(send_db.pcomm,&size);

    size_t rank_t    = send_db.pglobalrank;
    size_t blockrank = send_db.pblock_rank;
    size_t gridrank  = send_db.pgridrank;

    size_t* proc_grid    = send_db.pproc_grid;
    size_t* cyclic_block = send_db.pcyclic_block;

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

        for(size_t b=0; b<total_blocks; b++)
        {
            size_t tmp=b;


            #pragma omp unroll
            for(int d=rank_t-1; d>=0; d--)
            {
                bcoords[d] = tmp % grid[d];
                tmp /= grid[d];
            }

            int owner = compute_owner(
                            bcoords,
                            proc_grid,
                            cyclic_block,
                            gridrank);

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
                send_db.pcomm,
                &reqs[recv_idx++]);

            MPI_Type_free(&blocktype);

        }
        delete[]bcoords;
        delete[] sizes;
        delete[] subs;
        delete[] starts;
    }

    /* ---------- workers send blocks ---------- */

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
            send_db.pcomm,
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



