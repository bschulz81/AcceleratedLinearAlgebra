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

MPI_Datatype create_mpi_DataBlockConfig_type();


class MPI_CartesianContext
{
public:
    MPI_Comm comm;

    ptrdiff_t gridrank;
    int size;

    int* dims;
    int* periods;

public:

    MPI_CartesianContext(MPI_Comm comm_);

    ~MPI_CartesianContext();

    int rank_from_coords(int* coords) const;
};

class BlockMappingPolicy
{
public:
    ptrdiff_t gridrank;

    int* index_map;
    ptrdiff_t* cyclic_block;

public:

    BlockMappingPolicy(
        ptrdiff_t gridrank_,
        const int* index_map_ = nullptr,
        const ptrdiff_t* cyclic_block_ = nullptr);

    ~BlockMappingPolicy()
    {
        delete[] index_map;
        delete[] cyclic_block;
    }

    void create_coords(
        const ptrdiff_t* in,
        ptrdiff_t* out,
        ptrdiff_t in_rank) const;

    int owner(
        const ptrdiff_t* grid_coords,
        const MPI_CartesianContext& ctx,
        int* temp_coords) const;
};



class MPI_Policy
{
public:
    MPI_Comm comm = MPI_COMM_WORLD;
    bool mpi_enabled = true;

    int mpi_rank = 0;
    int mpi_size = 1;


    MPI_Policy(bool mpi=true, MPI_Comm com = MPI_COMM_WORLD);
};




class DataBlock_MPI_Functions;


class Math_Functions_MPI;

class Math_MPI_Functions_Policy;


template<typename T>
class DistributedDataBlock
{
    friend class DataBlock_MPI_Functions;
    friend class Math_Functions_MPI;
    friend class Math_MPI_Functions_Policy;
public:

    ptrdiff_t block_rank()const;

    bool global_rowmajor()const;

    ptrdiff_t* global_extents()const;

    ptrdiff_t* global_strides()const;

    ptrdiff_t local_blocknumber()const;

    DataBlockArray<T> & Blockarray();


    void print(int rootrank=0)const;

protected:

    DataBlockArray<T> Dblockarray;

    ptrdiff_t* pblock_grid_coords=nullptr;
    ptrdiff_t* pblock_starts=nullptr;
    ptrdiff_t* pblock_linear_idx = nullptr;
    ptrdiff_t  pblock_rank=0;
    ptrdiff_t* pglobal_extents=nullptr;
    ptrdiff_t* pglobal_strides=nullptr;
    ptrdiff_t* pblock_extents=nullptr;
    bool pmemmap=false;

    MPI_CartesianContext* pctx;
    BlockMappingPolicy* ppolicy;

    std::unordered_map<ptrdiff_t, ptrdiff_t> pglobal_to_local_index;
};

struct MPI_Sendlocation
{
    bool with_memmap=false;
    bool ondevice=false;
    int devicenum=-INT_MAX;
};

class DataBlock_MPI_Functions
{
public:
    template<typename T>
    inline static void MPI_Bcast_DataBlock (DataBlock<T> &db,MPI_Comm com, int rootrank);
    template<typename T>
    inline static void MPI_Bcast_DataBlock_meta (DataBlock<T> &db,MPI_Comm com, int rootrank);
    template<typename T>
    inline static void MPI_Bcast_DataBlock_extents_strides (DataBlock<T> &db,MPI_Comm com, int rootrank);
    template<typename T>
    inline static void MPI_Bcast_DataBlock_pdata (DataBlock<T> &db,MPI_Comm com, int rootrank);
    template<typename T>
    inline static void MPI_IBcast_DataBlock_pdata (DataBlock<T> &db,MPI_Comm com,MPI_Request*req, int rootrank);

    template<typename T>
    inline static void MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,MPI_Sendlocation loc,MPI_Comm com, int rootrank);

    template<typename T>
    inline static void MPI_Scatter_vector_to_subvectors_alloc(  ptrdiff_t blocksize,    DistributedDataBlock<T>& recv_db,
           MPI_Sendlocation loc,    MPI_CartesianContext *ctx,    BlockMappingPolicy* policy,   int rootrank,    const DataBlock<T>* send_db);

    template<typename T>
    inline static void MPI_Gather_vector_from_subvectors_alloc(    const DistributedDataBlock<T>& send_db,    int rootrank, MPI_Sendlocation loc,
            DataBlock<T>* recv_db=nullptr   );



    template<typename T>
    inline static void MPI_Scatter_matrix_to_rows_alloc( DistributedDataBlock<T>& recv_db,  MPI_Sendlocation loc,
            MPI_CartesianContext* ctx,  BlockMappingPolicy* policy,  int rootrank,   const DataBlock<T>* send_db);

    template<typename T>
    inline static void MPI_Gather_matrix_from_rows_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank, MPI_Sendlocation loc,   DataBlock<T>* recv_db=nullptr);

    template<typename T>
    inline static void MPI_Scatter_matrix_to_columns_alloc( DistributedDataBlock<T>& recv_db,  MPI_Sendlocation loc,
            MPI_CartesianContext* ctx,  BlockMappingPolicy* policy,  int rootrank,   const DataBlock<T>* send_db=nullptr);

    template<typename T>
    inline static void MPI_Gather_matrix_from_columns_alloc(   const DistributedDataBlock<T>& send_db,      int rootrank, MPI_Sendlocation loc,DataBlock<T>* recv_db = nullptr
           );

    template<typename T>
    inline static void MPI_Scatter_matrix_to_submatrices_alloc(    ptrdiff_t br,    ptrdiff_t bc,    DistributedDataBlock<T>& recv_db,   MPI_Sendlocation loc,     MPI_CartesianContext *ctx,    BlockMappingPolicy* policy, int rootrank,     const DataBlock<T>* send_db=nullptr  );


    template<typename T>
    inline static void MPI_Gather_matrix_from_submatrices_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank,MPI_Sendlocation loc, DataBlock<T>* recv_db = nullptr);

    template<typename T>
    inline static void MPI_Scatter_tensor_to_subtensors_alloc(    ptrdiff_t blockrank,    const ptrdiff_t* block_extents,
            DistributedDataBlock<T>& recv_db,    MPI_Sendlocation loc,    MPI_CartesianContext *ctx,    BlockMappingPolicy* policy, int rootrank,
            const DataBlock<T>* send_db=nullptr);



    template<typename T>
    inline static void MPI_Gather_tensor_from_subtensors_alloc(  const DistributedDataBlock<T>& send_db,  int rootrank,
            MPI_Sendlocation loc,DataBlock<T>* recv_db = nullptr  );

    template<typename T>
    inline static DataBlock<T> MPI_Recv_alloc_DataBlock( MPI_Sendlocation loc, const int source,const  int tag, MPI_Comm pcomm);

    template<typename T>
    inline static void MPI_Free_DataBlock(DataBlock<T>&m);

    template<typename T>
    inline static void MPI_Free_DistributedDataBlock(DistributedDataBlock<T>&m);

    template<typename T>
    inline static void MPI_Send_DataBlock(DataBlock<T> &m,const int dest, const int tag, MPI_Comm pcomm);

    template<typename T>
    inline static void MPI_Recv_DataBlock(DataBlock<T>& m, const int source,const  int tag, MPI_Comm pcomm);

    template<typename T>
    inline static void MPI_Send_DataBlock_meta(DataBlock<T> &m,const int dest, const int tag, MPI_Comm pcomm);

    template<typename T>
    inline static void MPI_Recv_DataBlock_meta(DataBlock<T>& m, const int source,const  int tag, MPI_Comm pcomm);

    template<typename T>
    inline static void MPI_Recv_DataBlock_pdata(DataBlock<T>& mds,const int source, const int tag,const  MPI_Comm pcomm);

    template<typename T>
    inline static void MPI_Send_DataBlock_pdata(DataBlock<T> &m,const int dest,const int tag,const MPI_Comm pcomm);

    template<typename T>
    inline static void MPI_Isend_DataBlock_pdata(DataBlock<T> &m,const int dest,const  int tag,const MPI_Comm pcomm,MPI_Request *request);

    template<typename T>
    inline static void MPI_Irecv_DataBlock_pdata(DataBlock<T> &mds, const int source, const int tag,const  MPI_Comm pcomm,MPI_Request *request);



    inline static std::optional<MPI_Policy> default_policy;


    static const MPI_Policy& get_default_policy(MPI_Comm com,ptrdiff_t blockrank);

    template<typename T>
    inline static void alloc_helper( MPI_Sendlocation loc, ptrdiff_t rank,ptrdiff_t datalength,ptrdiff_t* &pextents,ptrdiff_t *&pstrides,T *&pdata);
    template<typename T>
    inline static void alloc_helper2( MPI_Sendlocation loc, ptrdiff_t datalength,T *&pdata);

    template<typename T>
    inline static void free_helper( MPI_Sendlocation loc,ptrdiff_t datalength,ptrdiff_t* &pextents,ptrdiff_t *&pstrides,T *&pdata);

    template<typename T>
    inline static void free_helper2( MPI_Sendlocation loc, ptrdiff_t datalength,T *&pdata);

    template<typename T>
    inline static int compute_owner(const ptrdiff_t* bcoords,const ptrdiff_t* proc_grid, const ptrdiff_t* cyclic_block,ptrdiff_t gridrank);
};


#include "datablock_mpifunctions.hpp"

#endif



