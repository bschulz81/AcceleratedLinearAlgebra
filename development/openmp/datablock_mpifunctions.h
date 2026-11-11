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

template<typename T>
inline MPI_Datatype make_strided_2d_rowmajor_type(ptrdiff_t rows,ptrdiff_t cols,ptrdiff_t stride0,ptrdiff_t stride1)
{
    MPI_Datatype inner, outer;

    MPI_Type_create_hvector((int)(cols),1,(MPI_Aint)(stride1) * sizeof(T),mpi_get_type<T>(),&inner);

    MPI_Type_create_hvector((int)(rows),1,(MPI_Aint)(stride0) * sizeof(T),inner,&outer);

    MPI_Type_free(&inner);
    return outer;

}


template<typename T>
inline MPI_Datatype make_strided_nd_rowmajor_type(ptrdiff_t rank,const ptrdiff_t* extents,const ptrdiff_t* strides)
{
    if (rank <= 0 || !extents || !strides)
        throw std::invalid_argument("Invalid tensor rank or metadata");

    MPI_Datatype current = mpi_get_type<T>();
    bool owns_current = false;

    for (ptrdiff_t d = rank; d-- > 0;)
    {
        if (extents[d] <= 0 ||
                extents[d] > std::numeric_limits<int>::max())
        {
            if (owns_current)
                MPI_Type_free(&current);

            throw std::invalid_argument("Invalid extent");
        }

        MPI_Datatype next;

        MPI_Type_create_hvector((int)(extents[d]),1,(MPI_Aint)(strides[d] *sizeof(T)),current,&next);

        if (owns_current)
            MPI_Type_free(&current);

        current = next;
        owns_current = true;
    }

    return current;
}


class MPI_CartesianContext
{
public:
    MPI_Comm pcomm;

    ptrdiff_t pgridrank;
    int psize;

    int* pdims;
    int* pperiods;

public:

    MPI_CartesianContext(MPI_Comm comm_);

    ~MPI_CartesianContext();

    int rank_from_coords(int* coords) const;
};

class BlockMappingPolicy
{
public:
    ptrdiff_t pgridrank;

    int* pindex_map;
    ptrdiff_t* pcyclic_block;

    BlockMappingPolicy(ptrdiff_t gridrank,const int* index_map = nullptr,const ptrdiff_t* cyclic_block = nullptr);

    ~BlockMappingPolicy()
    {
        delete[] pindex_map;
        delete[] pcyclic_block;
    }

    inline void owner_mpi_cartesian_coords(const ptrdiff_t* global_block_grid_coords,ptrdiff_t block_grid_rank,const MPI_CartesianContext& ctx,int* cartesian_coords) const;

    inline int owner_mpi_rank(const ptrdiff_t* global_block_grid_coords,ptrdiff_t block_grid_rank,const MPI_CartesianContext& ctx) const;
};


struct TensorBlockLocation
{
    int owner_rank;
    ptrdiff_t global_block_index;
    ptrdiff_t local_flat_index;
};



class DataBlock_MPI_Functions;
class Math_Functions_MPI;
class Math_MPI_Functions_Policy;

class TensorRequestEngine;

template<typename T>
class DistributedDataBlock
{
    friend class DataBlock_MPI_Functions;
    friend class Math_Functions_MPI;
    friend class Math_MPI_Functions_Policy;

    friend class TensorRequestEngine;

public:

    // ------------------------------------------------------------
    // Global tensor
    // ------------------------------------------------------------

    ptrdiff_t tensor_rank() const;

    ptrdiff_t* global_extents() const;

    ptrdiff_t* global_strides() const;


    // ------------------------------------------------------------
    // Block grid
    // ------------------------------------------------------------

    ptrdiff_t block_grid_rank() const;

    ptrdiff_t* default_block_shape() const;


    const ptrdiff_t* block_grid_extents() const;

    ptrdiff_t block_grid_extent(ptrdiff_t dim) const;
    // ------------------------------------------------------------
    // Local blocks
    // ------------------------------------------------------------

    ptrdiff_t num_local_blocks() const;


    DataBlockArray<T>& block_array();

    const DataBlockArray<T>& block_array() const;

    DataBlock<T> local_block(ptrdiff_t local_block) const;


    // ------------------------------------------------------------
    // Global block-grid coordinates / indices
    // ------------------------------------------------------------

    // Local block index -> global block-grid coordinates
    const ptrdiff_t* global_block_grid_coords(ptrdiff_t local_block) const;

    // Global block-grid index -> global block-grid coordinates
    void global_block_grid_coords(ptrdiff_t global_block_grid_index,ptrdiff_t* global_block_grid_coords) const;

    // Global block-grid index
    ptrdiff_t global_block_grid_index(const ptrdiff_t* global_block_grid_coords) const;

    //local block index->global block grid index
    ptrdiff_t global_block_grid_index(ptrdiff_t local_block) const;

    // ------------------------------------------------------------
    // Block positions
    // ------------------------------------------------------------

    // Local block -> global tensor start
    const ptrdiff_t* block_grid_starts(ptrdiff_t local_block) const;

    // Global block-grid coordinates -> global tensor start
    void block_grid_starts(const ptrdiff_t* global_block_grid_coords,ptrdiff_t* starts) const;


    // ------------------------------------------------------------
    // Block geometry
    // ------------------------------------------------------------

    // Global block-grid coordinates -> actual tensor extent
    void global_block_extents(const ptrdiff_t* global_block_grid_coords,ptrdiff_t* extents) const;

    // Local block -> actual tensor extent
    const ptrdiff_t* local_block_extents(ptrdiff_t local_block) const;

    // Local block -> local DataBlock strides
    const ptrdiff_t* local_block_strides(ptrdiff_t local_block) const;


    // ------------------------------------------------------------
    // Global block-grid information
    // ------------------------------------------------------------

    ptrdiff_t total_block_num() const;


    // ------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------

    // Global block-grid coordinates -> owning MPI rank
    int owner_rank(const ptrdiff_t* global_block_grid_coords) const;
    int owner_rank_from_global_flat_index(const ptrdiff_t global_flat_index) const;

    void print(int rootrank = 0) const;

protected:

    // ------------------------------------------------------------
    // Local blocks
    // ------------------------------------------------------------

    DataBlockArray<T> Dblockarray;


    // ------------------------------------------------------------
    // Per-local-block metadata
    // ------------------------------------------------------------

    // [local_block * ptensor_rank]
    ptrdiff_t* plocal_block_grid_coords = nullptr;

    // [local_block * pblock_grid_rank]
    ptrdiff_t* plocal_block_global_starts = nullptr;

    // [local_block]
    ptrdiff_t* plocal_block_global_grid_indices = nullptr;


    // ------------------------------------------------------------
    // Global tensor metadata
    // ------------------------------------------------------------


    ptrdiff_t* pglobal_extents = nullptr;

    ptrdiff_t* pglobal_strides = nullptr;


    // ------------------------------------------------------------
    // Block-grid metadata
    // ------------------------------------------------------------

    // Number of dimensions actually being blocked.
    ptrdiff_t pblock_grid_rank = 0;

    // Nominal/default block shape.
    ptrdiff_t* pdefault_block_shape = nullptr;

    ptrdiff_t* pblock_grid_extents = nullptr;
    ptrdiff_t* pblock_grid_strides=nullptr;


    MPI_CartesianContext* pctx = nullptr;

    BlockMappingPolicy* ppolicy = nullptr;

    // global block-grid index -> local block index
    std::unordered_map<ptrdiff_t, ptrdiff_t> pglobal_grid_index_to_local_block;
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
    inline static void MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,DataBlockConfig loc,MPI_Comm com, int rootrank);

    template<typename T>
    inline static void MPI_Scatter_vector_to_subvectors_alloc(  ptrdiff_t blocksize,    DistributedDataBlock<T>& recv_db,
            DataBlockConfig loc,    MPI_CartesianContext *ctx,    BlockMappingPolicy* policy,   int rootrank,    const DataBlock<T>* send_db);

    template<typename T>
    inline static void MPI_Gather_vector_from_subvectors_alloc(    const DistributedDataBlock<T>& send_db,    int rootrank, DataBlockConfig loc,
            DataBlock<T>* recv_db=nullptr   );


    template<typename T>
    inline static void MPI_Scatter_matrix_to_rows_alloc( DistributedDataBlock<T>& recv_db,  DataBlockConfig loc,
            MPI_CartesianContext* ctx,  BlockMappingPolicy* policy,  int rootrank,   const DataBlock<T>* send_db);

    template<typename T>
    inline static void MPI_Gather_matrix_from_rows_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank, DataBlockConfig loc,   DataBlock<T>* recv_db=nullptr);

    template<typename T>
    inline static void MPI_Scatter_matrix_to_columns_alloc( DistributedDataBlock<T>& recv_db,  DataBlockConfig loc,
            MPI_CartesianContext* ctx,  BlockMappingPolicy* policy,  int rootrank,   const DataBlock<T>* send_db=nullptr);

    template<typename T>
    inline static void MPI_Gather_matrix_from_columns_alloc(   const DistributedDataBlock<T>& send_db,      int rootrank, DataBlockConfig loc,DataBlock<T>* recv_db = nullptr
                                                           );

    template<typename T>
    inline static void MPI_Scatter_matrix_to_submatrices_alloc(    ptrdiff_t br,    ptrdiff_t bc,    DistributedDataBlock<T>& recv_db,   DataBlockConfig loc,     MPI_CartesianContext *ctx,    BlockMappingPolicy* policy, int rootrank,     const DataBlock<T>* send_db=nullptr  );


    template<typename T>
    inline static void MPI_Gather_matrix_from_submatrices_alloc( const DistributedDataBlock<T>& send_db,
            int rootrank,DataBlockConfig loc, DataBlock<T>* recv_db = nullptr);

    template<typename T>
    inline static void MPI_Scatter_tensor_to_subtensors_alloc(    ptrdiff_t blockrank,    const ptrdiff_t* block_extents,
            DistributedDataBlock<T>& recv_db,    DataBlockConfig loc,    MPI_CartesianContext *ctx,    BlockMappingPolicy* policy, int rootrank,
            const DataBlock<T>* send_db=nullptr);

    template<typename T>
    inline static TensorBlockLocation get_tensor_block_location(const DistributedDataBlock<T>& source,ptrdiff_t global_flat_index);



    template<typename T>
    inline static void MPI_Gather_tensor_from_subtensors_alloc(  const DistributedDataBlock<T>& send_db,  int rootrank,
            DataBlockConfig loc,DataBlock<T>* recv_db = nullptr  );


    template<typename T>
    inline static void MPI_All_Gather_tensor_from_subtensors_alloc(
        const DistributedDataBlock<T>& send_db,DataBlockConfig loc,DataBlock<T>& recv_db);

    template<typename T>
    inline static void MPI_All_Gather_matrix_from_submatrices_alloc(
        const DistributedDataBlock<T>& send_db,DataBlockConfig loc,DataBlock<T>& recv_db);

    template<typename T>
    inline static void MPI_All_Gather_vector_from_subvectors_alloc(
        const DistributedDataBlock<T>& send_db,DataBlockConfig loc,DataBlock<T>& recv_db);

    template<typename T>
    inline static DataBlock<T> MPI_Recv_alloc_DataBlock( DataBlockConfig loc, const int source,const  int tag, MPI_Comm pcomm);

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

    template <typename T>
    inline static void MPI_Send_DataBlockArray(const DataBlockArray<T>& array,int dest,int tag,MPI_Comm pcomm);

    template <typename T>
    inline static DataBlockArray<T> MPI_Recv_alloc_DataBlockArray(DataBlockConfig loc,int source,int tag,MPI_Comm pcomm);

    template <typename T>
    inline static void MPI_Bcast_alloc_DataBlockArray(DataBlockArray<T>&array,DataBlockConfig loc,int rootrank,MPI_Comm pcomm);

    template <typename T>
    inline static void MPI_Free_DataBlockArray(DataBlockArray<T>&array);

    template<typename T>
    inline static void alloc_helper( DataBlockConfig loc, ptrdiff_t rank,ptrdiff_t datalength,ptrdiff_t* &pextents,ptrdiff_t *&pstrides,T *&pdata);
    template<typename T>
    inline static void alloc_helper2( DataBlockConfig loc, ptrdiff_t datalength,T *&pdata);


    template<typename T>
    inline static void free_helper( DataBlockConfig loc,ptrdiff_t datalength,ptrdiff_t* &pextents,ptrdiff_t *&pstrides,T *&pdata);

    template<typename T>
    inline static void free_helper2( DataBlockConfig loc, ptrdiff_t datalength,T *&pdata);
};


#include "datablock_mpifunctions.hpp"

#endif



