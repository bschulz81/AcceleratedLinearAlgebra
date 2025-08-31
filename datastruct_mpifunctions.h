#ifndef DATASTRUCT_MPIFUNCTIONS
#define DATASTRUCT_MPIFUNCTIONS

#include <mpi.h>


#include <complex>
#include <type_traits>
#include <cstdint>
#include <cstring>

#include "datastruct.h"
#include "datastruct_host_memory_functions.h"
#include "datastruct_gpu_memory_functions.h"
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
class Datastruct_MPI_Functions
{
public:

    inline static datastruct<T> MPI_Recv_alloc_datastruct(bool with_memmap, const int source,const  int tag, MPI_Comm pcomm);
    inline static datastruct<T> MPI_Recv_device_alloc_datastruct(bool with_memmap, int devicenum, const int source,const  int tag, MPI_Comm pcomm);
    inline static void MPI_Free_device_datastruct(datastruct<T>&m, int devicenum);
    inline static void MPI_Free_datastruct(datastruct<T>&m);

    inline static void MPI_Send_datastruct(datastruct<T> &m,const int dest, const int tag, MPI_Comm pcomm);
    inline static void MPI_Recv_datastruct(datastruct<T>& m, const int source,const  int tag, MPI_Comm pcomm);

    inline static void MPI_Isend_datastruct_pdata(datastruct<T> &m,const int dest,const  int tag,const MPI_Comm pcomm);
    inline static void MPI_Irecv_datastruct_pdata(datastruct<T> &mds, const int source, const int tag,const  MPI_Comm pcomm);

    inline static void MPI_Recv_datastruct_pdata(datastruct<T>& mds,const int source, const int tag,const  MPI_Comm pcomm);
    inline static void MPI_Send_datastruct_pdata(datastruct<T> &m,const int dest,const int tag,const MPI_Comm pcomm);


    inline static void mpi_broadcast_subtensor( const datastruct<T>& root_ds,    datastruct<T>& local_view,   const size_t* sub_extents,    const size_t* sub_strides,    const size_t* sub_offsets,    int root,    MPI_Comm comm);
    inline static void mpi_broadcast_subtensor(datastruct<T>& ds,    const size_t* sub_extents,    const size_t* sub_strides,  const size_t* sub_offsets,    int root, MPI_Comm comm);
    inline static void mpi_broadcast_datastruct(datastruct<T>& ds, int root, MPI_Comm comm);
    inline static MPI_Datatype create_strided_type(const datastruct<T>& m);


    inline static void mpi_scatter_subtensor(const datastruct<T>& send_ds,
            datastruct<T>& recv_ds,
            const size_t* send_extents,
            const size_t* send_strides,
            const size_t* send_offsets,
            const size_t* recv_extents,
            const size_t* recv_strides,
            const size_t* recv_offsets,
            int root, MPI_Comm comm);
    inline static void mpi_gather_subtensor(const datastruct<T>& send_ds,
                                            datastruct<T>& recv_ds,
                                            const size_t* send_extents,
                                            const size_t* send_strides,
                                            const size_t* send_offsets,
                                            const size_t* recv_extents,
                                            const size_t* recv_strides,
                                            const size_t* recv_offsets,
                                            int root, MPI_Comm comm);


    inline static MPI_Datatype mpi_row_type(const datastruct<T>& mat_ds, size_t row_index);

    inline static MPI_Datatype mpi_column_type(const datastruct<T>& mat_ds, size_t col_index);

    inline static void mpi_scatter_row(const datastruct<T>& send_mat,  datastruct<T>& recv_row, size_t row_index,  int root, MPI_Comm comm);


    inline static void mpi_gather_row(const datastruct<T>& send_row,  datastruct<T>& recv_mat,   size_t row_index,  int root, MPI_Comm comm);


    inline static void mpi_scatter_column(const datastruct<T>& send_mat,datastruct<T>& recv_col,size_t col_index, int root, MPI_Comm comm);

    inline static void mpi_gather_column(const datastruct<T>& send_col,  datastruct<T>& recv_mat, size_t col_index,   int root, MPI_Comm comm);

    inline static MPI_Datatype mpi_substruct_type(const datastruct<T>& sub_ds);
    inline static MPI_Datatype mpi_subspanmatrix_type(const datastruct<T>& mat_ds);
    inline static MPI_Datatype mpi_subtensor_type(const datastruct<T>& ds, const size_t* sub_extents, const size_t* sub_strides,   const size_t* sub_offsets);

    inline static MPI_Datatype mpi_datatype(const datastruct<T>& ds);
    inline static MPI_Datatype make_mpi_subtensor_type(const datastruct<T>& sub);


    inline static void mpi_scatter_substruct(const datastruct<T>& send_ds,  datastruct<T>& recv_ds, int root, MPI_Comm comm);
    inline static void mpi_gather_substruct(const datastruct<T>& send_ds, datastruct<T>& recv_ds, int root,   MPI_Comm comm);
    inline static void mpi_scatter_submatrix(const datastruct<T>& send_mat, datastruct<T>& recv_mat,int root,MPI_Comm comm);
    inline static void mpi_gather_submatrix(const datastruct<T>& send_mat, datastruct<T>& recv_mat,int root, MPI_Comm comm);

    inline static void mpi_allreduce_row(datastruct<T>& mat,
                                         size_t row_index,
                                         MPI_Op op,
                                         MPI_Comm comm);

    inline static void mpi_allreduce_column(datastruct<T>& mat,
                                            size_t col_index,
                                            MPI_Op op,
                                            MPI_Comm comm);

    inline static void mpi_reduce_row(const datastruct<T>& send_mat,
                                      datastruct<T>& recv_mat,
                                      size_t row_index,
                                      int root, MPI_Op op, MPI_Comm comm)   ;


    inline static void mpi_allreduce_subtensor(datastruct<T>& ds,
            const size_t* sub_extents,
            const size_t* sub_strides,   // unused by subarray, kept for symmetry
            const size_t* sub_offsets,
            MPI_Op op,
            MPI_Comm comm)   ;

    inline static void mpi_allreduce(datastruct<T>& ds,
                                     MPI_Op op,
                                     MPI_Comm comm);

    inline static void mpi_reduce(const datastruct<T>& send_ds,
                                  datastruct<T>& recv_ds,
                                  int root,
                                  MPI_Op op,
                                  MPI_Comm comm);


    inline static void mpi_reduce_subtensor(const datastruct<T>& send_ds,
                                            datastruct<T>& recv_ds,
                                            const size_t* send_extents,
                                            const size_t* send_strides,
                                            const size_t* send_offsets,
                                            const size_t* recv_extents,
                                            const size_t* recv_strides,
                                            const size_t* recv_offsets,
                                            int root, MPI_Op op, MPI_Comm comm);


    inline static void mpi_reduce_column(const datastruct<T>& send_mat,
                                         datastruct<T>& recv_mat,
                                         size_t col_index,
                                         int root, MPI_Op op, MPI_Comm comm);

};



template<typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::create_strided_type(const datastruct<T>& m)
{
    int ndims = m.rank();

    // Base case: 1D array
    if(ndims == 1)
    {
        MPI_Datatype type;
        MPI_Type_contiguous(m.dpextents[0], mpi_get_type<T>(), &type);
        MPI_Type_commit(&type);
        return type;
    }

    // Recursive: create type for inner dimensions
    datastruct<T> inner;
    // construct a "fake" datastruct for inner slice (extents+strides from 1..end)
    // or just pass extents+strides arrays offset by 1
    MPI_Datatype inner_type = create_strided_type_inner(ndims-1, m.extents()+1, m.strides()+1);

    // Wrap with hvector using stride of first dimension
    MPI_Datatype type;
    MPI_Type_create_hvector(m.dpextents[0], 1, m.dpstrides[0]*sizeof(T), inner_type, &type);
    MPI_Type_commit(&type);
    MPI_Type_free(&inner_type);
    return type;
}

template <typename T>
void Datastruct_MPI_Functions<T>::mpi_broadcast_subtensor(datastruct<T>& ds,
        const size_t* sub_extents,
        const size_t* sub_strides,
        const size_t* sub_offsets,
        int root,
        MPI_Comm comm)
{
    // Create datatype for the sub-block
    MPI_Datatype sub_type =
        Datastruct_MPI_Functions<T>::mpi_subtensor_type(ds, sub_extents, sub_strides, sub_offsets);

    // Broadcast the data from root
    MPI_Bcast(ds.dpdata, 1, sub_type, root, comm);

    // Clean up
    MPI_Type_free(&sub_type);
}


template <typename T>
inline static void mpi_broadcast_datastruct(datastruct<T>& ds, int root, MPI_Comm comm)
{
    // broad meta info (extents, strides, rank, etc.) if needed
    MPI_Bcast(&ds.dprank, 1, mpi_get_type<size_t>(), root, comm);
    MPI_Bcast(ds.dpextents, ds.dprank, mpi_get_type<size_t>(), root, comm);
    MPI_Bcast(ds.dpstrides, ds.dprank, mpi_get_type<size_t>(), root, comm);

    MPI_Bcast(ds.dpdata, ds.dpdatalength, mpi_get_type<T>(), root, comm);
}


template <typename T>
void Datastruct_MPI_Functions<T>::mpi_broadcast_subtensor(
    const datastruct<T>& root_ds,  // valid on root
    datastruct<T>& local_view,     // must point to allocated memory
    const size_t* sub_extents,
    const size_t* sub_strides,
    const size_t* sub_offsets,
    int root,
    MPI_Comm comm)
{
    MPI_Datatype sub_type =
        Datastruct_MPI_Functions<T>::mpi_subtensor_type(
            root_ds, sub_extents, sub_strides, sub_offsets);

    MPI_Bcast(local_view.dpdata, 1, sub_type, root, comm);

    MPI_Type_free(&sub_type);
}


template <typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::make_mpi_subtensor_type(const datastruct<T>& sub)
{
    int ndims = sub.dprank;
    // Base case: 1D array
    if (ndims == 1)
    {
        MPI_Datatype type;
        MPI_Type_contiguous(sub.dpextents[0], mpi_get_type<T>(), &type);
        MPI_Type_commit(&type);
        return type;
    }

    // Recursive creation for higher dimensions
    MPI_Datatype inner_type = create_strided_type_inner(ndims - 1, sub.dpextents + 1, sub.dpstrides + 1);

    // Wrap inner type in an hvector for the first dimension
    MPI_Datatype type;
    MPI_Type_create_hvector(
        sub.dpextents[0],            // number of blocks
        1,                                             // 1 inner block per outer block
        static_cast<MPI_Aint>(sub.dpstrides[0] * sizeof(T)), // stride in bytes
        inner_type,
        &type
    );

    MPI_Type_commit(&type);
    MPI_Type_free(&inner_type);  // free temporary inner type
    return type;
}



template<typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::mpi_subtensor_type(const datastruct<T>& ds,
        const size_t* sub_extents,
        const size_t* sub_strides,
        const size_t* sub_offsets)
{

    // Sizes = global tensor dimensions
    int* sizes=new int[ds.dprank];
    // Subsizes = sub-tensor dimensions
    int* subsizes=new int[ds.dprank];
    // Starts = offset (in units of elements, not bytes)
    int* starts=new int[ds.dprank];


    for (int i = 0; i < ds.dprank; i++)
    {
        sizes[i]    =ds.pdextents[i];
        subsizes[i] = sub_extents[i];
        starts[i]   = sub_offsets[i];
    }

    int order = ds.rowmajor() ? MPI_ORDER_C : MPI_ORDER_FORTRAN;

    MPI_Datatype newtype;
    MPI_Type_create_subarray(ds.dprank,
                             sizes,
                             subsizes,
                             starts,
                             order,
                             mpi_get_type<T>(),
                             &newtype);

    MPI_Type_commit(&newtype);

    delete []sizes;
    delete []subsizes;
    delete []starts;

    return newtype;
}

template<typename T>
void Datastruct_MPI_Functions<T>::mpi_scatter_subtensor(const datastruct<T>& send_ds,
        datastruct<T>& recv_ds,
        const size_t* send_extents,
        const size_t* send_strides,
        const size_t* send_offsets,
        const size_t* recv_extents,
        const size_t* recv_strides,
        const size_t* recv_offsets,
        int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Datatype send_type = MPI_DATATYPE_NULL;
    MPI_Datatype recv_type = mpi_subtensor_type(recv_ds, recv_extents, recv_strides, recv_offsets);

    if(rank == root)
    {
        send_type = mpi_subtensor_type(send_ds, send_extents, send_strides, send_offsets);
    }

    MPI_Scatter(rank == root ? send_ds.dpdata : nullptr,
                1, send_type,
                recv_ds.dpdata, 1, recv_type,
                root, comm);

    if(send_type != MPI_DATATYPE_NULL) MPI_Type_free(&send_type);
    MPI_Type_free(&recv_type);
}


template<typename T>
void Datastruct_MPI_Functions<T>::mpi_gather_subtensor(const datastruct<T>& send_ds,
        datastruct<T>& recv_ds,
        const size_t* send_extents,
        const size_t* send_strides,
        const size_t* send_offsets,
        const size_t* recv_extents,
        const size_t* recv_strides,
        const size_t* recv_offsets,
        int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Datatype send_type = mpi_subtensor_type(send_ds, send_extents, send_strides, send_offsets);
    MPI_Datatype recv_type = MPI_DATATYPE_NULL;

    if(rank == root)
    {
        recv_type = mpi_subtensor_type(recv_ds, recv_extents, recv_strides, recv_offsets);
    }

    MPI_Gather(send_ds.dpdata, 1, send_type,
               rank == root ? recv_ds.dpdata : nullptr,
               1, recv_type,
               root, comm);

    MPI_Type_free(&send_type);
    if(recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&recv_type);
}


template<typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::mpi_row_type(const datastruct<T>& mat_ds, size_t row_index)
{
    MPI_Datatype row_type;
    if (mat_ds.rowmajor())
    {
        // Row-major: rows are contiguous
        MPI_Type_contiguous(mat_ds.dpextents[1],
                            mpi_get_type<T>(),
                            &row_type);
    }
    else
    {
        // Column-major: rows are strided
        MPI_Type_vector(mat_ds.dpextents[1],
                        1,
                        mat_ds.dpstrides[1],
                        mpi_get_type<T>(),
                        &row_type);
    }
    MPI_Type_commit(&row_type);
    return row_type;
}

// -----------------------------
// Column of a matrix (Nx1 view)
// -----------------------------
template<typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::mpi_column_type(const datastruct<T>& mat_ds, size_t col_index)
{
    MPI_Datatype col_type;
    if (mat_ds.rowmajor())
    {
        // Row-major: columns are strided
        MPI_Type_vector(mat_ds.dpextents[0],
                        1,
                        mat_ds.dpstrides[0],
                        mpi_get_type<T>(),
                        &col_type);
    }
    else
    {
        // Column-major: columns are contiguous
        MPI_Type_contiguous(mat_ds.dpextents[0],
                            mpi_get_type<T>(),
                            &col_type);
    }
    MPI_Type_commit(&col_type);
    return col_type;
}

template<typename T>
void Datastruct_MPI_Functions<T>::mpi_scatter_row(const datastruct<T>& send_mat,
        datastruct<T>& recv_row,
        size_t row_index,
        int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype send_type = MPI_DATATYPE_NULL;
    MPI_Datatype recv_type = mpi_row_type(recv_row, 0);

    if(rank == root) send_type = mpi_row_type(send_mat, row_index);

    MPI_Scatter(rank == root ? send_mat.dpdata + row_index * send_mat.dpstrides[0] : nullptr,
                1, send_type,
                recv_row.dpdata, 1, recv_type, root, comm);

    if(send_type != MPI_DATATYPE_NULL) MPI_Type_free(&send_type);
    MPI_Type_free(&recv_type);
}

template<typename T>
void Datastruct_MPI_Functions<T>::mpi_gather_row(const datastruct<T>& send_row,
        datastruct<T>& recv_mat,
        size_t row_index,
        int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype send_type = mpi_row_type(send_row, 0);
    MPI_Datatype recv_type = MPI_DATATYPE_NULL;

    if(rank == root) recv_type = mpi_row_type(recv_mat, row_index);

    MPI_Gather(send_row.dpdata, 1, send_type,
               rank == root ? recv_mat.dpdata + row_index * recv_mat.dpstrides[0] : nullptr,
               1, recv_type, root, comm);

    MPI_Type_free(&send_type);
    if(recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&recv_type);
}

// -----------------------------
// Scatter / gather a column
// -----------------------------
template<typename T>
void Datastruct_MPI_Functions<T>::mpi_scatter_column(const datastruct<T>& send_mat,
        datastruct<T>& recv_col,
        size_t col_index,
        int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype send_type = MPI_DATATYPE_NULL;
    MPI_Datatype recv_type = mpi_subvector_type(recv_col);

    if(rank == root) send_type = mpi_column_type(send_mat, col_index);

    MPI_Scatter(rank == root ? send_mat.dpdata + col_index : nullptr,
                1, send_type,
                recv_col.dpdata, 1, recv_type, root, comm);

    if(send_type != MPI_DATATYPE_NULL) MPI_Type_free(&send_type);
    MPI_Type_free(&recv_type);
}

template<typename T>
void Datastruct_MPI_Functions<T>::mpi_gather_column(const datastruct<T>& send_col,
        datastruct<T>& recv_mat,
        size_t col_index,
        int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype send_type = mpi_subvector_type(send_col);
    MPI_Datatype recv_type = MPI_DATATYPE_NULL;

    if(rank == root) recv_type = mpi_column_type(recv_mat, col_index);

    MPI_Gather(send_col.dpdata, 1, send_type,
               rank == root ? recv_mat.dpdata + col_index : nullptr,
               1, recv_type, root, comm);

    MPI_Type_free(&send_type);
    if(recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&recv_type);
}


#include <mpi.h>

// -----------------------------
// MPI Datatype helpers
// -----------------------------
template<typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::mpi_datatype(const datastruct<T>& ds)
{
    int ndims = ds.dprank;
    int* sizes    = new int[ndims];
    int* subsizes = new int[ndims];
    int* starts   = new int[ndims] {0};

    for (int i = 0; i < ndims; ++i)
    {
        sizes[i]    = ds.dpextents[i];
        subsizes[i] = sizes[i];
    }

    int order = ds.rowmajor() ? MPI_ORDER_C : MPI_ORDER_FORTRAN;
    MPI_Datatype type;
    MPI_Type_create_subarray(ndims, sizes, subsizes, starts, order, mpi_get_type<T>(), &type);
    MPI_Type_commit(&type);

    delete[] sizes;
    delete[] subsizes;
    delete[] starts;
    return type;
}

template<typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::mpi_substruct_type(const datastruct<T>& sub_ds)
{
    // Uses substruct-style extents + strides
    int ndims = sub_ds.dprank;
    if (ndims == 1)
    {
        MPI_Datatype type;
        MPI_Type_contiguous(sub_ds.dpextents[0], mpi_get_type<T>(), &type);
        MPI_Type_commit(&type);
        return type;
    }

    // recursive hvector-based type
    MPI_Datatype inner_type;
    {
        size_t* offsets = new size_t[ndims-1] {0};
        size_t* strides = new size_t[ndims-1] {0};
        datastruct<T> inner_view = sub_ds.substruct_s(offsets, sub_ds.dpextents + 1, strides);
        inner_type = mpi_substruct_type(inner_view);
        delete[] offsets;
        delete[] strides;
    }

    MPI_Datatype type;
    MPI_Type_create_hvector(sub_ds.dpextents[0], 1, sub_ds.dpstrides[0] * sizeof(T), inner_type, &type);
    MPI_Type_commit(&type);
    MPI_Type_free(&inner_type);

    return type;
}

template<typename T>
MPI_Datatype Datastruct_MPI_Functions<T>::mpi_subspanmatrix_type(const datastruct<T>& mat_ds)
{
    // specialized 2-D view
    MPI_Datatype type;
    MPI_Type_create_hvector(mat_ds.dpextents[0], mat_ds.dpextents[1], mat_ds.dpstrides[0] * sizeof(T), mpi_get_type<T>(), &type);
    MPI_Type_commit(&type);
    return type;
}

// -----------------------------
// MPI Scatter/Gather helpers
// -----------------------------

// Scatter a subtensor (view)
template<typename T>
void Datastruct_MPI_Functions<T>::mpi_scatter_substruct(const datastruct<T>& send_ds,
        datastruct<T>& recv_ds,
        int root,
        MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype send_type = MPI_DATATYPE_NULL;
    MPI_Datatype recv_type = mpi_substruct_type(recv_ds);

    if (rank == root) send_type = mpi_substruct_type(send_ds);

    MPI_Scatter(rank == root ? send_ds.dpdata : nullptr, 1, send_type,
                recv_ds.dpdata, 1, recv_type, root, comm);

    if (send_type != MPI_DATATYPE_NULL) MPI_Type_free(&send_type);
    MPI_Type_free(&recv_type);
}

// Gather a subtensor (view)
template<typename T>
void Datastruct_MPI_Functions<T>::mpi_gather_substruct(const datastruct<T>& send_ds,
        datastruct<T>& recv_ds,
        int root,
        MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype send_type = mpi_substruct_type(send_ds);
    MPI_Datatype recv_type = MPI_DATATYPE_NULL;

    if (rank == root) recv_type = mpi_substruct_type(recv_ds);

    MPI_Gather(send_ds.dpdata, 1, send_type,
               rank == root ? recv_ds.dpdata : nullptr,
               1, recv_type, root, comm);

    MPI_Type_free(&send_type);
    if (recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&recv_type);
}

// Specialized 2-D scatter/gather for submatrices
template<typename T>
void Datastruct_MPI_Functions<T>::mpi_scatter_submatrix(const datastruct<T>& send_mat,
        datastruct<T>& recv_mat,
        int root,
        MPI_Comm comm)
{
    MPI_Datatype send_type = MPI_DATATYPE_NULL;
    MPI_Datatype recv_type = mpi_subspanmatrix_type(recv_mat);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == root) send_type = mpi_subspanmatrix_type(send_mat);

    MPI_Scatter(rank == root ? send_mat.dpdata : nullptr, 1, send_type,
                recv_mat.dpdata, 1, recv_type, root, comm);

    if (send_type != MPI_DATATYPE_NULL) MPI_Type_free(&send_type);
    MPI_Type_free(&recv_type);
}

template<typename T>
void Datastruct_MPI_Functions<T>::mpi_gather_submatrix(const datastruct<T>& send_mat,
        datastruct<T>& recv_mat,
        int root,
        MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype send_type = mpi_subspanmatrix_type(send_mat);
    MPI_Datatype recv_type = MPI_DATATYPE_NULL;

    if (rank == root) recv_type = mpi_subspanmatrix_type(recv_mat);

    MPI_Gather(send_mat.dpdata, 1, send_type,
               rank == root ? recv_mat.dpdata : nullptr,
               1, recv_type, root, comm);

    MPI_Type_free(&send_type);
    if (recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&recv_type);
}


template<typename T>
inline  void Datastruct_MPI_Functions<T>::MPI_Send_datastruct(datastruct<T> &m, int dest, int tag, MPI_Comm pcomm)
{
    MPI_Send(&m.dpdatalength, 1, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprank, 1, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(&m.dprowmajor, 1, mpi_get_type<bool>(), dest, tag, pcomm);
    MPI_Send(&m.dpdata_is_devptr,1, mpi_get_type<bool>(), dest, tag, pcomm);
    MPI_Send(m.dpextents, m.dprank, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(m.dpstrides, m.dprank, mpi_get_type<size_t>(), dest, tag, pcomm);
    MPI_Send(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}

template<typename T>
inline  void Datastruct_MPI_Functions<T>::MPI_Isend_datastruct_pdata(datastruct<T> &m,const int dest,const  int tag,const MPI_Comm pcomm)
{
    MPI_Request request;
    MPI_Isend(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm,&request);
}

template<typename T>
inline  void Datastruct_MPI_Functions<T>::MPI_Send_datastruct_pdata(datastruct<T> &m,const int dest,const int tag,const MPI_Comm pcomm)
{
    MPI_Send(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm);
}

template<typename T>
inline  datastruct<T> Datastruct_MPI_Functions<T>::MPI_Recv_alloc_datastruct(bool with_memmap, const int source,const  int tag, MPI_Comm pcomm)
{
    MPI_Status status;
    size_t pdatalength, prank;
    bool prowmajor;
    bool pdataisdevptr;

    MPI_Recv(&pdatalength, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&prank, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&prowmajor, 1, mpi_get_type<bool>(), source, tag, pcomm, &status);

    MPI_Recv(&pdataisdevptr,1, mpi_get_type<bool>(), source, tag, pcomm,&status);


    pdataisdevptr=false;


    size_t *pextents,
           *pstrides;

    T* pdata;

    pextents= (size_t*)malloc(sizeof(size_t)*prank);
    MPI_Recv(pextents,prank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    pstrides= (size_t*)malloc(sizeof(size_t)*prank);
    MPI_Recv(pstrides,prank, mpi_get_type<size_t>(), source, tag, pcomm, &status);


    if(with_memmap)
    {
        pdata=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(pdatalength);
    }
    else
    {
        pdata=(T*)malloc(sizeof(T)*pdatalength);
    }
    MPI_Recv(pdata,sizeof(T)*pdatalength, MPI_BYTE, source, tag, pcomm, &status);


    return datastruct<T>(pdata,pdatalength,prowmajor,prank,pextents,pstrides,false,false,pdataisdevptr);
}



template<typename T>
inline  datastruct<T> Datastruct_MPI_Functions<T>::MPI_Recv_device_alloc_datastruct(bool with_memmap, int devicenum, const int source,const  int tag, MPI_Comm pcomm)
{
#if defined(Unified_Shared_Memory)
    return MPI_Recv_alloc_datastruct( with_memmap, source, tag, pcomm);
#else

    MPI_Status status;
    size_t pdatalength, prank;
    bool prowmajor;
    bool pdataisdevptr;

    MPI_Recv(&pdatalength, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&prank, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&prowmajor, 1, mpi_get_type<bool>(), source, tag, pcomm, &status);

    MPI_Recv(&pdataisdevptr,1, mpi_get_type<bool>(), source, tag, pcomm,&status);

    pdataisdevptr=true;

    size_t *pextents,
           *pstrides;

    T* pdata;

    pextents=(size_t*)malloc(sizeof(size_t)*prank);
    pstrides=(size_t*)malloc(sizeof(size_t)*prank);

    pdata=(T*)omp_target_alloc(sizeof(T)*pdatalength,devicenum);

    MPI_Recv(pextents,prank, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(pstrides,prank, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(pdata,sizeof(T)*pdatalength, MPI_BYTE, source, tag, pcomm, &status);

    datastruct<T> m(pdata,pdatalength,prowmajor,prank,pextents,pstrides,false,false,pdataisdevptr);
    return m;

#endif
}

template <typename T>
void Datastruct_MPI_Functions<T>::MPI_Free_device_datastruct(datastruct<T>&m, int dev)
{
#if defined(Unified_Shared_Memory)
    MPI_Free_datastruct(m);
#else
    omp_target_free(m.dpdata,dev);
    free(m.dpextents);
    free(m.dpstrides);
#endif
}


template <typename T>
void Datastruct_MPI_Functions<T>::MPI_Free_datastruct(datastruct<T>&m)
{
    free(m.dpdata);
    free(m.dpextents);
    free(m.dpstrides);
}

template<typename T>
void Datastruct_MPI_Functions<T>::MPI_Recv_datastruct(datastruct<T>& m,const int source,const  int tag, MPI_Comm pcomm)
{
    MPI_Status status;

    MPI_Recv(&m.dpdatalength, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprank, 1, mpi_get_type<size_t>(), source, tag, pcomm, &status);
    MPI_Recv(&m.dprowmajor, 1, mpi_get_type<bool>(), source, tag, pcomm, &status);

    //ignore this since here, only the reciever can decide where to store the memory.
    bool dadatevptr;
    MPI_Recv(&dadatevptr,1, mpi_get_type<bool>(), source, tag, pcomm,&status);

    MPI_Recv(m.dpextents,m.dprank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    MPI_Recv(m.dpstrides,m.dprank, mpi_get_type<size_t>(), source, tag, pcomm, &status);

    MPI_Recv(m.dpdata,sizeof(T)*m.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);

}



template<typename T>
inline  void Datastruct_MPI_Functions<T>::MPI_Irecv_datastruct_pdata(datastruct<T> &mds, const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Request request;
    MPI_Irecv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &request);
}

template<typename T>
inline  void Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(datastruct<T>& mds,const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);
}


template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_allreduce_row(datastruct<T>& mat,
        size_t row_index,
        MPI_Op op,
        MPI_Comm comm)
{
    MPI_Datatype row_t = mpi_row_type(mat, row_index);
    T* row_ptr = mat.dpdata + row_index * mat.dpstrides[0];
    MPI_Allreduce(MPI_IN_PLACE, row_ptr, 1, row_t, op, comm);
    MPI_Type_free(&row_t);
}

template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_allreduce_column(datastruct<T>& mat,
        size_t col_index,
        MPI_Op op,
        MPI_Comm comm)
{
    MPI_Datatype col_t = mpi_column_type(mat, col_index);
    T* col_ptr = mat.dpdata + col_index * mat.dpstrides[1];
    MPI_Allreduce(MPI_IN_PLACE, col_ptr, 1, col_t, op, comm);
    MPI_Type_free(&col_t);
}

template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_reduce_row(const datastruct<T>& send_mat,
        datastruct<T>& recv_mat,
        size_t row_index,
        int root, MPI_Op op, MPI_Comm comm)
{
    MPI_Datatype t = mpi_row_type(send_mat, row_index);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == root)
    {
        T* recv_ptr = recv_mat.dpdata + row_index * recv_mat.dpstrides[0];
        // in-place allowed if same buffer
        if (send_mat.dpdata == recv_mat.dpdata)
        {
            MPI_Reduce(MPI_IN_PLACE, recv_ptr, 1, t, op, root, comm);
        }
        else
        {
            T* send_ptr = send_mat.dpdata + row_index * send_mat.dpstrides[0];
            MPI_Reduce(send_ptr, recv_ptr, 1, t, op, root, comm);
        }
    }
    else
    {
        T* send_ptr = send_mat.dpdata + row_index * send_mat.dpstrides[0];
        MPI_Reduce(send_ptr, nullptr, 1, t, op, root, comm);
    }
    MPI_Type_free(&t);
}

template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_allreduce_subtensor(datastruct<T>& ds,
        const size_t* sub_extents,
        const size_t* sub_strides,   // unused by subarray, kept for symmetry
        const size_t* sub_offsets,
        MPI_Op op,
        MPI_Comm comm)
{
    MPI_Datatype st = mpi_subtensor_type(ds, sub_extents, sub_strides, sub_offsets);
    // ds.dpdata must point to the base of the parent; subarray/starts locate the view
    MPI_Allreduce(MPI_IN_PLACE, ds.dpdata, 1, st, op, comm);
    MPI_Type_free(&st);
}


template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_allreduce(datastruct<T>& ds,
        MPI_Op op,
        MPI_Comm comm)
{
    // Works for contiguous or strided via derived type
    MPI_Datatype dtype = mpi_datatype(ds);
    MPI_Allreduce(MPI_IN_PLACE, ds.dpdata, 1, dtype, op, comm);
    MPI_Type_free(&dtype);
}

template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_reduce(const datastruct<T>& send_ds,
        datastruct<T>& recv_ds,
        int root,
        MPI_Op op,
        MPI_Comm comm)
{
    // send_ds and recv_ds must describe the same shape/layout
    MPI_Datatype dtype = mpi_datatype(send_ds);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == root)
    {
        // allow in-place at root
        if (send_ds.dpdata == recv_ds.dpdata)
        {
            MPI_Reduce(MPI_IN_PLACE, recv_ds.dpdata, 1, dtype, op, root, comm);
        }
        else
        {
            MPI_Reduce(send_ds.dpdata, recv_ds.dpdata, 1, dtype, op, root, comm);
        }
    }
    else
    {
        MPI_Reduce(send_ds.dpdata, nullptr, 1, dtype, op, root, comm);
    }
    MPI_Type_free(&dtype);
}



template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_reduce_subtensor(const datastruct<T>& send_ds,
        datastruct<T>& recv_ds,
        const size_t* send_extents,
        const size_t* send_strides,
        const size_t* send_offsets,
        const size_t* recv_extents,
        const size_t* recv_strides,
        const size_t* recv_offsets,
        int root, MPI_Op op, MPI_Comm comm)
{
    MPI_Datatype send_t = mpi_subtensor_type(send_ds, send_extents, send_strides, send_offsets);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == root)
    {
        MPI_Datatype recv_t = mpi_subtensor_type(recv_ds, recv_extents, recv_strides, recv_offsets);
        // If recv view aliases the same memory the root used as send, allow in-place.
        if (send_ds.dpdata == recv_ds.dpdata &&
                std::memcmp(send_extents, recv_extents, sizeof(size_t)*recv_ds.dprank)==0 &&
                std::memcmp(send_offsets, recv_offsets, sizeof(size_t)*recv_ds.dprank)==0)
        {
            MPI_Reduce(MPI_IN_PLACE, recv_ds.dpdata, 1, recv_t, op, root, comm);
        }
        else
        {
            MPI_Reduce(send_ds.dpdata, recv_ds.dpdata, 1, recv_t, op, root, comm);
        }
        MPI_Type_free(&recv_t);
    }
    else
    {
        MPI_Reduce(send_ds.dpdata, nullptr, 1, send_t, op, root, comm);
    }
    MPI_Type_free(&send_t);
}


template<typename T>
inline void Datastruct_MPI_Functions<T>::mpi_reduce_column(const datastruct<T>& send_mat,
        datastruct<T>& recv_mat,
        size_t col_index,
        int root, MPI_Op op, MPI_Comm comm)
{
    MPI_Datatype t = mpi_column_type(send_mat, col_index);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == root)
    {
        T* recv_ptr = recv_mat.dpdata + col_index * recv_mat.dpstrides[1];
        if (send_mat.dpdata == recv_mat.dpdata)
        {
            MPI_Reduce(MPI_IN_PLACE, recv_ptr, 1, t, op, root, comm);
        }
        else
        {
            T* send_ptr = send_mat.dpdata + col_index * send_mat.dpstrides[1];
            MPI_Reduce(send_ptr, recv_ptr, 1, t, op, root, comm);
        }
    }
    else
    {
        T* send_ptr = send_mat.dpdata + col_index * send_mat.dpstrides[1];
        MPI_Reduce(send_ptr, nullptr, 1, t, op, root, comm);
    }
    MPI_Type_free(&t);
}

#endif



