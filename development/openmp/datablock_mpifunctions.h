#ifndef DATABLOCK_MPIFUNCTIONS
#define DATABLOCK_MPIFUNCTIONS

#include <mpi.h>


#include <complex>
#include <type_traits>
#include <cstdint>
#include <cstring>

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

template<typename T>
class DataBlock_MPI_Functions
{
public:


    inline static void MPI_Bcast_DataBlock (DataBlock<T> &db,MPI_Comm com, int rootrank);
    inline static void MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,bool with_memmap,bool ondevice, int devicenum,MPI_Comm com, int rootrank);

    inline static void MPI_Scatter_matrix_as_rows_alloc(DataBlock<T>& recv_db,bool memmap, bool ondevice,int devicenum,
            MPI_Comm comm, int rootrank,  const DataBlock<T>* send_db = nullptr);

    inline static void MPI_Scatter_matrix_to_columns_alloc(DataBlock<T>& recv_db,
            MPI_Comm comm,bool memmap, bool ongpu, int devnum,
            int rootrank,const DataBlock<T>* send_db=nullptr);

    inline static void MPI_gather_matrix_from_rows_alloc(    const DataBlock<T>& send_db,    MPI_Comm comm,    int rootrank,
            DataBlock<T>* recv_db = nullptr,bool rowmajor=true,bool memmap=false, bool ongpu=false, int devicenum=-1);

    inline static void MPI_Gather_matrix_from_columns_alloc(   const DataBlock<T>& send_db,   MPI_Comm comm,    int rootrank,
            DataBlock<T>* recv_db = nullptr,  bool rowmajor=true,bool memmap=false, bool ongpu=false, int devicenum=-1);

    inline static void MPI_Scatter_matrix_to_submatrices_alloc( size_t br,  size_t bc,
            DataBlock<T>& recv_db,bool memmap, bool ongpu, int devicenum,
            MPI_Comm comm,   int rootrank,  const DataBlock<T>* send_db = nullptr);

    inline static void MPI_Scatter_subtensor_to_subtensors_alloc(
        const size_t* sub_extents, DataBlock<T>& recv_db, bool memmap, bool ondevice, int devicenum,
        MPI_Comm comm, int rootrank, const DataBlock<T>* send_db = nullptr);

    inline static void MPI_Gather_matrix_from_submatrices_alloc(
        const DataBlock<T>& send_db, MPI_Comm comm, int rootrank,DataBlock<T>* recv_db = nullptr,
        bool rowmajor=true, int M=0, int N=0,bool memmap=false, bool ongpu=false, int devicenum=-1 );

    inline static void MPI_Gather_tensor_from_subtensors_alloc(  const DataBlock<T>& send_db,  MPI_Comm comm,  int rootrank,
            bool rowmajor=true, size_t* global_extents=nullptr,
            DataBlock<T>* recv_db = nullptr, bool memmap=false, bool ongpu=false, int devicenum=-1);

    inline static DataBlock<T> MPI_Recv_alloc_DataBlock(bool with_memmap,bool ondevice, int devicenum, const int source,const  int tag, MPI_Comm pcomm);

    inline static void MPI_Free_DataBlock(DataBlock<T>&m);

    inline static void MPI_Send_DataBlock(DataBlock<T> &m,const int dest, const int tag, MPI_Comm pcomm);
    inline static void MPI_Recv_DataBlock(DataBlock<T>& m, const int source,const  int tag, MPI_Comm pcomm);

    inline static void MPI_Isend_DataBlock_pdata(DataBlock<T> &m,const int dest,const  int tag,const MPI_Comm pcomm);
    inline static void MPI_Irecv_DataBlock_pdata(DataBlock<T> &mds, const int source, const int tag,const  MPI_Comm pcomm);

    inline static void MPI_Recv_DataBlock_pdata(DataBlock<T>& mds,const int source, const int tag,const  MPI_Comm pcomm);
    inline static void MPI_Send_DataBlock_pdata(DataBlock<T> &m,const int dest,const int tag,const MPI_Comm pcomm);

protected:
    inline static void alloc_helper(bool &memmap,bool& ondevice, int& devnum, size_t rank,size_t datalength,size_t* pextents,size_t *pstrides,T *pdata);
};


template <typename T>
void DataBlock_MPI_Functions<T>::alloc_helper(bool &memmap,bool &ondevice, int& devicenum, size_t rank,size_t datalength,size_t* pextents,size_t *pstrides,T *pdata)
{
    pextents= (size_t*)malloc(sizeof(size_t)*rank);
    pstrides= (size_t*)malloc(sizeof(size_t)*rank);

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


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_DataBlock (DataBlock<T> &db,MPI_Comm com, int rootrank)
{
    int rank;
    MPI_Comm_rank(com, &rank);
    MPI_Bcast (&db.dpdatalength, 1,  mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (&db.dprank,1,  mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (&db.dprowmajor,1,  mpi_get_type<bool>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, MPI_COMM_WORLD);
}

template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Bcast_alloc_DataBlock (DataBlock<T> &db,bool memmap,bool ondevice, int devicenum,MPI_Comm com, int rootrank)
{
    int rank;
    MPI_Comm_rank(com, &rank);
    MPI_Bcast (&db.dpdatalength,1,  mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (&db.dprank,    1,    mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (&db.dprowmajor,1,  mpi_get_type<bool>(), rootrank, MPI_COMM_WORLD);

    if (rank != rootrank)
    {
        alloc_helper(memmap,ondevice,devicenum,db.dprank,db.dpdatalength,db.dpextents,db.dpstrides,db.dpdata);
        db.devptr_devicenum=devicenum;
        db.dpdata_is_devptr=ondevice;
        db.devptr_former_hostptr=nullptr;
    }
    MPI_Bcast (db.dpextents, db.dprank,  mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (db.dpstrides, db.dprank,  mpi_get_type<size_t>(), rootrank, MPI_COMM_WORLD);
    MPI_Bcast (db.dpdata, db.dpdatalength,  mpi_get_type<T>(), rootrank, MPI_COMM_WORLD);
}


template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_matrix_as_rows_alloc(DataBlock<T>& recv_db,bool memmap, bool ondevice,int devicenum,
        MPI_Comm comm,   int rootrank,  const DataBlock<T>* send_db )
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t rows = 0, cols = 0;

    if (rank == rootrank)
    {
        rows = send_db->dpextents[0];
        cols = send_db->dpextents[1];
    }

    MPI_Bcast(&rows, 1, mpi_get_type<size_t>(), rootrank, comm);
    MPI_Bcast(&cols, 1, mpi_get_type<size_t>(), rootrank, comm);

    bool receives = (rank < rows);

    T* recv_buffer = nullptr;

    if (receives)
    {
        size_t* ext=nullptr;
        size_t* str=nullptr;

        alloc_helper(memmap,ondevice,devicenum,1,cols,ext,str,recv_buffer);
        ext[0] = cols;
        str[0] = 1;

        recv_db = DataBlock<T>(recv_buffer,cols, true,1, ext, str,ondevice,devicenum);
    }
    else
    {
        recv_db.dpdata = nullptr;
        recv_db.dpextents = nullptr;
        recv_db.dpstrides = nullptr;
    }

    MPI_Datatype tmp, row_type=MPI_DATATYPE_NULL;

    if (rank == rootrank)
    {
        MPI_Type_vector(
            cols,
            1,
            send_db->dpstrides[1],
            mpi_get_type<T>(),
            &tmp);

        MPI_Type_create_resized(
            tmp,
            0,
            send_db->dpstrides[0] * sizeof(T),
            &row_type);

        MPI_Type_commit(&row_type);
        MPI_Type_free(&tmp);
    }

    int* sendcounts=nullptr, *displs=nullptr;

    if (rank == rootrank)
    {
        sendcounts=new int[size],
        displs=new int[size];

        for (int i = 0; i < size; ++i)
        {
            if (i < rows)
            {
                sendcounts[i] = 1;
                displs[i]     = i;
            }
            else
            {
                sendcounts[i] = 0;
                displs[i]     = 0;
            }
        }
    }

    MPI_Scatterv(
        rank == rootrank ? send_db->dpdata : nullptr,
        sendcounts,
        displs,
        row_type,
        receives ? recv_buffer : nullptr,
        receives ? cols : 0,
        mpi_get_type<T>(),
        rootrank,
        comm);

    if (rank == rootrank)
    {
        MPI_Type_free(&row_type);
        delete[] sendcounts;
        delete[] displs;
    }
}




template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_gather_matrix_from_rows_alloc(
    const DataBlock<T>& send_db,
    MPI_Comm comm,
    int rootrank,
    DataBlock<T>* recv_db,bool rowmajor,bool memmap, bool ondevice, int devicenum)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // --- Determine if this rank contributes ---
    int contributes = (send_db.dpdata != nullptr) ? 1: 0;

    // --- Gather row lengths on root ---
    size_t local_cols = contributes ? send_db.dpextents[0] : 0;

    size_t* all_cols = nullptr;

    if (rank == rootrank)
        all_cols = new size_t[size];

    MPI_Gather(&local_cols, 1, mpi_get_type<size_t>(),
               all_cols, 1, mpi_get_type<size_t>(),
               rootrank, comm);

    size_t rows = 0;
    size_t cols = 0;

    if (rank == rootrank)
    {
        rows = 0;

        for (int i = 0; i < size; ++i)
            if (all_cols[i] > 0)
            {
                rows++;
                cols = all_cols[0];
            }

        if (recv_db)
        {
            size_t* ext = nullptr;
            size_t* str =nullptr;

            T*recv_buffer=nullptr;
            alloc_helper(memmap,ondevice,devicenum,2,rows*cols,ext,str,recv_buffer);
            ext[0] = rows;
            ext[1] = cols;

            str[0] = rowmajor ? cols : 1;
            str[1] = rowmajor ? 1    : rows;

            recv_db = DataBlock<T>(recv_buffer,rows*cols, rowmajor,2, ext, str,ondevice,devicenum);
        }
    }

    // --- Prepare datatype on root ---
    MPI_Datatype tmp, row_type=MPI_DATATYPE_NULL;

    if (rank == rootrank)
    {
        MPI_Datatype tmp;

        MPI_Type_vector(
            cols,
            1,
            recv_db->dpstrides[1],
            mpi_get_type<T>(),
            &tmp);

        MPI_Type_create_resized(
            tmp,
            0,
            recv_db->dpstrides[0] * sizeof(T),
            &row_type);

        MPI_Type_commit(&row_type);
        MPI_Type_free(&tmp);
    }

    int* recvcounts = nullptr;
    int* displs     = nullptr;


    if (rank == rootrank)
    {
        recvcounts = new int[size];
        displs     = new int[size];

        int offset = 0;

        for (int i = 0; i < size; ++i)
        {
            if (all_cols[i] > 0)
            {
                recvcounts[i] = 1;
                displs[i]     = offset++;
            }
            else
            {
                recvcounts[i] = 0;
                displs[i]     = 0;
            }
        }
    }

    MPI_Gatherv(
        contributes ? send_db.dpdata : nullptr,
        contributes ? local_cols : 0,
        mpi_get_type<T>(),
        rank == rootrank ? recv_db->dpdata : nullptr,
        recvcounts,
        displs,
        row_type,
        rootrank,
        comm);

    if (rank == rootrank)
    {
        MPI_Type_free(&row_type);
        delete[] recvcounts;
        delete[] displs;
        delete[] all_cols;
    }

}





template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_matrix_to_columns_alloc(DataBlock<T>& recv_db,
        MPI_Comm comm,bool memmap, bool ondevice, int devicenum,
        int rootrank,const DataBlock<T>* send_db)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t rows = 0, cols = 0;

    if (rank == rootrank)
    {
        rows = send_db->dpextents[0];
        cols = send_db->dpextents[1];
    }

    MPI_Bcast(&rows, 1, mpi_get_type<size_t>(), rootrank, comm);
    MPI_Bcast(&cols, 1, mpi_get_type<size_t>(), rootrank, comm);

    bool receives = (rank < cols);

    T* recv_buffer = nullptr;

    if (receives)
    {
        size_t* ext = nullptr;
        size_t* str = nullptr;


        T* recv_buffer=nullptr;
        alloc_helper(memmap,ondevice,devicenum,1,rows,ext,str,recv_buffer);
        ext[0] = rows;
        str[0] = 1;

        recv_db = DataBlock<T>(recv_buffer,rows, true,1, ext, str,ondevice,devicenum);
    }
    else
    {
        recv_db.dpdata=nullptr;
        recv_db.dpextents=nullptr;
        recv_db.dpstrides=nullptr;
    }




    MPI_Datatype tmp, col_type=MPI_DATATYPE_NULL;


    if(rank==rootrank)
    {
        MPI_Type_vector(
            rows,
            1,
            send_db->dpstrides[0],
            mpi_get_type<T>(),
            &tmp);

        MPI_Type_create_resized(
            tmp,
            0,
            send_db->dpstrides[1] * sizeof(T),
            &col_type);

        MPI_Type_commit(&col_type);
        MPI_Type_free(&tmp);
    }

    int* sendcounts, *displs;

    if (rank == rootrank)
    {
        sendcounts=new int[size],
        displs=new int[size];

        for (int i = 0; i < size; ++i)
        {
            if (i < cols)
            {
                sendcounts[i] = 1;
                displs[i] = i ;
            }
            else
            {
                sendcounts[i] = 0;
                displs[i] = 0;
            }
        }
    }

    MPI_Scatterv(
        rank == rootrank ? send_db->dpdata : nullptr,
        rank == rootrank ? sendcounts : nullptr,
        rank == rootrank ? displs : nullptr,
        col_type,
        receives ? recv_buffer : nullptr,
        receives ? rows : 0,
        mpi_get_type<T>(),
        rootrank,
        comm);
    if(rank==rootrank)
    {
        MPI_Type_free(&col_type);
        delete[] sendcounts;
        delete[] displs;
    }
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_matrix_from_columns_alloc(
    const DataBlock<T>& send_db,
    MPI_Comm comm,
    int rootrank,
    DataBlock<T>* recv_db,  bool rowmajor,bool memmap, bool ondevice, int devicenum)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int contributes = (send_db.dpdata != nullptr) ? 1 : 0;

    size_t local_rows = contributes ? send_db.dpextents[0] : 0;

    size_t* all_rows = nullptr;

    if (rank == rootrank)
        all_rows = new size_t[size];

    MPI_Gather(&local_rows, 1, mpi_get_type<size_t>(),
               all_rows, 1, mpi_get_type<size_t>(),
               rootrank, comm);

    size_t rows = 0;
    size_t cols = 0;

    if (rank == rootrank)
    {
        cols = 0;
        for (int i = 0; i < size; ++i)
        {
            if (all_rows[i] > 0)
            {
                cols++;
                rows = all_rows[0];
            }
        }

        if (recv_db)
        {
            size_t* ext =nullptr;
            size_t* str = nullptr;


            T* recv_buffer=nullptr;
            alloc_helper(memmap,ondevice,devicenum,2,rows*cols,ext,str,recv_buffer);
            ext[0] = rows;
            ext[1] = cols;

            str[0] = rowmajor ? cols : 1;
            str[1] = rowmajor ? 1    : rows;

            recv_db = DataBlock<T>(recv_buffer,rows*cols, rowmajor,2, ext, str,ondevice,devicenum);
        }
    }

    MPI_Datatype tmp, col_type= MPI_DATATYPE_NULL;

    if (rank == rootrank)
    {
        MPI_Type_vector(
            rows,
            1,
            recv_db->dpstrides[0],
            mpi_get_type<T>(),
            &tmp);

        MPI_Type_create_resized(
            tmp,
            0,
            recv_db->dpstrides[1] * sizeof(T),
            &col_type);

        MPI_Type_commit(&col_type);
        MPI_Type_free(&tmp);
    }

    int* recvcounts = nullptr;
    int* displs     = nullptr;

    if (rank == rootrank)
    {
        recvcounts = new int[size];
        displs     = new int[size];

        int offset = 0;

        for (int i = 0; i < size; ++i)
        {
            if (all_rows[i] > 0)
            {
                recvcounts[i] = 1;
                displs[i]     = offset++;
            }
            else
            {
                recvcounts[i] = 0;
                displs[i]     = 0;
            }
        }
    }

    MPI_Gatherv(
        contributes ? send_db.dpdata : nullptr,
        contributes ? local_rows : 0,
        mpi_get_type<T>(),
        rank == rootrank ? recv_db->dpdata : nullptr,
        recvcounts,
        displs,
        col_type,
        rootrank,
        comm);

    if (rank == rootrank)
    {
        MPI_Type_free(&col_type);
        delete[] recvcounts;
        delete[] displs;
        delete[] all_rows;
    }
}




template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Scatter_matrix_to_submatrices_alloc(
    size_t br,
    size_t bc,
    DataBlock<T>& recv_db,bool memmap, bool ondevice, int devicenum,
    MPI_Comm comm,
    int rootrank,
    const DataBlock<T>* send_db )
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t M=0, N=0;
    bool rmajor;

    if(rank==rootrank)
    {
        M = send_db->dpextents[0];
        N = send_db->dpextents[1];
        rmajor=send_db->dpstrides[1]<send_db->dpstrides[0];
    }

    MPI_Bcast(&M,1,mpi_get_type<size_t>(),rootrank,comm);
    MPI_Bcast(&N,1,mpi_get_type<size_t>(),rootrank,comm);
    MPI_Bcast(&rmajor,1,mpi_get_type<bool>(),rootrank,comm);

    size_t grid_r = M / br;
    size_t grid_c = N / bc;

    bool receives = (rank < grid_r * grid_c);

    T* recv_buffer = nullptr;


    if(receives)
    {
        size_t *ext=nullptr;
        size_t *str=nullptr;

        alloc_helper(memmap,ondevice,devicenum,2,br*bc,ext,str,recv_buffer);

        ext[0]=br;
        ext[1]=bc;
        str[0]=rmajor? bc:1;
        str[1]=rmajor? 1:br;
        recv_db = DataBlock<T>(recv_buffer,br*bc, rmajor,2, ext, str,ondevice,devicenum);
    }
    else
    {
        recv_db.dpdata=nullptr;
        recv_db.dpextents=nullptr;
        recv_db.dpstrides=nullptr;
    }

    MPI_Datatype tmp, block_type=MPI_DATATYPE_NULL;

    if(rank==rootrank)
    {
        MPI_Type_vector(
            br,
            bc,
            rmajor?send_db->dpstrides[0]: send_db->dpstrides[1],
            mpi_get_type<T>(),
            &tmp);

        MPI_Type_create_resized(
            tmp,
            0,
            sizeof(T),
            &block_type);

        MPI_Type_commit(&block_type);
        MPI_Type_free(&tmp);
    }

    int* sendcounts, *displs;
    if(rank==rootrank)
    {
        sendcounts=new int[size],
        displs=new int[size];
        for(int i=0; i<size; i++)
        {
            if(i < grid_r*grid_c)
            {
                size_t bi = i / grid_c;
                size_t bj = i % grid_c;
                sendcounts[i] = 1;
                displs[i] =
                    bi * br * (rmajor? send_db->dpstrides[0]: send_db->dpstrides[1])
                    + bj * bc * (rmajor?  send_db->dpstrides[1]: send_db->dpstrides[0]);
            }
            else
            {
                sendcounts[i]=0;
                displs[i]=0;
            }
        }
    }

    MPI_Scatterv(
        rank==rootrank?send_db->dpdata:nullptr,
        rank==rootrank?sendcounts:nullptr,
        rank==rootrank?displs:nullptr,
        block_type,
        receives?recv_buffer:nullptr,
        receives?br*bc:0,
        mpi_get_type<T>(),
        rootrank,
        comm);

    if(rank==rootrank)
    {
        MPI_Type_free(&block_type);
        delete[] sendcounts;
        delete[] displs;
    }
}




template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_matrix_from_submatrices_alloc(
    const DataBlock<T>& send_db,
    MPI_Comm comm,
    int rootrank,DataBlock<T>* recv_db,bool rowmajor, int M, int N,bool memmap, bool ondevice, int devicenum )
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int contributes = (send_db.dpdata != nullptr) ? 1 : 0;


    size_t* all_nodes = nullptr;

    size_t local_gridsize_m = contributes ? send_db.dpextents[0] : 0;
    size_t local_gridsize_n = contributes ? send_db.dpextents[1] : 0;

    size_t *all_grids_m,*all_grids_n;

    if (rank == rootrank)
    {
        all_grids_m = new size_t[size];
        all_grids_n = new size_t[size];

    }

    MPI_Gather(&local_gridsize_m, 1, mpi_get_type<size_t>(),
               all_grids_m, 1, mpi_get_type<size_t>(),
               rootrank, comm);

    MPI_Gather(&local_gridsize_n, 1, mpi_get_type<size_t>(),
               all_grids_n, 1, mpi_get_type<size_t>(),
               rootrank, comm);

    MPI_Bcast(&rowmajor,1,mpi_get_type<bool>(),rootrank, comm);

    int filled_gridsize_m=0, filled_gridsize_n= 0;
    if(rank == rootrank && recv_db)
    {
        for (int i = 0; i < size; ++i)
        {
            if (all_grids_m[i] > 0 && all_grids_n[i] > 0)
            {
                filled_gridsize_m = all_grids_m[i];
                filled_gridsize_n = all_grids_n[i];
                break;
            }
        }

        size_t* ext = nullptr;

        size_t* str =nullptr;

        T* recv_buffer=nullptr;
        alloc_helper(memmap,ondevice,devicenum,2,N*M,ext,str,recv_buffer);
        ext[0]=M;
        ext[1]=N;
        str[0]= rowmajor ?N:1;
        str[1]=rowmajor ? 1:M;
        recv_db = DataBlock<T>(recv_buffer,N*M, rowmajor,2, ext, str,ondevice,devicenum);

    }


    MPI_Datatype tmp, block_type = MPI_DATATYPE_NULL;

    if(rank == rootrank)
    {
        MPI_Type_vector(
            filled_gridsize_m,
            filled_gridsize_n,
            rowmajor ? recv_db->dpstrides[0]
            : recv_db->dpstrides[1],
            mpi_get_type<T>(),
            &tmp);

        MPI_Type_create_resized(tmp, 0, sizeof(T), &block_type);
        MPI_Type_commit(&block_type);
        MPI_Type_free(&tmp);
    }



    int* recvcounts = nullptr;
    int* displs     = nullptr;

    if(rank == rootrank)
    {
        size_t grid_r = M / filled_gridsize_m;
        size_t grid_c = N / filled_gridsize_n;

        recvcounts = new int[size];
        displs     = new int[size];

        for(int i=0; i<size; i++)
        {
            if(i < grid_r*grid_c)
            {
                size_t bi = i / grid_c;
                size_t bj = i % grid_c;

                recvcounts[i] = 1;

                displs[i] =
                    bi * filled_gridsize_m * (rowmajor?  recv_db->dpstrides[0]: recv_db->dpstrides[1])
                    + bj * filled_gridsize_n * (rowmajor?  recv_db->dpstrides[1]: recv_db->dpstrides[0]);
            }
            else
            {
                recvcounts[i] = 0;
                displs[i] = 0;
            }
        }
    }

    MPI_Gatherv(
        contributes ? send_db.dpdata : nullptr,
        contributes ? local_gridsize_m*local_gridsize_n : 0,
        mpi_get_type<T>(),
        rank == rootrank ? recv_db->dpdata : nullptr,
        recvcounts,
        displs,
        block_type,
        rootrank,
        comm);

    if(rank == rootrank)
    {
        MPI_Type_free(&block_type);
        delete[] recvcounts;
        delete[] displs;
    }
}



template<typename T>
inline void DataBlock_MPI_Functions<T>:: MPI_Scatter_subtensor_to_subtensors_alloc(
    const size_t* sub_extents,
    DataBlock<T>& recv_db, bool memmap, bool ondevice, int devicenum,
    MPI_Comm comm,
    int rootrank,
    const DataBlock<T>* send_db )
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t R = 0;

    if (rank == rootrank)
        R = send_db->dprank;

    MPI_Bcast(&R, 1, mpi_get_type<size_t>(), rootrank, comm);

    size_t* global_ext=new size_t[R];
    size_t *global_str=new size_t[R];

    if (rank == rootrank)
    {
        for (size_t d = 0; d < R; ++d)
        {
            global_ext[d] = send_db->dpextents[d];
            global_str[d] = send_db->dpstrides[d];
        }
    }

    MPI_Bcast(global_ext, R, mpi_get_type<size_t>(), rootrank, comm);

    // ---- detect layout ----
    bool rowmajor = false;

    if (rank == rootrank)
        rowmajor = (send_db->dpstrides[R-1] == 1);

    MPI_Bcast(&rowmajor, 1, mpi_get_type<bool>(), rootrank, comm);

    // ---- compute block grid ----
    size_t * grid=new size_t [R];
    size_t total_blocks = 1;

    for (size_t d = 0; d < R; ++d)
    {
        grid[d] = global_ext[d] / sub_extents[d];
        total_blocks *= grid[d];
    }

    bool receives = (rank < total_blocks);



    if (receives)
    {
        size_t data_len = 1;
        for (size_t d = 0; d < R; ++d)
            data_len *= sub_extents[d];

        size_t* ext = nullptr;
        size_t* str = nullptr;
        T* recv_buffer=nullptr;

        alloc_helper(memmap,ondevice,devicenum,R,data_len,ext,str,recv_buffer);

        for (size_t d = 0; d < R; ++d)
            ext[d] = sub_extents[d];

        if (rowmajor)
        {
            str[R-1] = 1;
            for (int d = R-2; d >= 0; --d)
                str[d] = str[d+1] * sub_extents[d+1];
        }
        else
        {
            str[0] = 1;
            for (size_t d = 1; d < R; ++d)
                str[d] = str[d-1] * sub_extents[d-1];
        }


        recv_db = DataBlock<T>(recv_buffer,data_len, rowmajor,R, ext, str,ondevice,devicenum);
    }
    else
    {
        recv_db.dpdata = nullptr;
        recv_db.dpextents = nullptr;
        recv_db.dpstrides = nullptr;
    }


    MPI_Datatype subarray_type = MPI_DATATYPE_NULL;

    if (rank == rootrank)
    {
        int* sizes=new int[R];
        int* subs=new int[R];
        int* starts=new int[R];

        for (size_t d = 0; d < R; ++d)
        {
            starts[d]=0;
            sizes[d] = static_cast<int>(global_ext[d]);
            subs[d]  = static_cast<int>(sub_extents[d]);
        }

        MPI_Datatype tmp_type;

        MPI_Type_create_subarray(
            R,
            sizes,
            subs,
            starts,
            rowmajor ? MPI_ORDER_C : MPI_ORDER_FORTRAN,
            mpi_get_type<T>(),
            &tmp_type);


        MPI_Type_create_resized(
            tmp_type,
            0,
            sizeof(T),
            &subarray_type);

        MPI_Type_commit(&subarray_type);
        MPI_Type_free(&tmp_type);

        delete[] sizes;
        delete[] subs;
        delete[] starts;
    }


    int* sendcounts=nullptr;
    int* displs=nullptr;

    if (rank == rootrank)
    {
        sendcounts=new int[size];
        displs =new int[size];
        for (int p = 0; p < size; ++p)
        {
            if (p < total_blocks)
            {
                size_t *block_index=new size_t[R];
                size_t tmp = p;

                for (int d = R-1; d >= 0; --d)
                {
                    block_index[d] = tmp % grid[d];
                    tmp /= grid[d];
                }

                size_t offset = 0;

                for (size_t d = 0; d < R; ++d)
                    offset += block_index[d] *
                              sub_extents[d] *
                              global_str[d];

                sendcounts[p] = 1;
                displs[p] = static_cast<int>(offset);

                delete[] block_index;
            }
            else
            {
                sendcounts[p] = 0;
                displs[p] = 0;
            }
        }
    }


    MPI_Scatterv(
        rank == rootrank ? send_db->dpdata : nullptr,
        sendcounts,
        displs,
        subarray_type,
        receives ? recv_db.dpdata : nullptr,
        receives ? recv_db.dpdatalength : 0,
        mpi_get_type<T>(),
        rootrank,
        comm);

    if (rank == rootrank)
    {
        MPI_Type_free(&subarray_type);
        delete[] sendcounts;
        delete[] displs;
    }

    delete[] grid;
    delete[] global_ext;
    delete[] global_str;
}



template<typename T>
inline void DataBlock_MPI_Functions<T>::MPI_Gather_tensor_from_subtensors_alloc(
    const DataBlock<T>& send_db,
    MPI_Comm comm,
    int rootrank,
    bool rowmajor,
    size_t* global_extents,
    DataBlock<T>* recv_db, bool memmap, bool ondevice, int devicenum)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    bool contributes = (send_db.dpdata != nullptr);

    size_t R = contributes ? send_db.dprank : 0;

    MPI_Allreduce(MPI_IN_PLACE, &R, 1,
                  mpi_get_type<size_t>(),
                  MPI_MAX, comm);


    MPI_Bcast(&rowmajor, 1,
              mpi_get_type<bool>(),
              rootrank, comm);

    if (rank!= rootrank)
        global_extents = new size_t[R];


    MPI_Bcast(global_extents, R,
              mpi_get_type<size_t>(),
              rootrank, comm);


    size_t* local_ext = new size_t[R];

    for (size_t d = 0; d < R; ++d)
        local_ext[d] = contributes? send_db.dpextents[d]:0 ;

    size_t* sub_ext = new size_t[R];

    MPI_Reduce(local_ext, sub_ext,R, mpi_get_type<size_t>(), MPI_MAX, rootrank, comm);

    // ---- ROOT allocates global tensor ----
    size_t* grid = new size_t[R];
    size_t total_blocks = 1;

    if (rank == rootrank && recv_db)
    {

        size_t* ext =nullptr;
        size_t* str = nullptr;
        T* recv_buffer=nullptr;

        size_t total_size = 1;
        for (size_t d = 0; d < R; ++d)
        {
            total_size *= global_extents[d];
        }

        alloc_helper(memmap,ondevice,devicenum,R,total_size,ext,str,recv_buffer);

        for (size_t d = 0; d < R; ++d)
        {
            ext[d] = global_extents[d];
        }
        if (rowmajor)
        {
            str[R-1] = 1;
            for (int d = R-2; d >= 0; --d)
                str[d] = str[d+1] * global_extents[d+1];
        }
        else
        {
            str[0] = 1;
            for (size_t d = 1; d < R; ++d)
                str[d] = str[d-1] * global_extents[d-1];
        }


        recv_db = DataBlock<T>(recv_buffer,total_size, rowmajor,R, ext, str,ondevice,devicenum);

        for (size_t d = 0; d < R; ++d)
        {
            grid[d] = global_extents[d] / sub_ext[d];
            total_blocks *= grid[d];
        }
    }

    // ---- datatype ----
    MPI_Datatype block_type = MPI_DATATYPE_NULL;

    if (rank == rootrank)
    {
        int* sizes  = new int[R];
        int* subs   = new int[R];
        int* starts = new int[R];


        for (size_t d = 0; d < R; ++d)
        {
            starts[d]=0;
            sizes[d] = static_cast<int>(global_extents[d]);
            subs[d]  = static_cast<int>(sub_ext[d]);
        }


        MPI_Datatype tmp;

        MPI_Type_create_subarray(
            R,
            sizes,
            subs,
            starts,
            rowmajor ? MPI_ORDER_C : MPI_ORDER_FORTRAN,
            mpi_get_type<T>(),
            &tmp);

        MPI_Type_create_resized(tmp, 0, sizeof(T), &block_type);
        MPI_Type_commit(&block_type);
        MPI_Type_free(&tmp);


        delete[] starts;
        delete[] sizes;
        delete[]subs;
    }

    // ---- recvcounts + displs ----
    int* recvcounts = nullptr;
    int* displs     = nullptr;

    if (rank == rootrank)
    {
        recvcounts = new int[size];
        displs     = new int[size];

        size_t* block_index = new size_t[R];

        for (int p = 0; p < size; ++p)
        {
            if (p < total_blocks)
            {
                size_t tmp = p;

                for (int d = R-1; d >= 0; --d)
                {
                    block_index[d] = tmp % grid[d];
                    tmp /= grid[d];
                }

                size_t offset = 0;
                for (size_t d = 0; d < R; ++d)
                    offset += block_index[d] *
                              sub_ext[d] *
                              recv_db->dpstrides[d];

                recvcounts[p] = 1;
                displs[p]     = static_cast<int>(offset);
            }
            else
            {
                recvcounts[p] = 0;
                displs[p]     = 0;
            }
        }

        delete[] block_index;
    }

    MPI_Gatherv(
        contributes ? send_db.dpdata : nullptr,
        contributes ? send_db.dpdatalength : 0,
        mpi_get_type<T>(),
        rank == rootrank ? recv_db->dpdata : nullptr,
        recvcounts,
        displs,
        block_type,
        rootrank,
        comm);

    if (rank == rootrank)
    {
        MPI_Type_free(&block_type);
        delete[] recvcounts;
        delete[] displs;
    }
    else
    {
        delete[] global_extents;
    }

    delete[] grid;
    delete[] sub_ext;

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
inline  void DataBlock_MPI_Functions<T>::MPI_Isend_DataBlock_pdata(DataBlock<T> &m,const int dest,const  int tag,const MPI_Comm pcomm)
{
    MPI_Request request;
    MPI_Isend(m.dpdata,sizeof(T)* m.dpdatalength, MPI_BYTE, dest, tag, pcomm,&request);
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
void DataBlock_MPI_Functions<T>::MPI_Free_DataBlock(DataBlock<T>&m)
{

    if(m.dpdata!=nullptr)
    {
#if defined(Unified_Shared_Memory)
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
inline  void DataBlock_MPI_Functions<T>::MPI_Irecv_DataBlock_pdata(DataBlock<T> &mds, const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Request request;
    MPI_Irecv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &request);
}

template<typename T>
inline  void DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(DataBlock<T>& mds,const int source, const int tag,const  MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(mds.dpdata,sizeof(T)* mds.dpdatalength, MPI_BYTE, source, tag, pcomm, &status);
}


#endif



