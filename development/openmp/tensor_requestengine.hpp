#include "tensor_requestengine.h"
#include "datablock_mpifunctions.h"

TensorFlatIndexRequestSet::TensorFlatIndexRequestSet(int comm_size)
    : requests_by_owner(comm_size),
      lookup_by_owner(comm_size)
{}

ptrdiff_t TensorFlatIndexRequestSet::add(int owner_rank,ptrdiff_t global_index,ptrdiff_t global_block_index,ptrdiff_t local_flat_index)
{
    auto& lookup = lookup_by_owner[owner_rank];

    auto it = lookup.find(global_index);

    if (it != lookup.end())
        return it->second;

    auto& requests = requests_by_owner[owner_rank];

    const ptrdiff_t position =(ptrdiff_t)requests.size();

    requests.push_back({global_block_index,local_flat_index});

    lookup.emplace(global_index, position);

    return position;
}

ptrdiff_t TensorFlatIndexRequestSet::total_num_indices() const
{
    ptrdiff_t total = 0;

    for (const auto& requests : requests_by_owner)
        total += static_cast<ptrdiff_t>(requests.size());

    return total;
}



TensorFlatIndexReceivedRequests::TensorFlatIndexReceivedRequests(int comm_size)
    : requests_by_requester(comm_size)
{}



#pragma omp begin declare target
template<typename T>
T TensorFlatIndexData<T>::operator()(ptrdiff_t output_index) const
{
    T retval = pdata[output_index];

    if constexpr (is_complex<T>::value)
    {
        if (pconjugate)
            retval = std::conj(retval);
    }

    return retval;
}

#pragma omp end declare target




#pragma omp begin declare target
TensorFlatIndexLookup::TensorFlatIndexLookup(ptrdiff_t length)
{
    allocate(length);
}
#pragma omp end declare target

#pragma omp begin declare target
void TensorFlatIndexLookup::allocate(ptrdiff_t length)
{
    delete[] const_cast<ptrdiff_t*>(positions);

    this->length = length;

    ptrdiff_t total = length;

    ptrdiff_t* p = total > 0 ? new ptrdiff_t[total] : nullptr;

    positions = p;
}
#pragma omp end declare target

#pragma omp begin declare target
TensorFlatIndexLookup::~TensorFlatIndexLookup()
{
    delete[] const_cast<ptrdiff_t*>(positions);
}
#pragma omp end declare target

#pragma omp begin declare target
TensorFlatIndexLookup::TensorFlatIndexLookup(TensorFlatIndexLookup&& other) noexcept
 : positions(other.positions),length(other.length)
{
    other.positions = nullptr;
    other.length = 0;
}
#pragma omp end declare target

#pragma omp begin declare target
TensorFlatIndexLookup& TensorFlatIndexLookup::operator=(TensorFlatIndexLookup&& other) noexcept
{
    if (this != &other)
    {
        delete[] const_cast<ptrdiff_t*>(positions);

        positions = other.positions;
        length = other.length;


        other.positions = nullptr;
        other.length = 0;

    }

    return *this;
}
#pragma omp end declare target

#pragma omp begin declare target
ptrdiff_t& TensorFlatIndexLookup::operator()(ptrdiff_t output_index)
{
    return positions[output_index];
}
#pragma omp end declare target

#pragma omp begin declare target
ptrdiff_t TensorFlatIndexLookup::operator()(ptrdiff_t output_index) const
{
    return positions[output_index];
}
#pragma omp end declare target



TensorFlatIndexLookupBuilder::TensorFlatIndexLookupBuilder(int comm_size,ptrdiff_t output_length)
    : requests(comm_size),positions(output_length),owner_by_output(output_length, -1)
{}


ptrdiff_t TensorFlatIndexLookupBuilder::add(int owner_rank,ptrdiff_t global_index,ptrdiff_t global_block_index,ptrdiff_t local_flat_index)
{
    return requests.add(owner_rank,global_index,global_block_index,local_flat_index);
}

void TensorFlatIndexLookupBuilder::set_position(ptrdiff_t output_index,int owner_rank,ptrdiff_t local_position)
{
    positions(output_index) = local_position;
    owner_by_output[output_index] = owner_rank;
}

void TensorFlatIndexLookupBuilder::finalize_positions()
{
    const ptrdiff_t comm_size =(ptrdiff_t)requests.requests_by_owner.size();

    std::vector<ptrdiff_t> owner_offsets(comm_size, 0);

    ptrdiff_t offset = 0;
    #pragma omp unroll partial
    for (ptrdiff_t owner = 0; owner < comm_size; ++owner)
    {
        owner_offsets[owner] = offset;

        offset += (ptrdiff_t)requests.requests_by_owner[owner].size();
    }

    #pragma omp parallel for simd if(positions.length > 50)
    for (ptrdiff_t i = 0; i < positions.length; ++i)
    {
        const int owner = owner_by_output[i];
        if (owner >= 0)
        {
            positions(i) += owner_offsets[owner];
        }
    }
}




template<typename T>
inline void TensorRequestEngine::
MPI_Exchange_tensor_flat_index_data(const TensorFlatIndexReceivedRequests& incoming,const DistributedDataBlock<T>& source,
                                    DataBlockConfig location,TensorFlatIndexData<T>& result,MPI_Comm comm)
{
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);

    const auto& A = source.Dblockarray;

    const ptrdiff_t rank = A.ptensor_rank;

    int* sendcounts=new int[comm_size];
    int* recvcounts=new int[comm_size];

    #pragma omp parallel for simd if(parallel:comm_size>30)
    for (int r = 0; r < comm_size; ++r)
    {
        sendcounts[r] = (int)incoming.requests_by_requester[r].size();
    }

    MPI_Alltoall(sendcounts, 1, MPI_INT,recvcounts, 1, MPI_INT,comm);

    int* senddispls=new int[comm_size];
    int* recvdispls=new int[comm_size];

    ptrdiff_t total_send = 0;
    ptrdiff_t total_recv = 0;

    #pragma omp unroll partial
    for (int r = 0; r < comm_size; ++r)
    {
        senddispls[r] = (int)total_send;
        recvdispls[r] = (int)total_recv;

        total_send += sendcounts[r];
        total_recv += recvcounts[r];
    }


    ptrdiff_t* send_offsets=new ptrdiff_t[total_send];

    for (int requester = 0; requester < comm_size; ++requester)
    {
        ptrdiff_t position = senddispls[requester];

        for (const TensorBlockIndexRequest& request :incoming.requests_by_requester[requester])
        {
            const ptrdiff_t global_block =request.global_block_index;

            const ptrdiff_t local_block =source.pglobal_grid_index_to_local_block.find(global_block)->second;

            const ptrdiff_t local_offset =compute_offset_datablockarray(A.pextentsbuffer,A.pstridesbuffer,rank,local_block,request.local_flat_index);

            send_offsets[position++] =A.pblock_offsets[local_block] + local_offset;
        }
    }




    T* tempbuffer = nullptr;
    if(total_send>0)
        DataBlock_MPI_Functions::alloc_helper2<T>(A.conf,total_send,tempbuffer);


    if (total_send > 0)
    {
        if (A.conf.data_is_devptr)
        {
            const int devicenum = A.conf.devicenum;

            T* source_data = A.pdata;
            #if !defined(Unified_Shared_Memory)
                #pragma omp target enter data map(to: send_offsets[0:total_send])device(devicenum)
            #endif

            #pragma omp target teams distribute parallel for simd device(devicenum) is_device_ptr(source_data, tempbuffer)
            for (ptrdiff_t i = 0; i < total_send; ++i)
            {
                tempbuffer[i] = source_data[send_offsets[i]];
            }
            #if !defined(Unified_Shared_Memory)
                #pragma omp target exit data map(delete: send_offsets[0:total_send])device(devicenum)
            #endif
        }
        else
        {
            #pragma omp parallel for simd
            for (ptrdiff_t i = 0; i < total_send; ++i)
            {
                tempbuffer[i] = A.pdata[send_offsets[i]];
            }
        }
    }



    result.pdata = nullptr;
    result.pdatalength = total_recv;
    result.config = location;
    result.pconjugate = A.pconjugate;

    if(result.pdatalength>0)
        DataBlock_MPI_Functions::alloc_helper2<T>(result.config,result.pdatalength,result.pdata);



    T dummy{};

    const T* send_ptr =total_send > 0 ? tempbuffer : &dummy;

    T* recv_ptr =total_recv > 0 ? result.pdata : &dummy;

    MPI_Alltoallv(send_ptr,sendcounts,senddispls,mpi_get_type<T>(),
                  recv_ptr,recvcounts,recvdispls,mpi_get_type<T>(),comm);

    if(total_send>0)
        DataBlock_MPI_Functions::free_helper2<T>(A.conf,total_send,tempbuffer);


    delete[]sendcounts;
    delete[] recvcounts;

    delete[] senddispls;
    delete[] recvdispls;

    if(total_send>0)
        delete[] send_offsets;

    return;
}



inline void TensorRequestEngine::MPI_Exchange_tensor_flat_index_requests(const TensorFlatIndexRequestSet& outgoing,TensorFlatIndexReceivedRequests& incoming,MPI_Comm comm)
{
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);


    int* sendcounts = new int[comm_size];
    int* recvcounts = new int[comm_size];
    #pragma omp parallel for simd if(parallel:comm_size>30)
    for (int r = 0; r < comm_size; ++r)
    {
        sendcounts[r] =(int)(outgoing.requests_by_owner[r].size());
    }



    MPI_Alltoall(sendcounts, 1, MPI_INT,recvcounts, 1, MPI_INT,comm);




    int* senddispls = new int[comm_size];
    int* recvdispls = new int[comm_size];

    ptrdiff_t total_send = 0;
    ptrdiff_t total_recv = 0;

    #pragma omp unroll partial
    for (int r = 0; r < comm_size; ++r)
    {
        senddispls[r] = (int)total_send;
        recvdispls[r] = (int)total_recv;
        total_send += sendcounts[r];
        total_recv += recvcounts[r];
    }



    ptrdiff_t* send_buffer = new ptrdiff_t[2 * total_send];



    for (int r = 0; r < comm_size; ++r)
    {
        ptrdiff_t position = 2 * senddispls[r];

        for (const TensorBlockIndexRequest& request :outgoing.requests_by_owner[r])
        {
            send_buffer[position++] = request.global_block_index;
            send_buffer[position++] = request.local_flat_index;
        }
    }


    ptrdiff_t* recv_buffer = new ptrdiff_t[2 * total_recv];


    int* sendcounts_elements = new int[comm_size];
    int* recvcounts_elements = new int[comm_size];

    int* senddispls_elements = new int[comm_size];
    int* recvdispls_elements = new int[comm_size];

    #pragma omp parallel for simd if(parallel:comm_size>30)
    for (int r = 0; r < comm_size; ++r)
    {
        sendcounts_elements[r] = 2 * sendcounts[r];
        recvcounts_elements[r] = 2 * recvcounts[r];

        senddispls_elements[r] = 2 * senddispls[r];
        recvdispls_elements[r] = 2 * recvdispls[r];
    }

    ptrdiff_t dummy{};

    MPI_Alltoallv(total_send > 0 ? send_buffer : &dummy,sendcounts_elements,senddispls_elements,mpi_get_type<ptrdiff_t>(),
                  total_recv > 0 ? recv_buffer : &dummy,recvcounts_elements,recvdispls_elements,mpi_get_type<ptrdiff_t>(),comm);


    incoming.requests_by_requester.resize(comm_size);

    for (int r = 0; r < comm_size; ++r)
    {
        auto& requests = incoming.requests_by_requester[r];

        requests.resize(recvcounts[r]);

        ptrdiff_t position = 2 * recvdispls[r];

        for (int i = 0; i < recvcounts[r]; ++i)
        {
            requests[i].global_block_index = recv_buffer[position++];
            requests[i].local_flat_index = recv_buffer[position++];
        }
    }


    delete[] sendcounts;
    delete[] recvcounts;

    delete[] senddispls;
    delete[] recvdispls;

    delete[] send_buffer;
    delete[] recv_buffer;

    delete[] sendcounts_elements;
    delete[] recvcounts_elements;

    delete[] senddispls_elements;
    delete[] recvdispls_elements;

    return;
}
