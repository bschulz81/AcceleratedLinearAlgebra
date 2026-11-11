
#ifndef TENSORREQUESTENGINE
#define TENSORREQUESTENGINE

#include <unordered_map>
#include <vector>
#include "mpi.h"
#include "datablock_mpifunctions.h"

struct TensorBlockIndexRequest
{
    ptrdiff_t global_block_index;
    ptrdiff_t local_flat_index;
};

struct TensorFlatIndexRequestSet
{
    std::vector<std::vector<TensorBlockIndexRequest>> requests_by_owner;

    std::vector<std::unordered_map<ptrdiff_t, ptrdiff_t>> lookup_by_owner;

    explicit TensorFlatIndexRequestSet(int comm_size);

    ptrdiff_t add(int owner_rank,ptrdiff_t global_index,ptrdiff_t global_block_index,ptrdiff_t local_flat_index);

    ptrdiff_t total_num_indices() const;
};


struct TensorFlatIndexReceivedRequests
{
    std::vector<std::vector<TensorBlockIndexRequest>>requests_by_requester;

    explicit TensorFlatIndexReceivedRequests(int comm_size);
};






#pragma omp begin declare target
template<typename T>
class TensorFlatIndexData
{
public:
    T* pdata=nullptr;
    ptrdiff_t pdatalength=0;
    DataBlockConfig config;
    bool pconjugate=false;
    T operator()(ptrdiff_t output_index) const;
};
#pragma omp end declare target





#pragma omp begin declare target
class TensorFlatIndexLookup
{
public:
    TensorFlatIndexLookup()=default;
    explicit TensorFlatIndexLookup(ptrdiff_t length);
    TensorFlatIndexLookup(const ptrdiff_t* positions,ptrdiff_t length);

    void allocate(ptrdiff_t length);

    ~TensorFlatIndexLookup();

    TensorFlatIndexLookup(const TensorFlatIndexLookup&) = delete;
    TensorFlatIndexLookup& operator=(const TensorFlatIndexLookup&) = delete;
    TensorFlatIndexLookup(TensorFlatIndexLookup&& other) noexcept;
    TensorFlatIndexLookup& operator=(TensorFlatIndexLookup&& other) noexcept;


   ptrdiff_t& operator()(ptrdiff_t output_index);
ptrdiff_t operator()(ptrdiff_t output_index) const;

    ptrdiff_t* positions = nullptr;
    ptrdiff_t length = 0;
};
#pragma omp end declare target

class TensorFlatIndexLookupBuilder
{
public:
    TensorFlatIndexRequestSet requests;
    TensorFlatIndexLookup positions;

    std::vector<int> owner_by_output;

    explicit TensorFlatIndexLookupBuilder(int comm_size,ptrdiff_t output_length);

    ptrdiff_t add(int owner_rank,ptrdiff_t global_index,ptrdiff_t global_block_index,ptrdiff_t local_flat_index);
    void set_position(ptrdiff_t output_index,int owner_rank,ptrdiff_t local_position);
    void finalize_positions();

};

class TensorRequestEngine
{

public:

    template<typename T>
    inline static TensorBlockLocation get_tensor_block_location(const DistributedDataBlock<T>& source,ptrdiff_t global_flat_index);

    inline static void MPI_Exchange_tensor_flat_index_requests(const TensorFlatIndexRequestSet& outgoing,TensorFlatIndexReceivedRequests& incoming,MPI_Comm comm);

    template<typename T>
    inline static void MPI_Exchange_tensor_flat_index_data(const TensorFlatIndexReceivedRequests& incoming,const DistributedDataBlock<T>& source,
                                                            DataBlockConfig location,TensorFlatIndexData<T>& result,MPI_Comm comm);
};

#include "tensor_requestengine.hpp"
#endif
