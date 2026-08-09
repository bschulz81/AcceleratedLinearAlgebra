
#ifndef DATABLOCKSTRUCTS
#define DATABLOCKSTRUCTS

#if defined(Unified_Shared_Memory)
#pragma omp requires unified_shared_memory
#else
#pragma omp requires unified_address
#endif



#pragma omp begin declare target
enum class DataBlockObject
{
    Scalar,
    Vector,
    Matrix,
    Tensor
};
#pragma omp end declare target

#pragma omp begin declare target
struct ComputeMetadata
{
    bool ComputeStrides=true;
    bool ComputeLength=true;
};
#pragma omp end declare target

#pragma omp begin declare target
struct DataBlockConfig
{
    bool dprowmajor=true;
    bool pmemmap=        false;
    bool data_is_devptr= false;
    int devicenum =     -INT_MAX;

};
#pragma omp end declare target


#pragma omp begin declare target
DataBlockObject object_type(
    ptrdiff_t rank,
    const auto& extents)
{
    if (rank == 1)
    {
        if (abs(extents[0]) == 1)
            return DataBlockObject::Scalar;

        return DataBlockObject::Vector;
    }

    if (rank == 2)
    {
        if (abs(extents[0]) == 1 &&
            abs(extents[1]) == 1)
            return DataBlockObject::Scalar;

        if (abs(extents[0]) == 1 ||
            abs(extents[1]) == 1)
            return DataBlockObject::Vector;

        return DataBlockObject::Matrix;
    }

    if (rank > 2)
        return DataBlockObject::Tensor;

    return DataBlockObject::Scalar;
}
#pragma omp end declare target



struct ManagedDataBlockConfig
{
    bool dprowmajor=true;
    bool memmap=false;
    bool data_ondevice=false;
    bool default_device=true;
    int devicenum = -INT_MAX;


    inline DataBlockConfig Get_DataBlockConfig() const
    {
        return DataBlockConfig
        {
            .dprowmajor    = this->dprowmajor,
            .pmemmap=this->memmap,
            .data_is_devptr = this->data_ondevice,
            .devicenum     = this->devicenum
        };
    }

    inline static ManagedDataBlockConfig SetConfig( bool defaultdevice, const DataBlockConfig& config)
    {
        return ManagedDataBlockConfig
        {
            .dprowmajor     = config.dprowmajor,
            .memmap         =config.pmemmap,
            .data_ondevice  = config.data_is_devptr,
            .default_device = defaultdevice,
            .devicenum      = config.devicenum
        };
    }
};


#endif
