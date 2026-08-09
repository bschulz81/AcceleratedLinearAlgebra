#ifndef DATABLOCKGPUMEMHELPERS
#define DATABLOCKGPUMEMHELPERS


#include "datablock.h"

template<typename T>
class DataBlockArray;
template<typename T>
class BlockedDataView;


class GPU_Memory_Functions
{
public:

    template<typename T>
    class OffloadHelper
    {
    protected:
        bool pupdate_host;
        DataBlock<T> &pdL;
        int pdevicenum;
    public:
        OffloadHelper(DataBlock<T>& dL, int devicenum, bool just_alloc, bool update_host_on_exit);

        OffloadHelper(const DataBlock<T>& dL, int devicenum, bool just_alloc);

        ~OffloadHelper();

        OffloadHelper(const OffloadHelper&) = delete;
        OffloadHelper& operator=(const OffloadHelper&) = delete;
    };

    template<typename T>
    class OffloadHelperConst
    {
    protected:
        const DataBlock<T> &pdL;
        int pdevicenum;
    public:
        OffloadHelperConst(const DataBlock<T>& dL, int devicenum, bool just_alloc);

        ~OffloadHelperConst();

        OffloadHelperConst(const OffloadHelperConst&) = delete;
        OffloadHelperConst& operator=(const OffloadHelperConst&) = delete;
    };

    template <typename T>
    class DataBlockdpdataoffloader
    {
    public:
        DataBlockdpdataoffloader(const DataBlock<T>& block, int devicenum, bool is_output = false);

        DataBlockdpdataoffloader(DataBlock<T>& block, int devicenum, bool is_output = false);

        ~DataBlockdpdataoffloader();


        const DataBlock<T>& get() const;


        DataBlock<T>& get_mutable();


        DataBlockdpdataoffloader(const DataBlockdpdataoffloader&) = delete;
        DataBlockdpdataoffloader& operator=(const DataBlockdpdataoffloader&) = delete;

    private:
        DataBlock<T> m_block;
        bool m_copied;
        bool m_is_output;
    };



    template<typename T>
    class BlockedDataViewOffloadHelper
    {
    protected:
        const BlockedDataView<T> &pdL;
        int pdevicenum;
    public:
        BlockedDataViewOffloadHelper(const BlockedDataView<T>& dL, int devicenum);


        ~BlockedDataViewOffloadHelper();

        BlockedDataViewOffloadHelper(const BlockedDataViewOffloadHelper&) = delete;
        BlockedDataViewOffloadHelper& operator=(const BlockedDataViewOffloadHelper&) = delete;

    };

    template<typename T>
    class DataBlockArrayOffloadHelper
    {
    protected:
        bool pupdate_host;
        DataBlockArray<T> &pArr;
        int pdevicenum;
    public:
        DataBlockArrayOffloadHelper(DataBlockArray<T>& arr, int devicenum, bool just_alloc, bool update_host_on_exit);

        ~DataBlockArrayOffloadHelper();

        DataBlockArrayOffloadHelper(const DataBlockArrayOffloadHelper&) = delete;
        DataBlockArrayOffloadHelper& operator=(const DataBlockArrayOffloadHelper&) = delete;
    };

    template<typename T>
    class DataBlockArrayOffloadHelperConst
    {
    protected:
        const DataBlockArray<T> &pArr;
        int pdevicenum;
    public:
        DataBlockArrayOffloadHelperConst(const DataBlockArray<T>& arr, int devicenum);

        ~DataBlockArrayOffloadHelperConst();

        DataBlockArrayOffloadHelperConst(const DataBlockArrayOffloadHelperConst&) = delete;
        DataBlockArrayOffloadHelperConst& operator=(const DataBlockArrayOffloadHelperConst&) = delete;
    };

    template<typename T>
    inline static void create_out(DataBlockArray<T>& arr, int devicenum);
    template<typename T>
    inline static void create_in(DataBlockArray<T>& arr, int devicenum);
    template<typename T>
    inline static void create_in(const DataBlockArray<T>& arr, int devicenum);
    template<typename T>
    inline static void update_host(DataBlockArray<T>& arr, int devicenum);
    template<typename T>
    inline static void release(const DataBlockArray<T>& arr, int devicenum);




    template<typename T>
    inline static bool update_device(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static bool update_host(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static bool update_device_data(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static void update_device_metadata(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static bool update_host_data(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static void update_host_metadata(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static void set_data_to_device_ptr(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static void set_data_to_host_ptr(DataBlock<T>& dL,int devicenum);
    template<typename T>
    inline static void create_out(DataBlock<T>& dA,int devicenum);
    template<typename T>
    inline static void create_in(DataBlock<T>& dA,int devicenum);
    template<typename T>
    inline static void exit(DataBlock<T> &dA,int devicenum);
    template<typename T>
    inline static void release(DataBlock<T> &dA,int devicenum);
    template<typename T>
    inline static void create_in(const DataBlock<T>& dA,int devicenum);
    template<typename T>
    inline static void exit(const DataBlock<T> &dA,int devicenum);
    template<typename T>
    inline static void release(const DataBlock<T> &dA,int devicenum);

    template<typename T>
    inline static void create_in_blocked(const BlockedDataView<T>& dA,int devicenum);
    template<typename T>
    inline static void exit_blocked(const BlockedDataView<T> &dA,int devicenum);
    template<typename T>
    inline static void release_blocked(const BlockedDataView<T> &dA,int devicenum);

    template<typename T>
    inline static bool copy_data_to_device_set_devptr(DataBlock<T>&dL,int devicenum);
    template<typename T>
    inline static bool alloc_data_to_device_set_devptr(DataBlock<T>&dL,int devicenum);
    template<typename T>
    inline static bool copy_data_to_host_set_host_ptr(DataBlock<T>&dL);
    template<typename T>
    inline static bool free_device_data_set_host_ptr(DataBlock<T>&dL);

    template<typename T>
    inline static void copy_data_to_host_ptr(DataBlock<T>& dL);
    template<typename T>
    inline static void copy_data_to_device_ptr(DataBlock<T>& dL);
    template<typename T>
    inline static T* alloc_device_ptr(ptrdiff_t length, int devicenum);
    template<typename T>
    inline static void free_device_ptr(T* &deviceptr, int devicenum);

    template<typename T>
    inline static T* alloc_data_device_ptr(ptrdiff_t datalength,bool with_memmap, int devicenum);
    template<typename T>
    inline static void free_data_device_ptr(T*&pdata,ptrdiff_t datalength,bool with_memmap, int devicenum);

    template<typename T>
    inline static DataBlock<T> alloc_data_copy_strides_extents_device(ptrdiff_t datalength,bool rowmajor, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides,  DataBlockConfig conf);
    template<typename T>
    inline static DataBlock<T>alloc_data_strides_extents_device(ptrdiff_t datalength,bool rowmajor, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides,  DataBlockConfig conf);
    template<typename T>
    inline static void free_copy_device(DataBlock<T>&m);

    template<typename T>
    inline static bool is_on_gpu(const DataBlock<T>&m,const int devicenum);

     template<typename T>
    inline static bool is_on_gpu(const DataBlockArray<T>&m,const int devicenum);

    template<typename T>
    inline static bool is_on_gpu_ptr(const T* pdata,const int devicenum);
    template<typename T>
    inline static bool is_on_gpu_ptr(const ptrdiff_t* p,const int devicenum);

};

#include "gpu_memory_functions.hpp"

#endif


