#ifndef DATABLOCKHOSTMEMHELPERS
#define DATABLOCKHOSTMEMHELPERS
#include <filesystem>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


#include "datablock.h"


class Host_Memory_Functions
{
public:
    template<typename T>
    inline static void free_copy(DataBlock<T>&m);
    template<typename T>
    inline static DataBlock<T> alloc_data_copy_strides_extents(ptrdiff_t datalength, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides,  DataBlockConfig conf);
    template<typename T>
    inline static DataBlock<T> alloc_data_strides_extents(ptrdiff_t datalength, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides,  DataBlockConfig conf);
    template<typename T>
    inline static T*  alloc_data_ptr(ptrdiff_t length,bool create_memmap);
    template<typename T>
    inline static void free_data_ptr(T*&pdata,ptrdiff_t datalength,bool with_memmap);
    template<typename T>
    inline static T* create_temp_mmap(const ptrdiff_t array_size);
    template<typename T>
    inline static void delete_temp_mmap(T* &mmap_ptr,const ptrdiff_t array_size);
};

template<typename T>
T* Host_Memory_Functions::create_temp_mmap(const ptrdiff_t array_size)
{
    ptrdiff_t file_size = array_size * sizeof(T);


    FILE* tmpf = tmpfile();
    if (!tmpf)
    {
        perror("tmpfile");
        return NULL;
    }


    int fd = fileno(tmpf);
    if (fd == -1)
    {
        perror("fileno");
        fclose(tmpf);
        return NULL;
    }


    if (ftruncate(fd, file_size) == -1)
    {
        perror("ftruncate");
        fclose(tmpf);
        return NULL;
    }


    T* mmap_ptr = (T*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED)
    {
        perror("mmap");
        fclose(tmpf);
        return NULL;
    }


    fclose(tmpf);


    return mmap_ptr;
}

template<typename T>
void Host_Memory_Functions::delete_temp_mmap(T* &mmap_ptr,const ptrdiff_t array_size)
{
    ptrdiff_t file_size = array_size * sizeof(T);
    if (mmap_ptr!=nullptr)
    if (munmap(mmap_ptr, file_size) == -1)
    {
        perror("munmap");
    }
}





template<typename T>
void Host_Memory_Functions::free_data_ptr(T*&pdata,ptrdiff_t datalength,bool with_memmap)
{
    if(pdata!=nullptr)
    {
        if (with_memmap)
            Host_Memory_Functions::delete_temp_mmap(pdata,datalength);
        else
            if(pdata!=nullptr)
                omp_free(pdata,omp_default_mem_alloc);
    }
}


template<typename T>
T* Host_Memory_Functions::alloc_data_ptr(ptrdiff_t length,bool create_memmap)
{

    if (create_memmap)
        return Host_Memory_Functions::create_temp_mmap<T>(length);
    else
        return (T*) omp_alloc(sizeof(T)*length,omp_default_mem_alloc);

}


template<typename T>
DataBlock<T> Host_Memory_Functions::alloc_data_copy_strides_extents(ptrdiff_t datalength, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides,DataBlockConfig conf)
{
    ptrdiff_t*pextents;
    ptrdiff_t*pstrides;
    T* pdata;
    pextents=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);
    memcpy(pextents,extents,sizeof(ptrdiff_t)*rank);

    pstrides=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);
    memcpy(pstrides,strides,sizeof(ptrdiff_t)*rank);

    if (conf.pmemmap)
        pdata=Host_Memory_Functions::create_temp_mmap<T>(datalength);
    else
        pdata=(T*)omp_alloc(sizeof(T)*datalength,omp_default_mem_alloc);

    conf.data_is_devptr=false;
    conf.devicenum=-INT_MAX;
    return DataBlock<T>(pdata,datalength,rank,pextents,pstrides,conf);
}


template<typename T>
DataBlock<T> Host_Memory_Functions::alloc_data_strides_extents(ptrdiff_t datalength, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides, DataBlockConfig conf)
{
    ptrdiff_t*pextents;
    ptrdiff_t*pstrides;
    T* pdata;
    pextents=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);

    pstrides=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);

    if (conf.pmemmap)
        pdata=Host_Memory_Functions::create_temp_mmap<T>(datalength);
    else
        pdata=(T*)omp_alloc(sizeof(T)*datalength,omp_default_mem_alloc);


    conf.data_is_devptr=false;
    conf.devicenum=-INT_MAX;
    return DataBlock<T>(pdata,datalength,rank,pextents,pstrides,conf);
}


template<typename T>
void Host_Memory_Functions::free_copy(DataBlock<T>&m)
{
    if(m.dpextents!=nullptr)
    free(m.dpextents);
    if(m.dpstrides!=nullptr)
    free(m.dpstrides);

    if (m.dpconfig.pmemmap)
        Host_Memory_Functions::delete_temp_mmap(m.dpdata,m.dpdatalength);
    else
        if(m.dpdata!=nullptr)
            omp_free(m.dpdata,omp_default_mem_alloc);
}



#endif
