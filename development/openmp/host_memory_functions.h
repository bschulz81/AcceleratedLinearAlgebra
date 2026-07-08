#ifndef DATABLOCKHOSTMEMHELPERS
#define DATABLOCKHOSTMEMHELPERS
#include <filesystem>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

class Host_Memory_Functions
{
public:
    template<typename T>
    inline static void free_copy(DataBlock<T>&m, bool with_memmap);
    template<typename T>
    inline static DataBlock<T> alloc_data_copy_strides_extents(size_t datalength,bool rowmajor, size_t rank, size_t*extents,size_t *strides, bool with_memmap,bool conjugate);
    template<typename T>
    inline static T*  alloc_data_ptr(size_t length,bool create_memmap);
    template<typename T>
    inline static void free_data_ptr(T*&pdata,size_t datalength,bool with_memmap);
    template<typename T>
    inline static T* create_temp_mmap(const size_t array_size);
    template<typename T>
    inline static void delete_temp_mmap(T* &mmap_ptr,const size_t array_size);
};

template<typename T>
T* Host_Memory_Functions::create_temp_mmap(const size_t array_size)
{
    size_t file_size = array_size * sizeof(T);

    // Create a temporary file using std::tmpfile()
    FILE* tmpf = tmpfile();
    if (!tmpf)
    {
        perror("tmpfile");
        return NULL;
    }

    // Get the file descriptor from the FILE*
    int fd = fileno(tmpf);
    if (fd == -1)
    {
        perror("fileno");
        fclose(tmpf);
        return NULL;
    }

    // Resize the file to the required size
    if (ftruncate(fd, file_size) == -1)
    {
        perror("ftruncate");
        fclose(tmpf);
        return NULL;
    }

    // Memory map the file
    T* mmap_ptr = (T*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED)
    {
        perror("mmap");
        fclose(tmpf);
        return NULL;
    }

    // Close the FILE* but keep the memory mapping valid
    fclose(tmpf);

    // Return the pointer to the mapped memory
    return mmap_ptr;
}

template<typename T>
void Host_Memory_Functions::delete_temp_mmap(T* &mmap_ptr,const size_t array_size)
{
    size_t file_size = array_size * sizeof(T);
    if (mmap_ptr!=nullptr)
    if (munmap(mmap_ptr, file_size) == -1)
    {
        perror("munmap");
    }
}





template<typename T>
void Host_Memory_Functions::free_data_ptr(T*&pdata,size_t datalength,bool with_memmap)
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
T* Host_Memory_Functions::alloc_data_ptr(size_t length,bool create_memmap)
{

    if (create_memmap)
        return Host_Memory_Functions::create_temp_mmap<T>(length);
    else
        return (T*) omp_alloc(sizeof(T)*length,omp_default_mem_alloc);

}


template<typename T>
DataBlock<T> Host_Memory_Functions::alloc_data_copy_strides_extents(size_t datalength,bool rowmajor, size_t rank, size_t*extents,size_t *strides, bool with_memmap,bool conjugate)
{
    size_t*pextents;
    size_t*pstrides;
    T* pdata;
    pextents=(size_t*) malloc(sizeof(size_t)*rank);
    memcpy(pextents,extents,sizeof(size_t)*rank);

    pstrides=(size_t*) malloc(sizeof(size_t)*rank);
    memcpy(pstrides,strides,sizeof(size_t)*rank);

    if (with_memmap)
        pdata=Host_Memory_Functions::create_temp_mmap<T>(datalength);
    else
        pdata=(T*)omp_alloc(sizeof(T)*datalength,omp_default_mem_alloc);

    return DataBlock<T>(pdata,datalength,rowmajor,rank,pextents,pstrides,false,-1,conjugate);
}



template<typename T>
void Host_Memory_Functions::free_copy(DataBlock<T>&m, bool with_memmap)
{
    if(m.dpextents!=nullptr)
    free(m.dpextents);
    if(m.dpstrides!=nullptr)
    free(m.dpstrides);

    if (with_memmap)
        Host_Memory_Functions::delete_temp_mmap(m.dpdata,m.dpdatalength);
    else
        if(m.dpdata!=nullptr)
            omp_free(m.dpdata,omp_default_mem_alloc);
}


#endif
