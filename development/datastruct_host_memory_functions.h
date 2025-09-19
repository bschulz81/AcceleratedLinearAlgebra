#ifndef DATASTRUCTHOSTMEMHELPERS
#define DATASTRUCTHOSTMEMHELPERS
#include <filesystem>
#include <string.h>
#include <sys/mman.h>

template<typename T>
class Datastruct_Host_Memory_Functions
{
public:
    inline static void free_copy(datastruct<T>&m, bool with_memmap);
    inline static datastruct<T> alloc_data_copy_strides_extents(size_t datalength,bool rowmajor, size_t rank, size_t*extents,size_t *strides, bool with_memmap);

    inline static T*  alloc_data_ptr(size_t length,bool create_memmap);
    inline static void free_data_ptr(T*pdata,size_t datalength,bool with_memmap);

    inline static T* create_temp_mmap(const size_t array_size);
    inline static void delete_temp_mmap(T* mmap_ptr,const size_t array_size);
};

template<typename T>
T* Datastruct_Host_Memory_Functions<T>::create_temp_mmap(const size_t array_size)
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
void Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(T* mmap_ptr,const size_t array_size)
{
    size_t file_size = array_size * sizeof(T);
    if (munmap(mmap_ptr, file_size) == -1)
    {
        perror("munmap");
    }
}





template<typename T>
void Datastruct_Host_Memory_Functions<T>::free_data_ptr(T*pdata,size_t datalength,bool with_memmap)
{
    if(pdata!=nullptr)
    {
        if (with_memmap)
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(pdata,datalength);
        else
            free(pdata);
    }
}


template<typename T>
T*  Datastruct_Host_Memory_Functions<T>::alloc_data_ptr(size_t length,bool create_memmap)
{

    if (create_memmap)
        return Datastruct_Host_Memory_Functions<T>::create_temp_mmap(length);
    else
        return (T*) malloc(sizeof(T)*length);

}


template<typename T>
datastruct<T> Datastruct_Host_Memory_Functions<T>::alloc_data_copy_strides_extents(size_t datalength,bool rowmajor, size_t rank, size_t*extents,size_t *strides, bool with_memmap)
{
    size_t*pextents;
    size_t*pstrides;
    T* pdata;
    pextents=(size_t*) malloc(sizeof(size_t)*rank);
    memcpy(pextents,extents,sizeof(size_t)*rank);

    pstrides=(size_t*) malloc(sizeof(size_t)*rank);
    memcpy(pstrides,strides,sizeof(size_t)*rank);

    if (with_memmap)
        pdata=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(datalength);
    else
        pdata=(T*)malloc(sizeof(T)*datalength);


    return datastruct<T>(pdata,datalength,rowmajor,rank,pextents,pstrides,false, false,false);
}



template<typename T>
void Datastruct_Host_Memory_Functions<T>::free_copy(datastruct<T>&m, bool with_memmap)
{

    free(m.dpextents);
    free(m.dpstrides);

    if (with_memmap)
        Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(m.dpdata,m.dpdatalength);
    else
        free(m.dpdata);
}


#endif
