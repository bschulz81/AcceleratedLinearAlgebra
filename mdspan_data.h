#ifndef MDSPAN_DATAH
#define MDSPAN_DATAH

#include "mdspan_omp.h"


template<typename T>
class Datastruct_Functions;



template <typename T, typename Container>
class mdspan_data: public mdspan<T,Container>
{
public:
    friend class Datastruct_Functions<T>;

    mdspan_data() {};
    mdspan_data(const size_t datalength, const bool rowm, const bool memmap, const Container& extents, const Container& strides);
    mdspan_data(const bool rowm,const  bool memmap, const Container& extents, const Container& strides);
    mdspan_data(const bool rowm,const  bool memmap, const Container& extents);
    mdspan_data(const bool rowm, const bool memmap, const size_t rows, const size_t cols);

    mdspan_data<T, Container> subspan_copy(const Container& offsets, const Container& sub_extents, bool with_memmap) ;
    mdspan_data<T, Container> subspan_copy(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols, bool with_memmap);
    mdspan_data<T, Container> copy(bool memmap);

    mdspan_data<T, Container> subspan_view(const Container& offsets, const Container& sub_extents) ;
    mdspan_data<T, Container> subspan_view(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols);
    mdspan_data<T, Container>column_view(const size_t col_index);
    mdspan_data<T, Container>row_view(const size_t row_index);
    mdspan_data<T, Container>column_copy(const size_t col_index,bool with_memmap);
    mdspan_data<T, Container>row_copy(const size_t row_index, bool with_memmap );
    mdspan_data<T, Container>transpose_view();
    mdspan_data<T, Container>transpose_copy( bool with_memmap);

    inline mdspan_data<T, Container>&operator=(const datastruct<T> & other);

protected:
    shared_ptr<T> p_refctr;
    bool pwith_memmap=false;
    void allocate_data(const bool memmap,const size_t datalength);
};

template<typename T, typename Tag>
using mdspan_data_t = mdspan<T, typename container_for_tag<Tag>::type>;



template <typename T, typename Container>
mdspan_data<T, Container>::mdspan_data( const bool rowm,const bool memmap, const Container& extents, const Container& strides )
    : mdspan<T,Container>(nullptr, rowm,  extents,strides)
{
    allocate_data(memmap,this->dpdatalength);
}



template <typename T, typename Container>
mdspan_data<T, Container>::mdspan_data(const bool rowm,const bool memmap,const  Container& extents)
    :  mdspan<T,Container>(nullptr, rowm,  extents)
{
    allocate_data(memmap,this->dpdatalength);
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::copy(bool memmap)
{

    mdspan_data<T,Container>m(this->rowm,memmap,this->pextents,this->pstrides);
    memcpy(m.pdata,this->pd,this->pdatalength);
    m.pis_offloaded=this->pis_offloaded;
    for(auto& p : this->pis_offloaded)
    {
        if(p.second==true)
        {
            Datastruct_Functions<T>::create_in_struct(*m,p.first);
        }
    }
    return m;
}



template <typename T, typename Container>
mdspan_data<T, Container>::mdspan_data(const bool rowm,const bool memmap, const size_t rows, const size_t cols)
    :   mdspan<T,Container>(nullptr, rowm,  rows,cols)
{
    allocate_data(memmap,this->dpdatalength);
}


template <typename T, typename Container>
void mdspan_data<T, Container>::allocate_data(bool memmap, size_t dl)
{
    if (memmap)
    {
        this->dpdata = Datastruct_Host_Memory_Functions<T>::create_temp_mmap(dl);
        pwith_memmap = true;
    }
    else
    {
        this->dpdata = new T[dl];
        pwith_memmap = false;
    }

    p_refctr=shared_ptr<T>(this->dpdata,[this](T* p)
    {
        for(auto& p : this->pis_offloaded)
        {
            if(p.second==true)
            {
                Datastruct_GPU_Memory_Functions<T>::exit_struct(*this,p.first);
                p.second=false;
            }
        }
        if (pwith_memmap==true)
             Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(p,this->dpdatalength);
        else
            delete[] p;
    });
}
template <typename T, typename Container>
mdspan_data<T, Container>&mdspan_data<T, Container>::operator=(const datastruct<T> & other)
{
    this->transfer_extents_and_strides(other);
    if (other.pdata!=this->pdata)
    {
        datastruct<T>::operator=(other);
        p_refctr=shared_ptr<T>(this->p_refctr);
    }
    else
    {
        datastruct<T>::operator=(other);
    }
    return *this;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::subspan_copy(const Container& offsets, const Container& sub_extents,bool with_memmap)
{
    size_t size=1;
    #pragma omp parallel for simd reduction (*:size)
    for (size_t i=0; i< sub_extents.size(); i++)
    {
        size*=sub_extents[i];
    }
    mdspan_data<T, Container>  sub(this->pdrowmajor,with_memmap,size);
    sub= mdspan<T, Container>::subspan(offsets, sub_extents, this->dpdata);
    return sub;
}


template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container> ::subspan_view(const Container& offsets, const Container& sub_extents)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::subspan(offsets, sub_extents, nullptr);
    sub.addrefcounter(this);
    return sub;
}



template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::subspan_copy(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,tile_rows*tile_cols);
    sub= mdspan<T, Container>::subspan(row, col, tile_rows, tile_cols, this->dpdata);
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::column_copy(const size_t col_index,bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,this->pdextents[0]);
    sub= mdspan<T, Container>::column_copy(col_index, this->dpdata);
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::row_copy(const size_t row_index,bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,this->pdextents[1]);
    sub= mdspan<T, Container>::row_copy(row_index, this->dpdata);
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::transpose_copy(bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,this->dpextents[1]*this->dpextents[0]);
    sub= mdspan<T, Container>::transpose_copy(this->dpdata);
    return sub;
}






template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::subspan_view(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::subspan(row, col, tile_rows, tile_cols, nullptr);
    sub.addrefcounter(this);
    return sub;
}


template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::column_view(const size_t col_index)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::column_view(col_index, nullptr);
    sub.addrefcounter(this);
    return sub;
}
template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::row_view(const size_t row_index)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::row_view(row_index, nullptr);
    sub.addrefcounter(this);
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::transpose_view()
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::transpose_view();
    sub.addrefcounter(this);
    return sub;
}


#endif

