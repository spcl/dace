#ifndef __DACE_MEMORYPOOL_H
#define __DACE_MEMORYPOOL_H


template <bool IS_GPU>
class MemoryPool
{
private:
    size_t m_size, m_offset;
    void* m_mem;
    size_t block_size;

public:
    MemoryPool(size_t membytes, size_t size){
        block_size = size;
        m_mem = nullptr;
        m_size = 0;
        m_offset = 0;

        if (IS_GPU) {
            cudaMalloc(&m_mem, membytes);
        } else {
            m_mem = malloc(membytes);
            if (m_mem == NULL){
                printf("ERROR: MemoryPool allocation failed");
                exit(1);
            }
        }
        m_size = membytes;
        m_offset = 0;
    }

    ~MemoryPool(){
    }

    void* Alloc(size_t size)
    {
        if (m_size == 0){
            printf("ERROR: Please reserve Memory before allocating!");
            exit(1);
        }

        if (size > m_size - m_offset)
        {
            printf("\n\n Warning: bad context allocation, exiting. \n\n");
            exit(1);
        }

        size_t offset = m_offset;
        int int_size = (size / block_size + 1) * block_size;
        m_offset += int_size;
        return (void*)((char*)m_mem + offset);
    }

    void Dealloc(void* ptr)
    {
        return;
    }

    void Replace(size_t size, void* ptr)
    {
        return;
    }

};

#endif // __DACE_MEMORYPOOL_H