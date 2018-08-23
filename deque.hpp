#include <string>
#include <strstream>
#include <ctime>
#include <atomic>
#include <vector>
#include <random>
#include <future>
#include <thread>

std::atomic_int c_push_fail{0};
std::atomic_int c_pop_fail{0};

// t_buffer_size must be a power of 2
template <typename T, int t_buffer_size>
struct Deque {
    std::atomic_uint m_head, m_tail, size;
    std::vector< std::atomic<T*> > m_data;
    int m_mask;

    T* empty_token;

    Deque() : m_head(0), m_tail(0), size(0), m_data(t_buffer_size),
        m_mask(t_buffer_size - 1),
        empty_token(new T) /* could be nullptr as well */
    {
        assert(m_data.size() == t_buffer_size);
        for (int i=0; i<t_buffer_size; ++i) {
            m_data[i].store(empty_token);
        }
    }

    ~Deque()
    {
        delete empty_token;
    }

    /* Fails if:
     - there is no valid slot at index tail
     */
    bool push(T* a_x)
    {
        T* empty_ = empty_token;

        unsigned tail = m_tail.load();
        const bool cas = m_data[tail&m_mask].compare_exchange_strong(empty_, a_x);
        // cas is false if the slot contains an element
        m_tail.compare_exchange_strong(tail,tail+1);

        if (cas) {
            ++size;
        }
        else {
            ++c_push_fail;
        }

        return cas;
    }

    /* Fails if:
     - there is no valid item at index head
     */
    bool pop(T** a_x)
    {
        unsigned head = m_head.load();
        T* const val = m_data[head&m_mask].exchange(empty_token);
        m_head.compare_exchange_strong(head,head+1);

        const bool exchg = (val != empty_token);
        // val == empty_token if another thread has already retrieved the element

        if (exchg) {
            --size;
            // Successful pop
            *a_x = val;
        }
        else {
            ++c_pop_fail;
        }

        return exchg;
    }

    unsigned buffer_size()
    {
        return t_buffer_size;
    }

    bool empty()
    {
        return (size.load() == 0);
    }

    bool full()
    {
        return (size.load() == t_buffer_size);
    }

    bool empty_traverse()
    {
        for (int i=0; i<t_buffer_size; ++i) {
            if (m_data[i].load() != empty_token) {
                return false;
            }
        }
        return true;
    }

    bool full_traverse()
    {
        for (int i=0; i<t_buffer_size; ++i) {
            if (m_data[i].load() == empty_token) {
                return false;
            }
        }
        return true;
    }
};
