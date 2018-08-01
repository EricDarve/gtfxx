#include <iostream>
#include <utility>
#include <string>
#include <strstream>
#include <atomic>
#include <vector>
#include <list>
#include <future>
#include <thread>

#include "gtest/gtest.h"

/*
 Proposed algorithm:
 1. Use array based MPSC queue to enqueue tasks for threads
 2. Use array based MPMC queue for list of idle threads; this is queried during task pops to improve the load balancing
 3. When a thread enqueue is full, we use a private linked list used to postponed the insertion. The insertion is attempted again during the next post. This should have minor performance impact since the full queue corresponds to a busy thread.
 4. Each thread pops from their queues and use a private priority queue.
 5. Idle thread queue is used to deal work to idle threads.
 6. MPSC and MPMC array queues are implemented using CAS and a tail/head index with wrapping.
 7. Check that ABA is not a problem with this implementation.
 */

// buffer_size must be a power of 2
template <typename T, int buffer_size>
struct Deque {
    std::atomic_uint m_head, m_tail;
    std::vector< std::atomic<T*> > m_data;
    int m_mask;

    Deque() : m_head(0), m_tail(0), m_data(buffer_size)
    {
        m_mask = buffer_size - 1;
        assert(m_data.size() == buffer_size);
        for (int i=0; i<buffer_size; ++i) {
            m_data[i].store(static_cast<T*>(nullptr));
        }
        for (int i=0; i<buffer_size; ++i) {
            assert(m_data[i].load() == static_cast<T*>(nullptr));
        }
    }

    bool push(T* a_x)
    {
        assert(a_x != static_cast<T*>(nullptr));
        unsigned tail = m_tail.load();
        bool cas;
        {
            T* empty = static_cast<T*>(nullptr);
            cas = m_data[tail&m_mask].compare_exchange_strong(empty,a_x);
        }

        m_tail.compare_exchange_strong(tail,tail+1);

//        for (int i=0; i<buffer_size; ++i) {
//            if (m_data[i].load() != static_cast<T*>(nullptr))
//                std::cout << "push i " << i << " data " << *m_data[i].load() << std::endl;
//            else
//                std::cout << "push i " << i << " empty" << std::endl;
//        }

        return cas;
    }

    bool pop(T** a_x)
    {
        unsigned head = m_head.load();
        T* val;
        {
            T* empty = static_cast<T*>(nullptr);
            val = m_data[head&m_mask].exchange(empty);
        }

        m_head.compare_exchange_strong(head,head+1);

//        for (int i=0; i<buffer_size; ++i) {
//            if (m_data[i].load() != static_cast<T*>(nullptr))
//                std::cout << "pop " << i << " " << *m_data[i].load() << std::endl;
//        }

        if (val != static_cast<T*>(nullptr)) {
            // Successful pop
            *a_x = val;
            return true;
        }
        return false;
    }

    bool is_empty()
    {
        for (int i=0; i<buffer_size; ++i) {
            if (m_data[i].load() != static_cast<T*>(nullptr)) {
                return false;
            }
        }
        return true;
    }
};

struct int2 {
    int x, y;
};

TEST(Atomic, Lock_free)
{
    ASSERT_TRUE(std::atomic_bool{}.is_lock_free());
    ASSERT_TRUE(std::atomic_int{}.is_lock_free());
    ASSERT_TRUE(std::atomic_uint{}.is_lock_free());
    ASSERT_TRUE(std::atomic<float> {}.is_lock_free());
    ASSERT_TRUE(std::atomic<double> {}.is_lock_free());
    ASSERT_TRUE(std::atomic<int2> {}.is_lock_free());
    ASSERT_TRUE(std::atomic<void*> {}.is_lock_free());
}

TEST(Deque, Basic)
{
    const int queue_size = 4;
    const int loop_size = 2 * queue_size;

    Deque<int, queue_size> dq;
    std::vector< int > input(loop_size);
    int * output;

    for (int i=0; i<loop_size; ++i) {
        input[i] = i;
    }

    int sum_in = 0;
    int sum_out = 0;

    for (int i=0; i<loop_size; ++i) {
        if (dq.push(&input[i])) {
            sum_in += input[i];
        }
    }

    for (int i=0; i<loop_size; ++i) {
        if (dq.pop(&output)) {
            sum_out += *output;
        }
    }

    ASSERT_TRUE(sum_in == sum_out);
}

TEST(Deque, ManyThreads)
{
    const int queue_size = 16;
    const unsigned n_thread = 1024;
    Deque<int, queue_size> dq;

    std::atomic_int in{0};
    std::atomic_int out{0};

    std::vector< std::future<void> > future_push(n_thread), future_pop(n_thread);

    const unsigned loop_size_max = 1024;
    std::vector< int > input(loop_size_max);
    for (unsigned i=0; i<loop_size_max; ++i) {
        input[i] = i;
    }

    auto lambda_push = [=,&dq,&input,&in] (unsigned loop_size) {
        for (unsigned i=0; i<loop_size; ++i) {
            if (dq.push(&input[i]))
                in.fetch_add(input[i]);
        }
    };

    auto lambda_pop = [=,&dq,&future_push,&out] (int i, unsigned loop_size) {
        future_push[i].wait();
        int* output;
        for (unsigned i=0; i<loop_size; ++i) {
            if (dq.pop(&output))
                out.fetch_add(*output);
        }
    };

    for (unsigned outer=0; outer<32; ++outer) {
        const unsigned loop_size = 32 + outer;
        assert(loop_size <= loop_size_max);
        for (unsigned i=0; i<n_thread; ++i) {
            future_push[i] = std::async(std::launch::async, lambda_push, loop_size);
            future_pop[i]  = std::async(std::launch::async, lambda_pop, i, loop_size);
        }
    }

    for (unsigned i=0; i<n_thread; ++i) {
        future_pop[i].wait();
    }

    ASSERT_EQ(in.load(), out.load());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
