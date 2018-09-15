#include <string>
#include <strstream>
#include <ctime>
#include <atomic>
#include <vector>
#include <random>
#include <future>
#include <thread>

#include "gtest/gtest.h"

#include "threadpool.hpp"

/*
 Proposed algorithm:
 1. Use array based MPSC queue to enqueue tasks for threads
 2. Use array based MPMC queue for list of idle threads; this is queried during task pops to improve the load balancing
 3. When a thread enqueue is full, we use a private queue to postpone the insertion. The insertion is attempted again during the next post. This should have minor performance impact since the full queue corresponds to a busy thread.
 4. Each thread pops from their queues and use a private priority queue.
 5. Idle thread queue is used to deal work to idle threads.
 6. MPSC and MPMC array queues are implemented using CAS and a tail/head index with wrapping.
 */

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
//            printf("push success head/tail %d %d\n",m_head.load(),m_tail.load());
        }
        else {
            ++c_push_fail;
//            printf("push fail head/tail %d %d\n",m_head.load(),m_tail.load());
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
//            printf("pop success head/tail %d %d\n",m_head.load(),m_tail.load());
            // Successful pop
            *a_x = val;
        }
        else {
            ++c_pop_fail;
//            printf("pop fail head/tail %d %d\n",m_head.load(),m_tail.load());
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

TEST(Deque, Empty_Full)
{
    const unsigned loop_size = 4;
    Deque<int, loop_size> dq;

    ASSERT_TRUE(dq.empty_traverse());
    ASSERT_TRUE(dq.empty());
    ASSERT_FALSE(dq.full_traverse());
    ASSERT_FALSE(dq.full());

    std::vector< int > input(loop_size);
    int* output;

    for (int i=0; i<loop_size; ++i) {
        input[i] = i;
    }

    for (unsigned i=0; i<loop_size; ++i) {
        ASSERT_FALSE(dq.full_traverse());
        ASSERT_FALSE(dq.full());
        dq.push(&input[i]);
        ASSERT_FALSE(dq.empty_traverse());
        ASSERT_FALSE(dq.empty());
    }

    ASSERT_FALSE(dq.empty_traverse());
    ASSERT_FALSE(dq.empty());
    ASSERT_TRUE(dq.full_traverse());
    ASSERT_TRUE(dq.full());

    for (unsigned i=0; i<loop_size; ++i) {
        ASSERT_FALSE(dq.empty_traverse());
        ASSERT_FALSE(dq.empty());
        dq.pop(&output);
        ASSERT_FALSE(dq.full_traverse());
        ASSERT_FALSE(dq.full());
    }

    ASSERT_TRUE(dq.empty_traverse());
    ASSERT_TRUE(dq.empty());
    ASSERT_FALSE(dq.full_traverse());
    ASSERT_FALSE(dq.full());
}

template <int queue_size>
void basic_test(const int loop_size)
{
    Deque<int, queue_size> dq;
    std::vector< int > input(loop_size);
    int* output;

    for (int i=0; i<loop_size; ++i) {
        input[i] = i;
    }

    int sum_in = 0;
    int sum_out = 0;

    c_push_fail.store(0);
    c_pop_fail.store(0);

    for (int i=0; i<loop_size; ++i) {
        if (dq.push(&input[i])) {
            sum_in += input[i];
            if (i >= queue_size)
                ASSERT_TRUE(false) << "Push expected to fail in basic_test";
        }
        else {
            if (i < queue_size)
                ASSERT_TRUE(false) << "Push expected to succeed in basic_test";
        }
    }

    for (int i=0; i<loop_size; ++i) {
        if (dq.pop(&output)) {
            ASSERT_EQ(*output,input[i]);
            sum_out += *output;
            if (i >= queue_size)
                ASSERT_TRUE(false) << "Pop expected to fail in basic_test";
        }
        else {
            if (i < queue_size)
                ASSERT_TRUE(false) << "Pop expected to succeed in basic_test";
        }
    }

    if (loop_size <= queue_size) {
        ASSERT_EQ(c_push_fail.load(), 0);
        ASSERT_EQ(c_pop_fail.load(), 0);
    }

    ASSERT_TRUE(dq.empty_traverse());
    ASSERT_TRUE(dq.empty());
    ASSERT_EQ(sum_in, sum_out);
}

TEST(Basic, ShortLoop)
{
    const int queue_size = 1<<10;
    const int loop_size = queue_size>>1;
    basic_test<queue_size>(loop_size);
}

TEST(Basic, EqualLoop)
{
    const int queue_size = 1<<10;
    const int loop_size = queue_size;
    basic_test<queue_size>(loop_size);
}

TEST(Basic, LongLoop)
{
    const int queue_size = 1<<10;
    const int loop_size = queue_size<<1;
    basic_test<queue_size>(loop_size);
}

void thread_sleep()
{
    int microsecs = 10; // length of time to sleep, in microseconds
    struct timespec req {
        0
    };
    req.tv_sec = 0;
    req.tv_nsec = microsecs * 1000L;
    nanosleep(&req, (struct timespec *)NULL);
}

template <int queue_size>
void test_async(const unsigned n_async,
                unsigned outer_max,
                const unsigned loop_size_max)
{
    std::vector< int > input(loop_size_max);
    for (unsigned i=0; i<loop_size_max; ++i) {
        input[i] = i;
    }

    std::vector< std::future<int> > future_push(outer_max*n_async);
    std::vector< std::future<int> > future_pop(future_push.size());

    Deque<int, queue_size> dq;
    std::atomic_int in{0};
    std::atomic_int out{0};

    auto lambda_push = [=,&dq,&future_push,&future_pop,&input,&in] (unsigned f_count, unsigned loop_size) -> int {
        assert(loop_size > 0);
        for (unsigned i=0; i<loop_size; ++i)
        {
            if (dq.push(&input[i]))
                in.fetch_add(input[i]);
        }
        return static_cast<int>(f_count);
    };

    auto lambda_pop = [=,&dq,&future_push,&future_pop,&out] (unsigned f_count, unsigned loop_size) -> int {
        assert(loop_size > 0);
        int* output;
        for (unsigned i=0; i<loop_size; ++i)
        {
            if (dq.pop(&output))
                out.fetch_add(*output);
        }
        return static_cast<int>(f_count);
    };

    c_push_fail.store(0);
    c_pop_fail.store(0);

    int count_ops = 0;

    unsigned f_count = 0;
    for (unsigned outer=0; outer<outer_max; ++outer) {
        const unsigned loop_size = 1 + (outer*loop_size_max)/outer_max;
        ASSERT_GE(loop_size_max, loop_size);
        for (unsigned i=0; i<n_async; ++i, ++f_count) {
            ASSERT_GT(future_push.size(), f_count);
            future_push[f_count] = std::async(std::launch::async, lambda_push, f_count, loop_size);
            future_pop[f_count]  = std::async(std::launch::async, lambda_pop,  f_count, loop_size);
        }
        count_ops += loop_size*n_async;
    }

    for (unsigned i=0; i<outer_max*n_async; ++i) {
        future_push[i].wait();
        future_pop[i].wait();
    }

    printf("Rate of failed push: %g\n", float(c_push_fail.load()) / float(count_ops));
    printf("Rate of failed pop:  %g\n", float( c_pop_fail.load()) / float(count_ops));

    if (! dq.empty()) {
        unsigned more_pop{0};
        int* output;
        for(unsigned i=0; i<dq.buffer_size(); ++i) {
            if (dq.pop(&output)) {
                out.fetch_add(*output);
                ++more_pop;
            }
        }
        printf("No. of add'l pops at the end %d\n", more_pop);
    }

    EXPECT_TRUE(dq.empty_traverse());
    EXPECT_TRUE(dq.empty());
    ASSERT_EQ(in.load(), out.load());
}

TEST(ManyThreads, OneAsync)
{
    const int queue_size = 1<<20;
    const unsigned n_async = 1;
    unsigned outer_max = 1<<0;
    const unsigned loop_size_max = 1<<5;

    test_async<queue_size>(n_async,
                           outer_max,
                           loop_size_max);
}

TEST(ManyThreads, FewAsyncs)
{
    const int queue_size = 1<<20;
    const unsigned n_async = 4;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<5;

    test_async<queue_size>(n_async,
                           outer_max,
                           loop_size_max);
}

TEST(ManyThreads, LongQueue)
{
    const int queue_size = 1<<20;
    const unsigned n_async = 1<<10;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<10;

    test_async<queue_size>(n_async,
                           outer_max,
                           loop_size_max);
}

TEST(ManyThreads, MediumQueue)
{
    const int queue_size = 1<<10;
    const unsigned n_async = 1<<10;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<10;

    test_async<queue_size>(n_async,
                           outer_max,
                           loop_size_max);
}

TEST(ManyThreads, ShortQueue)
{

    const int queue_size = 1<<5;
    const unsigned n_async = 1<<10;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<10;

    test_async<queue_size>(n_async,
                           outer_max,
                           loop_size_max);
}

template <int queue_size>
void test_thread_pool(const unsigned n_thread,
                      unsigned outer_max,
                      const unsigned loop_size_max)
{
    std::vector< int > input(loop_size_max);
    for (unsigned i=0; i<loop_size_max; ++i) {
        input[i] = i;
    }

    Deque<int, queue_size> dq;
    std::vector< std::queue<int*> > private_q(n_thread);

    std::atomic_int in{0};
    std::atomic_int out{0};

    auto lambda_push = [=,&dq,&input,&private_q,&in] (int a_th, unsigned loop_size) {
        for (unsigned i=0; i<loop_size; ++i) {
            if (! dq.push(&input[i])) {
                private_q[a_th].push(&input[i]);
            }
            in.fetch_add(input[i]);
        }
    };

    auto lambda_pop = [=,&dq,&private_q,&out] (int a_th, unsigned loop_size) {
        int* output;
        unsigned j = 0;
        // Empty own private queue
        while (j < loop_size && !private_q[a_th].empty()) {
            output = private_q[a_th].front();
            ASSERT_GE(*output, 0);
            ASSERT_LT(*output, loop_size_max);
            out.fetch_add(*output);
            private_q[a_th].pop();
            ++j;
        }
        while (j < loop_size) {
            if (dq.pop(&output)) {
                ASSERT_GE(*output, 0);
                ASSERT_LT(*output, loop_size_max);
                out.fetch_add(*output);
                ++j;
            }
        }
    };

    c_push_fail.store(0);
    c_pop_fail.store(0);

    std::vector<Task> push_task(outer_max*n_thread);
    std::vector<Task> pop_task(push_task.size());

    unsigned count_ops = 0;

    Thread_pool th_pool(n_thread);
    th_pool.start();

    for (unsigned outer=0; outer<outer_max; ++outer) {

        const unsigned loop_size = 1 + (outer*loop_size_max)/outer_max;
        ASSERT_GE(loop_size_max, loop_size);
        for (unsigned i_th=0; i_th<n_thread; ++i_th) {
            const unsigned idx = i_th + n_thread*outer;
            ASSERT_GT(push_task.size(), idx);

            push_task[idx] = std::bind(lambda_push, i_th, loop_size);
            th_pool.spawn(i_th, &push_task[idx]);
            pop_task [idx] = std::bind(lambda_pop,  i_th, loop_size);
            th_pool.spawn(i_th, &pop_task[idx]);
        }
        count_ops += loop_size*n_thread;
    }

    th_pool.join();

    printf("Rate of failed push: %g\n", float(c_push_fail.load()) / float(count_ops));
    printf("Rate of failed pop:  %g\n", float( c_pop_fail.load()) / float(count_ops));

    for (auto pq : private_q) {
        ASSERT_TRUE(pq.empty());
    }

    ASSERT_TRUE(dq.empty_traverse());
    ASSERT_TRUE(dq.empty());
    ASSERT_EQ(in.load(), out.load());
}

TEST(ThreadPool, OneThread)
{
    const int queue_size = 1<<5;
    const unsigned n_thread = 1;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<5;

    test_thread_pool<queue_size>(n_thread,
                                 outer_max,
                                 loop_size_max);
}

TEST(ThreadPool, FewThreads)
{
    const int queue_size = 1<<5;
    const unsigned n_thread = 4;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<5;

    test_thread_pool<queue_size>(n_thread,
                                 outer_max,
                                 loop_size_max);
}

TEST(ThreadPool, LongQueue)
{
    const int queue_size = 1<<20;
    const unsigned n_thread = 1<<10;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<8;

    test_thread_pool<queue_size>(n_thread,
                                 outer_max,
                                 loop_size_max);
}

TEST(ThreadPool, MediumQueue)
{
    const int queue_size = 1<<10;
    const unsigned n_thread = 1<<10;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<8;

    test_thread_pool<queue_size>(n_thread,
                                 outer_max,
                                 loop_size_max);
}

TEST(ThreadPool, ShortQueue)
{
    const int queue_size = 1<<5;
    const unsigned n_thread = 1<<10;
    unsigned outer_max = 1<<3;
    const unsigned loop_size_max = 1<<8;

    test_thread_pool<queue_size>(n_thread,
                                 outer_max,
                                 loop_size_max);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
