//
// Created by Eric Darve on 9/16/18.
//

#ifndef GTFXX_GTFXX_H
#define GTFXX_GTFXX_H

#include "util.h"

#include "queue_lockfree.hpp"

namespace gtfxx {

// ---------------------
// UPC++ function calls

    void active_message_progress();

    void capture_master_thread();

    void release_master_thread();

// Base_task is a simple function with no argument
    typedef std::function<void()> Base_task;

// Tasks are functions executed by threads
    struct Task : public Base_task {
        float m_priority = 0.0; // task priority
        explicit Task(const Base_task &&a_f, float a_priority = 0.0);
    };

// Thread with priority queue

// Task comparison based on their priorities
    struct Task_comparison {
    public:
        // Comparison () operator
        bool operator()(const Task *a_lhs, const Task *a_rhs) const;
    };

// Thread with priority queue
    struct Thread_prio {
        Thread_pool *th_pool;
        int m_id; // thread id in a thread pool

        std::priority_queue<Task *, std::vector<Task *>, Task_comparison> ready_queue;
        std::mutex mtx; // Used to manage access to the priority queue

        Queue_lockfree<Task *> message_queue;

        std::thread th; // thread that will execute the tasks

        // For optimization to avoid testing ready_queue.empty() in some cases
        std::atomic_bool m_empty;

        Thread_prio();

        ~Thread_prio();

        // Thread starts executing the function spin()
        void start();

        // Add new task to queue; thread safe
        void spawn(Task *);

        // Not thread safe
        Task *pop_unsafe();

        // join() the thread
        void join();
    };

// ----------------------------------
// Progress thread for communications

    struct Thread_comm {
        Thread_pool *th_pool = nullptr;
        std::queue<Task *> ready_queue;
        std::thread th;
        std::mutex mtx;
        std::atomic_bool m_empty;

        Thread_comm();

        ~Thread_comm();

        void start();

        // Add new task to queue; thread safe
        void spawn(Task *a_t);

        // Not thread safe
        Task *pop_unsafe();

        // join() the thread
        void join();
    };

// -----------
// Thread pool

    struct Thread_pool {
        std::vector<Thread_prio> v_thread;
        unsigned long n_query_spawn = 4;  // Optimization parameter
        unsigned long n_query_steal = 16; // Optimization parameter
        std::atomic_int n_tasks; // Number of ready tasks in any thread queue
        std::atomic_bool m_stop;

        Thread_comm th_comm;

        explicit Thread_pool(int n_thread);

        void start();

        void join();

        void spawn(int a_id, Task *a_task);

        void steal(unsigned short a_id);
    };

// ------------------------
// Base class for task flow

    typedef std::array<int, 3> int3;
    typedef std::atomic_int Promise;

    struct Dependency_flow {
        typedef std::function<int(const int3)> Dependency_count;
        typedef std::function<void(const int3)> Build_task;

        // Thread pool
        Thread_pool *th_pool;

        // How to calculate the number of dependencies for promises
        Dependency_count m_dep_count;

        // How to construct a task
        Build_task m_build_task;

        Dependency_flow();

        explicit Dependency_flow(Thread_pool *);

        virtual ~Dependency_flow() = default;

        // Enqueue tasks
        void seed_task(int3);

        virtual void async_task_spawn(int3 idx, Task *a_tsk) = 0;

        // Decrement the dependency counter and spawn task if ready
        virtual void fulfill_promise(int3) = 0;

        void do_fulfill_promise(int3, Promise *);
    };

// ------------
// Promise grid

    struct Matrix3_promise : std::vector<Promise> {
        int n0, n1, n2;

        Matrix3_promise();

        // Initialize all promises with -1
        Matrix3_promise(int a_n0, int a_n1, int a_n2);

        std::atomic_int &operator()(int i, int j, int k);
    };

// -----------------------------------
// Task flow for computational kernels

    struct Task_flow : public Dependency_flow {

        typedef std::function<int(int3 &)> Map_task;

        // Tasks are indexed using a 3D grid of atomic dependencies
        Matrix3_promise promise_grid;

        // Mapping from task index to thread id
        Map_task m_map;

        Task_flow();

        Task_flow(Thread_pool *a_pool, int n0, int n1, int n2);

        // Defines the number of dependencies
        Task_flow &wait_on_promises(Dependency_count f);

        // Defines the task to be run asynchronously
        Task_flow &then_run(Build_task f);

        // Which thread should execute the task
        void on_thread(Map_task a_map);

        // Decrement the dependency counter and spawn task if ready
        void fulfill_promise(int3) override;

    private:
        // Spawn a task that is already initialized
        void async_task_spawn(int3, Task *) override;
    };

// -----------------
// Channel class: used to communicate data between nodes
// using active messages

    namespace hash_array {

// Another option is to specify the maximum sizes in each dimension
// for the sparse grid.
        inline void hash_combine(std::size_t &seed, int const &v) {
            seed ^= std::hash<int>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        struct hash {
            size_t
            operator()(std::array<int, 3> const &key) const {
                size_t seed = 0;
                hash_combine(seed, key[0]);
                hash_combine(seed, key[1]);
                hash_combine(seed, key[2]);
                return seed;
            }
        };
    }

    struct Channel : public Dependency_flow {

        typedef std::function<void(int3)> Finalize_task;

        // How to unpack after a data transfer is complete
        Finalize_task m_finalize;

        std::unordered_map<int3, std::atomic_int, hash_array::hash> promise_map;
        std::mutex mtx_graph; // Used to concurrently make changes to task_map

        Channel();

        explicit Channel(Thread_pool *a_pool);

        // Defines the number of dependencies
        Channel &wait_on_promises(Dependency_count f);

        // Defines the communication channel to run
        void then_send(Build_task f);

        // Defines the callback function.
        // The callback function may be called after communication completes.
        void set_finalize(Finalize_task a_finalize);

        // Run the m_finalize() callback function defined in set_finalize()
        // This function may be called in the active message.
        void finalize(int3 idx);

        // Find a promise in promise_map and return a pointer
        Promise *find_promise(int3 &);

        // Decrement the dependency counter and enqueue communication if ready
        void fulfill_promise(int3) override;

    private:
        // Enqueue a communication that is already initialized
        void async_task_spawn(int3, Task *) override;
    };

// ----------------
// Active messaging

    template<typename... T>
    class Active_message {

    private:
        std::tuple<T...> tuple_;
        upcxx::intrank_t dest;

    public:
        explicit Active_message(std::tuple<T...> a_tuple) : tuple_(std::move(a_tuple)), dest(-1) {}

        Active_message &to_rank(upcxx::intrank_t a_dest) {
            dest = a_dest;
            return *this;
        }

        template<typename Fn>
        void then_on_receiving_rank(Fn &&a_tsk_) {
            rpc_ff_tuple(a_tsk_,
                         upcxx::make_index_sequence<std::tuple_size<std::tuple<T...> >::value>()
            );
        }

    private:
        template<typename Fn, int... i>
        void rpc_ff_tuple(Fn &a_tsk_, upcxx::index_sequence<i...>) {
            upcxx::rpc_ff(dest, std::forward<Fn>(a_tsk_), std::forward<T>(std::get<i>(tuple_))...);
        }
    };

    template<typename T, typename Iter = T *>
    struct memblock {
        typedef upcxx::view<T, Iter> iterator;
    };

    template<typename T, typename Iter = T *>
    upcxx::view<T, Iter> memblock_view(std::size_t n, Iter begin) {
        Iter end = begin + n;
        return {static_cast<Iter &&>(begin), static_cast<Iter &&>(end), n};
    }

    template<typename... T>
    Active_message<T...> send(T... a_msg) {
        return Active_message<T...>(std::tuple<T...>(a_msg...));
    }
}

#endif //GTFXX_GTFXX_H
