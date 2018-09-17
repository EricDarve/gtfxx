//
// Created by Eric Darve on 9/16/18.
//

#include "gtfxx.h"

// ---------------------
// UPC++ function calls

void active_message_progress() {
    upcxx::progress();
}

void capture_master_thread() {
    if (upcxx::backend::initial_master_scope == nullptr) {
        upcxx::backend::initial_master_scope = new persona_scope{backend::master};
    }
}

void release_master_thread() {
    if (upcxx::backend::initial_master_scope != nullptr) {
        upcxx::liberate_master_persona();
    }
}

// ---------------------
// Task

Task::Task(const Base_task &&a_f, float a_priority) :
        Base_task(a_f), m_priority(a_priority) {}

bool Task_comparison::operator()(const Task *a_lhs, const Task *a_rhs) const {
    return (a_lhs->m_priority < a_rhs->m_priority);
};

// ---------------------
// Thread_prio

Thread_prio::Thread_prio() : th_pool(nullptr), m_id(-1), m_empty{false} {
}

Thread_prio::~Thread_prio() {
    join();
}

void spin_task(Thread_prio *);

void Thread_prio::start() {
    m_empty.store(true);
    th = thread(spin_task, this); // Execute tasks in queue
}

void Thread_prio::spawn(Task *a_t) {
    std::lock_guard<std::mutex> lck(mtx);
    m_empty.store(false);
    ready_queue.push(a_t); // Add task to queue
}

Task *Thread_prio::pop_unsafe() {
    Task *tsk = ready_queue.top();
    ready_queue.pop();
    if (ready_queue.empty())
        m_empty.store(true);
    return tsk;
}

void Thread_prio::join() {
    if (th.joinable()) {
        th.join();
    }
}

// Keep executing tasks until m_stop = true && there are no tasks left anywhere
void spin_task(Thread_prio *a_thread) {
    auto pe = profiler.start("overhead");

    std::unique_lock<std::mutex> lck(a_thread->mtx);

    while (true) {

        while (!a_thread->ready_queue.empty()) {
            Task *tsk = a_thread->pop_unsafe();
            lck.unlock();
            profiler.stop(pe);

            (*tsk)();

            pe = profiler.start("overhead");
            --(a_thread->th_pool->n_tasks);

            // Free memory
            delete tsk;

            lck.lock();
        }

        lck.unlock();
        // Try to steal a task
        a_thread->th_pool->steal(a_thread->m_id);

        while (a_thread->m_empty.load()) {

            // Return if stop=true and no tasks are left
            if (a_thread->th_pool->m_stop.load() && a_thread->th_pool->n_tasks.load() <= 0) {
                profiler.stop(pe);
                return;
            }

            profiler.stop(pe);

            // When queue is empty, yield
            pe = profiler.start("wait");
            //  std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::microseconds(40));
            profiler.stop(pe);

            pe = profiler.start("overhead");

            if (a_thread->m_empty.load()) {
                // Try to steal a task
                a_thread->th_pool->steal(a_thread->m_id);
            }
        }

        lck.lock();
    }
}

// ----------------------------------
// Progress thread for communications

Thread_comm::Thread_comm() : th_pool(nullptr), m_empty{false} {
}

Thread_comm::~Thread_comm() {
    join();
}

void spin_comm(Thread_comm *);

void Thread_comm::start() {
    m_empty.store(true);
    th = thread(spin_comm, this);
}

void Thread_comm::spawn(Task *a_t) {
    std::lock_guard<std::mutex> lck(mtx);
    m_empty.store(false);
    ready_queue.push(a_t); // Add task to queue
}

void Thread_comm::join() {
    if (th.joinable()) {
        th.join();
    }
}

Task *Thread_comm::pop_unsafe() {
    Task *tsk = ready_queue.front();
    ready_queue.pop();
    if (ready_queue.empty())
        m_empty.store(true);
    return tsk;
}

// Keep executing tasks until m_stop = true && there are no tasks left anywhere
void spin_comm(Thread_comm *a_thread) {

    // Acquiring master persona
    upcxx::persona_scope scope(upcxx::master_persona());

    std::unique_lock<std::mutex> lck(a_thread->mtx);
    while (true) {

        while (!a_thread->ready_queue.empty()) {
            Task *tsk = a_thread->pop_unsafe();
            lck.unlock();

            (*tsk)(); // Issue active message

            --(a_thread->th_pool->n_tasks);

            // Free memory
            delete tsk;

            // Make progress on active messages
            active_message_progress();

            lck.lock();
        }

        lck.unlock();

        while (a_thread->m_empty.load()) {
            // Return if stop=true and no tasks are left
            if (a_thread->th_pool->m_stop.load() && a_thread->th_pool->n_tasks.load() <= 0) {
                return;
            }

            // When queue is empty, sleep
            std::this_thread::sleep_for(std::chrono::microseconds(40));

            // Make progress on active messages
            active_message_progress();
        }

        lck.lock();
    }
}

// ---------------------
// Threadpool functions

Threadpool::Threadpool(const int n_thread) :
        v_thread(n_thread), n_tasks(0), m_stop(false) {
    for (int i = 0; i < n_thread; ++i) {
        v_thread[i].th_pool = this;
        v_thread[i].m_id = static_cast<unsigned short>(i);
    }
}

void Threadpool::start() {
    n_tasks.store(0);
    m_stop.store(false);
    for (auto &th : v_thread) th.start();
}

void Threadpool::spawn(const int a_id, Task *a_task) {
    assert(a_id >= 0 && static_cast<unsigned long>(a_id) < v_thread.size());

    ++n_tasks;

    int id_ = a_id;

    // Check if queue is empty
    if (!v_thread[a_id].m_empty.load()) {
        // Thread is already busy
        // Are there other threads that are idle?
        const unsigned long n_query = min(1 + n_query_spawn, v_thread.size());
        for (unsigned long i = a_id + 1; i < a_id + n_query; ++i) {
            auto j = i % v_thread.size();
            if (v_thread[j].m_empty.load()) {
                // We have found an idle thread
                id_ = static_cast<int>(j);
                break;
            }
        }
    }

    {
        char timestamp_message[80];
        if (id_ == a_id) {
            sprintf(timestamp_message, "spawn_to %d", a_id);
        } else {
            sprintf(timestamp_message, "spawn_other [%d->]%d", a_id, id_);
        }
        profiler.timestamp(string(timestamp_message));
    }

    v_thread[id_].spawn(a_task);
}

void Threadpool::steal(unsigned short a_id) {
    const unsigned long n_query = min(n_query_steal, v_thread.size());
    for (unsigned long i = a_id + 1; i < a_id + n_query; ++i) {
        auto j = i % v_thread.size();
        Thread_prio &thread_j = v_thread[j];
        if (!thread_j.m_empty.load()) {
            std::unique_lock<std::mutex> lck(thread_j.mtx);
            if (!thread_j.ready_queue.empty()) {
                // We have found a non empty task queue
                Task *tsk = thread_j.pop_unsafe();
                lck.unlock();
                {
                    char timestamp_message[80];
                    sprintf(timestamp_message, "steal %d[<-%ld]", a_id, j);
                    profiler.timestamp(timestamp_message);
                }
                v_thread[a_id].spawn(tsk);
                break;
            }
        }
    }
}

// ------------------------
// Base class for task flow

Dependency_flow::Dependency_flow() : th_pool(nullptr), m_dep_count(nullptr), m_build_task(nullptr) {}

Dependency_flow::Dependency_flow(Threadpool *a_pool) :
        th_pool(a_pool), m_dep_count(nullptr), m_build_task(nullptr) {}

// ------------
// Promise grid

Matrix3_promise::Matrix3_promise() : n0(0), n1(0), n2(0) {}

// Initialize with -1
Matrix3_promise::Matrix3_promise(int a_n0, int a_n1, int a_n2) :
        vector<Promise>(a_n0 * a_n1 * a_n2),
        n0(a_n0), n1(a_n1), n2(a_n2) {
    for (int i = 0; i < n0 * n1 * n2; ++i) {
        operator[](i).store(-1);
    }
}

std::atomic_int &Matrix3_promise::operator()(int i, int j, int k) {
    assert(i >= 0 && i < n0);
    assert(j >= 0 && j < n1);
    assert(k >= 0 && k < n2);
    return operator[](i + n0 * (j + n1 * k));
}

// -----------------------------------
// Task flow for computational kernels

Task_flow::Task_flow() : m_map(nullptr) {}

Task_flow::Task_flow(Threadpool *a_pool, int n0, int n1, int n2) :
        Dependency_flow(a_pool), promise_grid(n0, n1, n2), m_map(nullptr) {}

Task_flow &Task_flow::dependency_count(Dependency_count f) {
    m_dep_count = std::move(f);
    return *this;
}

Task_flow &Task_flow::define_task(Build_task f) {
    m_build_task = std::move(f);
    return *this;
}

void Task_flow::compute_on(Map_task a_map) {
    m_map = std::move(a_map);
}

// Spawn task
void Task_flow::async(int3 idx, Task *a_tsk) {
    // Basic sanity check
    assert(a_tsk != nullptr);
    assert(th_pool != nullptr);
    assert(m_map != nullptr);
    assert(m_map(idx) >= 0);

    th_pool->spawn(/*task map*/ m_map(idx), a_tsk);
}

// Enqueue task immediately
void Task_flow::async(int3 idx) {
    throw_assert(this->m_build_task != nullptr,
                 "define_task() was not called; the task cannot be defined");
    Build_task f = this->m_build_task;
    async(idx, new Task([f, idx]() { f(idx); }));
}

void Task_flow::fulfill_promise(int3 idx) {
    assert(0 <= idx[0] && idx[0] < promise_grid.n0);
    assert(0 <= idx[1] && idx[1] < promise_grid.n1);
    assert(0 <= idx[2] && idx[2] < promise_grid.n2);

    auto atomic_count = &(promise_grid(idx[0], idx[1], idx[2]));

    // Decrement counter
    int wait_count = std::atomic_fetch_sub(atomic_count, 1);

    if (wait_count == -1) { // Uninitialized task
        throw_assert(m_dep_count != nullptr,
                     "dependency_count() was not called; the promise cannot be initialized");
        int wait_count_init = m_dep_count(idx);
        wait_count = std::atomic_fetch_add(atomic_count, wait_count_init);
        // wait_count = value before add operation
        wait_count += wait_count_init + 1;
    }

    if (wait_count == 0) { // task is ready to run
        throw_assert(this->m_build_task != nullptr,
                     "define_task() was not called; the task cannot be defined");
        Build_task f = this->m_build_task;
        async(idx, new Task([f, idx]() { f(idx); }));
    }
}

// -------------
// Channel class

Channel::Channel() : m_finalize(nullptr) {}
Channel::Channel(Threadpool *a_pool) : Dependency_flow(a_pool), m_finalize(nullptr) {}

Channel &Channel::dependency_count(Dependency_count f) {
    m_dep_count = std::move(f);
    return *this;
}

void Channel::define_task(Build_task f) {
    m_build_task = std::move(f);
}

void Channel::set_finalize(Finalize_task a_finalize) {
    m_finalize = std::move(a_finalize);
}

void Channel::finalize(int3 idx) {
    throw_assert(m_finalize != nullptr,
                 "set_finalize() was not called; the finalize() task cannot be called");
    m_finalize(idx);
}

Promise *Channel::find_promise(int3 &idx) {
    Promise *t_ = nullptr;

    std::unique_lock<std::mutex> lck(mtx_graph);

    auto prom_it = promise_map.find(idx);

    if (prom_it == promise_map.end()) {
        // Promise does not exist; create it
        promise_map.emplace(std::piecewise_construct,
                            std::forward_as_tuple(idx),
                            std::forward_as_tuple(-1)); // Insert in promise_map
        prom_it = promise_map.find(idx); // Find location
    }

    t_ = &(prom_it->second); // Get promise

    lck.unlock();

    assert(t_ != nullptr);

    return t_;
}

// Initialize task and spawn it
void Channel::async(int3 idx) {
    throw_assert(this->m_build_task != nullptr,
                 "define_task() was not called; the task cannot be defined");
    Build_task f = this->m_build_task;
    async(idx, new Task([f, idx]() { f(idx); }));
}

/* TODO: fulfill_promise is nearly the same as Task_flow.
 * Need to make this part of the base class.
 * The only difference is find_task. */
void Channel::fulfill_promise(int3 idx) {
    auto atomic_count = find_promise(idx);

    // Decrement counter
    int wait_count = std::atomic_fetch_sub(atomic_count, 1);

    if (wait_count == -1) { // Uninitialized task
        throw_assert(m_dep_count != nullptr,
                     "dependency_count() was not called, the promise cannot be initialized");
        int wait_count_init = m_dep_count(idx) + 1;
        wait_count = std::atomic_fetch_add(atomic_count, wait_count_init);
        // wait_count = value before add operation
        wait_count += wait_count_init;
    }

    if (wait_count == 0) { // task is ready to run
        throw_assert(this->m_build_task != nullptr,
                     "define_task() was not called; the task cannot be defined");
        Build_task f = this->m_build_task;
        async(idx, new Task([f, idx]() { f(idx); }));
    }
}

// -----------------
// Task flow context

Context::Context() : empty_task{nullptr, 0, 0, 0}, empty_comm{nullptr} {}