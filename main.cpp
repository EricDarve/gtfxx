#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <cassert>
#include <exception>
#include <stdexcept>

#include <random>

#include <list>
#include <queue>
#include <array>
#include <unordered_map>
#include <vector>

#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <chrono>

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

using namespace std::chrono;

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::min;
using std::max;

using std::list;
using std::vector;
using std::unordered_map;
using std::array;

using std::thread;
using std::mutex;
using std::bind;
using std::atomic_int;

typedef std::function<void()> Base_task;

// Builds a logging message and outputs it in the destructor
class LogMessage {
public:
    LogMessage(const char * file, const char * function, int line)
    {
        os << file << ":" << line << " (" << function << ") ";
    }

    // output operator
    template<typename T>
    LogMessage & operator<<(const T & t)
    {
        os << t;
        return *this;
    }

    ~LogMessage()
    {
        os << "\n";
        std::cout << os.str();
        std::cout.flush();
    }
private:
    std::ostringstream os;
};

#if 0
#define LOG(out) do { \
  LogMessage(__FILE__,__func__,__LINE__) << out; \
} while (0)
#else
#define LOG(out)
#endif

void f_1()
{
}

void f_2()
{
}

void f_x(int * x)
{
    --(*x);
};

void f_count(std::atomic_int * c)
{
    ++(*c);
    std::this_thread::sleep_for(microseconds(20));
};

struct Fun_mv {
    MatrixXd *A;
    VectorXd *x, *y;
    void operator()()
    {
        (*y) = (*A) * (*x);
    }
};

struct Task : public Base_task {
    atomic_int indegree; // Number of incoming edges

    bool m_delete = false;
    // whether the object should be deleted after running the function
    // to completion.

    float priority = 0.0; // task priority

    Task() {}

    Task(Base_task a_f, int a_indegree = 0, bool a_del = false, float a_priority = 0.0) :
        Base_task(a_f), m_delete(a_del), priority(a_priority)
    {
        indegree.store(a_indegree);
    }

    ~Task() {};

    void operator=(Base_task a_tsk)
    {
        Base_task::operator=(a_tsk);
    }
};

// Task comparison based on their priorities
struct Task_comparison {
public:
    bool operator() (const Task* a_lhs, const Task* a_rhs) const
    {
        return (a_lhs->priority < a_rhs->priority);
    };
};

struct Thread_prio;
void spin(Thread_prio * a_thread);
struct Thread_team;

// Thread with priority queue management
struct Thread_prio {
    Thread_team * team;
    unsigned short m_id;

    std::priority_queue<Task*, vector<Task*>, Task_comparison> ready_queue;
    thread th;
    mutex mtx;
    std::condition_variable cv;
    std::atomic_bool m_empty;
    // For optimization to avoid testing ready_queue.empty() in some cases
    std::atomic_bool m_stop;

    // Thread starts executing the function spin()
    void start()
    {
        m_empty.store(true);
        m_stop.store(false); // Used to return from spin()
        th = thread(spin, this); // Execute tasks in queue
    };

    // Add new task to queue
    void run(Task * a_t)
    {
        LOG(m_id);
        std::lock_guard<std::mutex> lck(mtx);
        ready_queue.push(a_t); // Add task to queue
        m_empty.store(false);
        cv.notify_one(); // Wake up thread
    };

    Task* pop()
    {
        Task* tsk = ready_queue.top();
        ready_queue.pop();
        if (ready_queue.empty()) m_empty.store(true);
        return tsk;
    };

    // Set stop boolean to true so spin() can return
    void stop()
    {
        LOG(m_id);
        std::lock_guard<std::mutex> lck(mtx);
        m_stop.store(true);
        cv.notify_one(); // Wake up thread
    };

    // join() the thread
    void join()
    {
        stop();
        if (th.joinable()) {
            th.join();
        }
    }

    ~Thread_prio()
    {
        join();
    }
};

struct Thread_team : public vector<Thread_prio*> {
    vector<Thread_prio> v_thread;
    unsigned long n_thread_query = 16; // Optimization parameter

    Thread_team(const int n_thread) : v_thread(n_thread)
    {
        for (int i=0; i<n_thread; ++i) {
            v_thread[i].team = this;
            v_thread[i].m_id = static_cast<unsigned short>(i);
        }
    }

    void start()
    {
        for (auto& th : v_thread) th.start();
    }
    void stop()
    {
        for (auto& th : v_thread) th.stop();
    }
    void join()
    {
        for (auto& th : v_thread) th.join();
    }

    void run(const int a_id, Task * a_task)
    {
        assert(a_id >= 0 && static_cast<unsigned long>(a_id) < v_thread.size());
        int id_ = a_id;
        // Check if queue is empty
        if (!v_thread[a_id].m_empty.load()) {
            // Check whether other threads have empty queues
            const unsigned n_query = min(n_thread_query, v_thread.size());
            for (unsigned long i=a_id+1; i<a_id+n_query; ++i) {
                auto j = i%v_thread.size();
                if (v_thread[j].m_empty.load()) {
                    id_ = j;
                    break;
                }
            }
        }
        LOG("requested thread: " << a_id << " got " << id_);
        // Note that because we are not using locks, the queue may no longer be empty.
        // But this implementation is more efficient than using locks.
        v_thread[id_].run(a_task);
    }

    void steal(unsigned short a_id)
    {
        const unsigned n_query = min(n_thread_query, v_thread.size());
        for (unsigned long i=a_id+1; i<a_id+n_query; ++i) {
            auto j = i%v_thread.size();
            Thread_prio & thread_ = v_thread[j];
            if (!thread_.m_empty.load()) {
                std::unique_lock<std::mutex> lck(thread_.mtx);
                if (!thread_.ready_queue.empty()) {
                    Task * tsk = thread_.pop();
                    lck.unlock();
                    LOG(a_id << " from " << j);
                    v_thread[a_id].run(tsk);
                    break;
                }
            }
        }
    }
};

// Keep executing tasks until m_stop = true && queue is empty
void spin(Thread_prio * a_thread)
{
    LOG(a_thread->m_id);
    std::unique_lock<std::mutex> lck(a_thread->mtx);
    while (true) {
        while (!a_thread->ready_queue.empty()) {
            Task * tsk = a_thread->pop();
            lck.unlock();
            LOG(a_thread->m_id << " task()");
            (*tsk)();
            if (tsk->m_delete) delete tsk;
            lck.lock();
        }
        // Try to steal a task
        lck.unlock();
        a_thread->team->steal(a_thread->m_id);
        lck.lock();
        // Wait if queue is empty
        while (a_thread->ready_queue.empty()) {
            // Return if stop=true
            if (a_thread->m_stop.load()) {
                LOG(a_thread->m_id << " stop");
                return;
            }
            LOG(a_thread->m_id << " wait");
            a_thread->cv.wait(lck);
        }
    }
};

namespace hash_array {

inline void hash_combine(std::size_t& seed, int const& v)
{
    seed ^= std::hash<int>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct hash {
    size_t
    operator()(array<int,3> const& key) const
    {
        size_t seed = 0;
        hash_combine(seed, key[0]);
        hash_combine(seed, key[1]);
        hash_combine(seed, key[2]);
        return seed;
    }
};

}

typedef array<int,3> int3;

struct Task_graph {
    Thread_team * team = nullptr;

    unordered_map< int3, Task*, hash_array::hash > graph;
    mutex mtx_graph, mtx_done;
    atomic_int n_tasks;
    std::condition_variable cond_done;

    bool done = false; // boolean: all tasks have been posted

    virtual ~Task_graph() {};

    // How to initialize a task
    virtual void initialize_task(int3&,Task*) = 0;
    // Mapping from int3 task index to thread id
    virtual int task_map(int3 & idx) = 0;

    // Find a task in the graph and return pointer
    Task * find_task(int3&);
    // Is the task ready to run?
    void isready_then_run(int3&, Task*);
    // Decrement the dependency counter and run task if ready
    void decrement(int3 &);

    // Returns when all tasks have been posted
    void wait()
    {
        std::unique_lock<std::mutex> lck(mtx_done);
        while (!done) cond_done.wait(lck);
    }

    // Decrement task counter by 1; return true when all tasks have been posted
    bool decrement_alldone()
    {
        --n_tasks;
        LOG(n_tasks.load());
        assert(n_tasks.load() >= 0);
        return (n_tasks.load() == 0);
    }
};

Task * Task_graph::find_task(int3 & idx)
{
    Task * t_ = nullptr;

    std::lock_guard<std::mutex> lck(mtx_graph);
    auto tsk = graph.find(idx);

    // Task exists
    if (tsk != graph.end()) {
        t_ = tsk->second;
    }
    else {
        // Task does not exist; create it
        t_ = new Task;
        graph[idx] = t_; // Insert in graph
        initialize_task(idx,t_);
    }

    assert(t_ != nullptr);

    return t_;
}

void Task_graph::isready_then_run(int3 & idx, Task * a_tsk)
{
    assert(a_tsk != nullptr);
    assert(a_tsk->indegree.load() >= 0);

    if (a_tsk->indegree.load() == 0) { // task is ready to run

        LOG("run task " << idx[0] << " " << idx[1] << " " << idx[2]);

        team->run(task_map(idx), a_tsk);

        // Delete entry in graph
        {
            std::lock_guard<std::mutex> lck(mtx_graph);
            assert(graph.find(idx) != graph.end());
            graph.erase(idx);
        }

        // Test whether all tasks have been posted
        if (decrement_alldone()) {
            std::unique_lock<std::mutex> lck(mtx_done);
            done = true;
            cond_done.notify_one(); // Notify waiting thread
        }
    }
}

void Task_graph::decrement(int3 & idx)
{
    Task * t_ = find_task(idx);

    // Decrement counter
    --( t_->indegree );

    isready_then_run(idx, t_);
}

struct Block_matrix : vector<MatrixXd*> {
    int row, col;
    Block_matrix() {};
    Block_matrix(int a_row, int a_col) : vector<MatrixXd*>(a_row*a_col),
        row(a_row), col(a_col) {};
    void resize(int a_row, int a_col)
    {
        vector<MatrixXd*>::resize(a_row*a_col);
        row = a_row;
        col = a_col;
    }
    MatrixXd* & operator()(int i, int j)
    {
        assert(i>=0 && i<row);
        assert(j>=0 && j<col);
        return operator[](i + j*row);
    }
};

struct Gemm_graph: public Task_graph {
    int size_i = -1;
    int size_j = -1;
    int size_k = -1;
    Block_matrix A, B, C;

    void initialize_task(int3&, Task*);

    int task_map(int3 & idx)
    {
        return ( ( idx[0] + size_i * idx[1] ) % ( team->v_thread.size() ) );
    }

    void start();
};

void run_gemm(MatrixXd *A, MatrixXd *B, MatrixXd *C)
{
    *C += (*A) * (*B);
}

void fun_gemm(Gemm_graph * m_g, int i, int j, int k)
{
    assert(i>=0 && i<m_g->size_i);
    assert(j>=0 && j<m_g->size_j);
    assert(k>=0 && k<m_g->size_k);
    LOG(i << " " << j << " " << k);

    run_gemm(m_g->A(i,k), m_g->B(k,j), m_g->C(i,j));

    if (k < m_g->size_k - 1) {
        // Release dependency on the next task
        int3 idx = {i,j,k+1};
        m_g->decrement(idx);
    }
}

void Gemm_graph::initialize_task(int3 & idx, Task* a_tsk)
{
    assert(a_tsk != nullptr);
    if (idx[2] > 0) { // index k
        a_tsk->indegree.store(1);
    }
    else {
        a_tsk->indegree.store(0);
    }
    a_tsk->m_delete = true; // delete memory after execution
    (*a_tsk) = bind(fun_gemm,this,idx[0],idx[1],idx[2]);
}

void Gemm_graph::start()
{
    assert(size_i>0);
    assert(size_j>0);
    assert(size_k>0);
    assert(A.size() == static_cast<unsigned long>(size_i*size_k));
    assert(B.size() == static_cast<unsigned long>(size_k*size_j));
    assert(C.size() == static_cast<unsigned long>(size_i*size_j));
    assert(n_tasks.load() == size_i*size_j*size_k);
    assert(team != nullptr);

    for (int i=0; i<size_i; ++i) {
        for (int j=0; j<size_j; ++j) {
            int3 idx = {i,j,0};
            Task * t_ = find_task(idx);
            isready_then_run(idx, t_);
        }
    }
}

void test();

int main(void)
{

    try {
        test();
    }
    catch (std::exception& a_e) {
        std::cout << a_e.what() << '\n';
    }

    return 0;
}

void test()
{

    {
        Task t1( f_1 );
        assert(t1.indegree.load() == 0);

        Task t2( f_2, 4 );
        assert(t2.indegree.load() == 4);

        t1();
        t2();
    }

    {
        int x = 2;
        assert(x == 2);

        Task t(bind(f_x, &x));
        t();

        assert(x == 1);
    }

    {
        MatrixXd A(2,2);
        A << 1,1,2,-2;
        assert(A(0,0) == 1);
        assert(A(0,1) == 1);
        assert(A(1,0) == 2);
        assert(A(1,1) == -2);
        VectorXd x(2);
        x << 3,2;
        assert(x(0) == 3);
        assert(x(1) == 2);
        VectorXd y(2);

        // Demonstrating a function with arguments without bind()
        Fun_mv f_mv;
        f_mv.A = &A;
        f_mv.x = &x;
        f_mv.y = &y;

        Task t(f_mv);
        t();

        assert(y(0) == 5);
        assert(y(1) == 2);
    }

#if 0
    {
        const int n_thread = 4;
        const int max_count = 10;

        // Create thread team
        Thread_team team(n_thread);
        vector<std::atomic_int> counter(n_thread);

        for(auto & c : counter) {
            c.store(0);
        }

        vector<Task> tsk(n_thread);
        for(int nt=0; nt<n_thread; ++nt) {
            tsk[nt] = bind(f_count, &counter[nt]);
        }

#ifdef PROFILER
        ProfilerStart("ctxx.pprof");
#endif
        {
            team.start();
#ifdef PROFILER
            auto start = high_resolution_clock::now();
#endif
            for(int it=0; it < max_count; ++it) {
                for(int nt=0; nt<n_thread; ++nt) {
                    team.run(0, &tsk[nt]);
                }
            }
            team.join();
#ifdef PROFILER
            auto end = high_resolution_clock::now();
            auto duration_ = duration_cast<milliseconds>(end - start);
            std::cout << "Elapsed: " << duration_.count() << "\n";
#endif
        }

        for(int nt=0; nt<n_thread; ++nt) {
            assert(counter[nt].load() == max_count);
        }

        for(auto & c : counter) {
            c.store(0);
        }

        {
            team.start();
#ifdef PROFILER
            auto start = high_resolution_clock::now();
#endif
            for(int it=0; it < max_count; ++it) {
                for(int nt=0; nt<n_thread; ++nt) {
                    team.run(nt, &tsk[nt]);
                }
            }
            team.join();
#ifdef PROFILER
            auto end = high_resolution_clock::now();
            auto duration_ = duration_cast<milliseconds>(end - start);
            std::cout << "Elapsed: " << duration_.count() << "\n";
#endif
        }

#ifdef PROFILER
        ProfilerStop();
#endif

        for(int nt=0; nt<n_thread; ++nt) {
            assert(counter[nt].load() == max_count);
        }
    }
#endif

    {
        const int nb = 4; // number of blocks
        const int b = 128; // size of blocks

        const int n_thread = 4; // Number of threads to use

        const int n = b*nb; // matrix size

        MatrixXd A(n,n), B(n,n);

        LOG("matrix");

        // Initialize GEMM matrices
        std::mt19937 mers_rand;
        // Seed the engine
        mers_rand.seed(2018);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                A(i,j) = mers_rand()%3;
                B(i,j) = mers_rand()%3;
            }
        }

        auto start = high_resolution_clock::now();
        auto C = A*B;
        auto x = C(0,0);
        auto end = high_resolution_clock::now();
        auto duration_ = duration_cast<milliseconds>(end - start);
        std::cout << "A*B elapsed: " << duration_.count() << "\n";

        LOG("init");

        Gemm_graph gemm_g;
        gemm_g.size_i = nb;
        gemm_g.size_j = nb;
        gemm_g.size_k = nb;
        gemm_g.A.resize(nb,nb);
        gemm_g.B.resize(nb,nb);
        gemm_g.C.resize(nb,nb);
        gemm_g.n_tasks.store(nb*nb*nb);

        LOG("graph");

        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                gemm_g.A(i,j) = new MatrixXd(A.block(i*b,j*b,b,b));
                gemm_g.B(i,j) = new MatrixXd(B.block(i*b,j*b,b,b));
                gemm_g.C(i,j) = new MatrixXd(MatrixXd::Zero(b,b));
            }
        }

        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                assert(*gemm_g.A(i,j) == A.block(i*b,j*b,b,b));
                assert(*gemm_g.B(i,j) == B.block(i*b,j*b,b,b));
                assert(*gemm_g.C(i,j) == MatrixXd::Zero(b,b));
            }
        }


        start = high_resolution_clock::now();
        // Calculate matrix product using blocks
        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                for (int k=0; k <nb; ++k) {
                    *(gemm_g.C(i,j)) += *(gemm_g.A(i,k)) * *(gemm_g.B(k,j));
                }
            }
        }
        end = high_resolution_clock::now();
        duration_ = duration_cast<milliseconds>(end - start);
        std::cout << "Block A*B elapsed: " << duration_.count() << "\n";

        // First test
        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                assert(*gemm_g.C(i,j) == C.block(i*b,j*b,b,b));
            }
        }

        // Copy back for testing purposes
        MatrixXd C0(n,n);
        for (int i=0; i<nb; ++i) {
            for (int j=0; j<nb; ++j) {
                MatrixXd & M = *(gemm_g.C(i,j));
                for (int i0=0; i0<b; ++i0) {
                    for (int j0=0; j0<b; ++j0) {
                        C0(i0+b*i,j0+b*j) = M(i0,j0);
                    }
                }
            }
        }

        LOG("graph_init");

        // Second test
        assert(C == C0);

        // Re-doing calculation using task graph

        // Init C
        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                *(gemm_g.C(i,j)) = MatrixXd::Zero(b,b);
            }
        }

        // Create thread team
        Thread_team team(n_thread);

        gemm_g.team = &team;

        // Start team of threads
        team.start();

#ifdef PROFILER
        ProfilerStart("ctxx.pprof");
#endif
        start = high_resolution_clock::now();

        LOG("start");
        // Create seed tasks in graph and start
        gemm_g.start();

        LOG("wait");
        // Wait for all tasks to be posted
        gemm_g.wait();

        LOG("join");
        // Wait for end of task queue execution
        team.join();

#ifdef PROFILER
        ProfilerStop();
#endif

        end = high_resolution_clock::now();
        duration_ = duration_cast<milliseconds>(end - start);
        std::cout << "CTXX GEMM elapsed: " << duration_.count() << "\n";

        LOG("test");

        // Test output
        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                assert(*gemm_g.C(i,j) == C.block(i*b,j*b,b,b));
            }
        }
    }
}
