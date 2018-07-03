#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cassert>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <list>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

//#include <gperftools/profiler.h>

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::list;
using std::vector;
using std::thread;
using std::mutex;
using std::bind;
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
    int indegree = 0; // Number of incoming edges
    std::queue<Task*> outgoing; // List of outgoing edges
    float priority = 0; // task priority

    void operator()()
    {
        Base_task::operator()(); // Execute task
        decrement_outgoing(); // Decrement counters for outgoing edges
    }

    Task() {}

    Task( std::function<void()> a_f, int a_indegree = 0, float a_priority = 0.) :
        Base_task(a_f), indegree(a_indegree), priority(a_priority) {}
    ~Task() {};

    void decrement_outgoing()
    {
        while (!outgoing.empty()) {
            --outgoing.front()->indegree; // TODO race condition!
            outgoing.pop();
        }
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
    std::priority_queue<Task*, vector<Task*>, Task_comparison> ready_queue;
    thread th;
    mutex mtx;
    std::condition_variable cv;
    std::atomic_bool m_empty;
    // For optimization to avoid testing ready_queue.empty() in some cases
    std::atomic_bool m_stop;
    Thread_team * team;
    unsigned short m_id;

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
        LOG("lck " << m_id);
        std::lock_guard<std::mutex> lck(mtx);
        ready_queue.push(a_t); // Add task to queue
        m_empty.store(false);
        cv.notify_one(); // Wake up thread
        LOG("unlck " << m_id);
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
        int id_ = a_id;
        // Check if queue is empty
        if (!v_thread[a_id].m_empty.load()) {
            // Check whether other threads have empty queues
            for (unsigned long i=0; i<v_thread.size(); ++i) {
                if (static_cast<unsigned short>(i) != a_id && v_thread[i].m_empty.load()) {
                    id_ = i;
                    break;
                }
            }
        }
        // Note that because we are not using locks, the queue may no longer be empty.
        // But this implementation is more efficient than using locks.
        v_thread[id_].run(a_task);
    }

    void steal(unsigned short a_id)
    {
        for (unsigned long i=0; i<v_thread.size(); ++i) {
            if (static_cast<unsigned short>(i) != a_id) {
                Thread_prio & thread_ = v_thread[i];
                if (!thread_.m_empty.load()) {
                    LOG("lck " << a_id);
                    std::unique_lock<std::mutex> lck(thread_.mtx);
                    if (!thread_.ready_queue.empty()) {
                        Task * tsk = thread_.pop();
                        LOG("unlck " << a_id);
                        lck.unlock();
                        v_thread[a_id].run(tsk);
                        break;
                    }
                    LOG("unlck " << a_id);
                }
            }
        }
    }
};

// Keep executing tasks until m_stop = true && queue is empty
void spin(Thread_prio * a_thread)
{
    LOG("lck " << a_thread->m_id);
    std::unique_lock<std::mutex> lck(a_thread->mtx);
    while (true) {
        while (!a_thread->ready_queue.empty()) {
            Task * tsk = a_thread->pop();
            LOG("unlck " << a_thread->m_id);
            lck.unlock();
            (*tsk)();
            LOG("lck " << a_thread->m_id);
            lck.lock();
        }
        // Try to steal a task
        LOG("unlck " << a_thread->m_id);
        lck.unlock();
        a_thread->team->steal(a_thread->m_id);
        LOG("lck " << a_thread->m_id);
        lck.lock();
        // Wait if queue is empty
        while (a_thread->ready_queue.empty()) {
            // Return if stop=true
            if (a_thread->m_stop.load()) {
                LOG("unlck " << a_thread->m_id);
                return;
            }
            a_thread->cv.wait(lck);
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
        assert(t1.indegree == 0);

        Task t2( f_2, 4 );
        assert(t2.indegree == 4);

        t1.outgoing.push(&t2);

        t1();
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

    {
        const int n_thread = 4;

        // Create thread team
        Thread_team team(n_thread);
        vector<std::atomic_int> counter(n_thread);

        for(auto & c : counter) {
            c.store(0);
        }

        vector<Task> tsk(n_thread);
        for(int nt=0; nt<n_thread; ++nt) {
            tsk[nt] = static_cast<Task>(bind(f_count, &counter[nt]));
        }


        //ProfilerStart("ctxx.pprof");
        team.start();

        const int max_count = 1000;
        for(int it=0; it < max_count; ++it) {
            for(int nt=0; nt<n_thread; ++nt) {
                team.run(nt, &tsk[nt]);
            }
        }

        team.join();
        //ProfilerStop();

        for(int nt=0; nt<n_thread; ++nt) {
            assert(counter[nt].load() == max_count);
        }
    }
}
