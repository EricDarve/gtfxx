#include <cstdio>
#include <cassert>

#include <random>

#include <list>
#include <queue>
#include <vector>

#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "gtest/gtest.h"


using std::min;
using std::max;

using std::list;
using std::vector;

using std::thread;
using std::mutex;

typedef std::function<void()> Task;

// Small logging framework
// Message is output when the object is destroyed
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

// Macro that calls LogMessage
#if 1
#define LOG(out) do { \
  LogMessage(__FILE__,__func__,__LINE__) << out; \
} while (0)
#else
#define LOG(out)
#endif

struct Thread_lite;
void spin(Thread_lite * a_thread);
struct Thread_pool;

// Thread with priority queue management
struct Thread_lite {
    Thread_pool* team;
    unsigned short m_id;

    std::queue<Task*> queue;
    thread th;
    mutex mtx;
    std::condition_variable cv;

    // Thread starts executing the function spin()
    void start()
    {
        th = thread(spin, this); // Execute tasks in queue
    };

    // Add new task to queue
    void spawn(Task* a_t)
    {
        std::lock_guard<std::mutex> lck(mtx);
        queue.push(a_t); // Add task to queue
        cv.notify_one(); // Wake up thread
    };

    Task* pop()
    {
        Task* tsk = queue.front();
        queue.pop();
        return tsk;
    };

    void notify()
    {
        std::lock_guard<std::mutex> lck(mtx);
        cv.notify_one(); // Wake up thread
    }

    // thread join()
    void join()
    {
        if (th.joinable()) {
            th.join();
        }
    }

    ~Thread_lite()
    {
        join();
    }
};

struct Thread_pool : public vector<Thread_lite*> {
    vector<Thread_lite> v_thread;
    std::atomic_int ntasks; // number of ready tasks in any thread queue
    std::atomic_bool m_stop; // boolean used for join()

    Thread_pool(const int n_thread) : v_thread(n_thread)
    {
        for (int i=0; i<n_thread; ++i) {
            v_thread[i].team = this;
            v_thread[i].m_id = static_cast<unsigned short>(i);
        }
        ntasks.store(0);
        m_stop.store(false);
    }

    void start()
    {
        ntasks.store(0);
        m_stop.store(false);
        for (auto& th : v_thread) th.start();
    }

    void join()
    {
        m_stop.store(true);
        if (ntasks.load() == 0) {
            wake_all();
        }
        for (auto& th : v_thread) th.join();
    }

    void wake_all()
    {
        if (atomic_fetch_sub(&ntasks,1) == 0) {
            for (auto& th : v_thread) th.notify();
        }
    }

    void spawn(const int a_id, Task * a_task)
    {
        assert(a_id >= 0 && static_cast<unsigned long>(a_id) < v_thread.size());
        ++ntasks;
        v_thread[a_id].spawn(a_task);
    }
};

// Keep executing tasks until m_stop = true && there are no tasks left anywhere
void spin(Thread_lite * a_thread)
{
    std::unique_lock<std::mutex> lck(a_thread->mtx);
    while (true) {
        while (!a_thread->queue.empty()) {
            Task * tsk = a_thread->pop();
            lck.unlock();
            (*tsk)();
            --(a_thread->team->ntasks);
            lck.lock();
        }
        while (a_thread->queue.empty()) {
            // Return if stop=true and no tasks are left
            if (a_thread->team->m_stop.load() && a_thread->team->ntasks.load() <= 0) {
                lck.unlock();
                // Wake up all threads and exits
                a_thread->team->wake_all();
                return;
            }
            // Wait if queue is empty
            a_thread->cv.wait(lck);
        }
    }
}
