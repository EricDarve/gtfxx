//
// Created by Eric Darve on 9/16/18.
//

#include "util.h"

// --------------
// Logging utility

Log_message::Log_message(const char *file, const char *function, int line) {
    os << file << ":" << line << " (" << function << ") ";
}

Log_message::~Log_message() {
    os << "\n";
    std::cout << os.str();
    std::cout.flush();
}

// ----------------------------
// Assert with exception throws

Assertion_failure_exception::StreamFormatter::operator std::string() const {
    return stream.str();
}

// Log error before throwing
void Assertion_failure_exception::log_error() {
    std::cerr << m_report << std::endl;
}

// Construct an assertion failure exception
Assertion_failure_exception::Assertion_failure_exception(const char *expression, const char *file, int line,
                                                     const std::string &message)
        : m_expression(expression), m_file(file), m_line(line), m_message(message) {
    std::ostringstream outputStream;

    if (!message.empty()) {
        outputStream << message << ": ";
    }

    std::string expressionString = expression;

    if (expressionString == "false" || expressionString == "0" || expressionString == "FALSE") {
        outputStream << "Unreachable code assertion";
        /* We asserted false to abort at a line that code code
         * should not be able to reach. */
    } else {
        outputStream << "Assertion '" << expression << "'";
    }

    outputStream << " failed in file '" << file << "' line " << line;
    m_report = outputStream.str();

    log_error();
}

// The assertion message
const char *Assertion_failure_exception::what() const noexcept {
    return m_report.c_str();
}

// The expression which was asserted to be true
const char *Assertion_failure_exception::expression() const noexcept {
    return m_expression;
}

// Source file
const char *Assertion_failure_exception::file() const noexcept {
    return m_file;
}

// Source line
int Assertion_failure_exception::line() const noexcept {
    return m_line;
}

// Description of failure
const char *Assertion_failure_exception::message() const noexcept {
    return m_message.c_str();
}

// --------
// Profiler

Profiling_event::Profiling_event(std::string s, std::string t_id) :
        name(std::move(s)), thread_id(std::move(t_id)), ev_type(uninitialized) {}

void Profiling_event::timestamp() {
    ev_type = timepoint;
    m_start = std::chrono::high_resolution_clock::now();
}

void Profiling_event::start() {
    ev_type = duration;
    m_start = std::chrono::high_resolution_clock::now();
}

void Profiling_event::stop() {
    m_stop = std::chrono::high_resolution_clock::now();
}

std::string id_to_string(std::thread::id id) {
    std::stringstream s_id;
    s_id << id;
    return s_id.str();
}

std::string get_thread_id() {
    return id_to_string(std::this_thread::get_id());
}

void Profiler::open(std::string s) {
    // Open a new file and clear up the list of events
    file_name = std::move(s);
    events.clear();
    thread_id_map.clear();
}

void Profiler::timestamp(std::string s) {
    auto thread_id = get_thread_id();
    Profiling_event prof_ev(std::move(s), thread_id);
    prof_ev.timestamp();
    {
        std::lock_guard<std::mutex> lck(mtx);
        events.push_back(prof_ev);
    }
}

Profiling_event *Profiler::start(std::string s) {
    auto thread_id = get_thread_id();
    Profiling_event *prof_ev = new Profiling_event(std::move(s), thread_id);
    prof_ev->start();
    return prof_ev;
}

void Profiler::stop(Profiling_event *prof_ev) {
    prof_ev->stop();
    {
        std::lock_guard<std::mutex> lck(mtx);
        events.push_back(*prof_ev);
    }
    delete prof_ev;
}

void Profiler::dump() {
    FILE *fp = std::fopen(file_name.c_str(), "w");
    if (!fp) {
        std::perror("File opening failed");
        return;
    }

    fprintf(fp, "nthreads %ld\n", thread_id_map.size());
    for (auto &it : thread_id_map) {
        fprintf(fp, "tidmap %s %d\n", it.first.c_str(), it.second);
    }

    for (auto &it : events) {
        auto d_start = std::chrono::duration<long long, std::nano>(it.m_start.time_since_epoch());
        switch (it.ev_type) {
            case Profiling_event::duration : {
                auto d_end = std::chrono::duration<long long, std::nano>(it.m_stop.time_since_epoch());
                fprintf(fp, "tid %s start %lld end %lld name %s\n",
                        it.thread_id.c_str(),
                        d_start.count(), d_end.count(), it.name.c_str());
            }
                break;
            case Profiling_event::timepoint :
                fprintf(fp, "tid %s timestamp %lld name %s\n", it.thread_id.c_str(),
                        d_start.count(), it.name.c_str());
                break;
            case Profiling_event::uninitialized :
                printf("Fatal error; we found an uninitialized profiling event");
                exit(1);
        }
    }

    std::fclose(fp);
}