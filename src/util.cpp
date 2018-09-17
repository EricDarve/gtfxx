//
// Created by Eric Darve on 9/16/18.
//

#include "util.h"

// --------------
// Logging utility

LogMessage::LogMessage(const char *file, const char *function, int line) {
    os << file << ":" << line << " (" << function << ") ";
}

LogMessage::~LogMessage() {
    os << "\n";
    std::cout << os.str();
    std::cout.flush();
}

// ----------------------------
// Assert with exception throws

AssertionFailureException::StreamFormatter::operator std::string() const {
    return stream.str();
}

// Log error before throwing
void AssertionFailureException::LogError() {
    std::cerr << report << std::endl;
}

// Construct an assertion failure exception
AssertionFailureException::AssertionFailureException(const char *expression, const char *file, int line,
                                                     const std::string &message)
        : expression(expression), file(file), line(line), message(message) {
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
    report = outputStream.str();

    LogError();
}

// The assertion message
const char *AssertionFailureException::what() const noexcept {
    return report.c_str();
}

// The expression which was asserted to be true
const char *AssertionFailureException::Expression() const noexcept {
    return expression;
}

// Source file
const char *AssertionFailureException::File() const noexcept {
    return file;
}

// Source line
int AssertionFailureException::Line() const noexcept {
    return line;
}

// Description of failure
const char *AssertionFailureException::Message() const noexcept {
    return message.c_str();
}

// --------
// Profiler

Profiling_event::Profiling_event(string s, string t_id) :
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

void Profiler::open(string s) {
    // Open a new file and clear up the list of events
    file_name = std::move(s);
    events.clear();
    thread_id_map.clear();
}

void Profiler::timestamp(string s) {
    auto thread_id = get_thread_id();
    Profiling_event prof_ev(std::move(s), thread_id);
    prof_ev.timestamp();
    {
        std::lock_guard<std::mutex> lck(mtx);
        events.push_back(prof_ev);
    }
}

Profiling_event *Profiler::start(string s) {
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
                break;
        }
    }

    std::fclose(fp);
}
