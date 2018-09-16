//
// Created by Eric Darve on 9/16/18.
//

#ifndef GTFXX_UTIL_H
#define GTFXX_UTIL_H

#define LOGGER_OUTPUT_ON

// Builds a logging message and writes it out when calling the destructor
class LogMessage {
public:
    LogMessage(const char *file, const char *function, int line) {
        os << file << ":" << line << " (" << function << ") ";
    }

    // output operator
    template<typename T>
    LogMessage &operator<<(const T &t) {
        os << t;
        return *this;
    }

    ~LogMessage() {
        os << "\n";
        std::cout << os.str();
        std::cout.flush();
    }

private:
    std::ostringstream os;
};

#ifdef LOGGER_OUTPUT_ON

#define LOG(out) do { \
LogMessage(__FILE__,__func__,__LINE__) << out; \
} while (0)

#else
#define LOG(out)
#endif

/// Exception type for assertion failures
class AssertionFailureException : public std::exception {
private:
    const char *expression;
    const char *file;
    int line;
    std::string message;
    std::string report;

public:
    // Helper class for formatting assertion message
    class StreamFormatter {
    private:
        std::ostringstream stream;

    public:
        operator std::string() const {
            return stream.str();
        }

        template<typename T>
        StreamFormatter &operator<<(const T &value) {
            stream << value;
            return *this;
        }
    };

    // Log error before throwing
    void LogError() {
        std::cerr << report << std::endl;
    }

    // Construct an assertion failure exception
    AssertionFailureException(const char *expression, const char *file, int line, const std::string &message)
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
    virtual const char *what() const throw() {
        return report.c_str();
    }

    // The expression which was asserted to be true
    const char *Expression() const throw() {
        return expression;
    }

    // Source file
    const char *File() const throw() {
        return file;
    }

    // Source line
    int Line() const throw() {
        return line;
    }

    // Description of failure
    const char *Message() const throw() {
        return message.c_str();
    }

    ~AssertionFailureException() throw() {
    }
};

// Assert that EXPRESSION evaluates to true, otherwise raise AssertionFailureException
// with associated MESSAGE (which may use C++ stream-style message formatting).
#define throw_assert(EXPRESSION, MESSAGE) if(!(EXPRESSION)) { throw AssertionFailureException(#EXPRESSION, __FILE__, __LINE__, (AssertionFailureException::StreamFormatter() << MESSAGE)); }

#endif //GTFXX_UTIL_H
