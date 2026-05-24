#include <iostream>
#include <queue>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <windows.h>
#include <psapi.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// ==========================
// Patient Structure
// ==========================
struct Patient {
    int    id;
    double arrivalTime;
};

// ==========================
// CPU Sampler
// ==========================
class CpuSampler {
public:
    CpuSampler() : running_(false) {
        self_         = GetCurrentProcess();
        numProcs_     = omp_get_num_procs();
        lastKernel_   = {0};
        lastUser_     = {0};
        lastWall_     = {0};
        // Prime the baseline so the first real delta is valid
        primeBaseline();
    }

    void start() {
        running_ = true;
        thread_  = thread(&CpuSampler::loop, this);
    }

    void stop() {
        running_ = false;
        if (thread_.joinable())
            thread_.join();
    }

    // Returns time-averaged mean CPU % across all samples
    double mean() const {
        if (samples_.empty()) return 0.0;
        return accumulate(samples_.begin(), samples_.end(), 0.0)
               / samples_.size();
    }

    // Returns peak CPU % seen during the simulation
    double peak() const {
        if (samples_.empty()) return 0.0;
        return *max_element(samples_.begin(), samples_.end());
    }

    const vector<double>& samples() const { return samples_; }

private:
    void primeBaseline() {
        FILETIME creation, exit, kernel, user, wall;
        GetProcessTimes(self_, &creation, &exit, &kernel, &user);
        GetSystemTimeAsFileTime(&wall);

        lastKernel_.LowPart  = kernel.dwLowDateTime;
        lastKernel_.HighPart = kernel.dwHighDateTime;
        lastUser_.LowPart    = user.dwLowDateTime;
        lastUser_.HighPart   = user.dwHighDateTime;
        lastWall_.LowPart    = wall.dwLowDateTime;
        lastWall_.HighPart   = wall.dwHighDateTime;
    }

    double sample() {
        FILETIME creation, exit, kernel, user, wall;
        if (!GetProcessTimes(self_, &creation, &exit, &kernel, &user))
            return 0.0;
        GetSystemTimeAsFileTime(&wall);

        ULARGE_INTEGER k, u, w;
        k.LowPart  = kernel.dwLowDateTime;  k.HighPart = kernel.dwHighDateTime;
        u.LowPart  = user.dwLowDateTime;    u.HighPart = user.dwHighDateTime;
        w.LowPart  = wall.dwLowDateTime;    w.HighPart = wall.dwHighDateTime;

        ULONGLONG cpuDelta  = (k.QuadPart - lastKernel_.QuadPart)
                            + (u.QuadPart - lastUser_.QuadPart);
        ULONGLONG timeDelta = w.QuadPart  - lastWall_.QuadPart;

        double pct = 0.0;
        if (timeDelta > 0)
            pct = ((double)cpuDelta / (double)timeDelta / numProcs_) * 100.0;

        lastKernel_ = k;
        lastUser_   = u;
        lastWall_   = w;

        return pct;
    }

    void loop() {
        while (running_) {
            samples_.push_back(sample());
            this_thread::sleep_for(milliseconds(100)); // 10 Hz — matches Python
        }
    }

    atomic<bool>      running_;
    thread            thread_;
    vector<double>    samples_;
    HANDLE            self_;
    int               numProcs_;
    ULARGE_INTEGER    lastKernel_;
    ULARGE_INTEGER    lastUser_;
    ULARGE_INTEGER    lastWall_;
};

int main() {

    // ==========================
    // Scale Factor - Real-world equivalents:
    //   Arrival interval : 12 sec → 24 ms  (scaled)
    //   Mean service     : 10 min → 1200 ms (scaled)
    //   Std dev service  : 2 min  → 240 ms  (scaled)
    // ==========================
    const double scaleFactor = 500.0;

    // ==========================
    // Simulation Parameters
    // ==========================
    const int totalPatients = 100; // Trial 1
    //const int totalPatients = 300; // Trial 2
    //const int totalPatients = 500; // Trial 3

    const int numCounters = 4;

    // Arrival: λ = 5 patients/min → 12 sec interval → 24 ms scaled
    const int arrivalIntervalMs = 24;

    // Service Time (scaled 1/500 from real-world values)
    // mean = 10 min → 1200 ms scaled
    // std  = 2 min  → 240 ms scaled
    const double meanService   = 1200.0;
    const double stdDevService = 240.0;

    // ==========================
    // Per-thread Random Generators
    // Each thread gets its own mt19937 — eliminates shared-state data race
    // ==========================
    vector<mt19937> generators(numCounters);
    random_device rd;
    for (int i = 0; i < numCounters; i++)
        generators[i].seed(rd() + i);

    normal_distribution<double> serviceDistribution(meanService, stdDevService);

    queue<Patient> patientQueue;

    int    processedPatients   = 0;
    double totalWaitingTime    = 0.0;
    double totalTurnaroundTime = 0.0;

    // Per-thread busy time tracking (index = thread ID)
    vector<double> threadBusyTime(numCounters, 0.0);
    vector<int>    threadPatientCount(numCounters, 0);

    cout << "\n====================================================\n";
    cout << "HOSPITAL QUEUE SIMULATION\n";
    cout << "====================================================\n";
    cout << "\nPatients (Jobs)      : " << totalPatients        << endl;
    cout << "Worker Threads       : " << numCounters           << endl;
    cout << "Arrival Interval     : " << arrivalIntervalMs
         << " ms (scaled 1/500 from 12s)"                      << endl;
    cout << "Service mean/std     : " << (int)meanService
         << " / " << (int)stdDevService
         << " ms (scaled 1/500 from 10 min)"                   << endl;
    cout << "Traffic Intensity (p): 12.50 (Overloaded Queue)"  << endl;
    cout << "====================================================\n";

    auto simulationStart = high_resolution_clock::now();

    // ==========================
    // Start CPU Sampler Thread
    // ==========================
    CpuSampler cpuSampler;
    cpuSampler.start();

    // ==========================
    // Patient Arrival
    // ==========================
    cout << "\n===== PATIENT ARRIVAL =====\n" << endl;

    for (int i = 1; i <= totalPatients; i++) {

        auto now    = high_resolution_clock::now();
        double arrival = duration<double>(now - simulationStart).count();

        Patient p;
        p.id          = i;
        p.arrivalTime = arrival;

        patientQueue.push(p);

        cout << "Patient "
             << setw(3) << p.id
             << " arrived at "
             << fixed << setprecision(2)
             << p.arrivalTime << " sec"
             << endl;

        this_thread::sleep_for(milliseconds(arrivalIntervalMs));
    }

    cout << "\n==================== PROCESSING PATIENTS ====================\n" << endl;

    // ==========================
    // Parallel Processing
    // ==========================
    #pragma omp parallel num_threads(numCounters)
    {
        int threadID = omp_get_thread_num();

        while (true) {

            Patient currentPatient;
            bool    hasPatient       = false;
            double  serviceStartTime = 0.0;
            double  waitingTime      = 0.0;

            // ==========================
            // Queue Access (critical)
            // ==========================
            #pragma omp critical
            {
                if (!patientQueue.empty()) {

                    currentPatient = patientQueue.front();
                    patientQueue.pop();

                    auto now = high_resolution_clock::now();
                    serviceStartTime =
                        duration<double>(now - simulationStart).count();
                    waitingTime =
                        serviceStartTime - currentPatient.arrivalTime;

                    hasPatient = true;
                }
            }

            if (!hasPatient)
                break;

            // ==========================
            // Generate Service Time
            // Each thread uses its own generator — no data race
            // ==========================
            double serviceTime = serviceDistribution(generators[threadID]);

            // Truncate minimum to 10 ms (~5 real seconds)
            if (serviceTime < 10.0)
                serviceTime = 10.0;

            cout << "\nCounter " << threadID
                 << " assigned to Patient " << currentPatient.id
                 << endl;

            // Simulate service
            this_thread::sleep_for(milliseconds((int)serviceTime));

            auto processEnd    = high_resolution_clock::now();
            double departureTime =
                duration<double>(processEnd - simulationStart).count();
            double turnaroundTime =
                departureTime - currentPatient.arrivalTime;

            // ==========================
            // Update Metrics (critical)
            // ==========================
            #pragma omp critical
            {
                processedPatients++;
                totalWaitingTime      += waitingTime;
                totalTurnaroundTime   += turnaroundTime;
                threadBusyTime[threadID]    += serviceTime / 1000.0; // ms → sec
                threadPatientCount[threadID]++;

                cout << "Patient "               << currentPatient.id
                     << " processed by Counter " << threadID          << endl;
                cout << "Arrival Time       : "
                     << fixed << setprecision(2)
                     << currentPatient.arrivalTime << " sec"          << endl;
                cout << "Service Start Time : "
                     << serviceStartTime           << " sec"          << endl;
                cout << "Waiting Time       : "
                     << waitingTime                << " sec"          << endl;
                cout << "Service Duration   : "
                     << serviceTime / 1000.0       << " sec"          << endl;
                cout << "Departure Time     : "
                     << departureTime              << " sec"          << endl;
                cout << "Turnaround Time    : "
                     << turnaroundTime             << " sec"          << endl;
                cout << "---------------------------------------"           << endl;
            }
        }
    }

    auto simulationEnd = high_resolution_clock::now();

    double totalExecutionTime =
        duration<double>(simulationEnd - simulationStart).count();

    // ==========================
    // Stop CPU Sampler
    // ==========================
    cpuSampler.stop();

    // ==========================
    // Compute Summary Metrics
    // ==========================
    double averageWaitingTime    = totalWaitingTime    / processedPatients;
    double averageTurnaroundTime = totalTurnaroundTime / processedPatients;
    double throughput            = processedPatients   / totalExecutionTime;

    // Average worker utilization across all threads
    double totalBusy = 0.0;
    for (int i = 0; i < numCounters; i++)
        totalBusy += threadBusyTime[i];
    double avgWorkerUtil = (totalBusy / (numCounters * totalExecutionTime)) * 100.0;

    // Per-thread utilization
    vector<double> perThreadUtil(numCounters);
    for (int i = 0; i < numCounters; i++)
        perThreadUtil[i] = (threadBusyTime[i] / totalExecutionTime) * 100.0;

    // CPU stats from continuous sampler
    double avgCpu  = cpuSampler.mean();
    double peakCpu = cpuSampler.peak();
    int    numSamples = (int)cpuSampler.samples().size();

    // ==========================
    // Scale back to real-world
    // ==========================
    double scaledExecMin    = (totalExecutionTime    * scaleFactor) / 60.0;
    double scaledAvgWaitMin = (averageWaitingTime     * scaleFactor) / 60.0;
    double scaledAvgTurnMin = (averageTurnaroundTime  * scaleFactor) / 60.0;
    double scaledThroughput =  throughput * (60.0 / scaleFactor);

    // ==========================
    // Summary
    // ==========================
    cout << "\n\n================== SIMULATION SUMMARY ==================\n" << endl;

    cout << "Total Patients Processed : "
         << processedPatients << endl;

    cout << "Counters/Threads Used    : "
         << numCounters << endl;

    cout << "Total Execution Time     : "
         << fixed << setprecision(2)
         << totalExecutionTime
         << " sec (scaled: " << scaledExecMin << " real min)"
         << endl;

    cout << "Average Waiting Time     : "
         << averageWaitingTime
         << " sec (scaled: " << scaledAvgWaitMin << " real min)"
         << endl;

    cout << "Average Turnaround Time  : "
         << averageTurnaroundTime
         << " sec (scaled: " << scaledAvgTurnMin << " real min)"
         << endl;

    cout << "Throughput               : "
         << throughput
         << " patients/sec (scaled: "
         << scaledThroughput << " patients/real min)"
         << endl;

    // Worker Utilization — average then per-thread breakdown
    cout << "\nWorker Utilization --------------------------------------" << endl;
    cout << "Average (all counters)   : "
         << fixed << setprecision(2)
         << avgWorkerUtil << " %" << endl;
    for (int i = 0; i < numCounters; i++) {
        cout << "  Counter " << i
             << "  -> "
             << perThreadUtil[i] << " %"
             << "  (" << threadPatientCount[i] << " patients,"
             << " busy " << fixed << setprecision(2)
             << threadBusyTime[i] << " sec)"
             << endl;
    }

    // CPU Utilization — mean + peak from continuous sampling
    cout << "\nSystem CPU Utilization ----------------------------------" << endl;
    cout << "Mean CPU (simulation)    : "
         << fixed << setprecision(2)
         << avgCpu << " %"
         << "  (over " << numSamples << " samples @ 100ms)"
         << endl;
    cout << "Peak CPU (simulation)    : "
         << peakCpu << " %" << endl;

    cout << "\n========================================================\n";

    return 0;
}
