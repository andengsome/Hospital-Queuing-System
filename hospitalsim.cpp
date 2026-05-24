#include <iostream>
#include <queue>
#include <random>
#include <chrono>
#include <thread>
#include <iomanip>
#include <windows.h>
#include <psapi.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// ==========================
// Patient Structure
// ==========================
struct Patient {
    int id;
    double arrivalTime;
};

// ==========================
// Get CPU Usage Function
// ==========================
double getCPUUsage() {

    FILETIME idleTime, kernelTime, userTime;

    static ULARGE_INTEGER lastCPU = {0};
    static ULARGE_INTEGER lastSysCPU = {0};
    static ULARGE_INTEGER lastUserCPU = {0};

    static int numProcessors =
        omp_get_num_procs();

    static HANDLE self =
        GetCurrentProcess();

    if (!GetProcessTimes(
            self,
            &idleTime,
            &idleTime,
            &kernelTime,
            &userTime))
        return -1.0;

    ULARGE_INTEGER sysKernel, sysUser;

    sysKernel.LowPart = kernelTime.dwLowDateTime;
    sysKernel.HighPart = kernelTime.dwHighDateTime;

    sysUser.LowPart = userTime.dwLowDateTime;
    sysUser.HighPart = userTime.dwHighDateTime;

    ULARGE_INTEGER now;
    now.QuadPart =
        GetTickCount64();

    double percent =
        (double)(
            (sysKernel.QuadPart - lastSysCPU.QuadPart) +
            (sysUser.QuadPart - lastUserCPU.QuadPart)
        );

    percent /= (now.QuadPart - lastCPU.QuadPart);
    percent /= numProcessors;

    lastCPU = now;
    lastSysCPU = sysKernel;
    lastUserCPU = sysUser;

    return percent * 100.0;
}

// ==========================
// Main Function
// ==========================
int main() {

    // ==========================
    // Simulation Parameters
    // ==========================
    const int totalPatients = 100;

    const int numCounters = 4;

    // λ = 5 patients/min
    // Scaled to 120 ms interval
    const int arrivalIntervalMs = 120;

    // Service Time
    // mean = 10 min
    // std = 3 min
    const double meanService = 1000.0;
    const double stdDevService = 300.0;

    // ==========================
    // Random Generator
    // ==========================
    random_device rd;
    mt19937 gen(rd());

    normal_distribution<double>
        serviceDistribution(
            meanService,
            stdDevService
        );

    queue<Patient> patientQueue;

    int processedPatients = 0;

    double totalWaitingTime = 0;
    double totalTurnaroundTime = 0;

    cout << "\n=========================================\n";
    cout << "HOSPITAL QUEUE SIMULATION\n";
    cout << "=========================================\n";

    cout << "\nPatients : "
         << totalPatients << endl;

    cout << "Counters : "
         << numCounters << endl;

    auto simulationStart =
        high_resolution_clock::now();

    // Initialize CPU tracking
    getCPUUsage();

    // ==========================
    // Patient Arrival
    // ==========================
    cout << "\n===== PATIENT ARRIVAL =====\n"
         << endl;

    for (int i = 1; i <= totalPatients; i++) {

        auto now =
            high_resolution_clock::now();

        double arrival =
            duration<double>(
                now - simulationStart
            ).count();

        Patient p;

        p.id = i;
        p.arrivalTime = arrival;

        patientQueue.push(p);

        cout << "Patient "
             << setw(3)
             << p.id
             << " arrived at "
             << fixed
             << setprecision(2)
             << p.arrivalTime
             << " sec"
             << endl;

        this_thread::sleep_for(
            milliseconds(arrivalIntervalMs)
        );
    }

    cout << "\n===== PROCESSING PATIENTS =====\n"
         << endl;

    // ==========================
    // Parallel Processing
    // ==========================
    #pragma omp parallel num_threads(numCounters)
    {
        while (true) {

            Patient currentPatient;

            bool hasPatient = false;

            double serviceStartTime = 0;
            double waitingTime = 0;

            // ==========================
            // Queue Access
            // ==========================
            #pragma omp critical
            {
                if (!patientQueue.empty()) {

                    currentPatient =
                        patientQueue.front();

                    patientQueue.pop();

                    auto now =
                        high_resolution_clock::now();

                    serviceStartTime =
                        duration<double>(
                            now - simulationStart
                        ).count();

                    waitingTime =
                        serviceStartTime
                        - currentPatient.arrivalTime;

                    hasPatient = true;
                }
            }

            if (!hasPatient)
                break;

            int threadID =
                omp_get_thread_num();

            // ==========================
            // Generate Service Time
            // ==========================
            double serviceTime =
                serviceDistribution(gen);

            if (serviceTime < 100)
                serviceTime = 100;

            cout << "\nCounter "
                 << threadID
                 << " assigned to Patient "
                 << currentPatient.id
                 << endl;

            auto processStart =
                high_resolution_clock::now();

            // Simulate processing
            this_thread::sleep_for(
                milliseconds((int)serviceTime)
            );

            auto processEnd =
                high_resolution_clock::now();

            double departureTime =
                duration<double>(
                    processEnd - simulationStart
                ).count();

            double turnaroundTime =
                departureTime
                - currentPatient.arrivalTime;

            // ==========================
            // Update Metrics
            // ==========================
            #pragma omp critical
            {
                processedPatients++;

                totalWaitingTime +=
                    waitingTime;

                totalTurnaroundTime +=
                    turnaroundTime;

                cout << "Patient "
                     << currentPatient.id
                     << " processed by Counter "
                     << threadID
                     << endl;

                cout << "Arrival Time       : "
                     << currentPatient.arrivalTime
                     << " sec" << endl;

                cout << "Service Start Time : "
                     << serviceStartTime
                     << " sec" << endl;

                cout << "Waiting Time       : "
                     << waitingTime
                     << " sec" << endl;

                cout << "Service Duration   : "
                     << serviceTime / 1000.0
                     << " sec" << endl;

                cout << "Departure Time     : "
                     << departureTime
                     << " sec" << endl;

                cout << "Turnaround Time    : "
                     << turnaroundTime
                     << " sec" << endl;

                cout << "----------------------------------"
                     << endl;
            }
        }
    }

    auto simulationEnd =
        high_resolution_clock::now();

    double totalExecutionTime =
        duration<double>(
            simulationEnd - simulationStart
        ).count();

    double averageWaitingTime =
        totalWaitingTime
        / processedPatients;

    double averageTurnaroundTime =
        totalTurnaroundTime
        / processedPatients;

    double throughput =
        processedPatients
        / totalExecutionTime;

    double cpuUsage =
        getCPUUsage();

    // ==========================
    // Summary
    // ==========================
    cout << "\n===== SIMULATION SUMMARY =====\n"
         << endl;

    cout << "Total Patients Processed : "
         << processedPatients
         << endl;

    cout << "Counters/Threads Used    : "
         << numCounters
         << endl;

    cout << "Total Execution Time     : "
         << fixed
         << setprecision(2)
         << totalExecutionTime
         << " sec"
         << endl;

    cout << "Average Waiting Time     : "
         << averageWaitingTime
         << " sec"
         << endl;

    cout << "Average Turnaround Time  : "
         << averageTurnaroundTime
         << " sec"
         << endl;

    cout << "Throughput               : "
         << throughput
         << " patients/sec"
         << endl;

    cout << "CPU Utilization          : "
         << cpuUsage
         << " %"
         << endl;

    cout << "\n=========================================\n";

    return 0;
}
