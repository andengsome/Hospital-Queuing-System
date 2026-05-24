#include <iostream>
#include <queue>
#include <chrono>
#include <thread>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace std::chrono;

struct Patient {
    int id;
    double arrivalTime;
};

int main() {

    queue<Patient> patientQueue;

    const int totalPatients = 100;
    const int numCounters = 4;

    int processedPatients = 0;
    double totalWaitingTime = 0;

    auto simulationStart = high_resolution_clock::now();

    // Patient Arrival Simulation
    cout << "===== PATIENT ARRIVAL =====\n" << endl;

    for (int i = 1; i <= totalPatients; i++) {

        auto currentTime = high_resolution_clock::now();
        double arrival =
            duration<double>(currentTime - simulationStart).count();

        Patient p;
        p.id = i;
        p.arrivalTime = arrival;

        patientQueue.push(p);

        cout << "Patient " << p.id
             << " arrived at "
             << fixed << setprecision(2)
             << p.arrivalTime << " sec" << endl;

        // Simulate interval between arrivals
        this_thread::sleep_for(milliseconds(100));
    }

    cout << "\n===== PROCESSING PATIENTS =====\n" << endl;

    auto processingStart = high_resolution_clock::now();

    // Parallel Queue Processing
    #pragma omp parallel num_threads(numCounters)
    {
        while (true) {

            Patient currentPatient;
            bool hasPatient = false;

            double registrationTime = 0;
            double waitingTime = 0;
            double processingDuration = 0;

            // Access queue safely
            #pragma omp critical
            {
                if (!patientQueue.empty()) {

                    currentPatient = patientQueue.front();
                    patientQueue.pop();

                    auto now = high_resolution_clock::now();

                    registrationTime =
                        duration<double>(now - simulationStart).count();

                    waitingTime =
                        registrationTime - currentPatient.arrivalTime;

                    hasPatient = true;
                }
            }

            if (!hasPatient)
                break;

            int threadID = omp_get_thread_num();

            cout << "Counter/Thread "
                 << threadID
                 << " assigned to Patient "
                 << currentPatient.id
                 << endl;

            // Simulate registration and processing time
            auto processStart = high_resolution_clock::now();

            this_thread::sleep_for(milliseconds(500));

            auto processEnd = high_resolution_clock::now();

            processingDuration =
                duration<double>(processEnd - processStart).count();

            #pragma omp critical
            {
                processedPatients++;
                totalWaitingTime += waitingTime;

                cout << "Patient " << currentPatient.id
                     << " processed by Counter "
                     << threadID << endl;

                cout << "Arrival Time: "
                     << fixed << setprecision(2)
                     << currentPatient.arrivalTime
                     << " sec" << endl;

                cout << "Registration Time: "
                     << registrationTime
                     << " sec" << endl;

                cout << "Waiting Time: "
                     << waitingTime
                     << " sec" << endl;

                cout << "Processing Duration: "
                     << processingDuration
                     << " sec" << endl;

                cout << "-----------------------------------"
                     << endl;
            }
        }
    }

    auto simulationEnd = high_resolution_clock::now();

    double totalExecutionTime =
        duration<double>(simulationEnd - simulationStart).count();

    double averageWaitingTime =
        totalWaitingTime / processedPatients;

    cout << "\n===== SIMULATION SUMMARY =====\n" << endl;

    cout << "Total Patients: "
         << processedPatients << endl;

    cout << "Total Execution Time: "
         << fixed << setprecision(2)
         << totalExecutionTime
         << " sec" << endl;

    cout << "Average Waiting Time: "
         << averageWaitingTime
         << " sec" << endl;

    cout << "Number of Counters/Threads: "
         << numCounters << endl;

    return 0;
}
