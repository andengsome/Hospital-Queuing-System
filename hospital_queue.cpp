#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <omp.h>

using namespace std;

constexpr int    NUM_COUNTERS = 4;
constexpr double LAMBDA       = 5.0;
constexpr double SVC_MEAN     = 8.0;
constexpr double SVC_STD      = 2.0;

struct Patient {
    int    id;
    double arrival_time;
};

struct ProcessedPatient {
    int    id;
    double arrival_time;
    double svc_start;
    double service_dur;
    double finish_time;
    int    counter_id;
};

// CPU-load kernel: count primes <= n
static int count_primes(int n) {
    int cnt = 0;
    for (int i = 2; i <= n; i++) {
        bool ok = true;
        for (int j = 2; j * j <= i; j++) {
            if (i % j == 0) { ok = false; break; }
        }
        if (ok) cnt++;
    }
    return cnt;
}

struct TrialResult {
    int    n;
    double total_sim_time, throughput, avg_queue_wait, avg_service_time;
    double cpu_utilization, counter_util[NUM_COUNTERS], wall_clock_sec;
};

TrialResult run_trial(int n_patients, int trial_num) {
    // Generate all patients upfront (Poisson arrivals)
    vector<Patient> patients(n_patients);
    {
        mt19937 arr_rng(1234 + trial_num * 7);
        exponential_distribution<double> arr_dist(LAMBDA);
        double sim_time = 0.0;
        for (int i = 0; i < n_patients; i++) {
            sim_time += arr_dist(arr_rng);
            patients[i] = { i + 1, sim_time };
        }
    }

    // Per-counter state
    double counter_free_at[NUM_COUNTERS] = {};
    double counter_busy[NUM_COUNTERS]    = {};
    int    counter_patients[NUM_COUNTERS] = {};

     vector<ProcessedPatient> processed(n_patients);

    // Assignment index — round-robin with earliest-free selection
    // Process patients in arrival order; assign each to the counter
    // that will be free soonest (mimics a real parallel queue dispatcher)
    double wall_t0 = omp_get_wtime();

    // Sequential dispatch loop — assigns each patient to a counter
    for (int i = 0; i < n_patients; i++) {
        // Find counter that finishes earliest
        int best = 0;
        for (int c = 1; c < NUM_COUNTERS; c++)
            if (counter_free_at[c] < counter_free_at[best]) best = c;

        double svc_start =  max(patients[i].arrival_time, counter_free_at[best]);
        processed[i].id           = patients[i].id;
        processed[i].arrival_time = patients[i].arrival_time;
        processed[i].svc_start    = svc_start;
        processed[i].counter_id   = best;
        // Service duration sampled per-patient (seeded by patient id + counter)
        unsigned seed = (unsigned)(patients[i].id * 1327 + best * 31 + trial_num);
         mt19937 rng(seed);
         normal_distribution<double> svc_dist(SVC_MEAN, SVC_STD);
        double svc_dur =  max(1.0, (double)svc_dist(rng));
        processed[i].service_dur  = svc_dur;
        processed[i].finish_time  = svc_start + svc_dur;
        counter_free_at[best]     = processed[i].finish_time;
    }

    // Parallel processing: each patient's CPU-load work runs in parallel
    // OpenMP parallel for distributes patient processing across threads
    #pragma omp parallel for num_threads(NUM_COUNTERS) schedule(dynamic)
    for (int i = 0; i < n_patients; i++) {
        int prime_limit = static_cast<int>(processed[i].service_dur * 750);
        volatile int _x = count_primes(prime_limit);
    }

    double wall_t1 = omp_get_wtime();
    double wall_sec = wall_t1 - wall_t0;

    // Accumulate per-counter stats
    for (int i = 0; i < n_patients; i++) {
        int c = processed[i].counter_id;
        counter_busy[c]     += processed[i].service_dur;
        counter_patients[c]++;
    }

    // Find total sim time (last finish)
    double sim_end = 0.0;
    for (auto& pp : processed)
        if (pp.finish_time > sim_end) sim_end = pp.finish_time;

    // Compute averages
    double total_wait = 0.0, total_svc = 0.0;
    for (auto& pp : processed) {
        total_wait +=  max(0.0, pp.svc_start - pp.arrival_time);
        total_svc  += pp.service_dur;
    }

    double total_busy = 0.0;
    for (int c = 0; c < NUM_COUNTERS; c++) total_busy += counter_busy[c];

    TrialResult r;
    r.n                = n_patients;
    r.total_sim_time   = sim_end;
    r.throughput       = n_patients / sim_end;
    r.avg_queue_wait   = total_wait / n_patients;
    r.avg_service_time = total_svc  / n_patients;
    r.cpu_utilization  = (total_busy / (sim_end * NUM_COUNTERS)) * 100.0;
    r.wall_clock_sec   = wall_sec;
    for (int c = 0; c < NUM_COUNTERS; c++)
        r.counter_util[c] = (counter_busy[c] / sim_end) * 100.0;
    return r;
}

void print_result(const TrialResult& r, int t) {
    cout <<  fixed <<  setprecision(2);
    cout << "\n============= TRIAL " << t << ": " << r.n << " Patients =============\n";
    cout << "Total Simulation Time   : " << r.total_sim_time    << " min\n";
    cout << "Throughput              : " << r.throughput         << " patients/min\n";
    cout << "Avg Queue Wait Time     : " << r.avg_queue_wait     << " min\n";
    cout << "Avg Service Time        : " << r.avg_service_time   << " min\n";
    cout << "CPU Utilization         : " << r.cpu_utilization    << "%\n";
    for (int i = 0; i < NUM_COUNTERS; i++)
        cout << "Counter " << (i+1) << " Utilization   : " << r.counter_util[i] << "%\n";
    cout << "(Wall-clock time        : " << r.wall_clock_sec << " s)\n";
}

int main() {
    cout << "\n==========================================================\n";
    cout << "  Hospital Parallel Queue Simulation (C++/OpenMP)\n";
    cout << "==========================================================\n";
    cout << "Arrival rate     : lambda = " << LAMBDA    << " patients/min\n";
    cout << "Service time     : mean = "   << SVC_MEAN  << " min, std = " << SVC_STD << " min\n";
    cout << "Counters/threads : "          << NUM_COUNTERS << "\n";
    cout << "CPU-load kernel  : prime counting (proportional to service time)\n";
    cout << "OpenMP threads   : " << NUM_COUNTERS << "\n";

    int trials[] = {100, 300, 500};
    for (int t = 0; t < 3; t++)
        print_result(run_trial(trials[t], t+1), t+1);

    cout << "\n============== Simulation Complete ==============\n" << endl;
    return 0;
}