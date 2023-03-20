#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <omp.h>

std::vector<std::vector<int>> sequential_multiply_matrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    int n = A.size();
    int m = B.size();
    int k = B[0].size();

    std::vector<std::vector<int>> C(n, std::vector<int>(k));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int t = 0; t < m; ++t) {
                C[i][j] += A[i][t] * B[t][j];
            }
        }
    }
    return C;
}

std::vector<std::vector<int>> parallel_multiply_matrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    int n = A.size();
    int m = B.size();
    int k = B[0].size();

    std::vector<std::vector<int>> C(n, std::vector<int>(k));

    #pragma omp parallel for private
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int t = 0; t < m; ++t) {
                C[i][j] += A[i][t] * B[t][j];
            }
        }
    }

    return C;
}

std::vector<std::vector<int>> change_parallel_multiply_matrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    int n = A.size();
    int m = B.size();
    int k = B[0].size();

    std::vector<std::vector<int>> C(n, std::vector<int>(k));

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int t = 0; t < m; ++t){
            for (int j = 0; j < k; ++j) {
                C[i][j] += A[i][t] * B[t][j];
            }
        }
    }

    return C;
}

int main()
{
	int sizeMtx = 1000;
    int block_size = 10;
    srand(time(0));

    std::vector<std::vector<int>> A(sizeMtx, std::vector<int>(sizeMtx));
    std::vector<std::vector<int>> B(sizeMtx, std::vector<int>(sizeMtx));

    for (int i = 0; i < sizeMtx; ++i) {
        for (int j = 0; j < sizeMtx; ++j) {
            A[i][j] = rand() % 100 + 1;
        }
    }
    for (int i = 0; i < sizeMtx; ++i) {
        for (int j = 0; j < sizeMtx; ++j) {
            B[i][j] = rand() % 100 + 1;
        }
    }

    // Последовательное умножение
    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> seq_mtx = sequential_multiply_matrices(A, B);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds_par = end - start;
    std::cout << "Sequential multiplication took " << elapsed_seconds_par.count() << " seconds." << std::endl;

    // Параллельное умножение
    auto start1 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> par_mtx = parallel_multiply_matrices(A, B);
    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds_par1 = end1 - start1;
    std::cout << "Parallel multiplication took " << elapsed_seconds_par1.count() << " seconds." << std::endl;

    // Параллельное умножение c инверсией внутренних циклов
    auto start2 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> change_par_mtx = change_parallel_multiply_matrices(A, B);
    auto end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds_par2 = end2 - start2;
    std::cout << "Cycle swapping parallel multiplication took " << elapsed_seconds_par2.count() << " seconds." << std::endl;
}
