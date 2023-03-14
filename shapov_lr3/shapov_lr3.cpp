#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>

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

    #pragma omp paralell for private (j, t) 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int t = 0; t < m; ++t) {
                C[i][j] += A[i][t] * B[t][j];
            }
        }
    }

    return C;
}

std::vector<std::vector<int>> block_multiply_matrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, int block_size) {
    int n = A.size();
    int m = B.size();
    int k = B[0].size();

    std::vector<std::vector<int>> C(n, std::vector<int>(k));

    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < k; j += block_size) {
            for (int t = 0; t < m; t += block_size) {
                for (int ii = i; ii < std::min(i + block_size, n); ii++) {
                    for (int jj = j; jj < std::min(j + block_size, k); jj++) {
                        for (int tt = t; tt < std::min(t + block_size, m); tt++) {
                            C[ii][jj] += A[ii][tt] * B[tt][jj];
                        }
                    }
                }
            }
        }
    }

    return C;
}

std::vector<std::vector<int>> block_multiply_matrices_parallel(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, int block_size) {
    int n = A.size();
    int m = B.size();
    int k = B[0].size();

    std::vector<std::vector<int>> C(n, std::vector<int>(k));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < k; j += block_size) {
            for (int t = 0; t < m; t += block_size) {
                for (int ii = i; ii < std::min(i + block_size, n); ii++) {
                    for (int jj = j; jj < std::min(j + block_size, k); jj++) {
                        for (int tt = t; tt < std::min(t + block_size, m); tt++) {
                            C[ii][jj] += A[ii][tt] * B[tt][jj];
                        }
                    }
                }
            }
        }
    }

    return C;
}

int main()
{
	int sizeMtx = 500;
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

    // Блоковый алгоритм
    auto start2 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> block_mtx = block_multiply_matrices(A, B, block_size);
    auto end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds_par2 = end2 - start2;
    std::cout << "Block multiplication took " << elapsed_seconds_par2.count() << " seconds." << std::endl;

    //Блоковый алгоритм с OpenMP
    auto start3 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> block_mtx_parallel = block_multiply_matrices_parallel(A, B, block_size);
    auto end3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds_par3 = end3 - start3;
    std::cout << "Block parallel multiplication took " << elapsed_seconds_par3.count() << " seconds." << std::endl;
}
