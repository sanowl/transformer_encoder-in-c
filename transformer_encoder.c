#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#define EPSILON 1e-6f
#define EMBED_SIZE 512
#define SEQ_LEN 128
#define NUM_HEADS 8
#define HEAD_DIM (EMBED_SIZE / NUM_HEADS)

static inline float max_f(float a, float b) {
    return (a > b) ? a : b;
}

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

void initialize_weights(float *weights, int rows, int cols) {
    float scale = sqrtf(2.0f / (float)rows);
    #pragma omp parallel for
    for(int i = 0; i < rows * cols; i++) {
        weights[i] = rand_float() * scale;
    }
}

void initialize_bias(float *bias, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        bias[i] = rand_float() * 0.01f;
    }
}

void matmul_avx(const float *A, const float *B, float *C, int m, int k, int n) {
    #pragma omp parallel for
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            __m256 sum = _mm256_setzero_ps();
            int p;
            for(p = 0; p <= k - 8; p += 8) {
                __m256 a = _mm256_loadu_ps(&A[i * k + p]);
                __m256 b = _mm256_loadu_ps(&B[j * k + p]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            float total = 0.0f;
            for(int l = 0; l < 8; l++) total += temp[l];
            for(; p < k; p++) total += A[i * k + p] * B[j * k + p];
            C[i * n + j] = total;
        }
    }
}

void softmax_avx(float *matrix, int m, int n) {
    #pragma omp parallel for
    for(int i = 0; i < m; i++) {
        float max_val = matrix[i * n];
        for(int j = 1; j < n; j++) max_val = max_f(max_val, matrix[i * n + j]);
        __m256 max_vec = _mm256_set1_ps(max_val);
        __m256 sum_vec = _mm256_setzero_ps();
        int j;
        for(j = 0; j <= n - 8; j += 8) {
            __m256 val = _mm256_loadu_ps(&matrix[i * n + j]);
            val = _mm256_sub_ps(val, max_vec);
            __m256 exp_val = _mm256_exp_ps(val);
            _mm256_storeu_ps(&matrix[i * n + j], exp_val);
            sum_vec = _mm256_add_ps(sum_vec, exp_val);
        }
        float sum_temp[8];
        _mm256_storeu_ps(sum_temp, sum_vec);
        float sum = 0.0f;
        for(int l = 0; l < 8; l++) sum += sum_temp[l];
        for(; j < n; j++) {
            float val = expf(matrix[i * n + j] - max_val);
            matrix[i * n + j] = val;
            sum += val;
        }
        __m256 sum_div = _mm256_set1_ps(sum);
        for(j = 0; j <= n - 8; j += 8) {
            __m256 val = _mm256_loadu_ps(&matrix[i * n + j]);
            val = _mm256_div_ps(val, sum_div);
            _mm256_storeu_ps(&matrix[i * n + j], val);
        }
        for(; j < n; j++) matrix[i * n + j] /= sum;
    }
}

void layer_norm_avx(float *matrix, int m, int n, float epsilon) {
    #pragma omp parallel for
    for(int i = 0; i < m; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        int j;
        for(j = 0; j <= n - 8; j += 8) {
            __m256 val = _mm256_loadu_ps(&matrix[i * n + j]);
            sum_vec = _mm256_add_ps(sum_vec, val);
        }
        float sum_temp[8];
        _mm256_storeu_ps(sum_temp, sum_vec);
        float mean = 0.0f;
        for(int l = 0; l < 8; l++) mean += sum_temp[l];
        for(; j < n; j++) mean += matrix[i * n + j];
        mean /= n;
        __m256 mean_vec = _mm256_set1_ps(mean);
        __m256 var_sum_vec = _mm256_setzero_ps();
        for(j = 0; j <= n - 8; j += 8) {
            __m256 val = _mm256_loadu_ps(&matrix[i * n + j]);
            __m256 diff = _mm256_sub_ps(val, mean_vec);
            __m256 sq = _mm256_mul_ps(diff, diff);
            var_sum_vec = _mm256_add_ps(var_sum_vec, sq);
        }
        float var_sum_temp[8];
        _mm256_storeu_ps(var_sum_temp, var_sum_vec);
        float variance = 0.0f;
        for(int l = 0; l < 8; l++) variance += var_sum_temp[l];
        for(; j < n; j++) {
            float diff = matrix[i * n + j] - mean;
            variance += diff * diff;
        }
        variance /= n;
        float stddev = sqrtf(variance + epsilon);
        __m256 stddev_vec = _mm256_set1_ps(stddev);
        for(j = 0; j <= n - 8; j += 8) {
            __m256 val = _mm256_loadu_ps(&matrix[i * n + j]);
            __m256 normalized = _mm256_div_ps(_mm256_sub_ps(val, mean_vec), stddev_vec);
            _mm256_storeu_ps(&matrix[i * n + j], normalized);
        }
        for(; j < n; j++) matrix[i * n + j] = (matrix[i * n + j] - mean) / stddev;
    }
}

void add_bias_avx(float *matrix, const float *bias, int m, int n) {
    #pragma omp parallel for
    for(int i = 0; i < m; i++) {
        int j;
        for(j = 0; j <= n - 8; j += 8) {
            __m256 val = _mm256_loadu_ps(&matrix[i * n + j]);
            __m256 b = _mm256_loadu_ps(&bias[j]);
            val = _mm256_add_ps(val, b);
            _mm256_storeu_ps(&matrix[i * n + j], val);
        }
        for(; j < n; j++) matrix[i * n + j] += bias[j];
    }
}

void multi_head_attention_avx(const float *queries, const float *keys, const float *values, float *output,
                             const float *Wq, const float *Wk, const float *Wv, const float *Wo) {
    float *Q = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    float *K = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    float *V = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    matmul_avx(queries, Wq, Q, SEQ_LEN, EMBED_SIZE, EMBED_SIZE);
    matmul_avx(keys, Wk, K, SEQ_LEN, EMBED_SIZE, EMBED_SIZE);
    matmul_avx(values, Wv, V, SEQ_LEN, EMBED_SIZE, EMBED_SIZE);
    float *attn_outputs[NUM_HEADS];
    for(int h = 0; h < NUM_HEADS; h++) attn_outputs[h] = aligned_alloc(32, SEQ_LEN * HEAD_DIM * sizeof(float));
    #pragma omp parallel for
    for(int h = 0; h < NUM_HEADS; h++) {
        float energy[SEQ_LEN * SEQ_LEN] __attribute__((aligned(32)));
        for(int i = 0; i < SEQ_LEN; i++) {
            for(int j = 0; j < SEQ_LEN; j++) {
                __m256 q = _mm256_loadu_ps(&Q[i * EMBED_SIZE + h * HEAD_DIM]);
                __m256 k = _mm256_loadu_ps(&K[j * EMBED_SIZE + h * HEAD_DIM]);
                __m256 mul = _mm256_mul_ps(q, k);
                __m256 sum = _mm256_hadd_ps(mul, mul);
                sum = _mm256_hadd_ps(sum, sum);
                float temp[8];
                _mm256_storeu_ps(temp, sum);
                energy[i * SEQ_LEN + j] = temp[0] / sqrtf((float)HEAD_DIM);
            }
        }
        softmax_avx(energy, SEQ_LEN, SEQ_LEN);
        for(int i = 0; i < SEQ_LEN; i++) {
            for(int d = 0; d < HEAD_DIM; d++) {
                __m256 e = _mm256_loadu_ps(&energy[i * SEQ_LEN]);
                __m256 v = _mm256_loadu_ps(&V[d + h * HEAD_DIM + 0 * EMBED_SIZE]);
                __m256 mul = _mm256_mul_ps(e, v);
                __m256 sum = _mm256_setzero_ps();
                sum = _mm256_add_ps(sum, mul);
                float temp[8];
                _mm256_storeu_ps(temp, sum);
                attn_outputs[h][i * HEAD_DIM + d] = temp[0];
            }
        }
    }
    float *concat = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    memset(concat, 0, SEQ_LEN * EMBED_SIZE * sizeof(float));
    for(int h = 0; h < NUM_HEADS; h++) {
        for(int i = 0; i < SEQ_LEN; i++) {
            memcpy(&concat[i * EMBED_SIZE + h * HEAD_DIM], &attn_outputs[h][i * HEAD_DIM], HEAD_DIM * sizeof(float));
        }
    }
    matmul_avx(concat, Wo, output, SEQ_LEN, EMBED_SIZE, EMBED_SIZE);
    free(Q);
    free(K);
    free(V);
    for(int h = 0; h < NUM_HEADS; h++) free(attn_outputs[h]);
    free(concat);
}

void encoder_block_avx(float *input, float *output, const float *W1, const float *b1,
                      const float *W2, const float *b2) {
    float *linear1 = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    matmul_avx(input, W1, linear1, SEQ_LEN, EMBED_SIZE, EMBED_SIZE);
    add_bias_avx(linear1, b1, SEQ_LEN, EMBED_SIZE);
    layer_norm_avx(linear1, SEQ_LEN, EMBED_SIZE, EPSILON);
    matmul_avx(linear1, W2, output, SEQ_LEN, EMBED_SIZE, EMBED_SIZE);
    add_bias_avx(output, b2, SEQ_LEN, EMBED_SIZE);
    layer_norm_avx(output, SEQ_LEN, EMBED_SIZE, EPSILON);
    free(linear1);
}

int main() {
    srand((unsigned int)time(NULL));
    float *queries = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    float *keys = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    float *values = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    float *output = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    float *encoder_output = aligned_alloc(32, SEQ_LEN * EMBED_SIZE * sizeof(float));
    initialize_weights(queries, EMBED_SIZE, EMBED_SIZE);
    initialize_weights(keys, EMBED_SIZE, EMBED_SIZE);
    initialize_weights(values, EMBED_SIZE, EMBED_SIZE);
    float *encoder_W1 = aligned_alloc(32, EMBED_SIZE * EMBED_SIZE * sizeof(float));
    float *encoder_b1 = aligned_alloc(32, EMBED_SIZE * sizeof(float));
    float *encoder_W2 = aligned_alloc(32, EMBED_SIZE * EMBED_SIZE * sizeof(float));
    float *encoder_b2 = aligned_alloc(32, EMBED_SIZE * sizeof(float));
    initialize_weights(encoder_W1, EMBED_SIZE, EMBED_SIZE);
    initialize_bias(encoder_b1, EMBED_SIZE);
    initialize_weights(encoder_W2, EMBED_SIZE, EMBED_SIZE);
    initialize_bias(encoder_b2, EMBED_SIZE);
    float *Wq = aligned_alloc(32, EMBED_SIZE * EMBED_SIZE * sizeof(float));
    float *Wk = aligned_alloc(32, EMBED_SIZE * EMBED_SIZE * sizeof(float));
    float *Wv = aligned_alloc(32, EMBED_SIZE * EMBED_SIZE * sizeof(float));
    float *Wo = aligned_alloc(32, EMBED_SIZE * EMBED_SIZE * sizeof(float));
    initialize_weights(Wq, EMBED_SIZE, EMBED_SIZE);
    initialize_weights(Wk, EMBED_SIZE, EMBED_SIZE);
    initialize_weights(Wv, EMBED_SIZE, EMBED_SIZE);
    initialize_weights(Wo, EMBED_SIZE, EMBED_SIZE);
    multi_head_attention_avx(queries, keys, values, output, Wq, Wk, Wv, Wo);
    encoder_block_avx(output, encoder_output, encoder_W1, encoder_b1, encoder_W2, encoder_b2);
    printf("Encoder output size: %d x %d\n", SEQ_LEN, EMBED_SIZE);
    
    free(queries);
    free(keys);
    free(values);
    free(output);
    free(encoder_output);
    free(encoder_W1);
    free(encoder_b1);
    free(encoder_W2);
    free(encoder_b2);
    free(Wq);
    free(Wk);
    free(Wv);
    free(Wo);
    return 0;
}
