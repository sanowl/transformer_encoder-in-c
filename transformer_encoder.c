#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define EPSILON 1e-6

void initialize_weights(double** weights, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights[i][j] = ((double) rand() / RAND_MAX) * sqrt(2.0 / cols);
        }
    }
    printf("Initialized weights for matrix of size %d x %d\n", rows, cols);
}

void initialize_bias(double* bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = ((double) rand() / RAND_MAX);
    }
    printf("Initialized bias of size %d\n", size);
}

double** create_matrix(int rows, int cols) {
    double** matrix = (double**) malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*) calloc(cols, sizeof(double));
    }
    printf("Created matrix of size %d x %d\n", rows, cols);
    return matrix;
}

void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
    printf("Freed matrix of size %d\n", rows);
}

void matmul(double** a, int a_rows, int a_cols, double** b, int b_rows, int b_cols, double** result) {
    memset(*result, 0, a_rows * b_cols * sizeof(double));
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            double sum = 0;
            for (int k = 0; k < a_cols; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    printf("Performed matrix multiplication of size %d x %d and %d x %d\n", a_rows, a_cols, b_rows, b_cols);
}

void transpose(double** matrix, int rows, int cols, double** transposed) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
    printf("Transposed matrix of size %d x %d\n", rows, cols);
}

void softmax(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double max_val = matrix[i][0];
        for (int j = 1; j < cols; j++) {
            if (matrix[i][j] > max_val) {
                max_val = matrix[i][j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = exp(matrix[i][j] - max_val);
            sum += matrix[i][j];
        }
        for (int j = 0; j < cols; j++) {
            matrix[i][j] /= sum;
        }
    }
    printf("Applied softmax to matrix of size %d x %d\n", rows, cols);
}

void layer_norm(double** matrix, int rows, int cols, double epsilon) {
    for (int i = 0; i < rows; i++) {
        double mean = 0.0;
        for (int j = 0; j < cols; j++) {
            mean += matrix[i][j];
        }
        mean /= cols;

        double variance = 0.0;
        for (int j = 0; j < cols; j++) {
            variance += (matrix[i][j] - mean) * (matrix[i][j] - mean);
        }
        variance /= cols;

        double stddev = sqrt(variance + epsilon);
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (matrix[i][j] - mean) / stddev;
        }
    }
    printf("Applied layer normalization to matrix of size %d x %d\n", rows, cols);
}

void add_bias(double** matrix, double* bias, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] += bias[j];
        }
    }
    printf("Added bias to matrix of size %d x %d\n", rows, cols);
}

void multi_head_attention(double** queries, double** keys, double** values, double** output, int seq_len, int embed_size, int head_dim) {
    double** keys_transposed = create_matrix(embed_size, seq_len);
    transpose(keys, seq_len, embed_size, keys_transposed);

    double** energy = create_matrix(seq_len, seq_len);
    matmul(queries, seq_len, embed_size, keys_transposed, embed_size, seq_len, energy);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            energy[i][j] /= sqrt(head_dim);
        }
    }

    softmax(energy, seq_len, seq_len);
    matmul(energy, seq_len, seq_len, values, seq_len, embed_size, output);

    free_matrix(keys_transposed, embed_size);
    free_matrix(energy, seq_len);
    printf("Performed multi-head attention for sequence length %d and embedding size %d\n", seq_len, embed_size);
}

void encoder_block(double** input, double** output, double** weights, double* bias, int rows, int cols) {
    matmul(input, rows, cols, weights, cols, cols, output);
    add_bias(output, bias, rows, cols);
    layer_norm(output, rows, cols, EPSILON);
    printf("Completed encoder block for matrix of size %d x %d\n", rows, cols);
}

int main() {
    srand(time(0));

    int embed_size = 512;
    int heads = 8;
    int head_dim = embed_size / heads;
    int batch_size = 64;
    int seq_len = 128;

    double** values = create_matrix(seq_len, embed_size);
    double** keys = create_matrix(seq_len, embed_size);
    double** queries = create_matrix(seq_len, embed_size);
    double** output = create_matrix(seq_len, embed_size);

    initialize_weights(values, seq_len, embed_size);
    initialize_weights(keys, seq_len, embed_size);
    initialize_weights(queries, seq_len, embed_size);

    multi_head_attention(queries, keys, values, output, seq_len, embed_size, head_dim);

    double** encoder_weights = create_matrix(embed_size, embed_size);
    double* encoder_bias = (double*) malloc(embed_size * sizeof(double));
    initialize_weights(encoder_weights, embed_size, embed_size);
    initialize_bias(encoder_bias, embed_size);

    double** encoder_output = create_matrix(seq_len, embed_size);
    encoder_block(output, encoder_output, encoder_weights, encoder_bias, seq_len, embed_size);

    printf("Encoder output size: %d x %d\n", seq_len, embed_size);

    free_matrix(values, seq_len);
    free_matrix(keys, seq_len);
    free_matrix(queries, seq_len);
    free_matrix(output, seq_len);
    free_matrix(encoder_weights, embed_size);
    free(encoder_bias);
    free_matrix(encoder_output, seq_len);

    return 0;
}