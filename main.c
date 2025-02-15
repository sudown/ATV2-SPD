#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CANDIDATES 10000
#define MAX_QUESTIONS 30
#define MIN_GD 4.0

typedef struct {
    int registration;
    int test_code;
    char answers[MAX_QUESTIONS];
} Candidate;

typedef struct {
    int registration;
    double score;
} Result;

void load_answer_key(char *answer_key, const char *file_path) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        printf("Error opening answer key file.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < MAX_QUESTIONS; i++) {
        fscanf(file, " %c,", &answer_key[i]);
    }
    fclose(file);
}

int load_candidates(Candidate *candidates, const char *file_path, int test_code) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        printf("Error opening candidates file.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int count = 0;
    while (count < MAX_CANDIDATES && fscanf(file, "%d,%d,", &candidates[count].registration, &candidates[count].test_code) != EOF) {
        if (candidates[count].test_code == test_code) {
            for (int i = 0; i < MAX_QUESTIONS; i++) {
                fscanf(file, " %c,", &candidates[count].answers[i]);
            }
            count++;
        } else {
            char buffer[100];
            fgets(buffer, sizeof(buffer), file);
        }
    }
    fclose(file);
    return count;
}

void calculate_difficulties(Candidate *candidates, int total, char *answer_key, double *gd) {
    int correct_answers[MAX_QUESTIONS] = {0};
    for (int i = 0; i < total; i++) {
        for (int j = 0; j < MAX_QUESTIONS; j++) {
            if (candidates[i].answers[j] == answer_key[j]) {
                correct_answers[j]++;
            }
        }
    }
    int max_correct = 0;
    for (int j = 0; j < MAX_QUESTIONS; j++) {
        if (correct_answers[j] > max_correct) {
            max_correct = correct_answers[j];
        }
    }
    for (int j = 0; j < MAX_QUESTIONS; j++) {
        gd[j] = (correct_answers[j] > 0) ? ((double)max_correct / correct_answers[j]) * MIN_GD : 10.0;
    }
}

void calculate_scores(double *gd, double *scores) {
    double sum_gd[3] = {0.0};
    for (int j = 0; j < MAX_QUESTIONS; j++) {
        sum_gd[j / 10] += gd[j];
    }
    for (int j = 0; j < MAX_QUESTIONS; j++) {
        scores[j] = (gd[j] / sum_gd[j / 10]) * 100;
    }
}

double calculate_score(char *answers, char *answer_key, double *scores) {
    double score = 0.0;
    for (int i = 0; i < MAX_QUESTIONS; i++) {
        if (answers[i] == answer_key[i]) {
            score += scores[i];
        }
    }
    return score;
}

void sort_results(Result *results, int total) {
    for (int i = 0; i < total - 1; i++) {
        for (int j = i + 1; j < total; j++) {
            if (results[i].score < results[j].score) {
                Result temp = results[i];
                results[i] = results[j];
                results[j] = temp;
            }
        }
    }
}

void save_results(const char *file_path, Result *results, int total, int test_code) {
    FILE *file = fopen(file_path, "w");
    if (!file) {
        printf("Error creating file %s\n", file_path);
        return;
    }
    fprintf(file, "Position,Code,Registration,Score\n");
    for (int i = 0; i < total; i++) {
        fprintf(file, "%d,%d,%d,%.1f\n", i + 1, test_code, results[i].registration, results[i].score);
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <test_code>\n", argv[0]);
        return 1;
    }
    int test_code = atoi(argv[1]);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char answer_key[MAX_QUESTIONS];
    Candidate candidates[MAX_CANDIDATES];
    int total_candidates = 0;

    if (rank == 0) {
        load_answer_key(answer_key, "./DATA/gabarito.csv");
        total_candidates = load_candidates(candidates, "./DATA/respostas.csv", test_code);
    }

    MPI_Bcast(&total_candidates, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(answer_key, MAX_QUESTIONS, MPI_CHAR, 0, MPI_COMM_WORLD);

    double gd[MAX_QUESTIONS], scores[MAX_QUESTIONS];
    if (rank == 0) {
        calculate_difficulties(candidates, total_candidates, answer_key, gd);
        calculate_scores(gd, scores);
    }
    MPI_Bcast(gd, MAX_QUESTIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(scores, MAX_QUESTIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_count = total_candidates / size + (rank < total_candidates % size);
    Candidate *my_candidates = malloc(local_count * sizeof(Candidate));
    MPI_Scatter(candidates, local_count * sizeof(Candidate), MPI_BYTE,
                my_candidates, local_count * sizeof(Candidate), MPI_BYTE,
                0, MPI_COMM_WORLD);

    Result *local_results = malloc(local_count * sizeof(Result));
    for (int i = 0; i < local_count; i++) {
        local_results[i].registration = my_candidates[i].registration;
        local_results[i].score = calculate_score(my_candidates[i].answers, answer_key, scores);
    }

    Result *final_results = NULL;
    if (rank == 0) {
        final_results = malloc(total_candidates * sizeof(Result));
    }

    MPI_Gather(local_results, local_count * sizeof(Result), MPI_BYTE,
               final_results, local_count * sizeof(Result), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        sort_results(final_results, total_candidates);
        save_results("./RESULT/resultado.csv", final_results, total_candidates, test_code);
        free(final_results);
    }

    free(my_candidates);
    free(local_results);
    MPI_Finalize();
    return 0;
}