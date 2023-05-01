#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// функция потерь (здесь используем категориальную кросс-энтропию)
double loss_function(vector<vector<double>>& data, vector<int>& labels, vector<double>& weights) {
    int n = data.size();
    int num_features = data[0].size();
    int num_classes = 3;
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        vector<double> scores(num_classes, 0.0);
        for (int j = 0; j < num_classes; j++) {
            for (int k = 0; k < num_features; k++) {
                scores[j] += data[i][k] * weights[j * num_features + k];
            }
        }
        double sum_exp_scores = 0.0;
        for (int j = 0; j < num_classes; j++) {
            sum_exp_scores += exp(scores[j]);
        }
        for (int j = 0; j < num_classes; j++) {
            loss += labels[i] == j ? -scores[j] + log(sum_exp_scores) : 0.0;
        }
    }
    return loss / n;
}

// вычисление градиента функции потерь
vector<double> gradient(vector<vector<double>>& data, vector<int>& labels, vector<double>& weights) {
    int n = data.size();
    int num_features = data[0].size();
    int num_classes = 3;
    vector<double> grad(num_classes * num_features, 0.0);
    for (int i = 0; i < n; i++) {
        vector<double> scores(num_classes, 0.0);
        for (int j = 0; j < num_classes; j++) {
            for (int k = 0; k < num_features; k++) {
                scores[j] += data[i][k] * weights[j * num_features + k];
            }
        }
        double sum_exp_scores = 0.0;
        for (int j = 0; j < num_classes; j++) {
            sum_exp_scores += exp(scores[j]);
        }
        for (int j = 0; j < num_classes; j++) {
            double prob = exp(scores[j]) / sum_exp_scores;
            for (int k = 0; k < num_features; k++) {
                grad[j * num_features + k] += (prob - (labels[i] == j ? 1.0 : 0.0)) * data[i][k];
            }
        }
    }
    for (int i = 0; i < num_classes * num_features; i++) {
        grad[i] /= n;
    }
    return grad;
}

// метод градиентного спуска
void gradient_descent(vector<vector<double>>& train_data, vector<int>& train_labels, vector<double>& weights, double alpha, double tol, int max_iter) {
    int num_features = train_data[0].size();
    int num_classes = 3;
    int n_train = train_data.size();
    int iter = 0;
    double prev_loss = loss_function(train_data, train_labels, weights);
    while (true) {
        vector<double> grad = gradient(train_data, train_labels, weights);
        for (int i = 0; i < num_classes * num_features; i++) {
            weights[i] -= alpha * grad[i];
        }
        double loss = loss_function(train_data, train_labels, weights);
        if (fabs(loss - prev_loss) < tol || iter >= max_iter) {
            cout << "Finished after " << iter << " iterations with loss = " << loss << endl;
            break;
        }
        prev_loss = loss;
        iter++;
    }
}

int main() {
    // тренировочные данные
    vector<vector<double>> train_data = { {11, 41, 8, 22, 26, 24, 11},
                                          {16, 41, 9, 32, 24, 27, 13},
                                          {14, 42, 7, 28, 20, 22, 19},
                                          {19, 42, 10, 38, 20, 30, 16},
                                          {18, 44, 8, 36, 20, 23, 12},
                                          {11, 44, 9, 22, 20, 27, 11},
                                          {15, 44, 11, 30, 28, 32, 17},
                                          {19, 45, 10, 38, 28, 30, 18},
                                          {12, 46, 8, 24, 22, 25, 13},
                                          {12, 47, 7, 24, 24, 22, 11},
                                          {13, 47, 9, 26, 26, 28, 15},
                                          {17, 47, 10, 34, 24, 31, 16},
                                          {19, 48, 8, 38, 22, 25, 19},
                                          {11, 48, 10, 22, 20, 30, 17},
                                          {14, 49, 9, 28, 26, 28, 12},
                                          {19, 30, 16, 38, 22, 48, 13},
                                          {18, 31, 14, 36, 26, 43, 11},
                                          {11, 31, 16, 22, 22, 49, 15},
                                          {15, 31, 17, 30, 28, 52, 16},
                                          {19, 32, 15, 38, 22, 46, 19},
                                          {16, 33, 15, 32, 20, 44, 17},
                                          {14, 34, 14, 28, 24, 42, 13},
                                          {19, 34, 16, 38, 26, 49, 19},
                                          {18, 34, 17, 36, 28, 51, 16},
                                          {11, 35, 16, 22, 20, 47, 12},
                                          {15, 36, 15, 30, 24, 44, 11},
                                          {19, 37, 16, 38, 20, 49, 17},
                                          {12, 38, 14, 24, 26, 41, 18},
                                          {11, 38, 16, 22, 26, 47, 11},
                                          {16, 39, 15, 32, 22, 45, 14},
                                          {14, 48, 17, 28, 24, 50, 17},
                                          {19, 50, 18, 38, 28, 55, 13},
                                          {18, 51, 17, 36, 22, 52, 19},
                                          {11, 52, 17, 22, 28, 50, 16},
                                          {15, 52, 19, 30, 28, 56, 12},
                                          {19, 52, 19, 38, 26, 58, 11},
                                          {12, 52, 20, 24, 22, 61, 17},
                                          {19, 54, 16, 38, 24, 48, 18},
                                          {16, 54, 18, 32, 24, 55, 11},
                                          {14, 55, 17, 28, 22, 52, 14},
                                          {19, 55, 20, 38, 28, 59, 15},
                                          {18, 56, 18, 36, 20, 55, 13},
                                          {11, 57, 16, 22, 24, 49, 19},
                                          {15, 57, 19, 30, 26, 57, 17},
                                          {16, 58, 18, 32, 20, 54, 12} };
    // метки классов тренировочных данных
    vector<int> train_labels = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    // начальные веса
    vector<double> weights(21, 0.0);
    // обучение модели
    gradient_descent(train_data, train_labels, weights, 0.0001, 0.0001, 10000);

    // тестовые данные
    vector<vector<double>> test_data = { {19, 42, 9, 38, 20, 27, 16},
                                         {16, 43, 8, 32, 26, 24, 19},
                                         {14, 43, 10, 28, 26, 30, 17},
                                         {19, 46, 8, 38, 22, 23, 12},
                                         {18, 46, 9, 36, 24, 28, 13},
                                         {11, 32, 16, 22, 28, 49, 11},
                                         {15, 34, 15, 30, 22, 46, 15},
                                         {19, 35, 14, 38, 28, 43, 16},
                                         {12, 35, 17, 24, 28, 50, 19},
                                         {11, 37, 15, 22, 26, 46, 17},
                                         {16, 51, 18, 32, 22, 54, 13},
                                         {14, 53, 17, 28, 24, 52, 19},
                                         {19, 53, 20, 38, 24, 60, 16},
                                         {18, 55, 19, 36, 22, 57, 12},
                                         {11, 57, 17, 22, 28, 51, 11} };
    // предсказание меток классов
    int n_test = test_data.size();
    for (int i = 0; i < n_test; i++) {
        vector<double> scores(3, 0.0);
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 7; k++) {
                scores[j] += test_data[i][k] * weights[j * 7 + k];
            }
        }
        int pred_label = max_element(scores.begin(), scores.end()) - scores.begin();
        cout << "Predicted label for test sample " << i + 1 << " is " << pred_label << endl;
    }
    return 0;
}

/*В этом примере мы решаем задачу многоклассовой классификации с тремя классами и семью признаками. 
Мы используем категориальную кросс-энтропию как функционал потерь и метод градиентного спуска для его оптимизации. 

Мы начинаем с определения функции потерь и градиента этой функции. 
Затем мы реализуем метод градиентного спуска, который итеративно обновляет параметры модели, пока не достигнут критерий сходимости.

Мы также проводим тестирование модели на новых данных, используя обученные веса. 
Мы вычисляем сумму взвешенных значений признаков для каждого класса и выбираем класс с наибольшим значением как предсказанную метку. 

Важно отметить, что в данном примере мы использовали фиксированные значения для некоторых параметров, 
таких как скорость обучения (alpha, равный 0.0001) и критерий остановки (tol, равный 0.0001). 
Эти значения должны быть подобраны оптимально для каждой конкретной задачи и могут оказать значительное влияние на результаты обучения.*/