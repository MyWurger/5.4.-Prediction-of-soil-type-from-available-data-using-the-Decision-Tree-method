#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

struct Vector
{
    vector<double> values; // вектор значений
    double humus;          // значение гумуса
};

// Вычисление энтропии вектора меток
//функция entropy рассчитывает энтропию для заданного набора меток, где энтропия измеряет степень неопределенности или разнородности меток в наборе.
double entropy(const vector<double>& labels)
{
    // содержит только уникальные метки из исходного набора labels.
    unordered_set<double> unique_labels(labels.begin(), labels.end());  // неупорядоченное множество 

    double entropy = 0.0;
    for (const auto& label : unique_labels) // цикл по всем уникальным меткам в unique_labels
    {
        double count_1 = count(labels.begin(), labels.end(), label); // переменная count_1 инициализируется количеством вхождений текущей метки в исходном наборе labels с помощью функции count().
        double p = count_1 / labels.size();                          // вероятность появления текущей метки
        entropy -= p * log2(p);                                      // Это выражение вычисляет вклад каждой уникальной метки в общую меру неопределенности системы.
    }
    return entropy;
}


// Вычисление информационного выигрыша от разделения объекта (прироста информации)
// Вычисляет прирост информации при разбиении данных на две части по определенному признаку (feature).
double information_gain(const vector<Vector>& data, int feature)
{
    double total_entropy = entropy(vector<double>(data.size(), 0.0)); // хранит энтропию (меру неопределенности) для всего набора данных.
    vector<double> values(data.size());                               // хранит значения признаков feature для каждого объекта в наборе данных data

    for (int i = 0; i < data.size(); i++)
    {
        values[i] = data[i].values[feature];
    }

    sort(values.begin(), values.end());   // сортировка значений по возростанию 
    double best_gain = INT_MIN;           // переменная для хранения значения максимального прироста информации 
    double best_split_value = 0.0;        // переменная для хранения значения, по которому будет происходить разбиение 


    // цикл по всем значениям вектора values, кроме последнего элемента, т.к. разбиение на две части возможно только при наличии хотя бы двух различных значений
    for (int i = 0; i < data.size() - 1; i++)
    {
        double split_value = (values[i] + values[i + 1]) / 2.0; // среднее значение между текущим и следующим значением в values
        vector<double> left_labels, right_labels;               // векторы, в которых будут хранится метки объектов, соответствующие левой и правой частям разбиения

        for (const auto& vector : data)               // в цикле по всем объектам в data осуществляется проверка значения признака feature для каждого объекта
        {
            if (vector.values[feature] < split_value) // если значение меньше split_value, то метка добавляется в левое разбиение, 
            {
                left_labels.push_back(vector.humus);
            }

            else //иначе - в правое разбиение.
            {
                right_labels.push_back(vector.humus);
            }
        }

        double left_entropy = entropy(left_labels);   // вычисление энтропии левого разбиения
        double right_entropy = entropy(right_labels); // вычисление энтропии правого разбиения

        // вычисление прироста информации как разница между total_entropy и суммарным вкладом левой и правой части разбиения в общую энтропию
        double gain = total_entropy - (left_labels.size() * left_entropy + right_labels.size() * right_entropy) / data.size();

        if (gain > best_gain) // если прирост информации больше текущего прироста, то
        {
            best_gain = gain;
            best_split_value = split_value;
        }
    }

    return best_gain; // возвращение наилучшего прироста информации, которое удалось найти при разбиении данных по признаку feature
}


// функция нахождения наилучшего признака для разделения данных на основе полученной информации
int find_best_feature(const vector<Vector>& data)
{
    int p = data[0].values.size(); // определение кол-ва признаков первого вектора 
    double best_gain = INT_MIN;    // переменная для хранения значения максимального прироста информации 
    int best_feature = -1;         // переменная наилучшего признака

    for (int j = 0; j < p; j++) // цикл по всем признакам
    {
        double gain = information_gain(data, j); // вызывается функция information_gain для вычисления информационного выигрыша для признака j в данных data

        if (gain > best_gain)  // если информационный выигрыш gain больше текущего наилучшего выигрыша best_gain, то
        {
            best_gain = gain;
            best_feature = j;
        }
    }

    return best_feature; // возвращение признака с наилучшим информационным выигрышем
}

// структура для построения дерева решений
struct Node
{
    int feature;        // переменная для хранения признака
    double split_value; // переменная для хранения среднего значения 
    double prediction;  // переменная для хранения предсказанного значения
    Node* left;         // указатель на левого потомка
    Node* right;        // указатель на правого потомка
};


// функция построения дерева решений на основе набора данных и глубины дерева
Node* build_tree(const vector<Vector>& data, int depth)
{
    if (depth == 0 || data.size() < 2) // если глубина равна 0 или размер данных меньше 2, то 
    {
        double prediction = 0.0; // переменная, которая хранит предсказанное значение

        for (const auto& vector : data)
        {
            prediction = prediction + vector.humus;
        }

        prediction = prediction / data.size();                          // Вычисление среднего значения humus для каждого элемента вектора data
        Node* node = new Node{ -1, 0.0, prediction, nullptr, nullptr }; // создание нового узла

        return node; // возвращение указателя на новой узел
    }

    int best_feature = find_best_feature(data); //вызывается функция find_best_feature, которая находит наилучший признак (best_feature) для разделения данных

    if (best_feature == -1)  // если не нашли науилучший признак
    {
        double prediction = 0.0; // переменная, которая хранит предсказанное значение

        for (const auto& vector : data)
        {
            prediction = prediction + vector.humus;
        }

        prediction = prediction / data.size();                          // Вычисление среднего значения humus для каждого элемента вектора data
        Node* node = new Node{ -1, 0.0, prediction, nullptr, nullptr }; // создание нового узла

        return node; // возвращение указателя на новой узел                                                    
    }

    vector<double> values(data.size());

    for (int i = 0; i < data.size(); i++) // в вектор values записываются значения соответствующего признака best_feature для каждого элемента вектора data
    {
        values[i] = data[i].values[best_feature];
    }

    sort(values.begin(), values.end()); // сортировка вектора по возрастанию

    double split_value = (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0; // вычисление значения для разделения данных (split_value)

    vector<Vector> left_data, right_data;

    for (const auto& vector : data) // цикл по каждому элементу вектора data
    {
        if (vector.values[best_feature] < split_value) // если значение меньше split_value, то данные добавляется в левое разбиение,
        {
            left_data.push_back(vector);
        }
        else // иначе в правое разбиение
        {
            right_data.push_back(vector);
        }
    }

    Node* node = new Node{ best_feature, split_value, 0.0, nullptr, nullptr }; // создание новго узла

    node->left = build_tree(left_data, depth - 1);   // вызывов build_tree для построения левого поддерева на основе данных left_data и глубины depth - 1
    node->right = build_tree(right_data, depth - 1); // вызывов build_tree для построения правого поддерева на основе данных right_data и глубины depth - 1

    return node; // Возвращается указатель на созданный узел node
}



// Прогнозирование значения гумуса для тестового вектора, используя обученного дерева принятия решений
// В результате выполнения функции predict будет возвращено предсказание, полученное для тестового набора данных на основе обученного дерева принятия решений.
double predict(Node* node, const vector<double>& test_data)
{
    if (node->left == nullptr && node->right == nullptr) // если у узла нет потомков, то 
    {
        return node->prediction;  // возвращаем предсказанное значение
    }

    //  если значение тестовых данных по признаку node->feature меньше значения split_value текущего узла, то
    if (test_data[node->feature] < node->split_value)
    {
        return predict(node->left, test_data); // рекурсивный вызов с левым дочерним узлом
    }

    else
    {
        return predict(node->right, test_data);  // рекурсивный вызов с правым дочерним узлом
    }
}



int main() {
    // Load the data from the file
    ifstream file;
    file.open("data.txt");
    if (!file.is_open())
    {
        return -1;
    }
    int n = 70;
    int m = 8;
    vector<Vector> data(n);

    for (int i = 0; i < n; i++)
    {
        data[i].values.resize(m);

        for (int j = 0; j < m; j++)
        {
            file >> data[i].values[j];
        }

        file >> data[i].humus;
    }

    file.close();

    // Build a decision tree
    Node* root = build_tree(data, 50);

    // Normalize the data
    vector < double>  test_data(8);

    // ВХОДНЫЕ ДАННЫЕ МЕНЯТЬ ЗДЕСЬ
    test_data = { 1373,	1464,	1773,	1674,	2253,	3548,	3812,	4122 };

    double predicted_humus = predict(root, test_data);
    cout << "Predicted humus value: " << predicted_humus << endl;

    return 0;
}